"""
============================================================
番茄 6D 位姿估计 V7.0 版 (Grasp-Safety-Oriented Final)
============================================================

【坐标系约定 - 严格遵守 RealSense D435i 官方文档】
    相机系 (color_optical_frame):
        X_c: 右 (+u)
        Y_c: 下 (+v)
        Z_c: 前 (沿光轴, 出镜头方向)

    番茄系 (V7.0):
        Z_t: 沿主轴, 从球心 → 果梗 (calyx)
              ★ 果梗朝上时 Z_t = (0, -1, 0) (相机 -Y 方向)
              ★ 蓝色坐标轴线视觉上指向果梗, 直觉直观
        X_t: 沿赤道, 从 left → right
        Y_t: 由右手系确定 Y_t = Z_t × X_t

【俯仰角 ψ 约定 - 抓取安全导向 (V7.0 创新)】
    ψ = arctan2(R[2,1], -R[2,2])   ← 与标准 ZYX 仅差一个 R[2,2] 取负号

    物理含义: ψ 直接编码"果梗远离相机/夹爪"的角度
        ψ =   0°  : 果梗朝相机镜头     (frontal grasp 危险)
        ψ = +90°  : 果梗竖直朝上 (初始)  ★ 自然生长姿态
        ψ = +180° : 果梗背向相机/朝远方  (frontal grasp 最安全)
        ψ = -90°  : 果梗竖直朝下        (倒置)

【为什么这个约定对抓取最有利】
    ψ 把"果梗远离夹爪的程度"转化为单个数值, 机械臂可以:
        1. |ψ| < 30°  → 果梗冲着夹爪, 正面进入危险, 应侧向环绕
        2. 60° < |ψ| < 120° → 果梗朝上下, 应从赤道侧向进入
        3. |ψ| > 150° → 屁股冲着夹爪, 正面直接抓取最安全
    这把"位姿估计输出"直接变成"抓取风险评估器", 跳过几何变换.

【V7.0 创新点 】
    1. DOGCF: 双阶几何线索融合 (球心径向法线 + 关键点差向量)
    2. Ray-Casting 三维重建: 用射线-球面交点替代深度图查询, 边缘免疫
    3. CCIA: 跨任务 IoU 仲裁实现实例分割与关键点的协同
    4. 抓取安全导向 ψ 约定: 用单一角度编码进入风险, 直驱抓取规划
    5. 多级时间滤波: 关键点 EMA + 球心 EMA + SLERP + Z 翻转防护
    6. 线性化 RANSAC 球面拟合: 提取真实 3D 球心而非 mask 重心
============================================================
"""
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO


# ==========================================
# 1. RANSAC 球面拟合
# ==========================================
class SphereFitter:
    """对 mask 内 3D 点云做 RANSAC 球面拟合.
    线性化球面方程: 2c·p + κ = ‖p‖², 其中 κ = ‖c‖² - r²"""

    def __init__(self, iterations=60, threshold=0.004,
                 min_radius=0.015, max_radius=0.075):
        self.iterations = iterations
        self.threshold = threshold
        self.min_r = min_radius
        self.max_r = max_radius
        self.rng = np.random.default_rng(42)

    def _linear_fit(self, pts):
        A = np.hstack([2 * pts, np.ones((len(pts), 1))])
        b = np.sum(pts ** 2, axis=1)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, None
        c = sol[:3]
        r2 = sol[3] + c @ c
        if r2 <= 0:
            return None, None
        return c, np.sqrt(r2)

    def fit(self, points_3d):
        n = len(points_3d)
        if n < 20:
            return None, None, 0.0, None

        best_in, best_c, best_r = 0, None, None
        for _ in range(self.iterations):
            idx = self.rng.choice(n, 4, replace=False)
            c, r = self._linear_fit(points_3d[idx])
            if c is None or not (self.min_r < r < self.max_r):
                continue
            dists = np.abs(np.linalg.norm(points_3d - c, axis=1) - r)
            inliers = int(np.sum(dists < self.threshold))
            if inliers > best_in:
                best_in, best_c, best_r = inliers, c, r

        if best_c is None:
            return None, None, 0.0, None

        dists = np.abs(np.linalg.norm(points_3d - best_c, axis=1) - best_r)
        mask = dists < self.threshold
        P = points_3d[mask]
        if len(P) >= 10:
            c, r = self._linear_fit(P)
            if c is not None and self.min_r < r < self.max_r:
                best_c, best_r = c, r
        return best_c, best_r, best_in / n, mask


# ==========================================
# 2. Ray-Sphere Intersection (核心创新)
# ==========================================
class RayCaster:
    """把 2D 像素通过射线-球面交点投影到 3D 球面上,
    完全绕开 D435 在物体边缘的不可靠深度."""

    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def pixel_to_sphere(self, u, v, sphere_c, sphere_r):
        # 像素 → 归一化相机坐标射线
        dx = (u - self.intrinsics.ppx) / self.intrinsics.fx
        dy = (v - self.intrinsics.ppy) / self.intrinsics.fy
        ray_dir = np.array([dx, dy, 1.0])
        ray_dir /= np.linalg.norm(ray_dir)

        # 射线-球面交点二次方程
        b = -2.0 * np.dot(ray_dir, sphere_c)
        c_term = np.dot(sphere_c, sphere_c) - sphere_r ** 2
        delta = b ** 2 - 4 * c_term

        if delta >= 0:
            t = (-b - np.sqrt(delta)) / 2.0
            if t > 0:
                return ray_dir * t

        # 兜底: 射线不交球面时, 取球面上离射线最近点
        t_perp = np.dot(sphere_c, ray_dir)
        p_perp = ray_dir * t_perp
        delta_vec = p_perp - sphere_c
        delta_vec /= (np.linalg.norm(delta_vec) + 1e-9)
        return sphere_c + delta_vec * sphere_r


# ==========================================
# 3. 关键点 EMA 平滑 (含跳变保护)
# ==========================================
class KeypointSmoother:
    def __init__(self, alpha=0.35, timeout=0.5, jump_threshold=0.04):
        self.alpha = alpha
        self.timeout = timeout
        self.jump_threshold = jump_threshold
        self.cache = {}

    def smooth(self, track_id, kpt_name, pos_3d):
        now = time.time()
        key = (track_id, kpt_name)

        for k in list(self.cache.keys()):
            if now - self.cache[k]['t'] > self.timeout:
                del self.cache[k]

        pos_now = np.array(pos_3d, dtype=float)
        if key not in self.cache:
            self.cache[key] = {'pos': pos_now, 't': now}
            return pos_now

        prev = self.cache[key]
        jump = np.linalg.norm(pos_now - prev['pos'])
        eff_alpha = self.alpha if jump <= self.jump_threshold else self.alpha * 0.3
        new_pos = eff_alpha * pos_now + (1 - eff_alpha) * prev['pos']
        self.cache[key] = {'pos': new_pos, 't': now}
        return new_pos


# ==========================================
# 4. 球心 EMA 平滑
# ==========================================
class SphereCenterSmoother:
    def __init__(self, alpha=0.30, timeout=0.5, jump_threshold=0.03):
        self.alpha = alpha
        self.timeout = timeout
        self.jump_threshold = jump_threshold
        self.cache = {}

    def smooth(self, track_id, sphere_c, sphere_r):
        now = time.time()
        for k in list(self.cache.keys()):
            if now - self.cache[k]['t'] > self.timeout:
                del self.cache[k]

        c_now = np.array(sphere_c, dtype=float)
        if track_id not in self.cache:
            self.cache[track_id] = {'c': c_now, 'r': sphere_r, 't': now}
            return c_now, sphere_r

        prev = self.cache[track_id]
        jump = np.linalg.norm(c_now - prev['c'])
        eff_alpha = self.alpha if jump <= self.jump_threshold else self.alpha * 0.3
        c_s = eff_alpha * c_now + (1 - eff_alpha) * prev['c']
        r_s = eff_alpha * sphere_r + (1 - eff_alpha) * prev['r']
        self.cache[track_id] = {'c': c_s, 'r': r_s, 't': now}
        return c_s, r_s


# ==========================================
# 5. 几何主轴解算器 (DOGCF 简化版)
# ==========================================
class GeometricAxisSolver:
    """主轴 Z_visual = 球心 → 果梗"""

    def solve(self, sphere_center, sphere_radius,
              kpts_3d_sphere, kpts_conf):
        c = np.array(sphere_center)
        strategy = ""
        z_visual = None

        calyx = kpts_3d_sphere.get("tomato_calyx")
        bottom = kpts_3d_sphere.get("tomato_bottom")
        left = kpts_3d_sphere.get("tomato_left")
        right = kpts_3d_sphere.get("tomato_right")
        c_calyx = kpts_conf.get("tomato_calyx", 0.0)
        c_bottom = kpts_conf.get("tomato_bottom", 0.0)

        # ---------- Z 轴推导 ----------
        if calyx is not None and bottom is not None and \
                c_calyx > 0.5 and c_bottom > 0.5:
            v_diff = np.array(calyx) - np.array(bottom)
            v_diff_norm = v_diff / (np.linalg.norm(v_diff) + 1e-9)
            v_radial = np.array(calyx) - c
            v_radial_norm = v_radial / (np.linalg.norm(v_radial) + 1e-9)
            z_visual = 0.6 * v_diff_norm + 0.4 * v_radial_norm
            strategy = "P1: diff+radial"
        elif calyx is not None and c_calyx > 0.45:
            z_visual = np.array(calyx) - c
            strategy = "P2: center->calyx"
        elif bottom is not None and c_bottom > 0.45:
            z_visual = -(np.array(bottom) - c)
            strategy = "P3: -(center->bottom)"
        else:
            z_visual = np.array([0.0, -1.0, 0.0])
            strategy = "P4: gravity prior"

        z_visual /= (np.linalg.norm(z_visual) + 1e-9)

        # ---------- X 轴推导 ----------
        if left is not None and right is not None:
            x_visual = np.array(right) - np.array(left)
        elif left is not None:
            x_visual = c - np.array(left)
        elif right is not None:
            x_visual = np.array(right) - c
        else:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(z_visual, ref)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0])
            x_visual = ref - np.dot(ref, z_visual) * z_visual

        x_visual /= (np.linalg.norm(x_visual) + 1e-9)
        x_visual = x_visual - np.dot(x_visual, z_visual) * z_visual
        x_visual /= (np.linalg.norm(x_visual) + 1e-9)

        y_visual = np.cross(z_visual, x_visual)
        y_visual /= (np.linalg.norm(y_visual) + 1e-9)

        x_visual = np.cross(y_visual, z_visual)
        x_visual /= (np.linalg.norm(x_visual) + 1e-9)

        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x_visual, y_visual, z_visual, c
        return T, strategy


# ==========================================
# 6. 四元数 + SLERP 时间滤波
# ==========================================
class PoseTemporalFilter:
    def __init__(self, alpha_pos=0.25, alpha_rot=0.12, timeout=0.8):
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.timeout = timeout
        self.tracks = {}

    @staticmethod
    def rot_to_quat(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                             0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                             (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    @staticmethod
    def quat_to_rot(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])

    @staticmethod
    def slerp(q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / np.linalg.norm(q)
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_t0 = np.sin(theta_0)
        theta = theta_0 * t
        s0 = np.cos(theta) - dot * np.sin(theta) / sin_t0
        s1 = np.sin(theta) / sin_t0
        return s0 * q0 + s1 * q1

    def smooth(self, track_id, T_new):
        now = time.time()
        pos_new = T_new[:3, 3].copy()
        quat_new = self.rot_to_quat(T_new[:3, :3])

        expired = [k for k, v in self.tracks.items() if now - v["t"] > self.timeout]
        for k in expired:
            del self.tracks[k]

        if track_id not in self.tracks:
            self.tracks[track_id] = {"pos": pos_new, "quat": quat_new, "t": now}
            return T_new

        prev = self.tracks[track_id]
        pos_s = self.alpha_pos * pos_new + (1 - self.alpha_pos) * prev["pos"]
        quat_s = self.slerp(prev["quat"], quat_new, self.alpha_rot)
        quat_s /= (np.linalg.norm(quat_s) + 1e-9)

        self.tracks[track_id] = {"pos": pos_s, "quat": quat_s, "t": now}
        T_s = np.eye(4)
        T_s[:3, :3] = self.quat_to_rot(quat_s)
        T_s[:3, 3] = pos_s
        return T_s


# ==========================================
# 7. 实时流水线 V7.0
# ==========================================
class RealSenseTomato6DAppV7_0:
    """V7.0: 抓取安全导向 ψ 约定 + Ray-Casting + 多级稳定化"""

    def __init__(self, seg_weights, pose_weights,
                 color_w=1280, color_h=720, fps=30):
        print("正在加载 YOLO 双模型 (V7.0: grasp-safety-oriented)...")
        self.seg_model = YOLO(seg_weights)
        self.pose_model = YOLO(pose_weights)
        self.sphere_fitter = SphereFitter()
        self.solver = GeometricAxisSolver()
        self.kpt_smoother = KeypointSmoother(alpha=0.35, jump_threshold=0.04)
        self.center_smoother = SphereCenterSmoother(alpha=0.30, jump_threshold=0.03)
        self.tfilter = PoseTemporalFilter(alpha_pos=0.25, alpha_rot=0.12)
        self.last_z_visual = {}

        self.kpt_names = ["tomato_calyx", "tomato_bottom", "tomato_left", "tomato_right"]
        self.target_class_name = "ripe_tomato"

        self.color_w, self.color_h = color_w, color_h
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.K = None
        self.raycaster = None

    def start_camera(self):
        print(f"正在连接 RealSense D435 ({self.color_w}x{self.color_h}@{self.fps}) ...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.color_w, self.color_h, rs.format.z16, self.fps)
        profile = self.pipeline.start(config)

        self.intrinsics = profile.get_stream(rs.stream.color)\
            .as_video_stream_profile().get_intrinsics()
        self.K = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                           [0, self.intrinsics.fy, self.intrinsics.ppy],
                           [0, 0, 1]], dtype=np.float32)
        self.raycaster = RayCaster(self.intrinsics)
        self.align = rs.align(rs.stream.color)
        print(f"✅ 相机启动: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}")

    @staticmethod
    def calculate_iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / float(a1 + a2 - inter + 1e-6)

    def collect_mask_pointcloud(self, depth_array, mask_binary, orig_shape):
        mask_resized = cv2.resize(mask_binary, (orig_shape[1], orig_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        v_idx, u_idx = np.where(mask_resized > 0)
        if len(v_idx) == 0:
            return None, mask_resized

        z = depth_array[v_idx, u_idx] * 0.001
        valid = (z > 0.1) & (z < 1.0)
        if valid.sum() < 30:
            return None, mask_resized

        u_v, v_v, z_v = u_idx[valid], v_idx[valid], z[valid]
        x_v = (u_v - self.intrinsics.ppx) * z_v / self.intrinsics.fx
        y_v = (v_v - self.intrinsics.ppy) * z_v / self.intrinsics.fy
        return np.stack([x_v, y_v, z_v], axis=1), mask_resized

    @staticmethod
    def get_euler_angles_grasp(R):
        """
        【V7.0 创新核心】 抓取安全导向欧拉角分解.

        与标准 ZYX 仅有一个差别: ψ 公式中 R[2,2] 取负号.
            标准 ZYX:    ψ = atan2(R[2,1],  R[2,2])
            V7.0 抓取版: ψ = atan2(R[2,1], -R[2,2])

        物理意义:
            ψ = arctan2(sin(180°-ψ_std),  cos(180°-ψ_std)) = 180° - ψ_std
            等价于: 把"果梗朝相机"作为参考零位 (calyx-cam = 0°)
                   而非"果梗背向相机" (calyx-away = 0° 在标准 ZYX 中)

        验证 (θ=φ=0 简化情形):
            果梗朝相机    Z=(0,0,-1):  ψ = atan2(0, +1) =  0°  ✓
            果梗朝上      Z=(0,-1,0):  ψ = atan2(1, 0)  = +90° ✓
            果梗朝场景深处 Z=(0,0,+1):  ψ = atan2(0, -1) = ±180° ✓
            果梗朝下      Z=(0,+1,0):  ψ = atan2(-1, 0) = -90° ✓
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            psi   = np.arctan2(R[2, 1], -R[2, 2])     # ★V7.0 改动★
            phi   = np.arctan2(-R[2, 0], sy)
            theta = np.arctan2(R[1, 0], R[0, 0])
        else:
            psi   = np.arctan2(-R[1, 2], R[1, 1])
            phi   = np.arctan2(-R[2, 0], sy)
            theta = 0.0
        return np.degrees([theta, phi, psi])

    def stabilize_z_flip(self, track_id, T):
        z_now = T[:3, 2].copy()
        if track_id in self.last_z_visual:
            z_last = self.last_z_visual[track_id]
            if np.dot(z_now, z_last) < -0.3:
                T = T.copy()
                T[:3, 1] = -T[:3, 1]
                T[:3, 2] = -T[:3, 2]
                z_now = T[:3, 2].copy()
        self.last_z_visual[track_id] = z_now
        return T

    def draw_3d_axes(self, img, T, length=0.05):
        pts = np.float32([[0, 0, 0],
                          [length, 0, 0],
                          [0, length, 0],
                          [0, 0, length]]).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        imgpts, _ = cv2.projectPoints(pts, rvec, T[:3, 3], self.K, np.zeros(5))
        imgpts = imgpts.astype(int).squeeze()
        o = tuple(imgpts[0])
        cv2.line(img, o, tuple(imgpts[1]), (0, 0, 255), 4)    # X 红
        cv2.line(img, o, tuple(imgpts[2]), (0, 255, 0), 4)    # Y 绿
        cv2.line(img, o, tuple(imgpts[3]), (255, 0, 0), 5)    # Z 蓝→果梗

        cv2.putText(img, "X", tuple(imgpts[1] + np.array([6, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(img, "Y", tuple(imgpts[2] + np.array([6, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(img, "Z->Calyx", tuple(imgpts[3] + np.array([6, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 50), 2)

    @staticmethod
    def grasp_safety_advice(psi_deg):
        """根据 ψ 给出抓取安全建议"""
        a = abs(psi_deg)
        if a > 150:
            return "FRONTAL OK", (0, 255, 0)        # 屁股朝相机, 正面安全
        elif 60 < a < 120:
            return "SIDE GRASP", (0, 255, 255)      # 果梗朝上下, 赤道侧向
        elif a < 30:
            return "AVOID FRONT", (0, 100, 255)     # 果梗朝相机, 危险
        else:
            return "TILT GRASP", (180, 180, 180)    # 中间状态

    def draw_legend(self, img, fps):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 96), (0, 0, 0), -1)
        cv2.putText(img, f"Tomato 6D Pose V7.0 | FPS: {fps:.1f}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(img,
                    "Init pose (calyx-up, natural growth): (theta, phi, psi) = (0, 0, +90)",
                    (15, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
        cv2.putText(img,
                    "psi: 0 = calyx->cam (DANGER) | +90 = up | +-180 = calyx-away (SAFE)",
                    (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 220), 1)

    def get_pose_for_ros(self, T):
        """输出 ROS 兼容位姿信息"""
        pos = T[:3, 3]
        R = T[:3, :3]
        q_wxyz = PoseTemporalFilter.rot_to_quat(R)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler = self.get_euler_angles_grasp(R)

        # 抓取建议
        psi = euler[2]
        advice, _ = self.grasp_safety_advice(psi)

        return {
            "position_m": pos.tolist(),
            "rotation_matrix": R.tolist(),
            "quaternion_xyzw": q_xyzw.tolist(),
            "euler_zyx_deg": euler.tolist(),
            "grasp_advice": advice,
        }

    def run(self):
        self.start_camera()
        win_name = 'Tomato 6D Pose V7.0 (Grasp-Safety-Oriented)'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self.color_w, self.color_h)
        print("🎥 按 Q/ESC 退出, P 打印位姿(供 ROS)")
        prev_t = time.time()
        last_poses = {}

        try:
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(5000)
                except RuntimeError as e:
                    print(f"⚠️ 相机超时: {e}")
                    continue

                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_array = np.asanyarray(depth_frame.get_data())
                orig_shape = color_image.shape

                seg = self.seg_model.predict(color_image, conf=0.5,
                                             retina_masks=True, verbose=False)[0]
                pose = self.pose_model.predict(color_image, conf=0.5, verbose=False)[0]

                seg_boxes = seg.boxes.xyxy.cpu().numpy() if seg.boxes else []
                seg_masks = seg.masks.data.cpu().numpy() if seg.masks else []
                pose_boxes = pose.boxes.xyxy.cpu().numpy() if pose.boxes else []
                pose_kpts = pose.keypoints.data.cpu().numpy() if pose.keypoints else []

                for p_idx, p_box in enumerate(pose_boxes):
                    best_iou, best_s = 0, -1
                    for s_idx, s_box in enumerate(seg_boxes):
                        iou = self.calculate_iou(p_box, s_box)
                        if iou > best_iou:
                            best_iou, best_s = iou, s_idx
                    if best_iou < 0.45 or best_s == -1:
                        continue

                    target_mask = seg_masks[best_s]

                    pts_3d, resized_mask = self.collect_mask_pointcloud(
                        depth_array, target_mask, orig_shape)
                    if pts_3d is None:
                        continue

                    sphere_c_raw, sphere_r_raw, inlier_ratio, _ = \
                        self.sphere_fitter.fit(pts_3d)
                    if sphere_c_raw is None:
                        continue

                    sphere_c, sphere_r = self.center_smoother.smooth(
                        p_idx, sphere_c_raw, sphere_r_raw)

                    kpts_3d_sphere = {}
                    kpts_2d = {}
                    kpts_conf = {}
                    for i in range(4):
                        u, v, conf = pose_kpts[p_idx][i]
                        if conf > 0.45:
                            pos_3d = self.raycaster.pixel_to_sphere(
                                int(u), int(v), sphere_c, sphere_r)
                            pos_smoothed = self.kpt_smoother.smooth(
                                p_idx, self.kpt_names[i], pos_3d)
                            kpts_3d_sphere[self.kpt_names[i]] = pos_smoothed
                            kpts_2d[self.kpt_names[i]] = (int(u), int(v))
                            kpts_conf[self.kpt_names[i]] = float(conf)
                            cv2.circle(color_image, (int(u), int(v)),
                                       6, (0, 255, 255), -1)

                    T, strategy = self.solver.solve(
                        sphere_c, sphere_r, kpts_3d_sphere, kpts_conf)
                    T = self.stabilize_z_flip(p_idx, T)
                    T = self.tfilter.smooth(p_idx, T)
                    last_poses[p_idx] = T.copy()

                    bx1, by1, bx2, by2 = map(int, p_box)
                    cm = np.zeros_like(color_image, dtype=np.uint8)
                    cm[resized_mask > 0] = [0, 0, 200]
                    cv2.addWeighted(cm, 0.30, color_image, 1.0, 0, color_image)

                    if T[2, 3] > 1e-6:
                        u_c = int(T[0, 3] * self.intrinsics.fx / T[2, 3] + self.intrinsics.ppx)
                        v_c = int(T[1, 3] * self.intrinsics.fy / T[2, 3] + self.intrinsics.ppy)
                        cv2.circle(color_image, (u_c, v_c), 7, (255, 255, 255), -1)
                        cv2.circle(color_image, (u_c, v_c), 8, (0, 0, 0), 1)

                    cv2.rectangle(color_image, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    self.draw_3d_axes(color_image, T)

                    t_x, t_y, t_z = T[0, 3], T[1, 3], T[2, 3]
                    yaw, roll, pitch = self.get_euler_angles_grasp(T[:3, :3])
                    advice, advice_color = self.grasp_safety_advice(pitch)

                    label = f"{self.target_class_name} | r={sphere_r * 100:.1f}cm | inlier={inlier_ratio:.2f}"
                    info1 = f"P(m): [{t_x:+.3f}, {t_y:+.3f}, {t_z:+.3f}]  | {strategy}"
                    info2 = f"theta={yaw:+6.1f}  phi={roll:+6.1f}  psi={pitch:+6.1f}  (deg)"
                    info3 = f"GRASP: {advice}"

                    panel_w, panel_h = 700, 102
                    by1_panel = max(98, by1 - panel_h)
                    cv2.rectangle(color_image, (bx1, by1_panel),
                                  (bx1 + panel_w, by1), (0, 0, 0), -1)
                    cv2.putText(color_image, label, (bx1 + 8, by1_panel + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    cv2.putText(color_image, info1, (bx1 + 8, by1_panel + 47),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.putText(color_image, info2, (bx1 + 8, by1_panel + 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(color_image, info3, (bx1 + 8, by1_panel + 96),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, advice_color, 2)

                cur_t = time.time()
                fps = 1 / max(cur_t - prev_t, 1e-6)
                prev_t = cur_t
                self.draw_legend(color_image, fps)

                cv2.imshow(win_name, color_image)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    break
                elif key & 0xFF == ord('p'):
                    print("\n=== 当前位姿 (V7.0 抓取安全导向) ===")
                    for tid, Tp in last_poses.items():
                        info = self.get_pose_for_ros(Tp)
                        print(f"track {tid}:")
                        print(f"  position(m)        = {info['position_m']}")
                        print(f"  euler ZYX(deg)     = {info['euler_zyx_deg']}")
                        print(f"  quaternion (xyzw)  = {info['quaternion_xyzw']}")
                        print(f"  grasp advice       = {info['grasp_advice']}")
        finally:
            print("正在安全关闭相机 ...")
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    SEG_WEIGHTS = r"D:\code\Tomato_Grasp_Project\2026\20260321图像分割重做\runs\segmentation\train_yolo11s_seg_v1\weights\best.pt"
    POSE_WEIGHTS = r"D:\code\Tomato_Grasp_Project\2026\20260318关键点训练\Tomato_Pose_Runs\train_pose_0412\weights\best.pt"

    app = RealSenseTomato6DAppV7_0(
        SEG_WEIGHTS, POSE_WEIGHTS,
        color_w=1280, color_h=720,
        fps=30
    )
    app.run()

