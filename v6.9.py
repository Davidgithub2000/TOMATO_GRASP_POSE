"""
============================================================
番茄 6D 位姿估计 V6.6 版 (终极修复版)
============================================================

【坐标系约定 - 严格遵守 RealSense D435i 官方文档】
    相机系 (color_optical_frame):
        X_c: 右 (+u)
        Y_c: 下 (+v)
        Z_c: 前 (沿光轴, 出镜头方向)

    番茄系 (V6.6 直觉化定义):
        Z_t: 沿主轴, 从球心 → 果梗 (calyx)
              ★ 果梗朝上时 Z_t = (0, -1, 0) (相机 -Y 方向)
              ★ 蓝色坐标轴线视觉上指向果梗, 符合直觉
              ★ 与 Google V6.7 思路一致
        X_t: 沿赤道, 从 left → right
        Y_t: 由右手系确定 Y_t = Z_t × X_t

    旋转: R_visual ∈ SO(3), 列分别为 [X_t | Y_t | Z_t]

【欧拉角约定 - 与师兄 (0,0,-90)初始角对齐】
    标准 ZYX 顺序:  R_euler = R_z(θ) · R_y(φ) · R_x(ψ)
    但 R_visual 直接做 ZYX 分解会得到 +90°(果梗朝上时), 不符师兄约定.
    引入欧拉计算矩阵:
        R_euler = diag(1, -1, -1) · R_visual
    再对 R_euler 做 ZYX 分解, 得 ψ = -90° (果梗朝上).

【ψ 物理含义(绕相机系 X_c 轴旋转)】
    ψ = -90°  : 果梗竖直朝上 (初始, 自然生长姿态) ★
    ψ =   0°  : 果梗朝相机   (番茄水平, calyx 面向镜头)
    ψ = +90°  : 果梗竖直朝下 (完全倒置)
    ψ = ±180° : 果梗背向相机 (番茄水平, calyx 远离镜头)

【V6.6 相对 V6.5 的关键修复】
    1. ★禁用 EPC★ — 番茄是扁球(赤道直径 > 主轴长), mask 长轴水平、
                    主轴投影垂直, EPC 假设反向, 会产生 80°+ 系统误差.
    2. ★Ray-Casting 3D 重建★ — 关键点 3D 位置不再依赖深度图边缘
                    (深度边缘多飞点), 改用射线-球面交点几何法.
    3. ★双 R 机制★ — R_visual 用于绘制坐标轴, R_euler 用于角度输出.
                    满足"蓝线朝果梗 + ψ=-90"两个约束.
    4. ★强稳定化★ — 关键点 EMA(α=0.35) + 球心 EMA(α=0.30)
                    + SLERP(α=0.12) + Z 翻转防护.
    5. ★主轴方向二阶段融合★ — 球心+果梗驱动(强), 关键点差向量(中).
                    底部仅作兜底, 不作主驱动.

【番茄球心(白点)的算法流程】
    Step 1: YOLO 实例分割得到 mask
    Step 2: 反投影 mask 内有效深度像素 → 3D 点云 (~1k-5k 点)
    Step 3: RANSAC 球面拟合 (4-点最小拟合, 60 次迭代, 4mm 内点阈值)
    Step 4: 全部内点最小二乘精修
    Step 5: 跨帧 EMA(alpha=0.30) + 跳变保护
    输出: 真实 3D 球心 — 不是 mask 2D 重心, 不是 bbox 中心
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

        # 内点全部参与最终精修
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
    """将 2D 像素点通过射线-球面交点投影到 3D 球面上.
    避开了深度图在物体边缘的不可靠数据."""

    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def pixel_to_sphere(self, u, v, sphere_c, sphere_r):
        """
        相机原点出发, 经过像素 (u,v) 的射线与球面的交点.
        返回靠近相机的交点 (球的"前表面").
        """
        # 像素 → 归一化相机坐标射线方向 (相机原点为起点)
        dx = (u - self.intrinsics.ppx) / self.intrinsics.fx
        dy = (v - self.intrinsics.ppy) / self.intrinsics.fy
        ray_dir = np.array([dx, dy, 1.0])
        ray_dir /= np.linalg.norm(ray_dir)

        # 球面方程: ‖t·d - c‖² = r², 即 t² - 2(d·c)t + (c·c - r²) = 0
        b = -2.0 * np.dot(ray_dir, sphere_c)
        c_term = np.dot(sphere_c, sphere_c) - sphere_r ** 2
        delta = b ** 2 - 4 * c_term

        if delta >= 0:
            t = (-b - np.sqrt(delta)) / 2.0   # 取靠近相机的根
            if t > 0:
                return ray_dir * t

        # 兜底: 射线不与球相交时, 取球面上离射线最近的点
        # (像素恰好在 mask 边缘外一点点的情况)
        t_perp = np.dot(sphere_c, ray_dir)
        p_perp = ray_dir * t_perp
        delta_vec = p_perp - sphere_c
        delta_vec /= (np.linalg.norm(delta_vec) + 1e-9)
        return sphere_c + delta_vec * sphere_r


# ==========================================
# 3. 关键点 EMA 平滑
# ==========================================
class KeypointSmoother:
    """每个 (track_id, kpt_name) 独立 EMA, 跳变保护."""
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
        eff_alpha = self.alpha
        if jump > self.jump_threshold:
            eff_alpha = self.alpha * 0.3   # 大跳变样本权重压低

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
        eff_alpha = self.alpha
        if jump > self.jump_threshold:
            eff_alpha = self.alpha * 0.3

        c_s = eff_alpha * c_now + (1 - eff_alpha) * prev['c']
        r_s = eff_alpha * sphere_r + (1 - eff_alpha) * prev['r']
        self.cache[track_id] = {'c': c_s, 'r': r_s, 't': now}
        return c_s, r_s


# ==========================================
# 5. 几何主轴解算器 (V6.6 简化, 无 EPC)
# ==========================================
class GeometricAxisSolver:
    """
    主轴 Z_visual 推导策略 (按可见性优先级):
      P1) calyx + bottom 都可见 → 用差向量 + 球心校正
      P2) 仅 calyx 可见 → 球心 → calyx 方向
      P3) 仅 bottom 可见 → -(球心 → bottom) 方向
      P4) 都不可见 → 重力先验 (calyx-up 方向)
    """

    def solve(self, sphere_center, sphere_radius,
              kpts_3d_sphere, kpts_conf):
        c = np.array(sphere_center)
        strategy = ""
        z_visual = None       # 球心 → 果梗方向

        calyx = kpts_3d_sphere.get("tomato_calyx")
        bottom = kpts_3d_sphere.get("tomato_bottom")
        left = kpts_3d_sphere.get("tomato_left")
        right = kpts_3d_sphere.get("tomato_right")
        c_calyx = kpts_conf.get("tomato_calyx", 0.0)
        c_bottom = kpts_conf.get("tomato_bottom", 0.0)

        # ---------- Z 轴推导 ----------
        if calyx is not None and bottom is not None and \
                c_calyx > 0.5 and c_bottom > 0.5:
            # P1: 双端关键点融合 + 球心约束
            v_diff = np.array(calyx) - np.array(bottom)
            v_diff_norm = v_diff / (np.linalg.norm(v_diff) + 1e-9)
            v_radial = np.array(calyx) - c
            v_radial_norm = v_radial / (np.linalg.norm(v_radial) + 1e-9)
            # 双源加权融合 (差向量权重 0.6, 径向权重 0.4)
            z_visual = 0.6 * v_diff_norm + 0.4 * v_radial_norm
            strategy = "P1: diff+radial"
        elif calyx is not None and c_calyx > 0.45:
            # P2: 仅 calyx, 球心 → calyx
            z_visual = np.array(calyx) - c
            strategy = "P2: center->calyx"
        elif bottom is not None and c_bottom > 0.45:
            # P3: 仅 bottom, 反向
            z_visual = -(np.array(bottom) - c)
            strategy = "P3: -(center->bottom)"
        else:
            # P4: 重力先验, 假设果梗朝上
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
            # 兜底: 选与 z 不平行的相机 X 或 Z
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(z_visual, ref)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0])
            x_visual = ref - np.dot(ref, z_visual) * z_visual

        x_visual /= (np.linalg.norm(x_visual) + 1e-9)

        # 用 Gram-Schmidt 正交化
        # X 应当垂直于 Z, 否则强制正交化
        x_visual = x_visual - np.dot(x_visual, z_visual) * z_visual
        x_visual /= (np.linalg.norm(x_visual) + 1e-9)

        # Y = Z × X (右手系)
        y_visual = np.cross(z_visual, x_visual)
        y_visual /= (np.linalg.norm(y_visual) + 1e-9)

        # 确保正交 (重新算 X 防数值漂移)
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
# 7. 实时流水线 V6.6
# ==========================================
class RealSenseTomato6DAppV6_6:
    """
    番茄系约定: Z_visual = 球心 → 果梗 (蓝色轴朝上 当果梗朝上时)
    欧拉角约定: ψ = -90° 当果梗朝上 (与师兄约定一致)
    """

    # 欧拉角变换矩阵: 用于让"蓝线朝果梗"和"ψ=-90 当果梗朝上"同时成立
    # 数学验证: R_visual(果梗朝上) = [[1,0,0],[0,0,-1],[0,1,0]]
    #          M @ R_visual = [[1,0,0],[0,0,1],[0,-1,0]]
    #          ZYX 分解后 ψ = atan2(-1, 0) = -π/2 = -90°  ✓
    M_EULER = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ], dtype=np.float64)

    def __init__(self, seg_weights, pose_weights,
                 color_w=1280, color_h=720, fps=30):
        print("正在加载 YOLO 双模型 (V6.6: ray-cast + no-EPC + dual-R)...")
        self.seg_model = YOLO(seg_weights)
        self.pose_model = YOLO(pose_weights)
        self.sphere_fitter = SphereFitter()
        self.solver = GeometricAxisSolver()
        self.kpt_smoother = KeypointSmoother(alpha=0.35, jump_threshold=0.04)
        self.center_smoother = SphereCenterSmoother(alpha=0.30, jump_threshold=0.03)
        self.tfilter = PoseTemporalFilter(alpha_pos=0.25, alpha_rot=0.12)
        self.last_z_visual = {}     # Z 翻转防护

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

    def get_euler_angles_zyx(self, R_visual):
        """
        【V6.6 双 R 机制核心】
            R_euler = M_EULER · R_visual   (M = diag(1, -1, -1))
        然后对 R_euler 做标准 ZYX 分解, 返回 [θ, φ, ψ] (deg).

        验证: R_visual(果梗朝上) = [[1,0,0],[0,0,-1],[0,1,0]]
              R_euler = M·R_visual = [[1,0,0],[0,0,1],[0,-1,0]]
              R[2,0] = 0   → φ = 0
              R[2,1] = -1, R[2,2] = 0 → ψ = atan2(-1, 0) = -π/2 = -90°  ✓
              R[1,0] = 0,  R[0,0] = 1 → θ = atan2(0, 1) = 0   ✓
        """
        R = self.M_EULER @ R_visual
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            psi = np.arctan2(R[2, 1], R[2, 2])
            phi = np.arctan2(-R[2, 0], sy)
            theta = np.arctan2(R[1, 0], R[0, 0])
        else:
            psi = np.arctan2(-R[1, 2], R[1, 1])
            phi = np.arctan2(-R[2, 0], sy)
            theta = 0.0
        return np.degrees([theta, phi, psi])

    def stabilize_z_flip(self, track_id, T):
        """Z 翻转防护: 若 Z 与上一帧反向, 整体翻转 Y/Z (保右手系)"""
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
        """绘制番茄系坐标轴 (R_visual: Z 朝果梗)"""
        pts = np.float32([[0, 0, 0],
                          [length, 0, 0],
                          [0, length, 0],
                          [0, 0, length]]).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        imgpts, _ = cv2.projectPoints(pts, rvec, T[:3, 3], self.K, np.zeros(5))
        imgpts = imgpts.astype(int).squeeze()
        o = tuple(imgpts[0])
        cv2.line(img, o, tuple(imgpts[1]), (0, 0, 255), 4)    # X 红 (right)
        cv2.line(img, o, tuple(imgpts[2]), (0, 255, 0), 4)    # Y 绿
        cv2.line(img, o, tuple(imgpts[3]), (255, 0, 0), 5)    # Z 蓝 (toward calyx)

        cv2.putText(img, "X", tuple(imgpts[1] + np.array([6, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(img, "Y", tuple(imgpts[2] + np.array([6, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(img, "Z->Calyx", tuple(imgpts[3] + np.array([6, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 50), 2)

    def draw_legend(self, img, fps):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 96), (0, 0, 0), -1)
        cv2.putText(img, f"Tomato 6D Pose V6.6 | FPS: {fps:.1f}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(img,
                    "Init pose (calyx-up, natural): (theta, phi, psi) = (0, 0, -90)",
                    (15, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
        cv2.putText(img,
                    "psi: -90 = calyx-up (initial) | 0 = calyx->cam | +90 = calyx-down | +-180 = calyx-away",
                    (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 220), 1)

    def get_pose_for_ros(self, T):
        """输出 ROS 兼容的位姿信息"""
        pos = T[:3, 3]
        R = T[:3, :3]
        q_wxyz = PoseTemporalFilter.rot_to_quat(R)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler = self.get_euler_angles_zyx(R)
        return {
            "position_m": pos.tolist(),
            "rotation_matrix": R.tolist(),
            "quaternion_xyzw": q_xyzw.tolist(),
            "euler_zyx_deg": euler.tolist(),
        }

    def run(self):
        self.start_camera()
        win_name = 'Tomato 6D Pose V6.6'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self.color_w, self.color_h)
        print("🎥 按 Q/ESC 退出, P 打印位姿")
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
                    # CCIA 仲裁
                    best_iou, best_s = 0, -1
                    for s_idx, s_box in enumerate(seg_boxes):
                        iou = self.calculate_iou(p_box, s_box)
                        if iou > best_iou:
                            best_iou, best_s = iou, s_idx
                    if best_iou < 0.45 or best_s == -1:
                        continue

                    target_mask = seg_masks[best_s]

                    # 步骤 1: 反投影 mask 点云
                    pts_3d, resized_mask = self.collect_mask_pointcloud(
                        depth_array, target_mask, orig_shape)
                    if pts_3d is None:
                        continue

                    # 步骤 2: RANSAC 球面拟合
                    sphere_c_raw, sphere_r_raw, inlier_ratio, _ = \
                        self.sphere_fitter.fit(pts_3d)
                    if sphere_c_raw is None:
                        continue

                    # 步骤 3: 球心 EMA 平滑
                    sphere_c, sphere_r = self.center_smoother.smooth(
                        p_idx, sphere_c_raw, sphere_r_raw)

                    # 步骤 4: 关键点 2D 提取 + 射线投射到球面 (V6.6 核心)
                    kpts_3d_sphere = {}
                    kpts_2d = {}
                    kpts_conf = {}
                    for i in range(4):
                        u, v, conf = pose_kpts[p_idx][i]
                        if conf > 0.45:
                            # 射线-球面交点 (避开深度图边缘飞点)
                            pos_3d = self.raycaster.pixel_to_sphere(
                                int(u), int(v), sphere_c, sphere_r)
                            # 关键点 EMA 平滑
                            pos_smoothed = self.kpt_smoother.smooth(
                                p_idx, self.kpt_names[i], pos_3d)
                            kpts_3d_sphere[self.kpt_names[i]] = pos_smoothed
                            kpts_2d[self.kpt_names[i]] = (int(u), int(v))
                            kpts_conf[self.kpt_names[i]] = float(conf)
                            cv2.circle(color_image, (int(u), int(v)),
                                       6, (0, 255, 255), -1)

                    # 步骤 5: 几何主轴解算
                    T, strategy = self.solver.solve(
                        sphere_c, sphere_r, kpts_3d_sphere, kpts_conf)

                    # 步骤 6: Z 翻转防护
                    T = self.stabilize_z_flip(p_idx, T)

                    # 步骤 7: 时间滤波
                    T = self.tfilter.smooth(p_idx, T)
                    last_poses[p_idx] = T.copy()

                    # ---------------- 可视化 ----------------
                    bx1, by1, bx2, by2 = map(int, p_box)
                    cm = np.zeros_like(color_image, dtype=np.uint8)
                    cm[resized_mask > 0] = [0, 0, 200]
                    cv2.addWeighted(cm, 0.30, color_image, 1.0, 0, color_image)

                    # 球心反投影 → 白点
                    if T[2, 3] > 1e-6:
                        u_c = int(T[0, 3] * self.intrinsics.fx / T[2, 3] + self.intrinsics.ppx)
                        v_c = int(T[1, 3] * self.intrinsics.fy / T[2, 3] + self.intrinsics.ppy)
                        cv2.circle(color_image, (u_c, v_c), 7, (255, 255, 255), -1)
                        cv2.circle(color_image, (u_c, v_c), 8, (0, 0, 0), 1)

                    cv2.rectangle(color_image, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    self.draw_3d_axes(color_image, T)

                    t_x, t_y, t_z = T[0, 3], T[1, 3], T[2, 3]
                    yaw, roll, pitch = self.get_euler_angles_zyx(T[:3, :3])

                    label = f"{self.target_class_name} | r={sphere_r * 100:.1f}cm | inlier={inlier_ratio:.2f}"
                    info1 = f"P(m): [{t_x:+.3f}, {t_y:+.3f}, {t_z:+.3f}]  | {strategy}"
                    info2 = f"theta={yaw:+6.1f}  phi={roll:+6.1f}  psi={pitch:+6.1f}  (deg)"

                    panel_w, panel_h = 700, 80
                    by1_panel = max(98, by1 - panel_h)
                    cv2.rectangle(color_image, (bx1, by1_panel),
                                  (bx1 + panel_w, by1), (0, 0, 0), -1)
                    cv2.putText(color_image, label, (bx1 + 8, by1_panel + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    cv2.putText(color_image, info1, (bx1 + 8, by1_panel + 47),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.putText(color_image, info2, (bx1 + 8, by1_panel + 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cur_t = time.time()
                fps = 1 / max(cur_t - prev_t, 1e-6)
                prev_t = cur_t
                self.draw_legend(color_image, fps)

                cv2.imshow(win_name, color_image)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    break
                elif key & 0xFF == ord('p'):
                    print("\n=== 当前位姿 (供 ROS 调试) ===")
                    for tid, Tp in last_poses.items():
                        info = self.get_pose_for_ros(Tp)
                        print(f"track {tid}:")
                        print(f"  position(m)        = {info['position_m']}")
                        print(f"  euler ZYX(deg)     = {info['euler_zyx_deg']}")
                        print(f"  quaternion (xyzw)  = {info['quaternion_xyzw']}")
        finally:
            print("正在安全关闭相机 ...")
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    SEG_WEIGHTS = r"D:\code\Tomato_Grasp_Project\2026\20260321图像分割重做\runs\segmentation\train_yolo11s_seg_v1\weights\best.pt"
    POSE_WEIGHTS = r"D:\code\Tomato_Grasp_Project\2026\20260318关键点训练\Tomato_Pose_Runs\train_pose_0412\weights\best.pt"

    app = RealSenseTomato6DAppV6_6(
        SEG_WEIGHTS, POSE_WEIGHTS,
        color_w=1280, color_h=720,
        fps=30
    )
    app.run()

