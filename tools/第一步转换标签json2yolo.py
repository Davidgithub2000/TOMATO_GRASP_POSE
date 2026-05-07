import json
import os
import shutil
from pathlib import Path


def labelme2yolo_pose(input_dir, output_dir):
    # 1. 核心配置：严格对应你的 Labelme 标签
    BOX_LABEL = "red"  # 你的目标框标签
    # 关键点顺序必须绝对固定：果梗 -> 底部 -> 左侧 -> 右侧
    KPT_LABELS = ["tomato_calyx", "tomato_bottom", "tomato_left", "tomato_right"]
    CLASS_ID = 0  # 类别设为0 (代表番茄)

    # 2. 创建标准的 YOLO 数据集目录结构
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    images_dir = output_path / 'images' / 'train'
    labels_dir = output_path / 'labels' / 'train'

    # exist_ok=True 保证如果目录已存在不会报错
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))
    print(f"扫描到 {len(json_files)} 个 JSON 文件，开始转换...")

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_w = data['imageWidth']
        img_h = data['imageHeight']

        # 3. 按 group_id 聚合同一个番茄的框和关键点
        instances = {}
        for shape in data['shapes']:
            g_id = shape.get('group_id')
            if g_id is None:
                # 兼容处理：如果没有标 group_id，默认设为 0（针对图中只有一个番茄的情况）
                g_id = 0

            if g_id not in instances:
                # 初始化一个实例：准备存放它的外接矩形和关键点字典
                instances[g_id] = {'box': None, 'kpts': {}}

            label = shape['label']
            shape_type = shape['shape_type']
            points = shape['points']

            # 提取边界框并归一化
            if shape_type == 'rectangle' and label == BOX_LABEL:
                x1, y1 = points[0]
                x2, y2 = points[1]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)

                xc = (xmin + xmax) / 2.0 / img_w
                yc = (ymin + ymax) / 2.0 / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                instances[g_id]['box'] = [xc, yc, w, h]

            # 提取关键点并归一化
            elif shape_type == 'point' and label in KPT_LABELS:
                x = points[0][0] / img_w
                y = points[0][1] / img_h
                instances[g_id]['kpts'][label] = [x, y, 2]  # 2 表示该点可见且已标注

        # 4. 组装 YOLO 格式的 TXT 内容
        yolo_lines = []
        for g_id, inst in instances.items():
            if inst['box'] is None:
                continue  # 如果这个 group_id 只有点没有框，视为废弃数据，直接跳过

            line = [CLASS_ID] + inst['box']

            # 严格按照 KPT_LABELS 的顺序写入关键点
            for kpt_name in KPT_LABELS:
                if kpt_name in inst['kpts']:
                    # 正常标出的点：写入归一化坐标和状态 2
                    line.extend(inst['kpts'][kpt_name])
                else:
                    # 被遮挡、没画的点：写入 0 0 0 (YOLO会自动忽略这部分的损失计算)
                    line.extend([0.000000, 0.000000, 0])

            # 格式化列表，浮点数保留 6 位小数
            line_str = " ".join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in line])
            yolo_lines.append(line_str)

        # 5. 保存 TXT 文件并拷贝原图
        if yolo_lines:
            txt_name = json_file.stem + '.txt'
            txt_path = labels_dir / txt_name
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

            # 匹配原图（兼容 png 和 jpg）
            img_found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                src_img = input_path / (json_file.stem + ext)
                if src_img.exists():
                    shutil.copy(src_img, images_dir / src_img.name)
                    img_found = True
                    break

            if not img_found:
                print(f"警告: 找不到 {json_file.stem} 对应的图片文件，请检查原文件夹。")

    print(f"\n转换大功告成！请前往 {output_path} 查看你的 YOLO 数据集。")


if __name__ == '__main__':
    # 你截图中的原始标注文件夹路径
    INPUT_DIR = r"C:\Users\mechrevo\Desktop\20260317test_guanjiandian"
    # 转换后 YOLO 数据集的输出路径（会在桌面上新建一个文件夹）
    OUTPUT_DIR = r"C:\Users\mechrevo\Desktop\yolo11_tomato_pose_improvement"

    labelme2yolo_pose(INPUT_DIR, OUTPUT_DIR)