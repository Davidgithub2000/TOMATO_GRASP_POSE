import random
import shutil
from pathlib import Path


def split_train_val_test(dataset_dir, val_ratio=0.1, test_ratio=0.1):
    base_path = Path(dataset_dir)

    # 定义源文件夹（假设初始状态下所有数据都在 train 里）
    train_img_dir = base_path / 'images' / 'train'
    train_lbl_dir = base_path / 'labels' / 'train'

    # 定义目标文件夹
    val_img_dir = base_path / 'images' / 'val'
    val_lbl_dir = base_path / 'labels' / 'val'
    test_img_dir = base_path / 'images' / 'test'
    test_lbl_dir = base_path / 'labels' / 'test'

    # 1. 创建 val 和 test 专属文件夹
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 2. 统计目前 train 里面的所有图片
    images = [f for f in train_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    if not images:
        print("在 train 文件夹里没有找到图片，请检查图片是否都在 images/train 目录下！")
        return

    total_count = len(images)

    # 3. 计算要抽取的数量
    val_count = int(total_count * val_ratio)
    test_count = int(total_count * test_ratio)
    train_count = total_count - val_count - test_count

    print(f"总共有 {total_count} 张图片。")
    print(f"计划划分 -> 训练集: {train_count} 张, 验证集: {val_count} 张, 测试集: {test_count} 张")

    # 4. 随机打乱列表以保证抽样均匀
    random.shuffle(images)

    # 切片划分图片列表
    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]

    # 剩下的 images[val_count + test_count:] 自然就留在 train 里了，不需要移动

    # 5. 定义移动文件的辅助函数
    def move_files(file_list, target_img_dir, target_lbl_dir, set_name):
        print(f"正在将 {len(file_list)} 张图片移入 {set_name} 集...")
        for img_path in file_list:
            lbl_path = train_lbl_dir / (img_path.stem + '.txt')

            # 移动图片
            shutil.move(str(img_path), str(target_img_dir / img_path.name))
            # 移动对应的标签（如果存在）
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(target_lbl_dir / lbl_path.name))

    # 6. 执行移动
    move_files(val_images, val_img_dir, val_lbl_dir, "验证(Val)")
    move_files(test_images, test_img_dir, test_lbl_dir, "测试(Test)")

    print("划分大功告成！现在的比例是 8:1:1。")


if __name__ == '__main__':
    # 指向你生成 YOLO 数据集的总目录
    DATASET_DIR = r"C:\Users\mechrevo\Desktop\yolo11_tomato_pose_improvement"

    # 注意：运行此脚本前，请确保所有图片和标签都集中在 images/train 和 labels/train 中
    # 如果你之前已经跑过一次划分脚本，建议把 val 里的数据移回 train 再运行，否则多次抽取会导致比例不对。
    split_train_val_test(DATASET_DIR, val_ratio=0.1, test_ratio=0.1)