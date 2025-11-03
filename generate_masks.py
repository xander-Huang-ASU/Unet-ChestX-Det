import json
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

def generate_masks(json_path, images_dir, output_mask_dir):
    """
    根据 ChestX-Det 格式的 JSON 文件生成掩膜图像。
    每张图像对应一张 mask（白色表示病灶区域，黑色表示背景）。

    Args:
        json_path (str): JSON 文件路径，例如 /scratch/.../train.json
        images_dir (str): 图像目录，例如 /scratch/.../train_data/train
        output_mask_dir (str): 掩膜输出目录，例如 /scratch/.../train_data/masks
    """

    os.makedirs(output_mask_dir, exist_ok=True)

    # ===== 1. 读取 JSON 文件 =====
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} annotations from {json_path}")

    missing_images = 0
    mask_count = 0
    no_box_count = 0

    # ===== 2. 遍历所有样本 =====
    for item in tqdm(data, desc="Generating masks"):
        filename = item["file_name"]
        boxes = item.get("boxes", [])

        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            missing_images += 1
            continue

        # 读取图像尺寸
        with Image.open(img_path) as img:
            w, h = img.size

        # 创建全黑掩膜
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # 如果有框，则绘制白色矩形
        if boxes:
            for box in boxes:
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], fill=255)
        else:
            no_box_count += 1

        # 保存掩膜
        mask.save(os.path.join(output_mask_dir, filename))
        mask_count += 1

    # ===== 3. 打印总结 =====
    print("\n=== Summary ===")
    print(f"Total annotations: {len(data)}")
    print(f"✅ Masks generated: {mask_count}")
    print(f"⚠️ Missing images: {missing_images}")
    print(f"ℹ️  No-box (healthy) samples: {no_box_count}")
    print(f"Masks saved to: {output_mask_dir}\n")

if __name__ == "__main__":
    # ===== 你可以修改下面三行路径 =====
    json_path = "/scratch/yhuan591/chestX-Det/train.json"
    images_dir = "/scratch/yhuan591/chestX-Det/train_data/train"
    output_mask_dir = "/scratch/yhuan591/chestX-Det/train_data/masks"

    generate_masks(json_path, images_dir, output_mask_dir)
