from generate_masks import generate_masks
from train_unet_chestxdet import train_unet_with_val_plot
import os

if __name__ == "__main__":
    # ===== Step 1: 生成 Train 集掩膜 =====
    print("=== Step 1: Generating Train Masks ===")
    train_mask_dir = "/scratch/yhuan591/chestX-Det/train_data/masks"
    if not os.path.exists(train_mask_dir) or len(os.listdir(train_mask_dir)) == 0:
        generate_masks(
            json_path="/scratch/yhuan591/chestX-Det/train.json",
            images_dir="/scratch/yhuan591/chestX-Det/train_data/train",
            output_mask_dir=train_mask_dir
        )
    else:
        print(f"✅ Train masks already exist at: {train_mask_dir}")

    # ===== Step 2: 生成 Test 集掩膜（用于验证） =====
    print("\n=== Step 2: Generating Test Masks ===")
    test_mask_dir = "/scratch/yhuan591/chestX-Det/test_data/masks"
    if not os.path.exists(test_mask_dir) or len(os.listdir(test_mask_dir)) == 0:
        generate_masks(
            json_path="/scratch/yhuan591/chestX-Det/test.json",
            images_dir="/scratch/yhuan591/chestX-Det/test_data/test",
            output_mask_dir=test_mask_dir
        )
    else:
        print(f"✅ Test masks already exist at: {test_mask_dir}")

    # ===== Step 3: 开始训练并在 Test 数据集上验证 =====
    print("\n=== Step 3: Training U-Net with Validation ===")
    train_unet_with_val_plot(
        train_img_dir="/scratch/yhuan591/chestX-Det/train_data/train",
        train_mask_dir="/scratch/yhuan591/chestX-Det/train_data/masks",
        val_img_dir="/scratch/yhuan591/chestX-Det/test_data/test",
        val_mask_dir="/scratch/yhuan591/chestX-Det/test_data/masks",
        epochs=30,
        lr=1e-4,
        batch_size=4
    )

    print("\n🏁 All steps completed successfully!")
