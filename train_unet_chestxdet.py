import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt


# =======================
# 1. 数据集定义
# =======================
class ChestXDetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [
            f for f in os.listdir(image_dir)
            if os.path.exists(os.path.join(mask_dir, f))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# =======================
# 2. Dice & IoU 计算函数
# =======================
def dice_coef(pred, target):
    smooth = 1e-7
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target):
    smooth = 1e-7
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# =======================
# 3. 训练函数 + 绘图
# =======================
def train_unet_with_val_plot(train_img_dir, train_mask_dir,
                             val_img_dir, val_mask_dir,
                             epochs=10, lr=1e-4, batch_size=4):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_dataset = ChestXDetDataset(train_img_dir, train_mask_dir, transform)
    val_dataset = ChestXDetDataset(val_img_dir, val_mask_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(device)

    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 存储指标
    train_losses, val_dices, val_ious = [], [], []
    best_dice = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== 验证 =====
        model.eval()
        dice_scores, iou_scores = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                dice_scores.append(dice_coef(preds, masks).item())
                iou_scores.append(iou_score(preds, masks).item())

        mean_dice, mean_iou = np.mean(dice_scores), np.mean(iou_scores)
        val_dices.append(mean_dice)
        val_ious.append(mean_iou)

        print(f"✅ Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}")

        # 保存最优模型
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 Best model saved (Dice={best_dice:.4f})")

    # =======================
    # 4. 绘制训练曲线
    # =======================
    plt.figure(figsize=(8, 6))
    epochs_range = range(1, epochs + 1)

    plt.plot(epochs_range, train_losses, 'r-o', label="Train Loss")
    plt.plot(epochs_range, val_dices, 'g-o', label="Val Dice")
    plt.plot(epochs_range, val_ious, 'b-o', label="Val IoU")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("U-Net Training and Validation Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

    print("📊 Training curves saved to training_curves.png")
    print(f"🔥 Best Val Dice: {best_dice:.4f}")


# =======================
# 5. 主入口
# =======================
if __name__ == "__main__":
    train_unet_with_val_plot(
        train_img_dir="/scratch/yhuan591/chestX-Det/train_data/train",
        train_mask_dir="/scratch/yhuan591/chestX-Det/train_data/masks",
        val_img_dir="/scratch/yhuan591/chestX-Det/test_data/test",
        val_mask_dir="/scratch/yhuan591/chestX-Det/test_data/masks",
        epochs=10,
        lr=1e-4,
        batch_size=4
    )
