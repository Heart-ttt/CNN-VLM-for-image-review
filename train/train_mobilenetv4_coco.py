import os
import json
import shutil
from collections import defaultdict
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm


# =========================
# 配置
# =========================

COCO_IMAGE_DIR = "COCO/train2017"
COCO_ANN_FILE = "COCO/annotations/instances_train2017.json"
OUTPUT_DIR = "dataset"
BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Step1: 生成 simple/complex 数据
# =========================

def generate_dataset():
    if os.path.exists(OUTPUT_DIR):
        print("Dataset already exists. Skip generation.")
        return

    print("Generating simple/complex dataset from COCO...")

    os.makedirs(f"{OUTPUT_DIR}/train/simple", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/train/complex", exist_ok=True)

    with open(COCO_ANN_FILE, "r") as f:
        coco = json.load(f)

    img_id_to_filename = {}
    for img in coco["images"]:
        img_id_to_filename[img["id"]] = img["file_name"]

    annotation_count = defaultdict(int)

    for ann in coco["annotations"]:
        annotation_count[ann["image_id"]] += 1

    for img_id, count in tqdm(annotation_count.items()):
        filename = img_id_to_filename.get(img_id)
        if filename is None:
            continue

        src_path = os.path.join(COCO_IMAGE_DIR, filename)

        if not os.path.exists(src_path):
            continue

        if count <= 1:
            dst_path = os.path.join(OUTPUT_DIR, "train/simple", filename)
        elif count >= 3:
            dst_path = os.path.join(OUTPUT_DIR, "train/complex", filename)
        else:
            continue

        shutil.copy(src_path, dst_path)

    print("Dataset generation complete!")

# =========================
# Step2: 训练模型
# =========================

def train():
    print("prepare for training!!")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(f"{OUTPUT_DIR}/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = timm.create_model("mobilenetv4_conv_small", pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        pbar = tqdm(
                train_loader,
                file=sys.stdout,
                dynamic_ncols=True,
                leave=False
        )

        for imgs, labels in pbar:   #pbar 调用
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            acc = correct / total
            avg_loss = running_loss / len(pbar)

            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "classify.pth")

    print("Training complete. Best Acc:", best_acc)

# =========================
# 主函数
# =========================

if __name__ == "__main__":
    generate_dataset()
    train()