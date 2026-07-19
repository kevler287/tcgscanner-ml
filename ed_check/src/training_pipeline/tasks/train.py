"""
Binary Image Classifier Training with ResNet18 (Transfer Learning)

Expected folder structure under data_dir:
    data_dir/
        train/
            first_ed/
                img001.jpg
                ...
            other_ed/
                img002.jpg
                ...
        val/
            first_ed/
                ...
            other_ed/
                ...

Outputs (written into dest_dir):
    dest_dir/best.pt          - best checkpoint (by val_acc)
    dest_dir/metrics.csv      - per-epoch train/val loss & accuracy
"""

import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from ed_check.src.config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_dataloaders(data_dir: Path):
    # ImageNet normalization since we're using pretrained weights
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize(CONFIG.train_cfg.imgsz),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(CONFIG.train_cfg.imgsz),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG.train_cfg.batch, shuffle=True,
        num_workers=CONFIG.train_cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG.train_cfg.batch, shuffle=False,
        num_workers=CONFIG.train_cfg.num_workers, pin_memory=True,
    )

    logger.info(f"Classes: {train_ds.classes}")
    logger.info(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def train(data_dir: Path, results_dir: Path):
    """
    Train a ResNet18 binary classifier.

    Args:
        data_dir: path to dataset root, must contain train/ and val/ subfolders
                   with one folder per class (e.g. first_ed/, other_ed/)
        results_dir: destination for weights and epochs csv
    """
    weigths_path = results_dir / "best.pt"
    train_csv_path = results_dir / "metrics.csv"

    train_loader, val_loader, classes = get_dataloaders(data_dir)

    model = build_model(num_classes=len(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG.train_cfg.lr0
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0

    with open(train_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "lr", "duration_sec",
        ])

        for epoch in range(1, CONFIG.train_cfg.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            scheduler.step(val_loss)
            dt = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:02d}/{CONFIG.train_cfg.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{dt:.1f}s"
            )

            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", current_lr, f"{dt:.2f}",
            ])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "val_acc": val_acc,
                }, weigths_path)
                logger.info(f"  -> new best model saved ({weigths_path}), val_acc={val_acc:.4f}")

    logger.info(f"\nDone. Best val_acc: {best_val_acc:.4f}")
    logger.info(f"Metrics written to {train_csv_path}")
    return weigths_path, train_csv_path