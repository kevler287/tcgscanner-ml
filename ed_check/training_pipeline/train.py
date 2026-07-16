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
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(data_dir, batch_size, num_workers):
    # ImageNet normalization since we're using pretrained weights
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((64, 192)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 192)),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

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


def train(
    data_dir,
    output_path="best.pt",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    num_workers=8,
):
    """
    Train a ResNet18 binary classifier.

    Args:
        data_dir: path to dataset root, must contain train/ and val/ subfolders
                   with one folder per class (e.g. first_ed/, other_ed/)
        output_path: where to save the best checkpoint
        epochs: number of training epochs
        batch_size: batch size for train/val loaders
        lr: learning rate
        num_workers: dataloader worker processes
    """

    train_loader, val_loader, classes = get_dataloaders(
        data_dir, batch_size, num_workers
    )

    model = build_model(num_classes=len(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        scheduler.step(val_loss)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
            }, output_path)
            print(f"  -> new best model saved ({output_path}), val_acc={val_acc:.4f}")

    print(f"\nDone. Best val_acc: {best_val_acc:.4f}")
    return best_val_acc
