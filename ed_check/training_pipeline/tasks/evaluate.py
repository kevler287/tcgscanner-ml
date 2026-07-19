"""
Evaluation script for the ResNet18 binary classifier trained with train_resnet18.py

Expected folder structure under data_dir:
    data_dir/
        first_ed/
            img001.jpg
            ...
        other_ed/
            img002.jpg
            ...
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ed_check.config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_dataloader(data_dir: Path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    eval_transform = transforms.Compose([
        transforms.Resize(CONFIG.train_cfg.imgsz),
        transforms.ToTensor(),
        normalize,
    ])

    eval_ds = datasets.ImageFolder(data_dir, transform=eval_transform)

    eval_loader = DataLoader(
        eval_ds, batch_size=CONFIG.train_cfg.batch, shuffle=False,
        num_workers=CONFIG.train_cfg.num_workers, pin_memory=True,
    )

    logger.info(f"Classes (from folder): {eval_ds.classes}")
    logger.info(f"Eval set: {len(eval_ds)} images")

    return eval_loader, eval_ds.classes


def build_model(num_classes=2):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(data_dir: Path, checkpoint_path: Path):
    """
    Evaluate a trained ResNet18 binary classifier.

    Args:
        data_dir: path to eval dataset root, must contain one subfolder per
                   class directly (e.g. first_ed/, other_ed/) - no train/val split
        checkpoint_path: path to the .pt checkpoint saved by train()

    Returns:
        dict with accuracy, and per-class precision/recall/f1/support
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_classes = checkpoint["classes"]

    eval_loader, eval_classes = get_dataloader(data_dir)

    # Sanity check: eval folder class order must match training class order,
    # since the model's output indices correspond to train_classes order.
    if eval_classes != train_classes:
        raise ValueError(
            f"Class mismatch between checkpoint ({train_classes}) and "
            f"eval folder ({eval_classes}). Make sure eval subfolders are "
            f"named identically to the training subfolders."
        )

    model = build_model(num_classes=len(train_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(train_classes))), zero_division=0
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    for i, cls in enumerate(train_classes):
        logger.info(f"  {cls}: precision={precision[i]:.4f} recall={recall[i]:.4f} "
              f"f1={f1[i]:.4f} support={support[i]}")

    return {
        "accuracy": accuracy,
        "precision": dict(zip(train_classes, precision.tolist())),
        "recall": dict(zip(train_classes, recall.tolist())),
        "f1": dict(zip(train_classes, f1.tolist())),
        "support": dict(zip(train_classes, support.tolist())),
    }