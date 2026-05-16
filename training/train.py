from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet
from training.dataset import GreenAreaDataset
from training.metrics import DiceLoss, dice_coefficient, iou_score, pixel_accuracy


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_accuracy = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_iou += iou_score(logits, masks, threshold) * batch_size
        total_dice += dice_coefficient(logits, masks, threshold) * batch_size
        total_accuracy += pixel_accuracy(logits, masks, threshold) * batch_size

    dataset_size = len(loader.dataset)
    return {
        "loss": total_loss / dataset_size,
        "iou": total_iou / dataset_size,
        "dice": total_dice / dataset_size,
        "pixel_accuracy": total_accuracy / dataset_size,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = get_device(config["training"]["device"])
    dataset_root = PROJECT_ROOT / config["dataset"]["root"]
    image_size = config["dataset"]["image_size"]
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])

    train_dataset = GreenAreaDataset(
        dataset_root,
        split="train",
        image_size=image_size,
        augment=True,
        mean=mean,
        std=std,
    )
    val_dataset = GreenAreaDataset(
        dataset_root,
        split="val",
        image_size=image_size,
        augment=False,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        features=config["model"]["features"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    best_iou = -1.0
    best_path = PROJECT_ROOT / config["paths"]["best_model"]
    last_path = PROJECT_ROOT / config["paths"]["last_model"]

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            bce_loss,
            dice_loss,
            device,
        )
        val_metrics = validate_epoch(
            model,
            val_loader,
            bce_loss,
            dice_loss,
            device,
            config["training"]["threshold"],
        )

        save_checkpoint(last_path, model, optimizer, epoch, val_metrics, config)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics, config)

        print(
            f"Epoch {epoch:03d}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"IoU={val_metrics['iou']:.4f} | "
            f"Dice={val_metrics['dice']:.4f} | "
            f"PixelAcc={val_metrics['pixel_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
