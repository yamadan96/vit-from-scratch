"""
Training script for Vision Transformer on CIFAR-10.

Trains a ViT-Small model from scratch using:
- AdamW optimizer with cosine annealing schedule
- Mixed precision training (AMP)
- Standard CIFAR-10 augmentations
- Optional Weights & Biases logging

Usage:
    uv run python src/train.py --epochs 100 --batch-size 128
    WANDB_PROJECT=vit-cifar10 uv run python src/train.py  # with W&B
"""

import argparse
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .model import ViTConfig, VisionTransformer

logger = logging.getLogger(__name__)

# CIFAR-10 channel-wise statistics (precomputed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and validation transforms for CIFAR-10.

    Training uses standard augmentations:
    - Random crop with padding (simulates translation)
    - Random horizontal flip

    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, val_transform


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders from CIFAR-10.

    Splits the 50,000 training images into train/val sets.
    Uses the standard CIFAR-10 test set is NOT used here -- we hold it
    out for final evaluation.

    Args:
        data_dir: Root directory for CIFAR-10 download
        batch_size: Mini-batch size
        num_workers: Number of data loading workers
        val_ratio: Fraction of training data to use for validation

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform, val_transform = build_transforms()

    # Download CIFAR-10 training set
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # Split into train/val
    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Override transform for validation split (no augmentation)
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=val_transform
    )
    val_set.dataset = val_dataset  # pyright: ignore[reportAttributeAccessIssue]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Data loaded: %d train / %d val samples, batch_size=%d",
        n_train,
        n_val,
        batch_size,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """Train for one epoch with mixed precision.

    Args:
        model: ViT model
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scaler: AMP GradScaler
        device: Compute device

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation set.

    Args:
        model: ViT model
        loader: Validation DataLoader
        criterion: Loss function
        device: Compute device

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def setup_wandb(config: ViTConfig, args: argparse.Namespace) -> bool:
    """Initialize Weights & Biases if WANDB_PROJECT env var is set.

    Args:
        config: ViT configuration
        args: Training arguments

    Returns:
        True if W&B is active, False otherwise
    """
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb_project is None:
        logger.info("WANDB_PROJECT not set, skipping W&B logging")
        return False

    try:
        import wandb

        wandb.init(
            project=wandb_project,
            config={**asdict(config), "epochs": args.epochs, "batch_size": args.batch_size},
        )
        logger.info("W&B initialized: project=%s", wandb_project)
        return True
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Vision Transformer on CIFAR-10"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for CIFAR-10 data (default: ./data)",
    )
    return parser.parse_args()


def main() -> None:
    """Main training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Model configuration and instantiation
    config = ViTConfig()
    model = VisionTransformer(config).to(device)

    # Data
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=device.type)

    # W&B (optional)
    use_wandb = setup_wandb(config, args)

    # Checkpoint directory
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    best_val_acc = 0.0
    logger.info("Starting training for %d epochs", args.epochs)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.monotonic() - epoch_start

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.4f | lr=%.2e | time=%.1fs",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # W&B logging
        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": asdict(config),
                },
                best_checkpoint_path,
            )
            logger.info(
                "New best model saved: val_acc=%.4f -> %s",
                val_acc,
                best_checkpoint_path,
            )

    logger.info(
        "Training complete. Best val_acc=%.4f, checkpoint=%s",
        best_val_acc,
        best_checkpoint_path,
    )

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
