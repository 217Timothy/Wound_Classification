import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.datasets import (
    ClassificationDataset,
    compute_class_weights,
    create_weight_sampler,
    get_train_transforms,
    get_val_transforms
)
from src.models import Model
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.engine import train_one_epoch, validate


# ==========================================
# Device
# ==========================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PIN_MEMORY = DEVICE == "cuda"

# ==========================================
# Args
# ==========================================
def get_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None)
    known_args, remaining_args = conf_parser.parse_known_args()

    defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument("--version", type=str)
    parser.add_argument("--run_name", type=str)
    
    parser.add_argument("--model", type=str, default="efficientnet")
    parser.add_argument("--loss_name", type=str, default="cross_entropy")

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_args)
    return args


# ==========================================
# Dataset
# ==========================================
def build_datasets(split, transform):
    return ClassificationDataset(
        root_dir=f"data/split/{split}",
        transform=transform
    )


# ==========================================
# Model
# ==========================================
def build_model(backbone_name, num_classes=5):
    return Model(
        backbone_name=backbone_name,
        num_classes=num_classes
    ).to(DEVICE)


# ==========================================
# Loss
# ==========================================
def build_criterion(loss_name, class_weights=None):
    if loss_name == "cross_entropy":
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")


def main():
    args = get_args()
    print(args)
    
    # ==========================================
    # Checkpoint
    # ==========================================
    checkpoint_path = f"checkpoints/{args.version}/{args.run_name}"
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"[CheckPoint] Checkpoint directory: {checkpoint_path}")
    
    # ==========================================
    # Logging
    # ==========================================
    log_dir = f"logs/{args.version}"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{args.run_name}.csv")

    # 如果檔案不存在 → 建 header
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_acc,val_recall\n")
    
    # ==========================================
    # Dataset and Dataloader
    # ==========================================
    train_dataset = build_datasets(
        split="train",
        transform=get_train_transforms()
    )
    val_dataset = build_datasets(
        split="val",
        transform=get_val_transforms()
    )
    
    class_names = list(train_dataset.class_to_idx.keys())
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    train_sampler = create_weight_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY
    )
    
    # ==========================================
    # Model
    # ==========================================
    model = build_model(args.model, num_classes=5)
    
    # ==========================================
    # Loss
    # ==========================================
    criterion = build_criterion(loss_name=args.loss_name, class_weights=class_weights.to(DEVICE))
    
    # ==========================================
    # Optimizer
    # ==========================================
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # ==========================================
    # Training Loop
    # ==========================================
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ==========================================
        # Train
        # ==========================================
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )
        
        # ==========================================
        # Validate
        # ==========================================
        val_loss, val_results = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
            class_names=class_names
        )

        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_results['accuracy']:.4f}")
        print(f"Val Recall: {val_results['macro_recall']:.4f}")
        
        # ==========================================
        # Save log
        # ==========================================
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_loss:.6f},"
                f"{val_loss:.6f},"
                f"{val_results['accuracy']:.6f},"
                f"{val_results['macro_recall']:.6f}\n"
            )
        
        is_best = val_results['accuracy'] > best_acc
        if is_best:
            best_acc = val_results['accuracy']
        checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_results['accuracy'],
                "val_recall": val_results['macro_recall']
            }
        save_checkpoint(
            state=checkpoint,
            is_best=is_best,
            checkpoint_dir=checkpoint_path
        )


if __name__ == "__main__":
    main()