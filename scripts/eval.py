import json
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.datasets import (
    ClassificationDataset,
    get_val_transforms
)
from src.models import Model
from src.utils.checkpoint import load_checkpoint
from src.engine import validate


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
    parser.add_argument("--datasets", type=str, nargs="+")
    
    parser.add_argument("--model", type=str, default="efficientnet")
    parser.add_argument("--loss_name", type=str, default="cross_entropy")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

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
    ckpt_path = f"checkpoints/{args.version}/{args.run_name}/best.pt"
    save_path = f"results/{args.version}/{args.run_name}/metrics.json"

    # =============================
    # Datasets & Dataloaders
    # =============================
    val_transform = get_val_transforms()
    val_dataset = build_datasets(split="val", transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY
    )

    # =============================
    # Model
    # =============================
    model = build_model(backbone_name=args.model)

    # =============================
    # Criterion
    # =============================
    criterion = build_criterion(loss_name=args.loss_name)

    # =============================
    # Load checkpoint
    # =============================
    load_checkpoint(checkpoint_path=ckpt_path, model=model, device=DEVICE)

    # =============================
    # Validate
    # =============================
    class_names = list(val_dataset.class_to_idx.keys())
    _, results = validate(model, val_loader, criterion, DEVICE, class_names)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== Evaluation Result =====")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"\n✅ Saved to {save_path}")


if __name__ == "__main__":
    main()