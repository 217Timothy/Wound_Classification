import json
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

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

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_args)
    return args

args = get_args()
JSON_PATH = f"results/{args.version}/{args.run_name}/metrics.json"
SAVE_PATH = f"results/{args.version}/{args.run_name}/confusion_matrix.png"

# ==========================================
# Load Data
# ==========================================
with open(JSON_PATH, "r") as f:
    data = json.load(f)
labels = data["labels"]
preds = data["preds"]
class_names = list(data["per_class_recall"].keys())

# ==========================================
# Compute Confusion Matrix
# ==========================================
cm = confusion_matrix(labels, preds)
# Normalized version（比例）
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

# ==========================================
# Plot
# ==========================================
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix (Normalized with Counts)")
plt.tight_layout()

# ==========================================
# Save
# ==========================================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300)
plt.show()
print(f"\n✅ Confusion matrix saved to: {SAVE_PATH}")