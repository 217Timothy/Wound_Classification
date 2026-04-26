import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import yaml

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
# ==========================================
# Config
# ==========================================
LOG_PATH = f"logs/{args.version}/{args.run_name}.csv"
SAVE_PATH = f"results/{args.version}/{args.run_name}/curves.png"

# ==========================================
# Load CSV
# ==========================================
df = pd.read_csv(LOG_PATH)
epochs = df["epoch"]

# ==========================================
# Plot (3 subplots)
# ==========================================
plt.figure(figsize=(15, 5))

# -----------------------------
# 1️⃣ Loss
# -----------------------------
plt.subplot(1, 3, 1)
plt.plot(epochs, df["train_loss"], label="Train Loss", linestyle="-", color="royalblue", marker="o")
plt.plot(epochs, df["val_loss"], label="Val Loss", linestyle="-", color="tab:orange", marker="s")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# -----------------------------
# 2️⃣ Accuracy
# -----------------------------
plt.subplot(1, 3, 2)
plt.plot(epochs, df["val_acc"], label="Val Accuracy", linestyle="-", color="royalblue", marker="o")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# -----------------------------
# 3️⃣ Recall
# -----------------------------
plt.subplot(1, 3, 3)
plt.plot(epochs, df["val_recall"], label="Val Recall", linestyle="-", color="royalblue", marker="o")
plt.title("Recall Curve")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

# ==========================================
# Save
# ==========================================
plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300)
plt.show()

print(f"\n✅ Saved to: {SAVE_PATH}")