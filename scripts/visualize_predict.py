import argparse
import os
import sys
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
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
JSON_PATH = f"results/{args.version}/{args.run_name}/metrics.json"
DATA_ROOT = f"data/split/val"
SAVE_ROOT = f"results/{args.version}/{args.run_name}/viz_predictions/"

MAX_IMAGES = 100  # 可調

class_names = ["abrasion", "chronic", "cut", "dfu", "laceration"]

# ==========================================
# Load JSON
# ==========================================
with open(JSON_PATH, "r") as f:
    data = json.load(f)

preds = data["preds"]
labels = data["labels"]

# ==========================================
# Load image paths（順序要一致）
# ==========================================
image_paths = []

for cls in sorted(os.listdir(DATA_ROOT)):
    cls_path = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for img_name in sorted(os.listdir(cls_path)):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(cls_path, img_name))

assert len(image_paths) == len(preds), "❌ 數量不一致"

# ==========================================
# 建立資料夾
# ==========================================
correct_dir = os.path.join(SAVE_ROOT, "correct")
wrong_dir = os.path.join(SAVE_ROOT, "wrong")

os.makedirs(correct_dir, exist_ok=True)
os.makedirs(wrong_dir, exist_ok=True)

# ==========================================
# 隨機抽樣（避免太多）
# ==========================================
indices = list(range(len(preds)))
random.shuffle(indices)
indices = indices[:MAX_IMAGES]

# ==========================================
# 開始分類存圖
# ==========================================
for idx in indices:
    img_path = image_paths[idx]
    img = Image.open(img_path).convert("RGB")

    gt_idx = labels[idx]
    pred_idx = preds[idx]

    gt = class_names[gt_idx]
    pred = class_names[pred_idx]

    # --------------------------------------
    # 分 correct / wrong
    # --------------------------------------
    if gt_idx == pred_idx:
        save_folder = os.path.join(correct_dir, gt)
    else:
        save_folder = os.path.join(wrong_dir, f"{gt}_to_{pred}")

    os.makedirs(save_folder, exist_ok=True)

    # --------------------------------------
    # 畫圖
    # --------------------------------------
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")

    color = "green" if gt_idx == pred_idx else "red"
    plt.title(f"GT: {gt}\nPred: {pred}", color=color)

    save_path = os.path.join(
        save_folder,
        os.path.basename(img_path)
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

print(f"\n✅ Visualization saved to {SAVE_ROOT}")