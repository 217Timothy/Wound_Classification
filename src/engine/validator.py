import torch
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

def validate(model, loader, criterion, device, class_names=None):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    recall = recall_score(all_labels, all_preds, average="macro")

    # ==========================================
    # 🔥 Per-class recall
    # ==========================================
    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None
    )

    print("\n=== Per-class Recall ===")
    if class_names:
        for i, r in enumerate(per_class_recall): # type: ignore
            print(f"{class_names[i]:12s}: {r:.4f}")
    else:
        for i, r in enumerate(per_class_recall): # type: ignore
            print(f"class {i}: {r:.4f}")

    # ==========================================
    # 🔥 Confusion Matrix
    # ==========================================
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== Confusion Matrix ===")
    print(cm)

    return total_loss / len(loader), acc, recall