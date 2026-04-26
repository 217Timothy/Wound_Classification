import torch
from sklearn.metrics import recall_score, confusion_matrix


def validate(model, loader, criterion, device, class_names=None):
    model.eval()

    total_loss = 0.0
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

            # 🔥 修這裡
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # =============================
    # Metrics
    # =============================
    acc = correct / total if total > 0 else 0.0

    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )

    # =============================
    # Convert to dict（JSON-ready）
    # =============================
    if class_names is not None:
        per_class_dict = {
            class_names[i]: float(per_class_recall[i]) # type: ignore
            for i in range(len(per_class_recall))  # type: ignore
        }
    else:
        per_class_dict = {
            f"class_{i}": float(per_class_recall[i]) # type: ignore
            for i in range(len(per_class_recall))  # type: ignore
        }

    results = {
        "accuracy": float(acc),
        "macro_recall": float(macro_recall),
        "per_class_recall": per_class_dict,
        
        "preds": all_preds, 
        "labels": all_labels
    }

    # =============================
    # Print（debug / train用）
    # =============================
    print("\n=== Per-class Recall ===")
    for k, v in per_class_dict.items():
        print(f"{k:12s}: {v:.4f}")

    return total_loss / len(loader), results