import torch
from torch.utils.data import DataLoader

from src.datasets import ClassificationDataset
from src.datasets import get_train_transforms


def main():
    dataset = ClassificationDataset(
        root_dir="data/split/train",
        transform=get_train_transforms()
    )

    print("\n=== Dataset Info ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Class mapping: {dataset.class_to_idx}")

    # 測一筆
    image, label = dataset[0]

    print("\n=== Single Sample ===")
    print(f"Image type: {type(image)}")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")

    # DataLoader 測試
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("\n=== DataLoader Batch ===")

    for images, labels in loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break

    print("\n✅ Dataset test passed!")


if __name__ == "__main__":
    main()