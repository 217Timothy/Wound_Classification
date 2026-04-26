import os

DATA_DIR = "data/split"


def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])


def main():
    splits = ["train", "val", "test"]

    total_all = 0

    for split in splits:
        split_path = os.path.join(DATA_DIR, split)

        if not os.path.exists(split_path):
            continue

        print(f"\n=== {split.upper()} ===")

        classes = os.listdir(split_path)
        split_total = 0

        for cls in classes:
            cls_path = os.path.join(split_path, cls)

            if not os.path.isdir(cls_path):
                continue

            num_images = count_images(cls_path)
            split_total += num_images

            print(f"{cls:15s}: {num_images}")

        print(f"Total {split}: {split_total}")
        total_all += split_total

    print(f"\n=== TOTAL DATASET ===")
    print(f"Total images: {total_all}")


if __name__ == "__main__":
    main()