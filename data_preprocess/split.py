import os
import shutil
import random
from tqdm import tqdm


INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/split"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

SEED = 42


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def split_class(cls_name):
    class_input_path = os.path.join(INPUT_DIR, cls_name)
    images = os.listdir(class_input_path)
    
    random.shuffle(images)
    
    total = len(images)
    train_end = total * SPLIT_RATIO["train"]
    val_end = total * SPLIT_RATIO["val"]
    
    split_map = {
        "train": images[:int(train_end)],
        "val": images[int(train_end):int(train_end + val_end)],
        "test": images[int(train_end + val_end):]
    }

    for split_name, image_list in split_map.items():
        split_output_path = os.path.join(OUTPUT_DIR, split_name, cls_name)
        create_dir(split_output_path)
        for image in image_list:
            src = os.path.join(class_input_path, image)
            dst = os.path.join(split_output_path, image)
            shutil.copy(src, dst)


def main():
    random.seed(SEED)
    create_dir(OUTPUT_DIR)
    
    classes = os.listdir(INPUT_DIR)
    for cls in tqdm(classes, desc="Splitting classes"):
        class_input_path = os.path.join(INPUT_DIR, cls)
        if os.path.isdir(class_input_path):
            split_class(cls)


if __name__ == "__main__":
    main()