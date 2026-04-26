import os
import cv2
from tqdm import tqdm

INPUT_DIR = "data_raw"
OUTPUT_DIR = "data/processed"

IMAGE_SIZE = 224


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))


def preprocess_image(input_path, output_path):
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"[SKIP] Cannot read: {input_path}")
            return False
        
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        output_path = os.path.splitext(output_path)[0] + ".jpg"
        cv2.imwrite(output_path, img)
        
        return True
    
    except Exception as e:
        print(f"[ERROR] {input_path}: {e}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classes = os.listdir(INPUT_DIR)
    
    for cls in classes:
        class_input_path = os.path.join(INPUT_DIR, cls)
        if not os.path.isdir(class_input_path):
            continue
        
        class_name = cls.lower()
        class_output_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        
        images = os.listdir(class_input_path)
        print(f"\nProcessing class: {class_name} ({len(images)} images)")
        
        for img_name in tqdm(images):
            if not is_image_file(img_name):
                continue
            
            input_path = os.path.join(class_input_path, img_name)
            output_path = os.path.join(class_output_path, img_name)
            preprocess_image(input_path, output_path)


if __name__ == "__main__":
    main()