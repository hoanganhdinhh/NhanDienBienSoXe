

#tạo cấu trúc thư mục theo chuẩn YOLO
import os
import random
import shutil
from pathlib import Path

# Thư mục chứa ảnh + label ban đầu
SOURCE_DIR = Path("data")

# Thư mục output theo chuẩn YOLO
DATASET_DIR = Path("dataset")

# Tỷ lệ train / val
TRAIN_RATIO = 0.8  # 80% train, 20% val

def main():
    # Tạo các thư mục đích
    (DATASET_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Lấy danh sách tất cả ảnh (đuôi jpg, png tuỳ bạn)
    image_exts = [".jpg", ".jpeg", ".png"]
    images = [f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in image_exts]

    print(f"Tổng số ảnh: {len(images)}")

    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)
    train_images = images[:train_count]
    val_images = images[train_count:]

    def move_pair(img_path: Path, split: str):
        # split = "train" hoặc "val"
        label_path = img_path.with_suffix(".txt")  # đổi .jpg -> .txt

        if not label_path.exists():
            print(f"⚠️ Không tìm thấy label cho {img_path.name}, bỏ qua.")
            return

        # Copy ảnh
        dst_img = DATASET_DIR / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # Copy label
        dst_label = DATASET_DIR / "labels" / split / label_path.name
        shutil.copy2(label_path, dst_label)

    for img in train_images:
        move_pair(img, "train")

    for img in val_images:
        move_pair(img, "val")

    print("✅ Done! Đã tạo cấu trúc dataset/ theo chuẩn YOLO.")

if __name__ == "__main__":
    main()