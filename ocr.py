from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# ================== CONFIG ==================
MODEL_PATH = "best.pt"          # đường dẫn tới model YOLO biển số
IMAGE_PATH = "test_BienSo.jpg"  # đường dẫn tới ảnh muốn nhận diện
OCR_LANGS = ['en']  # biển số chủ yếu là số + chữ cái, 'en' là đủ
# ===========================================


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Không thể đọc ảnh. Kiểm tra lại đường dẫn: {path}")
        exit(1)
    return img


def init_yolo(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print("❌ Lỗi khi load model YOLO, kiểm tra lại MODEL_PATH trong code.")
        print("Chi tiết lỗi:", e)
        exit(1)


def init_ocr(langs):
    # gpu=False để chắc chắn chạy được trên Mac không có GPU CUDA
    reader = easyocr.Reader(langs, gpu=False)
    return reader


def preprocess_plate(plate_img):
    """Tiền xử lý biển số trước khi OCR cho nét hơn"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # phóng to cho dễ đọc
    scale = 2.0
    gray = cv2.resize(
        gray, None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR
    )

    # làm mịn nhẹ
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold (tuỳ ảnh, có thể bật/tắt để thử)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th


def detect_and_ocr_plate():
    # 1. Load model + OCR
    model = init_yolo(MODEL_PATH)
    reader = init_ocr(OCR_LANGS)

    # 2. Đọc ảnh
    img = load_image(IMAGE_PATH)
    img_draw = img.copy()

    # 3. Chạy YOLO detect biển số
    results = model(img)[0]  # lấy kết quả cho ảnh đầu tiên

    if results.boxes is None or len(results.boxes) == 0:
        print("⚠ Không tìm thấy biển số nào trong ảnh.")
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        return

    print(f"🔎 Tìm thấy {len(results.boxes)} biển số.")

    # 4. Lặp qua từng bounding box biển số
    for i, box in enumerate(results.boxes):
        # box.xyxy: [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Giới hạn trong size ảnh cho an toàn
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        plate_img = img[y1:y2, x1:x2]

        if plate_img.size == 0:
            print(f"⚠ Biển số {i} bị crop lỗi (kích thước 0). Bỏ qua.")
            continue

        # 5. Tiền xử lý ảnh biển số
        plate_proc = preprocess_plate(plate_img)

        # 6. OCR
        ocr_result = reader.readtext(plate_proc, detail=0, paragraph=True)
        text = " ".join(ocr_result).strip()

        print(f"📌 Biển số {i}: {text}")

        # 7. Vẽ bounding box + text lên ảnh gốc
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # đặt text ngay trên bbox
        cv2.putText(
            img_draw,
            text if text != "" else "N/A",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 8. Hiển thị riêng từng biển số (tùy, có thể tắt)
        cv2.imshow(f"Plate {i}", plate_proc)

    # 9. Hiển thị ảnh final
    cv2.imshow("Plate detection + OCR", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_ocr_plate()