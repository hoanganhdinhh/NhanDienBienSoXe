from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

img = cv2.imread("test_BienSo.jpg")
results = model(img)[0]

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"plate {conf:.2f}", (x1, y1-10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()