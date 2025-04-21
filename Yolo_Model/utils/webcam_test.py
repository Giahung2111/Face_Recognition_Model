import cv2
from ultralytics import YOLO

# Load mô hình đã train
model = YOLO("../best.pt")  # Đường dẫn tới file best.pt

# Khởi động webcam
cap = cv2.VideoCapture(0)  # Webcam mặc định (0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán với YOLO
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]
            color = (255, 255, 0) if label == 'with_mask' else (0, 0, 255)

            # Vẽ bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Xử lý theo class
            if label == "with_mask":
                cv2.putText(frame, "Yeu cau nguoi dung thao khau trang",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif label == "without_mask":
                # Đưa ảnh khuôn mặt vào face recognition ở bước sau
                face_img = frame[y1:y2, x1:x2]
                # TODO: Gọi function nhận diện khuôn mặt tại đây
                pass

    # Hiển thị kết quả
    cv2.imshow("Mask Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
