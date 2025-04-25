# Hệ thống Nhận diện Khuôn mặt và Phát hiện Đối tượng

## Tổng quan

Hệ thống này bao gồm hai phần chính:
1. **Nhận diện khuôn mặt (Face Recognition)**: Sử dụng Siamese Neural Network để xác minh danh tính khuôn mặt.
2. **Phát hiện đối tượng (Yolo Model)**: Sử dụng YOLO để phát hiện khuôn mặt và kiểm tra trạng thái “mask”/“no mask”.

### Mục tiêu
- Xây dựng hệ thống Face Recognition để xác minh danh tính khuôn mặt từ webcam, tích hợp:
    - **Object Detection (Face Detection)**: Dùng YOLO để phát hiện khuôn mặt và kiểm tra trạng thái “mask”/“no mask”.
    - **Face Recognition**: Dùng Siamese Neural Network với Binary Cross-Entropy Loss để xác minh liệu khuôn mặt từ webcam có khớp với cơ sở dữ liệu khuôn mặt hợp lệ (anchor database) hay không.

### Đặc điểm của Siamese Network hiện tại
- Nhận hai ảnh đầu vào (anchor và test).
- Tính khoảng cách L1 giữa embedding của hai ảnh.
- Xuất xác suất (sigmoid) rằng hai ảnh thuộc cùng một người.
- Loss: Binary Cross-Entropy (nhãn 1: cùng người, 0: khác người).
- Ứng dụng: Face Verification (kiểm tra cặp ảnh), nhưng có thể mở rộng cho recognition bằng cách so sánh với cơ sở dữ liệu.

---

## Chia bài toán thành hai phần

### 1. Phát hiện đối tượng (Face Detection)
- Phát hiện vùng khuôn mặt trong khung hình webcam.
- Phân loại khuôn mặt: “mask” (đeo khẩu trang) hoặc “no mask” (không đeo khẩu trang).
- **Output**:
    - Nếu “mask”: Hiển thị thông báo “Yêu cầu tháo khẩu trang”.
    - Nếu “no mask”: Cắt vùng khuôn mặt và chuyển sang Face Recognition.

### 2. Nhận diện khuôn mặt (Verification)
- So sánh khuôn mặt từ webcam với cơ sở dữ liệu khuôn mặt hợp lệ (anchor_database).
- Dùng Siamese Network để xác minh từng cặp ảnh (webcam vs anchor), trả về xác suất.
- Lấy danh tính từ ảnh anchor có xác suất cao nhất (nếu vượt ngưỡng).

---

## Cài đặt

### Yêu cầu
- Python 3.x
- Các thư viện cần thiết được liệt kê trong `requirements.txt`.

### Cách sử dụng

#### 1. Nhận diện khuôn mặt (Face Recognition)

1. **Tạo thư mục dữ liệu**:
   - Tạo thư mục `data` trong thư mục gốc của dự án.

2. **Chạy mã để chụp khuôn mặt**:
   - Mở Jupyter Notebook `main.ipynb` và chạy các ô để chụp khuôn mặt của bạn vào cơ sở dữ liệu.

3. **Chạy inference**:
   - Tải mô hình đã huấn luyện từ https://drive.google.com/file/d/1TL1wryed38ZZUeFu2PiuoE7PzZeg0fH8/view?usp=sharing và chạy mã để thực hiện nhận diện khuôn mặt.

4. **Dataset**:
   - Tập dữ liệu CelebA có thể được tải từ https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

#### 2. Phát hiện đối tượng (Yolo Model)

1. **Chạy mã để chia dữ liệu**:
   - Chạy file `C:\Python\Face_Recognition_Smart_Home\utils\yolo_utils\split_data.py` để chia dữ liệu thành train/val.

2. **Huấn luyện mô hình**:
   - Chạy file `train_yolo.py` để huấn luyện mô hình phát hiện đối tượng.

3. **Kiểm tra webcam**:
   - Sử dụng file `webcam_test.py` để kiểm tra phát hiện khuôn mặt và trạng thái “mask”/“no mask”.

4. **Kết quả huấn luyện**:
   - Kết quả huấn luyện có thể được xem bên dưới.

---

## Thông số huấn luyện

### 1. Yolo Model
- **Train Loss**: 0.0105
- **Train Precision**: 0.9979
- **Train Recall**: 0.9971
- **Test Loss**: 0.0077
- **Test Precision**: 0.9970
- **Test Recall**: 1.0000

### 2. Face Recognition
- **Train Loss**: 0.2348
- **Train Precision**: 0.8863
- **Train Recall**: 0.9254
- **Test Loss**: 0.2479
- **Test Precision**: 0.9199
- **Test Recall**: 0.8739

---

## Tài liệu tham khảo

- [YOLO: You Only Look Once](https://pjreddie.com/darknet/yolo/)
- [Siamese Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
