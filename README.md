# Hệ thống Nhận diện Khuôn mặt với Siamese Neural Network

## Tổng quan

### Mục tiêu
Hệ thống này được xây dựng để xác minh danh tính khuôn mặt từ webcam, tích hợp các thành phần sau:
- **Phát hiện đối tượng (Face Detection)**: Sử dụng YOLO để phát hiện khuôn mặt và kiểm tra trạng thái “mask”/“no mask”. Dữ liệu đã được thu thập từ tập dataset phân loại mask và no-mask. Mô hình MTCNN được sử dụng để phát hiện khuôn mặt, từ đó tạo ra tập dataset detection hoàn chỉnh cho bài toán Detection.
- **Nhận diện khuôn mặt (Face Recognition)**: Sử dụng Siamese Neural Network với Binary Cross-Entropy Loss để xác minh xem khuôn mặt từ webcam có khớp với cơ sở dữ liệu khuôn mặt hợp lệ (anchor database) hay không.

### Đặc điểm của Siamese Network
- Nhận hai ảnh đầu vào (anchor và test).
- Tính khoảng cách L1 giữa embedding của hai ảnh.
- Xuất xác suất (sigmoid) rằng hai ảnh thuộc cùng một người.
- Loss: Binary Cross-Entropy (nhãn 1: cùng người, 0: khác người).
- Ứng dụng: Face Verification (kiểm tra cặp ảnh), có thể mở rộng cho recognition bằng cách so sánh với cơ sở dữ liệu.

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
- Sử dụng Siamese Network để xác minh từng cặp ảnh (webcam vs anchor), trả về xác suất.
- Lấy danh tính từ ảnh anchor có xác suất cao nhất (nếu vượt ngưỡng).

---

## Cài đặt

### Yêu cầu
- Python 3.x
- Các thư viện cần thiết được liệt kê trong `requirements.txt`.

### Cách sử dụng
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy mô hình:
   ```bash
   python train_yolo.py
   ```
3. Sử dụng Jupyter Notebook `main.ipynb` để thực hiện các bước nhận diện khuôn mặt.

---

## Tài liệu tham khảo
- Dữ liệu sẽ được upload chi tiết sau này !
- [YOLO: You Only Look Once](https://pjreddie.com/darknet/yolo/)
- [Siamese Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

---

## Liên hệ
Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ với tôi qua email: [ghung21112004@gmail.com].