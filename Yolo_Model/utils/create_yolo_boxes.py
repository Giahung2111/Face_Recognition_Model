import cv2
import os
import numpy as np
import glob

def draw_yolo_bboxes(image_path, label_path, class_names=None):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Lấy kích thước ảnh
    height, width, _ = image.shape
    
    # Đọc file nhãn
    if not os.path.exists(label_path):
        print(f"Không tìm thấy file nhãn: {label_path}")
        return image
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Xử lý từng dòng trong file nhãn
    for line in lines:
        # Tách các giá trị: class_id, center_x, center_y, width, height
        values = line.strip().split()
        if len(values) < 5:
            continue
        class_id, center_x, center_y, bbox_width, bbox_height = map(float, values[:5])
        
        # Chuyển đổi tọa độ YOLO sang pixel
        center_x = int(center_x * width)
        center_y = int(center_y * height)
        bbox_width = int(bbox_width * width)
        bbox_height = int(bbox_height * height)
        
        # Tính toán tọa độ góc trên-trái và dưới-phải của bounding box
        x_min = int(center_x - bbox_width / 2)
        y_min = int(center_y - bbox_height / 2)
        x_max = int(center_x + bbox_width / 2)
        y_max = int(center_y + bbox_height / 2)
        
        # Vẽ bounding box lên ảnh
        color = (0, 255, 0)  # Màu xanh lá (BGR)
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Nếu có tên lớp, hiển thị tên lớp
        if class_names:
            label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class {int(class_id)}"
            cv2.putText(image, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return image

# Hàm hiển thị hoặc lưu ảnh
def show_or_save_image(image, output_path=None):
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Đã lưu ảnh tại: {output_path}")
    else:
        cv2.imshow("Image with BBoxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thư mục chứa ảnh và nhãn
    image_dir = "C:\\Python\\Face_Recognition_Smart_Home\\Yolo_Model\\dataset\\train\\images"
    label_dir = "C:\\Python\\Face_Recognition_Smart_Home\\Yolo_Model\\dataset\\train\\labels"
    output_dir = "output/"
    class_names = ["with_mask", "without_mask"]

    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    # Lấy danh sách file ảnh
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

    # Kiểm tra xem có ảnh nào không
    if not image_files:
        print(f"Không tìm thấy file ảnh .jpg trong thư mục: {image_dir}")
    else:
        print(f"Đã tìm thấy {len(image_files)} file ảnh.")
        for image_path in image_files:
            # Tạo đường dẫn file nhãn tương ứng
            image_name = os.path.basename(image_path).split('.')[0]
            label_path = os.path.join(label_dir, f"{image_name}.txt")
            
            # Vẽ và lưu ảnh
            image_with_bboxes = draw_yolo_bboxes(image_path, label_path, class_names)
            output_path = os.path.join(output_dir, f"{image_name}_bbox.jpg")
            show_or_save_image(image_with_bboxes, output_path=output_path)