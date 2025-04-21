"""
Module chuyển đổi dataset từ định dạng gốc sang định dạng YOLO.
Sử dụng MTCNN để phát hiện khuôn mặt và tạo bounding box.
"""

import os
import cv2
import torch
from facenet_pytorch import MTCNN
from typing import Dict, List, Tuple, Optional

class DatasetConverter:
    def __init__(self, dataset_path: str, output_path: str):
        """
        Khởi tạo converter
        
        Args:
            dataset_path: Đường dẫn đến dataset gốc
            output_path: Đường dẫn để lưu dataset YOLO
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.mtcnn = MTCNN(keep_all=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = {"with_mask": 0, "without_mask": 1}
        
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "labels"), exist_ok=True)

    def convert_bbox_to_yolo(self, box: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Chuyển đổi bounding box sang định dạng YOLO
        
        Args:
            box: [x1, y1, x2, y2]
            img_width: Chiều rộng ảnh
            img_height: Chiều cao ảnh
            
        Returns:
            Tuple chứa (x_center, y_center, width, height) theo định dạng YOLO
        """
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        bbox_width = (x2 - x1) / img_width
        bbox_height = (y2 - y1) / img_height
        return x_center, y_center, bbox_width, bbox_height

    def process_image(self, img_path: str, class_id: int) -> bool:
        """
        Xử lý một ảnh và tạo file label tương ứng
        
        Args:
            img_path: Đường dẫn đến ảnh
            class_id: ID của class
            
        Returns:
            bool: True nếu xử lý thành công, False nếu thất bại
        """
        # Đọc ảnh
        image = cv2.imread(img_path)
        if image is None:
            return False
            
        height, width, _ = image.shape
        
        # Phát hiện khuôn mặt
        boxes, _ = self.mtcnn.detect(image)
        if boxes is None:
            return False
            
        # Lưu ảnh
        img_name = os.path.basename(img_path)
        yolo_img_path = os.path.join(self.output_path, "images", img_name)
        cv2.imwrite(yolo_img_path, image)
        
        # Lưu label
        yolo_label_path = os.path.join(
            self.output_path, 
            "labels", 
            img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        
        with open(yolo_label_path, "w") as f:
            for box in boxes:
                x_center, y_center, bbox_width, bbox_height = self.convert_bbox_to_yolo(box, width, height)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                
        return True

    def convert(self) -> None:
        """Chuyển đổi toàn bộ dataset"""
        for class_name, class_id in self.classes.items():
            class_path = os.path.join(self.dataset_path, class_name)
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.process_image(img_path, class_id)
                
        print("Dataset YOLO đã được tạo thành công!")

def main():
    """Hàm main để chạy converter"""
    DATASET_PATH = "C:\\Python\\Face_Recognition_Smart_Home\\data\\face_detection_data\\raw"
    OUTPUT_PATH = "C:\\Python\\Face_Recognition_Smart_Home\\data\\face_detection_data\\processed"
    
    converter = DatasetConverter(DATASET_PATH, OUTPUT_PATH)
    converter.convert()

if __name__ == "__main__":
    main() 