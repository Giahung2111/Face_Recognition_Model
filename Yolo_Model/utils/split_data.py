import os
import shutil
import random
from pathlib import Path

def split_data(source_dir: str, output_dir: str, train_ratio: float = 0.8):
    """
    Chia dữ liệu YOLO format thành tập train và validation
    
    Args:
        source_dir: Thư mục chứa dữ liệu (phải có cấu trúc images/ và labels/)
        output_dir: Thư mục đầu ra
        train_ratio: Tỷ lệ dữ liệu training (mặc định 0.8)
    """
    # Kiểm tra cấu trúc thư mục
    src_images_path = os.path.join(source_dir, "images")
    src_labels_path = os.path.join(source_dir, "labels")
    
    if not os.path.exists(src_images_path) or not os.path.exists(src_labels_path):
        raise ValueError(f"Thư mục {source_dir} phải có cả images/ và labels/")
    
    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(src_images_path) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Kiểm tra xem mỗi ảnh có file label tương ứng không
    valid_files = []
    for img_file in image_files:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(src_labels_path, label_file)):
            valid_files.append(img_file)
        else:
            print(f"Warning: Không tìm thấy label cho {img_file}")
    
    if not valid_files:
        raise ValueError("Không tìm thấy cặp ảnh-label hợp lệ nào!")
    
    # Xáo trộn dữ liệu
    random.shuffle(valid_files)
    
    # Chia tập train/val
    train_size = int(len(valid_files) * train_ratio)
    train_files = valid_files[:train_size]
    val_files = valid_files[train_size:]
    
    # Tạo thư mục đầu ra
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "labels"), exist_ok=True)
    
    # Hàm copy files
    def copy_files(files: list, subset: str):
        for img_file in files:
            # Copy ảnh
            src_img = os.path.join(src_images_path, img_file)
            dst_img = os.path.join(output_dir, subset, "images", img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(src_labels_path, label_file)
            dst_label = os.path.join(output_dir, subset, "labels", label_file)
            shutil.copy2(src_label, dst_label)
    
    # Copy files vào thư mục tương ứng
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    
    # In thống kê
    print(f"Tổng số ảnh hợp lệ: {len(valid_files)}")
    print(f"Số ảnh training: {len(train_files)} ({train_ratio*100}%)")
    print(f"Số ảnh validation: {len(val_files)} ({(1-train_ratio)*100}%)")

    # Tạo file data.yaml
    yaml_content = f"""
        train: {os.path.join(output_dir, 'train')}
        val: {os.path.join(output_dir, 'val')}
        nc: 2
        names: ['with_mask', 'without_mask']
    """
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
    
    print("✅ Đã tạo file data.yaml")

if __name__ == "__main__":
    source_dir = "../../data/face_detection_data/processed"
    output_dir = "../../data/yolo_dataset"
    split_data(source_dir, output_dir)
