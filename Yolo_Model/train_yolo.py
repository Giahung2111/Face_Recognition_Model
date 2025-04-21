# train_yolo.py
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import time
import os
from utils.custom_callbacks import MetricsCallback
from utils.plot_utils import plot_training_metrics

def load_config(config_path="configs/yolo_config.yaml"):
    """
    Load config từ file yaml
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) # Hàm trong thư viện yaml

def train_yolo(config_path="configs/yolo_config.yaml"):
    """
    Huấn luyện model YOLOv8
    
    Args:
        config_path: Đường dẫn đến file config
    """
    # Load config
    config = load_config(config_path)
    
    # Khởi tạo model
    model = YOLO(config['model']['name']) #Yolov8s.pt
    
    # Tạo thư mục lưu kết quả
    run_dir = Path("runs/train")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks
    metrics_callback = MetricsCallback(save_dir=str(run_dir))

    # Đăng ký callback với model
    model.add_callback("on_train_epoch_end", metrics_callback.on_train_epoch_end)
    
    # Cấu hình tham số training
    train_args = {
        "data": config['data']['yaml_path'],
        "epochs": config['training']['epochs'],
        "batch": config['training']['batch_size'],
        "imgsz": config['model']['image_size'],
        "save_period": config.get('callbacks', {}).get('save_period', 1),
        "device": config['training']['device'],
        "patience": config.get('callbacks', {}).get('early_stopping', {}).get('patience', 10),
        "project": str(run_dir.parent), # Nơi lưu kết quả huấn luyện mô hình
        "name": run_dir.name,
    }
    
    # Bắt đầu training
    print("Bắt đầu training...")
    start_time = time.time()
    
    try:
        results = model.train(**train_args)
        
        training_time = time.time() - start_time
        print(f"Training hoàn thành sau {training_time:.2f} giây!")
        
        # Vẽ đồ thị metrics (sử dụng hàm từ plot_utils.py)
        plot_training_metrics(str(run_dir / "metrics_history.json"))
        
        return results
    except Exception as e:
        print(f"Lỗi trong quá trình training: {str(e)}")
        return None

def load_model_from_epoch(epoch, run_dir="runs/train"):
    """
    Load model từ một epoch cụ thể
    
    Args:
        epoch: Số epoch muốn load
        run_dir: Thư mục chứa kết quả training
    """
    checkpoint_path = Path(run_dir) / "weights" / f"epoch{epoch}.pt"
    
    if not checkpoint_path.exists():
        print(f"Không tìm thấy checkpoint cho epoch {epoch}")
        return None
    
    print(f"Loading model từ epoch {epoch}...")
    return YOLO(str(checkpoint_path))

if __name__ == "__main__":
    # Train model
    results = train_yolo()
    
    # Ví dụ load model từ epoch 50
    # model = load_model_from_epoch(50)    