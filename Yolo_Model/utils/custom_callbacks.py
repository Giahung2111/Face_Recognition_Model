# custom_callbacks.py
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

class MetricsCallback:
    def __init__(self, save_dir="runs"):
        self.save_dir = Path(save_dir)
        self.metrics_history = {
            'train': {'loss': [], 'map': []},
            'val': {'loss': [], 'map': []}
        }
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer):
        """Được gọi sau mỗi epoch training"""
        metrics = trainer.metrics
        # Lưu metrics train
        self.metrics_history['train']['loss'].append(metrics.get('train/loss', 0.0))
        self.metrics_history['train']['map'].append(metrics.get('metrics/mAP50-95', 0.0))
        
        # Lưu metrics val (nếu có)
        self.metrics_history['val']['loss'].append(metrics.get('val/loss', 0.0))
        self.metrics_history['val']['map'].append(metrics.get('val/mAP50-95', 0.0))
        
        # Lưu metrics vào file
        json_path = self.save_dir / 'metrics_history.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        # Vẽ đồ thị
        self.plot_metrics()

    def plot_metrics(self):
        """Vẽ đồ thị metrics"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics_history['train']['loss'], label='Train Loss')
        if self.metrics_history['val']['loss']:
            plt.plot(self.metrics_history['val']['loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot mAP
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics_history['train']['map'], label='Train mAP50-95')
        if self.metrics_history['val']['map']:
            plt.plot(self.metrics_history['val']['map'], label='Val mAP50-95')
        plt.title('Mean Average Precision')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_plot.png')
        plt.close()