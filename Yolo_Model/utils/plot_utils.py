# plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np

def plot_training_metrics(metrics_file: str):
    """
    Vẽ đồ thị từ file metrics đã lưu
    
    Args:
        metrics_file: Đường dẫn đến file metrics JSON
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create plots 
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    axes[0,0].plot(metrics['train']['loss'])
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    
    # Plot mAP
    axes[0,1].plot(metrics['train']['map'])
    axes[0,1].set_title('Mean Average Precision')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('mAP50-95')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def analyze_model_performance(results_file: str):
    """
    Phân tích chi tiết performance của model
    
    Args:
        results_file: File kết quả training
    """
    # Load results
    results = json.load(open(results_file))
    
    # Tạo các visualization khác nhau
    # ... (thêm code phân tích chi tiết)
