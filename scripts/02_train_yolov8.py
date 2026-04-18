"""
YOLOv8 Training Script for Vehicle Detection
Using Pre-trained weights and Fine-tuning
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolov8():
    """Train YOLOv8 model with GPU acceleration"""
    
    # Check GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load pre-trained YOLOv8n (nano - fastest)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    print("\nLoading pre-trained YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Using nano for speed with RTX 4060
    
    # Training configuration
    results = model.train(
        data='data/dataset.yaml',  # Dataset configuration
        epochs=50,                  # Number of epochs (50 cukup untuk fine-tuning)
        batch=16,                   # Batch size (adjust based on GPU memory)
        imgsz=640,                  # Image size
        device=device,              # GPU device
        
        # Optimizer settings
        optimizer='auto',           # Auto-select optimizer
        lr0=0.01,                   # Initial learning rate
        lrf=0.01,                   # Final learning rate factor
        momentum=0.937,             # SGD momentum
        weight_decay=0.0005,        # Weight decay
        
        # Training settings
        workers=8,                  # Number of workers for data loading
        patience=50,                # Early stopping patience
        save=True,                  # Save checkpoints
        save_period=10,             # Save checkpoint every N epochs
        
        # Validation
        val=True,                   # Validate during training
        plots=True,                 # Generate plots
        
        # Augmentation
        mosaic=1.0,                 # Mosaic augmentation
        mixup=0.0,                  # Mixup augmentation
        copy_paste=0.0,             # Copy-paste augmentation
        
        # Project settings
        project='outputs/yolov8',
        name='exp1',
        exist_ok=True
    )
    
    print("\n" + "="*60)
    print("YOLOv8 Training Completed!")
    print("="*60)
    print(f"Best model saved to: outputs/yolov8/exp1/weights/best.pt")
    
    return results

def validate_yolov8():
    """Validate trained YOLOv8 model"""
    
    print("\n" + "="*60)
    print("Validating YOLOv8 Model")
    print("="*60)
    
    model = YOLO('outputs/yolov8/exp1/weights/best.pt')
    
    results = model.val(
        data='data/dataset.yaml',
        batch=16,
        imgsz=640,
        device='cuda:0',
        plots=True,
        save_json=True
    )
    
    print(f"\nValidation Results:")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results

def export_yolov8():
    """Export model to ONNX for deployment"""
    
    model = YOLO('outputs/yolov8/exp1/weights/best.pt')
    
    # Export to ONNX
    model.export(format='onnx', imgsz=640)
    print("Model exported to ONNX format")
    
    # Export to TorchScript
    model.export(format='torchscript', imgsz=640)
    print("Model exported to TorchScript format")

if __name__ == "__main__":
    print("="*60)
    print("YOLOv8 TRAINING PIPELINE")
    print("="*60)
    
    # Train model
    train_yolov8()
    
    # Validate model
    validate_yolov8()
    
    # Export model
    export_yolov8()
    
    print("\n✅ YOLOv8 training complete!")