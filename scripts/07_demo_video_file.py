"""
Demo dengan video file yang diupload sendiri - MENGGUNAKAN 2 MODEL
Output: 2 video (YOLOv8 dan Faster R-CNN)
"""

import cv2
import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import numpy as np
from pathlib import Path

def load_faster_rcnn_model(weights_path='outputs/faster-rcnn/final_model.pth'):
    """Load Faster R-CNN model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Faster R-CNN model on {device}...")
    
    # Load pre-trained model structure
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace classifier head for 2 classes (background + car)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    # Load trained weights
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Faster R-CNN model loaded from {weights_path}")
    else:
        print(f"❌ Faster R-CNN weights not found at {weights_path}")
        return None
    
    model.to(device)
    model.eval()
    
    return model, device

def process_video_with_yolo(video_path, output_path, model):
    """Process video using YOLOv8"""
    
    print(f"\n📹 Processing with YOLOv8...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    
    frame_count = 0
    total_time = 0
    detections_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection every 2 frames for speed
        if frame_count % 2 == 0:
            start_time = time.time()
            results = model(frame, conf=0.5)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Draw detections
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections_count += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Car: {conf:.2f}', (int(x1), int(y1)-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS counter
        if frame_count % 10 == 0 and total_time > 0:
            avg_fps = frame_count / total_time
            cv2.putText(frame, f'YOLOv8 - FPS: {avg_fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add model label
        cv2.putText(frame, 'Model: YOLOv8', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"   ✅ YOLOv8: {frame_count} frames processed, {avg_fps:.1f} FPS, {detections_count} detections")
    
    return {'frames': frame_count, 'fps': avg_fps, 'detections': detections_count}

def process_video_with_faster_rcnn(video_path, output_path, model, device):
    """Process video using Faster R-CNN"""
    
    print(f"\n📹 Processing with Faster R-CNN...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    
    frame_count = 0
    total_time = 0
    detections_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection every 2 frames for speed
        if frame_count % 2 == 0:
            # Preprocess for Faster R-CNN
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                predictions = model(image_tensor)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Draw detections
            for pred in predictions:
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score >= 0.5 and label == 1:  # Car class
                        x1, y1, x2, y2 = box.astype(int)
                        detections_count += 1
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Car: {score:.2f}', (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add FPS counter
        if frame_count % 10 == 0 and total_time > 0:
            avg_fps = frame_count / total_time
            cv2.putText(frame, f'Faster R-CNN - FPS: {avg_fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Add model label
        cv2.putText(frame, 'Model: Faster R-CNN', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"   ✅ Faster R-CNN: {frame_count} frames processed, {avg_fps:.1f} FPS, {detections_count} detections")
    
    return {'frames': frame_count, 'fps': avg_fps, 'detections': detections_count}

def demo_on_uploaded_video(video_path="demo_video.mp4"):
    """Run detection on uploaded video using BOTH models"""
    
    print("="*60)
    print("🎬 VEHICLE DETECTION DEMO - 2 MODELS")
    print("="*60)
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("\nPlease place your video file as 'demo_video.mp4' in the project folder")
        print("Or change the video_path variable to your file path")
        return
    
    # Load YOLOv8 model
    print("\n📦 Loading YOLOv8 model...")
    yolo_model = YOLO('outputs/yolov8/exp1/weights/best.pt')
    print("✅ YOLOv8 model loaded")
    
    # Load Faster R-CNN model
    frcnn_model, device = load_faster_rcnn_model()
    if frcnn_model is None:
        print("❌ Cannot proceed without Faster R-CNN model")
        return
    
    # Process video with YOLOv8
    print("\n" + "="*60)
    yolo_output = "outputs/results/demo_output_yolov8.mp4"
    yolo_stats = process_video_with_yolo(video_path, yolo_output, yolo_model)
    
    # Process video with Faster R-CNN
    print("\n" + "="*60)
    frcnn_output = "outputs/results/demo_output_faster_rcnn.mp4"
    frcnn_stats = process_video_with_faster_rcnn(video_path, frcnn_output, frcnn_model, device)
    
    # Print comparison summary
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    if yolo_stats and frcnn_stats:
        print(f"\n{'Metric':<20} {'YOLOv8':<20} {'Faster R-CNN':<20}")
        print("-" * 60)
        print(f"{'FPS':<20} {yolo_stats['fps']:.1f}{'':<15} {frcnn_stats['fps']:.1f}")
        print(f"{'Total Detections':<20} {yolo_stats['detections']:<20} {frcnn_stats['detections']:<20}")
        print(f"{'Frames Processed':<20} {yolo_stats['frames']:<20} {frcnn_stats['frames']:<20}")
        
        # Speed comparison
        speed_ratio = yolo_stats['fps'] / frcnn_stats['fps'] if frcnn_stats['fps'] > 0 else 0
        print(f"\n🏆 YOLOv8 is {speed_ratio:.1f}x FASTER than Faster R-CNN")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print(f"\n📁 Output videos saved to:")
    print(f"   - YOLOv8: {yolo_output}")
    print(f"   - Faster R-CNN: {frcnn_output}")
    print(f"\n🎥 You can now use these videos for your submission!")

if __name__ == "__main__":
    # Create output directory
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    
    # Method 1: Use default filename (place your video as 'demo_video.mp4' in project folder)
    demo_on_uploaded_video("demo_video.mp4")
    
    # Method 2: Or specify your full video path (uncomment and use this instead)
    # video_path = r"D:\Kuliah 2\Semester 2 Periode 1\vehicle-detection-final\Cars Moving On Road Stock Footage - Free Download.mp4"
    # demo_on_uploaded_video(video_path)