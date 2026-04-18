"""
Inference Demo Script
Untuk video demo - test model on new images/videos
"""

import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

class VehicleDetectorDemo:
    def __init__(self):
        print("Loading models...")
        self.yolo_model = YOLO('outputs/yolov8/exp1/weights/best.pt')
        print("✅ YOLOv8 loaded")
        
        # Load Faster R-CNN
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        weights_path = 'outputs/faster-rcnn/final_model.pth'
        if Path(weights_path).exists():
            self.faster_rcnn.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("✅ Faster R-CNN loaded")
        
        self.faster_rcnn.to(self.device)
        self.faster_rcnn.eval()
    
    def detect_yolo(self, image, conf_threshold=0.5):
        """Run YOLOv8 detection"""
        results = self.yolo_model(image, conf=conf_threshold)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': 'car' if cls == 0 else 'unknown'
                })
        
        return detections
    
    def detect_faster_rcnn(self, image, conf_threshold=0.5):
        """Run Faster R-CNN detection"""
        
        # Preprocess
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.faster_rcnn(image_tensor)
        
        detections = []
        for pred in predictions:
            boxes = pred['boxes'].cpu().tolist()
            scores = pred['scores'].cpu().tolist()
            labels = pred['labels'].cpu().tolist()
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= conf_threshold and label > 0:
                    detections.append({
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'confidence': score,
                        'class': 'car'
                    })
        
        return detections
    
    def visualize_detection(self, image, detections, title="Detection Result"):
        """Visualize detection results"""
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        ax.imshow(image_rgb)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            ax.text(x1, y1-5, label, color='green', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def test_on_images(self, image_dir="data/test/images", num_samples=5):
        """Test on multiple images"""
        
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        selected = image_files[:num_samples]
        
        print(f"\nTesting on {len(selected)} images...")
        
        for img_path in selected:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            print(f"\nProcessing: {img_path.name}")
            
            # YOLO detection
            yolo_dets = self.detect_yolo(image)
            print(f"  YOLOv8 detected {len(yolo_dets)} objects")
            
            # Faster R-CNN detection
            frcnn_dets = self.detect_faster_rcnn(image)
            print(f"  Faster R-CNN detected {len(frcnn_dets)} objects")
            
            # Visualize
            fig1 = self.visualize_detection(image, yolo_dets, f"YOLOv8: {img_path.name}")
            fig1.savefig(f"outputs/results/yolo_{img_path.stem}.png")
            plt.close(fig1)
            
            fig2 = self.visualize_detection(image, frcnn_dets, f"Faster R-CNN: {img_path.name}")
            fig2.savefig(f"outputs/results/frcnn_{img_path.stem}.png")
            plt.close(fig2)
        
        print("\n✅ Results saved to outputs/results/")
    
    def test_on_video(self, video_path=None):
        """Test on video file"""
        
        if video_path is None:
            # Use test images as slideshow
            self.test_on_images()
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        print(f"\nProcessing video: {video_path}")
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 30th frame to save time
            if frame_count % 30 == 0:
                detections = self.detect_yolo(frame)
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['class']}: {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Resize for display
                frame_resized = cv2.resize(frame, (960, 540))
                cv2.imshow('Vehicle Detection Demo', frame_resized)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Processed {frame_count} frames")

if __name__ == "__main__":
    print("="*60)
    print("VEHICLE DETECTION DEMO")
    print("="*60)
    
    detector = VehicleDetectorDemo()
    
    # Test on test images
    detector.test_on_images()
    
    # Uncomment to test on your own video
    # detector.test_on_video("path/to/your/video.mp4")
    
    print("\n✅ Demo complete! Check outputs/results/ for results")