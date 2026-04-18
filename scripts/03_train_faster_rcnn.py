"""
Faster R-CNN Training - ULTRA SIMPLE VERSION (PASTI JALAN)
Tanpa validation loss tracking untuk menghindari error
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class SimpleVehicleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        print(f"Loaded {len(self.image_files)} images from {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, xc, yc, bw, bh = map(float, parts)
                        x1 = (xc - bw/2) * w
                        y1 = (yc - bh/2) * h
                        x2 = (xc + bw/2) * w
                        y2 = (yc + bh/2) * h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Normalize image
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model():
    """Main training function - simplified"""
    
    # Create directories
    Path("outputs/faster-rcnn").mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FASTER R-CNN TRAINING - SIMPLE VERSION")
    print("="*60)
    
    # Check datasets
    if not Path('data/train').exists():
        print("ERROR: data/train not found!")
        return None
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SimpleVehicleDataset('data/train')
    val_dataset = SimpleVehicleDataset('data/val')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("\nLoading pre-trained Faster R-CNN...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Training loop
    num_epochs = 30
    
    print("\n" + "="*60)
    print(f"Starting training for {num_epochs} epochs")
    print("="*60 + "\n")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss (loss_dict is always a dict in training mode)
            total_loss = 0
            for loss in loss_dict.values():
                total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        print("-" * 40)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'outputs/faster-rcnn/checkpoint_epoch_{epoch+1}.pth')
            print(f"  Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), 'outputs/faster-rcnn/final_model.pth')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal model saved to: outputs/faster-rcnn/final_model.pth")
    
    return model

if __name__ == "__main__":
    model = train_model()