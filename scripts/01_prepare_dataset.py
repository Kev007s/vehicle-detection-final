"""
Data Preparation Script untuk Car Object Detection Dataset
Convert single object detection ke multi-object format untuk YOLO
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# Konfigurasi
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

# Buat direktori
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)

def load_and_analyze_dataset():
    """Load dataset dan analisis"""
    
    # Load CSV bounding boxes
    train_csv = RAW_DIR / "train_solution_bounding_boxes (1).csv"
    
    if not train_csv.exists():
        print(f"File tidak ditemukan: {train_csv}")
        print("Mencari file CSV...")
        csv_files = list(RAW_DIR.glob("*.csv"))
        print(f"CSV files found: {csv_files}")
        for f in csv_files:
            if 'train' in str(f).lower():
                train_csv = f
                break
    
    df = pd.read_csv(train_csv)
    print(f"\nDataset Info:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Cek unique images
    unique_images = df['image'].nunique()
    print(f"\nUnique images: {unique_images}")
    
    return df

def convert_to_yolo_format(df, img_dir, output_dir, is_train=True):
    """Convert bounding boxes ke YOLO format (normalized)"""
    
    images_dir = Path(img_dir)
    output_images_dir = Path(output_dir) / "images"
    output_labels_dir = Path(output_dir) / "labels"
    
    # Dapatkan semua gambar
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    print(f"\nProcessing {len(image_files)} images from {img_dir}")
    
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files, desc="Converting"):
        img_name = img_path.name
        
        # Cari bounding box untuk gambar ini
        bbox_rows = df[df['image'] == img_name]
        
        if len(bbox_rows) == 0:
            # Tidak ada bounding box, skip
            failed += 1
            continue
        
        # Load image untuk dapatkan dimensi
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue
            
        h, w = img.shape[:2]
        
        # Copy image ke output
        shutil.copy2(img_path, output_images_dir / img_name)
        
        # Buat label file (YOLO format)
        label_path = output_labels_dir / f"{img_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for _, row in bbox_rows.iterrows():
                # Get bounding box coordinates
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                
                # Convert to YOLO format (normalized)
                x_center = (xmin + xmax) / (2 * w)
                y_center = (ymin + ymax) / (2 * h)
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                
                # Class ID: 0 untuk car (single class)
                class_id = 0
                
                # Write to file
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        successful += 1
    
    print(f"Success: {successful}, Failed: {failed}")
    return successful

def split_dataset(df):
    """Split dataset menjadi train, val, test"""
    
    unique_images = df['image'].unique()
    
    # Split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    train_df = df[df['image'].isin(train_imgs)]
    val_df = df[df['image'].isin(val_imgs)]
    test_df = df[df['image'].isin(test_imgs)]
    
    print(f"\nSplit Results:")
    print(f"Train: {len(train_imgs)} images, {len(train_df)} bounding boxes")
    print(f"Val: {len(val_imgs)} images, {len(val_df)} bounding boxes")
    print(f"Test: {len(test_imgs)} images, {len(test_df)} bounding boxes")
    
    return train_df, val_df, test_df

def visualize_sample(output_dir, num_samples=5):
    """Visualisasi sample hasil konversi"""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    
    image_files = list(images_dir.glob("*.jpg"))[:num_samples]
    
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 5))
    if len(image_files) == 1:
        axes = [axes]
    
    for idx, img_path in enumerate(image_files):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, xc, yc, bw, bh = map(float, parts)
                        
                        # Convert to pixel coordinates
                        x = (xc - bw/2) * w
                        y = (yc - bh/2) * h
                        box_w = bw * w
                        box_h = bh * h
                        
                        # Draw bounding box
                        rect = patches.Rectangle(
                            (x, y), box_w, box_h,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        axes[idx].add_patch(rect)
                        axes[idx].text(x, y-5, f'Car', color='red', fontsize=10)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(img_path.name)
    
    plt.tight_layout()
    plt.savefig("outputs/logs/sample_detections.png", dpi=150)
    plt.show()
    print("Sample visualization saved to outputs/logs/sample_detections.png")

def create_dataset_yaml():
    """Create dataset.yaml for YOLO"""
    
    yaml_content = """
# Dataset configuration untuk YOLOv8
# Vehicle Detection Dataset

path: ./data
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 1

# Class names
names:
  0: car
"""
    
    with open("data/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print("dataset.yaml created successfully!")

if __name__ == "__main__":
    print("="*60)
    print("STEP 1: Loading and analyzing dataset")
    print("="*60)
    df = load_and_analyze_dataset()
    
    print("\n" + "="*60)
    print("STEP 2: Splitting dataset")
    print("="*60)
    train_df, val_df, test_df = split_dataset(df)
    
    print("\n" + "="*60)
    print("STEP 3: Converting training data")
    print("="*60)
    convert_to_yolo_format(train_df, RAW_DIR / "training_images", TRAIN_DIR, is_train=True)
    
    print("\n" + "="*60)
    print("STEP 4: Converting validation data")
    print("="*60)
    convert_to_yolo_format(val_df, RAW_DIR / "training_images", VAL_DIR, is_train=True)
    
    print("\n" + "="*60)
    print("STEP 5: Converting test data")
    print("="*60)
    convert_to_yolo_format(test_df, RAW_DIR / "training_images", TEST_DIR, is_train=True)
    
    print("\n" + "="*60)
    print("STEP 6: Visualizing samples")
    print("="*60)
    visualize_sample(TRAIN_DIR, num_samples=5)
    
    print("\n" + "="*60)
    print("STEP 7: Creating dataset.yaml")
    print("="*60)
    create_dataset_yaml()
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    
    # Print summary
    print("\nFinal Dataset Summary:")
    print(f"Train images: {len(list(TRAIN_DIR.glob('images/*')))}")
    print(f"Train labels: {len(list(TRAIN_DIR.glob('labels/*')))}")
    print(f"Val images: {len(list(VAL_DIR.glob('images/*')))}")
    print(f"Val labels: {len(list(VAL_DIR.glob('labels/*')))}")
    print(f"Test images: {len(list(TEST_DIR.glob('images/*')))}")
    print(f"Test labels: {len(list(TEST_DIR.glob('labels/*')))}")