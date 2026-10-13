"""
Medical Image Dataset Download & Preparation Script
====================================================
This script downloads and prepares medical image datasets for training.
Supports: Chest X-ray (Pneumonia), Brain MRI (Tumor), Bone Fracture datasets.

For demo purposes, it also generates synthetic training data if downloads fail.
"""

import os
import sys
import json
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import urllib.request

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed")

CATEGORIES = ["normal", "pneumonia", "tumor", "fracture"]
IMG_SIZE = 224
TRAIN_SPLIT = 0.8

# ─── Synthetic Data Generator ─────────────────────────────────────────────────

def generate_synthetic_xray(category, size=(IMG_SIZE, IMG_SIZE)):
    """Generate synthetic medical-like images for training/demo purposes."""
    img = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(img)
    
    if category == "normal":
        # Clear lung-like region
        draw.ellipse([40, 30, 100, 190], fill=60)
        draw.ellipse([124, 30, 184, 190], fill=60)
        # Spine
        draw.rectangle([105, 20, 119, 210], fill=120)
        # Ribs
        for i in range(6):
            y = 40 + i * 25
            draw.arc([35, y, 105, y+20], 0, 180, fill=100)
            draw.arc([119, y, 189, y+20], 0, 180, fill=100)
        # Add noise
        noise = np.random.normal(0, 8, size).astype(np.uint8)
        img_array = np.clip(np.array(img).astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
            
    elif category == "pneumonia":
        # Lung region with opacity (white patches = fluid/infection)
        draw.ellipse([40, 30, 100, 190], fill=60)
        draw.ellipse([124, 30, 184, 190], fill=60)
        draw.rectangle([105, 20, 119, 210], fill=120)
        # Pneumonia patches (consolidation)
        num_patches = np.random.randint(3, 7)
        for _ in range(num_patches):
            cx = np.random.randint(50, 180)
            cy = np.random.randint(50, 180)
            rx = np.random.randint(10, 30)
            ry = np.random.randint(10, 30)
            brightness = np.random.randint(100, 180)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=brightness)
        noise = np.random.normal(0, 12, size).astype(np.uint8)
        img_array = np.clip(np.array(img).astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
    elif category == "tumor":
        # Brain MRI-like image
        draw.ellipse([30, 20, 194, 204], fill=80)
        draw.ellipse([50, 40, 174, 184], fill=50)
        # Tumor mass (bright region)
        tx = np.random.randint(70, 150)
        ty = np.random.randint(50, 160)
        tr = np.random.randint(15, 35)
        draw.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill=200)
        # Edema around tumor
        draw.ellipse([tx-tr-8, ty-tr-8, tx+tr+8, ty+tr+8], outline=140, width=3)
        noise = np.random.normal(0, 6, size).astype(np.uint8)
        img_array = np.clip(np.array(img).astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
    elif category == "fracture":
        # Bone X-ray-like image
        draw.rectangle([80, 10, 144, 214], fill=160)
        # Fracture line
        fy = np.random.randint(70, 160)
        points = [(80, fy), (95, fy+np.random.randint(-10, 10)),
                  (112, fy+np.random.randint(-15, 15)),
                  (130, fy+np.random.randint(-10, 10)), (144, fy)]
        draw.line(points, fill=30, width=2)
        # Displacement
        if np.random.random() > 0.5:
            draw.rectangle([80, fy, 144, fy+3], fill=20)
        # Soft tissue
        draw.ellipse([60, 20, 164, 214], outline=40, width=2)
        noise = np.random.normal(0, 10, size).astype(np.uint8)
        img_array = np.clip(np.array(img).astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    return img.convert('RGB')


def generate_dataset(num_per_class=250):
    """Generate synthetic dataset for all categories."""
    print("\n" + "="*60)
    print("  MEDICAL IMAGE DATASET GENERATOR")
    print("="*60)
    
    for split in ["train", "test"]:
        for category in CATEGORIES:
            dir_path = os.path.join(DATA_DIR, split, category)
            os.makedirs(dir_path, exist_ok=True)
    
    for category in CATEGORIES:
        train_count = int(num_per_class * TRAIN_SPLIT)
        test_count = num_per_class - train_count
        
        print(f"\n  Generating {category.upper()} images...")
        print(f"    Train: {train_count} | Test: {test_count}")
        
        # Generate training images
        for i in range(train_count):
            img = generate_synthetic_xray(category)
            img_path = os.path.join(DATA_DIR, "train", category, f"{category}_{i:04d}.png")
            img.save(img_path)
            if (i + 1) % 50 == 0:
                print(f"    ✓ Generated {i+1}/{train_count} training images")
        
        # Generate test images
        for i in range(test_count):
            img = generate_synthetic_xray(category)
            img_path = os.path.join(DATA_DIR, "test", category, f"{category}_test_{i:04d}.png")
            img.save(img_path)
        
        print(f"    ✅ {category.upper()} complete!")
    
    # Save dataset info
    info = {
        "categories": CATEGORIES,
        "img_size": IMG_SIZE,
        "train_per_class": int(num_per_class * TRAIN_SPLIT),
        "test_per_class": num_per_class - int(num_per_class * TRAIN_SPLIT),
        "total_images": num_per_class * len(CATEGORIES),
    }
    
    with open(os.path.join(DATA_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "="*60)
    print(f"  ✅ Dataset generated successfully!")
    print(f"  📁 Location: {DATA_DIR}")
    print(f"  📊 Total images: {info['total_images']}")
    print("="*60 + "\n")


def download_kaggle_datasets():
    """
    Attempt to download real medical datasets from Kaggle.
    Requires kaggle API credentials (~/.kaggle/kaggle.json).
    Falls back to synthetic data if unavailable.
    """
    try:
        import kaggle
        print("Kaggle API found. Downloading datasets...")
        
        datasets = {
            "chest-xray": "paultimothymooney/chest-xray-pneumonia",
            "brain-mri": "navoneel/brain-mri-images-for-brain-tumor-detection",
        }
        
        for name, dataset_id in datasets.items():
            print(f"  Downloading {name}...")
            kaggle.api.dataset_download_files(
                dataset_id,
                path=os.path.join(BASE_DIR, "raw", name),
                unzip=True
            )
            print(f"  ✅ {name} downloaded!")
        
        return True
        
    except (ImportError, Exception) as e:
        print(f"\n  ⚠️  Kaggle API not available: {e}")
        print("  📌 Falling back to synthetic dataset generation...")
        return False


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏥 Medical AI Project - Dataset Preparation")
    print("-" * 50)
    
    # Try Kaggle first, fall back to synthetic
    if not download_kaggle_datasets():
        generate_dataset(num_per_class=250)
    
    print("\n✅ Dataset preparation complete!")
    print("   Run 'python models/train_model.py' to train the model.\n")
