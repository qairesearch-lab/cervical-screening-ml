#!/usr/bin/env python3
"""
SIPaKMeD Dataset Preprocessing Script
=====================================
This script preprocesses the SIPaKMeD cervical cell dataset for multimodal classification.

Usage:
    python -m src.preprocess

The script performs:
    1. Data validation and loading
    2. Image preprocessing (resize, normalization)
    3. Train/val/test split (70/15/15)
    4. Feature normalization (if morphological features available)
    5. Data split preservation for reproducibility
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Set random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Path configuration
RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
SPLITS_DIR = Path("./data/splits")

# Class mapping
CLASS_NAMES = {
    0: "superficial-intermediate",
    1: "parabasal",
    2: "koilocytes",
    3: "dyskeratotic",
    4: "metaplastic"
}


def check_raw_data():
    """Validate raw data existence"""
    print("=" * 60)
    print("Checking Raw Data")
    print("=" * 60)

    if not RAW_DIR.exists():
        print(f"[ERROR] Raw data directory not found: {RAW_DIR}")
        print("\nPlease download the SIPaKMeD dataset:")
        print("1. Kaggle: kaggle datasets download -d mohaliy2016/papsinglecell")
        print("2. Extract to: ./data/raw/")
        return False

    image_dirs = list(RAW_DIR.glob("*/"))
    if not image_dirs:
        print("[ERROR] No image subdirectories found")
        return False

    print(f"[OK] Found {len(image_dirs)} class directories")
    for d in image_dirs:
        images = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.bmp"))
        print(f"  {d.name}: {len(images)} images")

    return True


def load_images_and_labels():
    """Load image paths and labels"""
    print("\n" + "=" * 60)
    print("Loading Image Data")
    print("=" * 60)

    image_paths = []
    labels = []

    # Iterate through each class directory
    for class_idx, class_name in CLASS_NAMES.items():
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            # Try alternative naming
            class_dir = RAW_DIR / class_name.replace("-", "_")

        if not class_dir.exists():
            print(f"[ERROR] Directory not found: {class_name}")
            continue

        # Get all images in this class
        images = list(class_dir.glob("*.png")) + \
                 list(class_dir.glob("*.jpg")) + \
                 list(class_dir.glob("*.bmp"))

        print(f"[OK] {class_name}: {len(images)} images")

        for img_path in images:
            image_paths.append(str(img_path))
            labels.append(class_idx)

    print(f"\nTotal: {len(image_paths)} images")

    return np.array(image_paths), np.array(labels)


def extract_or_load_features(image_paths, labels):
    """Extract or load 26 morphological features"""
    print("\n" + "=" * 60)
    print("Processing Morphological Features")
    print("=" * 60)

    # Check if precomputed features file exists
    feature_file = RAW_DIR / "features.csv"
    if feature_file.exists():
        print(f"[OK] Found precomputed features: {feature_file}")
        features_df = pd.read_csv(feature_file)
        
        # Assuming features file contains image names and 26 features
        # May need adjustment based on actual file format
        print(f"[OK] Loaded {len(features_df)} feature records")
        
        # Extract feature matrix (assuming last 26 columns are features)
        feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
        if len(feature_cols) == 0:
            # If no feature_ prefix, assume last 26 columns are features
            feature_cols = features_df.columns[-26:].tolist()
        
        features = features_df[feature_cols].values
        print(f"[OK] Feature dimension: {features.shape}")
        
        return features
    
    else:
        print("[WARNING] Precomputed features not found")
        print("\nThe SIPaKMeD dataset includes 26 hand-crafted morphological features:")
        print("- Intensity features: mean, std, skewness, kurtosis, etc.")
        print("- Texture features: GLCM contrast, correlation, energy, homogeneity")
        print("- Shape features: area, perimeter, circularity, eccentricity")
        print("\nPlease ensure downloaded data contains features.xlsx or features.csv")
        print("If not, manual calculation is required or contact dataset authors")
        
        # Return None for features (will be handled later)
        return None


def preprocess_images(image_paths, target_size=(224, 224)):
    """Preprocess images"""
    print("\n" + "=" * 60)
    print("Preprocessing Images")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_paths = []

    for idx, img_path in enumerate(image_paths):
        # Open image
        img = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)
        
        # Save preprocessed image
        relative_path = Path(img_path).relative_to(RAW_DIR)
        save_path = PROCESSED_DIR / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        img.save(save_path, quality=95)
        processed_paths.append(str(save_path))
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images")

    print(f"[OK] Image preprocessing complete, saved to {PROCESSED_DIR}")

    return np.array(processed_paths)


def split_data(image_paths, labels, features=None):
    """Split dataset"""
    print("\n" + "=" * 60)
    print("Splitting Dataset")
    print("=" * 60)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # First split: train + temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    if features is not None:
        train_features, temp_features = train_test_split(
            features,
            test_size=0.3,
            random_state=RANDOM_SEED,
            stratify=labels
        )

    # Second split: val + test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    if features is not None:
        val_features, test_features = train_test_split(
            temp_features,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_labels
        )

    splits = {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels}
    }

    if features is not None:
        splits['train']['features'] = train_features
        splits['val']['features'] = val_features
        splits['test']['features'] = test_features

    # Print split information
    for split_name, split_data in splits.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {len(split_data['paths'])}")
        for class_idx, class_name in CLASS_NAMES.items():
            count = np.sum(split_data['labels'] == class_idx)
            print(f"    {class_name}: {count}")

    return splits


def save_splits(splits):
    """Save split data"""
    print("\n" + "=" * 60)
    print("Saving Data Splits")
    print("=" * 60)

    for split_name, split_data in splits.items():
        # Save as numpy arrays
        np.save(SPLITS_DIR / f"{split_name}_paths.npy", split_data['paths'])
        np.save(SPLITS_DIR / f"{split_name}_labels.npy", split_data['labels'])
        
        if 'features' in split_data:
            np.save(SPLITS_DIR / f"{split_name}_features.npy", split_data['features'])
        
        # Save as CSV (for easy viewing)
        df = pd.DataFrame({
            'path': split_data['paths'],
            'label': split_data['labels'],
            'class_name': [CLASS_NAMES[l] for l in split_data['labels']]
        })
        df.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)
        
        print(f"[OK] {split_name} saved")

    print(f"\n[OK] All splits saved to {SPLITS_DIR}")


def normalize_features(splits):
    """Normalize features"""
    if 'features' not in splits['train']:
        return splits

    print("\n" + "=" * 60)
    print("Normalizing Features")
    print("=" * 60)

    # Use training set statistics for normalization
    scaler = StandardScaler()
    splits['train']['features'] = scaler.fit_transform(splits['train']['features'])
    splits['val']['features'] = scaler.transform(splits['val']['features'])
    splits['test']['features'] = scaler.transform(splits['test']['features'])

    # Save scaler
    import joblib
    joblib.dump(scaler, SPLITS_DIR / 'feature_scaler.pkl')
    print("[OK] Features normalized, scaler saved")

    return splits


def generate_metadata(splits):
    """Generate metadata"""
    print("\n" + "=" * 60)
    print("Generating Metadata")
    print("=" * 60)

    metadata = {
        'dataset': 'SIPaKMeD',
        'total_samples': sum(len(s['paths']) for s in splits.values()),
        'num_classes': 5,
        'class_names': CLASS_NAMES,
        'splits': {},
        'features': {
            'count': 26 if 'features' in splits['train'] else 0,
            'types': ['intensity', 'texture', 'shape'] if 'features' in splits['train'] else [],
            'normalized': True
        }
    }

    for split_name, split_data in splits.items():
        class_counts = {}
        for class_idx, class_name in CLASS_NAMES.items():
            class_counts[class_name] = int(np.sum(split_data['labels'] == class_idx))

        metadata['splits'][split_name] = {
            'num_samples': len(split_data['paths']),
            'class_distribution': class_counts
        }

    # Save metadata
    with open(SPLITS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("[OK] Metadata saved")
    print(json.dumps(metadata, indent=2))


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("SIPaKMeD Data Preprocessing")
    print("=" * 60)

    # Check raw data
    if not check_raw_data():
        return

    # Load images and labels
    image_paths, labels = load_images_and_labels()

    # Load/extract features
    features = extract_or_load_features(image_paths, labels)

    # Preprocess images
    processed_paths = preprocess_images(image_paths)

    # Split data
    splits = split_data(processed_paths, labels, features)

    # Normalize features
    splits = normalize_features(splits)

    # Save splits
    save_splits(splits)

    # Generate metadata
    generate_metadata(splits)

    print("\n" + "=" * 60)
    print("[OK] Preprocessing complete!")
    print("=" * 60)
    print("\nNext: Run training scripts")
    print("  python -m src.train")


if __name__ == "__main__":
    main()