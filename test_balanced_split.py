"""
Test script for balanced train/val splitting.
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path

print("=" * 60)
print("Testing Balanced Train/Val Split")
print("=" * 60)

# Test without torch to check split logic
from dataset import create_balanced_splits

# Create splits
train_samples, val_samples = create_balanced_splits(
    image_dir='datasets/train_images',
    mask_dir='datasets/train_masks',
    supplemental_image_dir='datasets/supplemental_images',
    supplemental_mask_dir='datasets/supplemental_masks',
    val_split=0.2,
    random_seed=42
)

print("\n" + "=" * 60)
print("Split Details")
print("=" * 60)

print("\nTrain samples:")
for sample in train_samples:
    label = "FORGED" if sample['is_forged'] else "AUTHENTIC"
    print(f"  {sample['case_id']}: {label}")

print("\nVal samples:")
for sample in val_samples:
    label = "FORGED" if sample['is_forged'] else "AUTHENTIC"
    print(f"  {sample['case_id']}: {label}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
