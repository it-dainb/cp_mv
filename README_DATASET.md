# Dataset Preprocessing Guide

This guide explains how to use the dataset preprocessing workflow.

## Overview

The dataset creation logic has been split into two parts:
1. **`create_dataset.py`** - Creates preprocessed dataset splits (train/val) and saves them to JSON files
2. **`train.py`** - Loads preprocessed splits from JSON files

## Benefits of Preprocessing

- **Consistency**: Same train/val split across multiple training runs
- **Speed**: Skip split creation during training
- **Reproducibility**: Share exact splits with team members
- **Flexibility**: Easy to modify splits without changing training code

## Usage

### Step 1: Create Preprocessed Splits

```bash
# Basic usage (without supplemental data)
python create_dataset.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --output-dir datasets/splits \
    --val-split 0.2 \
    --seed 42

# With supplemental data
python create_dataset.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --supplemental-images datasets/supplemental_images \
    --supplemental-masks datasets/supplemental_masks \
    --output-dir datasets/splits \
    --val-split 0.2 \
    --seed 42
```

This creates:
- `datasets/splits/train_samples.json` - Training samples
- `datasets/splits/val_samples.json` - Validation samples
- `datasets/splits/metadata.json` - Split metadata

### Step 2: Train with Preprocessed Splits

```bash
python train.py \
    --train-split datasets/splits/train_samples.json \
    --val-split datasets/splits/val_samples.json \
    --imgsz 512 \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4
```

## File Structure

After running `create_dataset.py`:

```
datasets/
├── train_images/
│   ├── forged/*.png
│   └── authentic/*.png
├── train_masks/*.npy
├── supplemental_images/*.png
├── supplemental_masks/*.npy
└── splits/
    ├── train_samples.json  # Preprocessed train split
    ├── val_samples.json    # Preprocessed val split
    └── metadata.json       # Split information
```

## JSON Format

The JSON files contain lists of samples:

```json
[
  {
    "image_path": "datasets/train_images/forged/123.png",
    "mask_path": "datasets/train_masks/123.npy",
    "is_forged": true,
    "case_id": "123"
  },
  {
    "image_path": "datasets/train_images/authentic/456.png",
    "mask_path": null,
    "is_forged": false,
    "case_id": "456"
  }
]
```

## Creating Multiple Splits

You can create different splits for experimentation:

```bash
# 80/20 split
python create_dataset.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --output-dir datasets/splits_80_20 \
    --val-split 0.2 \
    --seed 42

# 90/10 split
python create_dataset.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --output-dir datasets/splits_90_10 \
    --val-split 0.1 \
    --seed 42

# Different random seed
python create_dataset.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --output-dir datasets/splits_seed_123 \
    --val-split 0.2 \
    --seed 123
```

## Migration Guide

### Old Training Command
```bash
python train.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --supplemental-images datasets/supplemental_images \
    --supplemental-masks datasets/supplemental_masks \
    --val-split 0.2 \
    --epochs 100
```

### New Training Commands

**Step 1: Create splits once**
```bash
python create_dataset.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --supplemental-images datasets/supplemental_images \
    --supplemental-masks datasets/supplemental_masks \
    --val-split 0.2
```

**Step 2: Train multiple times with same splits**
```bash
python train.py \
    --train-split datasets/splits/train_samples.json \
    --val-split datasets/splits/val_samples.json \
    --epochs 100
```

## Tips

1. **Reproducibility**: Always use the same seed when creating splits for consistent results
2. **Sharing**: Share the `splits/` directory with team members for consistency across runs
3. **Validation**: Check `metadata.json` to verify class balance and split statistics
4. **Experimentation**: Create multiple split directories with different ratios or seeds for experimentation
