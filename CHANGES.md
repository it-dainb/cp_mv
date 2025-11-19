# Recent Changes

## 1. Added Supplemental Data Support

### Dataset (`dataset.py`)
- **New parameters** in `ForgeryDetectionDataset.__init__()`:
  - `supplemental_image_dir`: Optional path to supplemental images
  - `supplemental_mask_dir`: Optional path to supplemental masks
  
- **Behavior**: When provided, supplemental data is automatically merged with training data
  - Supplemental images are treated as forged (all have masks)
  - Case IDs are prefixed with `supp_` to avoid conflicts
  - Flat directory structure (no forged/authentic subdirs)

### Training Script (`train.py`)
- **New arguments**:
  - `--supplemental-images`: Path to supplemental_images directory (optional)
  - `--supplemental-masks`: Path to supplemental_masks directory (optional)
  
- **Usage**: Pass both arguments to include supplemental data in training
  ```bash
  python train.py \
      --train-images datasets/train_images \
      --train-masks datasets/train_masks \
      --supplemental-images datasets/supplemental_images \
      --supplemental-masks datasets/supplemental_masks \
      ...
  ```

## 2. Removed Data Augmentation

### Before
- Geometric augmentations: flips, rotation, shift-scale-rotate
- Color augmentations: brightness/contrast, HSV, CLAHE
- Noise: Gaussian noise, Gaussian blur, Median blur

### After
- **Train transforms**: Only normalization + tensor conversion
- **Val transforms**: Only normalization + tensor conversion
- Both use ImageNet normalization: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

### Rationale
- Training without augmentation as requested
- Keeps only essential preprocessing (normalization for EfficientNet backbone)

## Current Dataset Structure

```
datasets/
├── train_images/
│   ├── forged/*.png          # Forged images (have masks)
│   └── authentic/*.png        # Authentic images (no masks)
├── train_masks/
│   └── *.npy                  # Instance masks for forged images
├── supplemental_images/
│   └── *.png                  # Additional forged images (flat structure)
└── supplemental_masks/
    └── *.npy                  # Masks for supplemental images
```

## Training Examples

### Without supplemental data:
```bash
python train.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --val-images datasets/val_images \
    --val-masks datasets/val_masks \
    --epochs 100 --batch-size 8 --imgsz 512
```

### With supplemental data:
```bash
python train.py \
    --train-images datasets/train_images \
    --train-masks datasets/train_masks \
    --supplemental-images datasets/supplemental_images \
    --supplemental-masks datasets/supplemental_masks \
    --val-images datasets/val_images \
    --val-masks datasets/val_masks \
    --epochs 100 --batch-size 8 --imgsz 512
```

## Data Loading Summary

Current data:
- Training forged: 1 image
- Training authentic: 1 image  
- Training masks: 1 file
- Supplemental images: 1 image
- Supplemental masks: 1 file

**Total with supplemental**: 3 images (2 forged + 1 authentic)
