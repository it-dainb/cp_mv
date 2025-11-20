"""
Create preprocessed dataset splits for training.

This script creates balanced train/val splits from the raw data and physically
copies images and masks to a new directory structure, so training doesn't depend
on original data locations.

Usage:
    python create_dataset.py --train-images datasets/train_images \
                             --train-masks datasets/train_masks \
                             --supplemental-images datasets/supplemental_images \
                             --supplemental-masks datasets/supplemental_masks \
                             --output-dir datasets/processed \
                             --val-split 0.2 \
                             --seed 42
"""

import argparse
import json
from pathlib import Path
import sys
import shutil
from tqdm import tqdm

from dataset import create_balanced_splits, load_samples as load_samples_from_json


def copy_samples_to_output(samples, output_dir, split_name):
    """
    Copy images and masks to output directory and update sample paths.
    
    Structure:
        output_dir/
            train/
                images/
                    case_id.png
                masks/
                    case_id.npy
            val/
                images/
                    case_id.png
                masks/
                    case_id.npy
    
    Args:
        samples: List of sample dictionaries
        output_dir: Base output directory (Path object)
        split_name: 'train' or 'val'
    
    Returns:
        List of updated sample dictionaries with new paths
    """
    # Create directories
    split_dir = output_dir / split_name
    images_dir = split_dir / 'images'
    masks_dir = split_dir / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    updated_samples = []
    
    print(f"\nCopying {split_name} split files...")
    for sample in tqdm(samples, desc=f"Copying {split_name}"):
        case_id = sample['case_id']
        
        # Copy image
        src_image = sample['image_path']
        # Keep original extension
        image_ext = src_image.suffix
        dst_image = images_dir / f"{case_id}{image_ext}"
        shutil.copy2(src_image, dst_image)
        
        # Copy mask if exists
        dst_mask = None
        if sample['mask_path'] is not None:
            src_mask = sample['mask_path']
            dst_mask = masks_dir / f"{case_id}.npy"
            shutil.copy2(src_mask, dst_mask)
        
        # Create updated sample dict
        updated_sample = {
            'image_path': dst_image,
            'mask_path': dst_mask,
            'is_forged': sample['is_forged'],
            'case_id': case_id
        }
        updated_samples.append(updated_sample)
    
    return updated_samples


def save_samples(samples, output_path):
    """
    Save sample list to JSON file.
    
    Args:
        samples: List of sample dictionaries
        output_path: Path to save JSON file
    """
    # Convert Path objects to strings for JSON serialization
    samples_serializable = []
    for sample in samples:
        sample_copy = sample.copy()
        sample_copy['image_path'] = str(sample_copy['image_path'])
        if sample_copy['mask_path'] is not None:
            sample_copy['mask_path'] = str(sample_copy['mask_path'])
        samples_serializable.append(sample_copy)
    
    with open(output_path, 'w') as f:
        json.dump(samples_serializable, f, indent=2)
    
    print(f"Saved {len(samples)} sample references to {output_path}")


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dataset splits with:")
    print(f"  Train images: {args.train_images}")
    print(f"  Train masks: {args.train_masks}")
    print(f"  Supplemental images: {args.supplemental_images}")
    print(f"  Supplemental masks: {args.supplemental_masks}")
    print(f"  Validation split: {args.val_split}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Create balanced splits from original data
    train_samples, val_samples = create_balanced_splits(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        supplemental_image_dir=args.supplemental_images,
        supplemental_mask_dir=args.supplemental_masks,
        val_split=args.val_split,
        random_seed=args.seed
    )
    
    # Copy files to output directory and get updated paths
    print(f"\nCopying files to {output_dir}...")
    train_samples_updated = copy_samples_to_output(train_samples, output_dir, 'train')
    val_samples_updated = copy_samples_to_output(val_samples, output_dir, 'val')
    
    # Save sample references to JSON files (for convenience, optional)
    train_split_path = output_dir / 'train_samples.json'
    val_split_path = output_dir / 'val_samples.json'
    
    save_samples(train_samples_updated, train_split_path)
    save_samples(val_samples_updated, val_split_path)
    
    # Save metadata
    metadata = {
        'source_train_images': str(args.train_images),
        'source_train_masks': str(args.train_masks),
        'source_supplemental_images': str(args.supplemental_images) if args.supplemental_images else None,
        'source_supplemental_masks': str(args.supplemental_masks) if args.supplemental_masks else None,
        'val_split': args.val_split,
        'seed': args.seed,
        'num_train_samples': len(train_samples_updated),
        'num_val_samples': len(val_samples_updated),
        'num_train_forged': sum(1 for s in train_samples_updated if s['is_forged']),
        'num_train_authentic': sum(1 for s in train_samples_updated if not s['is_forged']),
        'num_val_forged': sum(1 for s in val_samples_updated if s['is_forged']),
        'num_val_authentic': sum(1 for s in val_samples_updated if not s['is_forged']),
        'output_structure': {
            'train_images': str(output_dir / 'train' / 'images'),
            'train_masks': str(output_dir / 'train' / 'masks'),
            'val_images': str(output_dir / 'val' / 'images'),
            'val_masks': str(output_dir / 'val' / 'masks'),
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dataset splits created successfully!")
    print(f"{'='*60}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/  ({len(train_samples_updated)} images)")
    print(f"    │   └── masks/   ({sum(1 for s in train_samples_updated if s['is_forged'])} masks)")
    print(f"    ├── val/")
    print(f"    │   ├── images/  ({len(val_samples_updated)} images)")
    print(f"    │   └── masks/   ({sum(1 for s in val_samples_updated if s['is_forged'])} masks)")
    print(f"    ├── train_samples.json")
    print(f"    ├── val_samples.json")
    print(f"    └── metadata.json")
    print(f"\nTo use these splits in training, run:")
    print(f"  python train.py --train-split {train_split_path} --val-split {val_split_path} [other args...]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create preprocessed dataset splits for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create splits with default settings
  python create_dataset.py --train-images datasets/train_images \\
                           --train-masks datasets/train_masks

  # Create splits with supplemental data
  python create_dataset.py --train-images datasets/train_images \\
                           --train-masks datasets/train_masks \\
                           --supplemental-images datasets/supplemental_images \\
                           --supplemental-masks datasets/supplemental_masks \\
                           --val-split 0.2 \\
                           --seed 42
        """
    )
    
    # Required arguments
    parser.add_argument('--train-images', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--train-masks', type=str, required=True,
                        help='Path to training masks directory')
    
    # Optional arguments
    parser.add_argument('--supplemental-images', type=str, default=None,
                        help='Path to supplemental images directory (optional)')
    parser.add_argument('--supplemental-masks', type=str, default=None,
                        help='Path to supplemental masks directory (optional)')
    parser.add_argument('--output-dir', type=str, default='datasets/processed',
                        help='Output directory for processed dataset (default: datasets/processed)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.train_images).exists():
        print(f"Error: Train images directory does not exist: {args.train_images}")
        sys.exit(1)
    
    if not Path(args.train_masks).exists():
        print(f"Error: Train masks directory does not exist: {args.train_masks}")
        sys.exit(1)
    
    if args.supplemental_images and not Path(args.supplemental_images).exists():
        print(f"Warning: Supplemental images directory does not exist: {args.supplemental_images}")
        args.supplemental_images = None
    
    if args.supplemental_masks and not Path(args.supplemental_masks).exists():
        print(f"Warning: Supplemental masks directory does not exist: {args.supplemental_masks}")
        args.supplemental_masks = None
    
    if args.val_split <= 0 or args.val_split >= 1:
        print(f"Error: val_split must be between 0 and 1, got {args.val_split}")
        sys.exit(1)
    
    main(args)
