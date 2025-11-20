"""
Create preprocessed dataset splits for training.

This script creates balanced train/val splits from the raw data and saves
the sample paths to files, so training can directly load from preprocessed splits.

Usage:
    python create_dataset.py --train-images datasets/train_images \
                             --train-masks datasets/train_masks \
                             --supplemental-images datasets/supplemental_images \
                             --supplemental-masks datasets/supplemental_masks \
                             --output-dir datasets/splits \
                             --val-split 0.2 \
                             --seed 42
"""

import argparse
import json
from pathlib import Path
import sys

from dataset import create_balanced_splits


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
    
    print(f"Saved {len(samples)} samples to {output_path}")


def load_samples(input_path):
    """
    Load sample list from JSON file.
    
    Args:
        input_path: Path to JSON file
    
    Returns:
        List of sample dictionaries with Path objects
    """
    with open(input_path, 'r') as f:
        samples = json.load(f)
    
    # Convert strings back to Path objects
    for sample in samples:
        sample['image_path'] = Path(sample['image_path'])
        if sample['mask_path'] is not None:
            sample['mask_path'] = Path(sample['mask_path'])
    
    return samples


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
    
    # Create balanced splits
    train_samples, val_samples = create_balanced_splits(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        supplemental_image_dir=args.supplemental_images,
        supplemental_mask_dir=args.supplemental_masks,
        val_split=args.val_split,
        random_seed=args.seed
    )
    
    # Save splits to JSON files
    train_split_path = output_dir / 'train_samples.json'
    val_split_path = output_dir / 'val_samples.json'
    
    save_samples(train_samples, train_split_path)
    save_samples(val_samples, val_split_path)
    
    # Save metadata
    metadata = {
        'train_images': str(args.train_images),
        'train_masks': str(args.train_masks),
        'supplemental_images': str(args.supplemental_images) if args.supplemental_images else None,
        'supplemental_masks': str(args.supplemental_masks) if args.supplemental_masks else None,
        'val_split': args.val_split,
        'seed': args.seed,
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'num_train_forged': sum(1 for s in train_samples if s['is_forged']),
        'num_train_authentic': sum(1 for s in train_samples if not s['is_forged']),
        'num_val_forged': sum(1 for s in val_samples if s['is_forged']),
        'num_val_authentic': sum(1 for s in val_samples if not s['is_forged']),
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset splits created successfully!")
    print(f"  Train samples: {train_split_path}")
    print(f"  Val samples: {val_split_path}")
    print(f"  Metadata: {metadata_path}")
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
    parser.add_argument('--output-dir', type=str, default='datasets/splits',
                        help='Output directory for dataset splits (default: datasets/splits)')
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
