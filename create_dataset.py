"""
Create preprocessed dataset splits for training.

This script creates balanced train/val/test splits from the raw data and physically
copies images and masks to a new directory structure. The test split contains only
original (non-augmented) data for fair evaluation.

Flow:
1. create_dataset.py → train/val/test splits (original data only)
2. aug_dataset.py --dataset-dir → augments train/val, keeps test untouched
3. Training uses: (train + aug_train), (val + aug_val), test

Output structure:
    output_dir/
    ├── train/
    │   ├── images/
    │   │   ├── forged/*.png
    │   │   └── authentic/*.png
    │   └── masks/*.npy
    ├── val/
    │   └── (same structure)
    ├── test/
    │   └── (same structure)
    ├── train_samples.json
    ├── val_samples.json
    ├── test_samples.json
    └── metadata.json

Usage:
    python create_dataset.py --train-images datasets/train_images \
                             --train-masks datasets/train_masks \
                             --supplemental-images datasets/supplemental_images \
                             --supplemental-masks datasets/supplemental_masks \
                             --output-dir datasets/processed \
                             --val-split 0.15 \
                             --test-split 0.15 \
                             --seed 42
"""

import argparse
import json
from pathlib import Path
import sys
import shutil
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def collect_samples(
    train_images_dir: Path,
    train_masks_dir: Path,
    supplemental_images_dir: Path = None,
    supplemental_masks_dir: Path = None,
) -> tuple:
    """
    Collect all samples from train and supplemental directories.

    Args:
        train_images_dir: Path to train_images (has forged/ and authentic/)
        train_masks_dir: Path to train_masks
        supplemental_images_dir: Optional path to supplemental_images
        supplemental_masks_dir: Optional path to supplemental_masks

    Returns:
        forged_samples: List of forged sample dicts
        authentic_samples: List of authentic sample dicts
    """
    forged_samples = []
    authentic_samples = []

    # Collect train forged samples
    forged_dir = train_images_dir / "forged"
    authentic_dir = train_images_dir / "authentic"

    if forged_dir.exists():
        for img_path in sorted(forged_dir.glob("*.png")):
            case_id = img_path.stem
            mask_path = train_masks_dir / f"{case_id}.npy"

            if mask_path.exists():
                # Also check for authentic pair (needed for augmentation)
                authentic_path = authentic_dir / f"{case_id}.png"
                has_authentic_pair = authentic_path.exists()

                forged_samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "authentic_path": authentic_path
                        if has_authentic_pair
                        else None,
                        "is_forged": True,
                        "case_id": case_id,
                        "source": "train",
                    }
                )

    # Collect train authentic samples
    if authentic_dir.exists():
        for img_path in sorted(authentic_dir.glob("*.png")):
            case_id = img_path.stem
            authentic_samples.append(
                {
                    "image_path": img_path,
                    "mask_path": None,
                    "authentic_path": None,
                    "is_forged": False,
                    "case_id": case_id,
                    "source": "train",
                }
            )

    # Collect supplemental forged samples
    if supplemental_images_dir and supplemental_masks_dir:
        supp_img_dir = Path(supplemental_images_dir)
        supp_mask_dir = Path(supplemental_masks_dir)

        if supp_img_dir.exists():
            for img_path in sorted(supp_img_dir.glob("*.png")):
                case_id = img_path.stem
                mask_path = supp_mask_dir / f"{case_id}.npy"

                if mask_path.exists():
                    forged_samples.append(
                        {
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "authentic_path": None,  # Supplemental has no authentic pair
                            "is_forged": True,
                            "case_id": f"supp_{case_id}",
                            "source": "supplemental",
                        }
                    )

    print(f"\nCollected samples:")
    print(
        f"  Forged: {len(forged_samples)} (train: {sum(1 for s in forged_samples if s['source'] == 'train')}, supplemental: {sum(1 for s in forged_samples if s['source'] == 'supplemental')})"
    )
    print(f"  Authentic: {len(authentic_samples)}")

    return forged_samples, authentic_samples


def create_three_way_split(
    forged_samples: list,
    authentic_samples: list,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> tuple:
    """
    Create balanced train/val/test splits.

    Strategy:
    1. Split forged samples into train/val/test
    2. Split authentic samples into train/val/test
    3. Balance each split to have equal forged/authentic counts

    Args:
        forged_samples: List of forged sample dicts
        authentic_samples: List of authentic sample dicts
        val_split: Fraction for validation (default: 0.15)
        test_split: Fraction for test (default: 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        train_samples, val_samples, test_samples: Lists of sample dicts
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    train_ratio = 1.0 - val_split - test_split
    if train_ratio <= 0:
        raise ValueError(
            f"Invalid splits: val={val_split}, test={test_split}, leaves no data for train"
        )

    # Split forged samples
    if len(forged_samples) > 2:
        # First split: separate test set
        forged_trainval, forged_test = train_test_split(
            forged_samples,
            test_size=test_split,
            random_state=random_seed,
        )
        # Second split: separate train and val
        relative_val = val_split / (1 - test_split)
        forged_train, forged_val = train_test_split(
            forged_trainval,
            test_size=relative_val,
            random_state=random_seed,
        )
    else:
        forged_train, forged_val, forged_test = forged_samples, [], []

    # Split authentic samples
    if len(authentic_samples) > 2:
        auth_trainval, auth_test = train_test_split(
            authentic_samples,
            test_size=test_split,
            random_state=random_seed,
        )
        relative_val = val_split / (1 - test_split)
        auth_train, auth_val = train_test_split(
            auth_trainval,
            test_size=relative_val,
            random_state=random_seed,
        )
    else:
        auth_train, auth_val, auth_test = authentic_samples, [], []

    # Balance each split
    def balance_split(forged, authentic, split_name):
        n_forged = len(forged)
        n_auth = len(authentic)
        min_count = min(n_forged, n_auth)

        if min_count > 0 and n_forged != n_auth:
            if n_forged > min_count:
                forged = random.sample(forged, min_count)
            if n_auth > min_count:
                authentic = random.sample(authentic, min_count)

        combined = forged + authentic
        random.shuffle(combined)

        print(
            f"  {split_name}: {len(combined)} total (forged: {len(forged)}, authentic: {len(authentic)})"
        )
        return combined

    print("\nBalanced splits:")
    train_samples = balance_split(forged_train, auth_train, "train")
    val_samples = balance_split(forged_val, auth_val, "val")
    test_samples = balance_split(forged_test, auth_test, "test")

    return train_samples, val_samples, test_samples


def copy_samples_to_output(samples: list, output_dir: Path, split_name: str) -> list:
    """
    Copy images and masks to output directory maintaining forged/authentic structure.

    Structure:
        output_dir/
            {split_name}/
                images/
                    forged/*.png
                    authentic/*.png
                masks/*.npy

    Args:
        samples: List of sample dictionaries
        output_dir: Base output directory
        split_name: 'train', 'val', or 'test'

    Returns:
        List of updated sample dictionaries with new paths
    """
    split_dir = output_dir / split_name
    images_forged_dir = split_dir / "images" / "forged"
    images_authentic_dir = split_dir / "images" / "authentic"
    masks_dir = split_dir / "masks"

    images_forged_dir.mkdir(parents=True, exist_ok=True)
    images_authentic_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    updated_samples = []
    skipped = []

    print(f"\nCopying {split_name} split files...")
    for sample in tqdm(samples, desc=f"Copying {split_name}"):
        case_id = sample["case_id"]
        src_image = Path(sample["image_path"])

        if not src_image.exists():
            skipped.append((case_id, f"Image not found: {src_image}"))
            continue

        # Determine destination based on forged/authentic
        if sample["is_forged"]:
            dst_image = images_forged_dir / f"{case_id}.png"
        else:
            dst_image = images_authentic_dir / f"{case_id}.png"

        # Copy image
        try:
            shutil.copy2(src_image, dst_image)
        except Exception as e:
            skipped.append((case_id, f"Failed to copy image: {e}"))
            continue

        # Copy mask if exists (only for forged)
        dst_mask = None
        if sample["mask_path"] is not None:
            src_mask = Path(sample["mask_path"])
            if src_mask.exists():
                dst_mask = masks_dir / f"{case_id}.npy"
                try:
                    shutil.copy2(src_mask, dst_mask)
                except Exception as e:
                    skipped.append((case_id, f"Failed to copy mask: {e}"))
                    continue

        # Copy authentic pair if exists (for augmentation Phase A)
        dst_authentic = None
        if sample.get("authentic_path") is not None:
            src_authentic = Path(sample["authentic_path"])
            if src_authentic.exists():
                dst_authentic = images_authentic_dir / f"{case_id}.png"
                # Only copy if not already there (avoid overwriting)
                if not dst_authentic.exists():
                    try:
                        shutil.copy2(src_authentic, dst_authentic)
                    except Exception as e:
                        pass  # Non-critical, authentic might be copied separately

        # Create updated sample with relative paths
        rel_image_path = dst_image.relative_to(output_dir)
        rel_mask_path = dst_mask.relative_to(output_dir) if dst_mask else None
        rel_authentic_path = (
            dst_authentic.relative_to(output_dir)
            if dst_authentic and dst_authentic.exists()
            else None
        )

        updated_sample = {
            "image_path": rel_image_path,
            "mask_path": rel_mask_path,
            "authentic_path": rel_authentic_path,
            "is_forged": sample["is_forged"],
            "case_id": case_id,
            "source": sample.get("source", "train"),
        }
        updated_samples.append(updated_sample)

    if skipped:
        print(f"\nWarning: Skipped {len(skipped)} samples in {split_name}:")
        for case_id, reason in skipped[:5]:
            print(f"  - {case_id}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return updated_samples


def save_samples(samples: list, output_path: Path):
    """
    Save sample list to JSON file with relative paths.

    Args:
        samples: List of sample dictionaries with relative Path objects
        output_path: Path to save JSON file
    """
    samples_serializable = []
    for sample in samples:
        sample_copy = sample.copy()
        sample_copy["image_path"] = sample_copy["image_path"].as_posix()
        if sample_copy["mask_path"] is not None:
            sample_copy["mask_path"] = sample_copy["mask_path"].as_posix()
        if sample_copy.get("authentic_path") is not None:
            sample_copy["authentic_path"] = sample_copy["authentic_path"].as_posix()
        else:
            sample_copy["authentic_path"] = None
        samples_serializable.append(sample_copy)

    with open(output_path, "w") as f:
        json.dump(samples_serializable, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Dataset Splits (train/val/test)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Train images: {args.train_images}")
    print(f"  Train masks: {args.train_masks}")
    print(f"  Supplemental images: {args.supplemental_images}")
    print(f"  Supplemental masks: {args.supplemental_masks}")
    print(f"  Val split: {args.val_split}")
    print(f"  Test split: {args.test_split}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")

    # Collect all samples
    forged_samples, authentic_samples = collect_samples(
        train_images_dir=Path(args.train_images),
        train_masks_dir=Path(args.train_masks),
        supplemental_images_dir=Path(args.supplemental_images)
        if args.supplemental_images
        else None,
        supplemental_masks_dir=Path(args.supplemental_masks)
        if args.supplemental_masks
        else None,
    )

    # Create 3-way split
    train_samples, val_samples, test_samples = create_three_way_split(
        forged_samples=forged_samples,
        authentic_samples=authentic_samples,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
    )

    # Copy files to output directory
    train_updated = copy_samples_to_output(train_samples, output_dir, "train")
    val_updated = copy_samples_to_output(val_samples, output_dir, "val")
    test_updated = copy_samples_to_output(test_samples, output_dir, "test")

    # Save sample JSON files
    save_samples(train_updated, output_dir / "train_samples.json")
    save_samples(val_updated, output_dir / "val_samples.json")
    save_samples(test_updated, output_dir / "test_samples.json")

    # Save metadata
    metadata = {
        "source_train_images": str(args.train_images),
        "source_train_masks": str(args.train_masks),
        "source_supplemental_images": str(args.supplemental_images)
        if args.supplemental_images
        else None,
        "source_supplemental_masks": str(args.supplemental_masks)
        if args.supplemental_masks
        else None,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "splits": {
            "train": {
                "total": len(train_updated),
                "forged": sum(1 for s in train_updated if s["is_forged"]),
                "authentic": sum(1 for s in train_updated if not s["is_forged"]),
            },
            "val": {
                "total": len(val_updated),
                "forged": sum(1 for s in val_updated if s["is_forged"]),
                "authentic": sum(1 for s in val_updated if not s["is_forged"]),
            },
            "test": {
                "total": len(test_updated),
                "forged": sum(1 for s in test_updated if s["is_forged"]),
                "authentic": sum(1 for s in test_updated if not s["is_forged"]),
            },
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Created Successfully!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(
        f"    │   ├── images/forged/  ({metadata['splits']['train']['forged']} images)"
    )
    print(
        f"    │   ├── images/authentic/  ({metadata['splits']['train']['authentic']} images)"
    )
    print(f"    │   └── masks/  ({metadata['splits']['train']['forged']} masks)")
    print(f"    ├── val/")
    print(f"    │   ├── images/forged/  ({metadata['splits']['val']['forged']} images)")
    print(
        f"    │   ├── images/authentic/  ({metadata['splits']['val']['authentic']} images)"
    )
    print(f"    │   └── masks/  ({metadata['splits']['val']['forged']} masks)")
    print(f"    ├── test/")
    print(
        f"    │   ├── images/forged/  ({metadata['splits']['test']['forged']} images)"
    )
    print(
        f"    │   ├── images/authentic/  ({metadata['splits']['test']['authentic']} images)"
    )
    print(f"    │   └── masks/  ({metadata['splits']['test']['forged']} masks)")
    print(f"    ├── train_samples.json")
    print(f"    ├── val_samples.json")
    print(f"    ├── test_samples.json")
    print(f"    └── metadata.json")

    print(f"\nNext step - augment train/val splits:")
    print(f"  python aug_dataset.py --dataset-dir {output_dir} \\")
    print(f"                        --output-dir aug_datasets \\")
    print(f"                        --num-augmentations 2 \\")
    print(f"                        --save-viz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create preprocessed dataset splits (train/val/test) for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 3-way splits with default settings (70/15/15)
  python create_dataset.py --train-images datasets/train_images \\
                           --train-masks datasets/train_masks

  # With supplemental data and custom splits
  python create_dataset.py --train-images datasets/train_images \\
                           --train-masks datasets/train_masks \\
                           --supplemental-images datasets/supplemental_images \\
                           --supplemental-masks datasets/supplemental_masks \\
                           --val-split 0.15 \\
                           --test-split 0.15 \\
                           --seed 42

Output structure maintains forged/authentic separation for augmentation:
  output_dir/
  ├── train/images/forged/, train/images/authentic/, train/masks/
  ├── val/images/forged/, val/images/authentic/, val/masks/
  ├── test/images/forged/, test/images/authentic/, test/masks/
  └── {train,val,test}_samples.json, metadata.json
        """,
    )

    parser.add_argument(
        "--train-images",
        type=str,
        required=True,
        help="Path to training images directory (must have forged/ and authentic/ subdirs)",
    )
    parser.add_argument(
        "--train-masks",
        type=str,
        required=True,
        help="Path to training masks directory",
    )
    parser.add_argument(
        "--supplemental-images",
        type=str,
        default=None,
        help="Path to supplemental images directory (optional)",
    )
    parser.add_argument(
        "--supplemental-masks",
        type=str,
        default=None,
        help="Path to supplemental masks directory (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/processed",
        help="Output directory for processed dataset (default: datasets/processed)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction for test set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.train_images).exists():
        print(f"Error: Train images directory not found: {args.train_images}")
        sys.exit(1)

    if not Path(args.train_masks).exists():
        print(f"Error: Train masks directory not found: {args.train_masks}")
        sys.exit(1)

    if args.val_split + args.test_split >= 1.0:
        print(
            f"Error: val_split ({args.val_split}) + test_split ({args.test_split}) must be < 1.0"
        )
        sys.exit(1)

    main(args)
