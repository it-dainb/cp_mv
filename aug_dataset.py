"""
Advanced Internal Copy-Move Augmentation Pipeline.

This script generates high-quality synthetic forgeries following the strategy
outlined in AUG_PROPOSAL.md.

Two modes of operation:

1. Dataset Mode (Recommended):
   Takes a preprocessed dataset from create_dataset.py and creates an augmented
   version with the SAME structure. The output can be used as a drop-in replacement.

   python aug_dataset.py --dataset-dir datasets/processed \
                         --output-dir datasets/augmented \
                         --num-augmentations 2 \
                         --save-viz --viz-samples 10

   Output structure (same as input):
       augmented_dataset/
       ├── train/          # original train + augmented train
       │   ├── images/forged/*.png
       │   ├── images/authentic/*.png
       │   └── masks/*.npy
       ├── val/            # original val + augmented val
       │   └── (same structure)
       ├── test/           # original test (no augmentation, just copied)
       │   └── (same structure)
       ├── train_samples.json
       ├── val_samples.json
       ├── test_samples.json
       └── metadata.json

2. Legacy Mode:
   Takes raw directories directly (for backward compatibility).

   python aug_dataset.py --train-images datasets/train_images \
                         --train-masks datasets/train_masks \
                         --output-dir aug_datasets
"""

import argparse
import json
import random
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


def create_mask_overlay(
    image: np.ndarray, mask_green: np.ndarray, mask_red: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Create visualization overlay with green and red masks.

    Args:
        image: RGB image [H, W, C]
        mask_green: Binary mask for green overlay (source/new objects) [H, W]
        mask_red: Binary mask for red overlay (target/original objects) [H, W]
        alpha: Transparency for overlay

    Returns:
        Overlay image [H, W, C]
    """
    overlay = image.copy().astype(np.float32)

    # Green overlay (source objects / new augmented objects)
    if mask_green is not None and np.any(mask_green > 0):
        green_mask = (mask_green > 0).astype(np.float32)
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - alpha * green_mask)  # R
        overlay[:, :, 1] = (
            overlay[:, :, 1] * (1 - alpha * green_mask) + 255 * alpha * green_mask
        )  # G
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - alpha * green_mask)  # B

    # Red overlay (target objects / original mask)
    if mask_red is not None and np.any(mask_red > 0):
        red_mask = (mask_red > 0).astype(np.float32)
        overlay[:, :, 0] = (
            overlay[:, :, 0] * (1 - alpha * red_mask) + 255 * alpha * red_mask
        )  # R
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - alpha * red_mask)  # G
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - alpha * red_mask)  # B

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_train_visualization(
    output_path: Path,
    forged_img: np.ndarray,
    source_mask: np.ndarray,
    target_masks: np.ndarray,
    aug_img: np.ndarray,
    aug_mask: np.ndarray,
    case_id: str,
    aug_idx: int,
):
    """Save visualization for train augmentation (Phase A)."""
    orig_overlay = create_mask_overlay(forged_img, source_mask, target_masks)
    aug_overlay = create_mask_overlay(aug_img, aug_mask, None)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        orig_overlay, f"Original: {case_id}", (10, 30), font, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        orig_overlay,
        "Green=Source, Red=Targets",
        (10, 60),
        font,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        aug_overlay, f"Augmented: aug{aug_idx}", (10, 30), font, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        aug_overlay, "Green=New Copy-Move", (10, 60), font, 0.5, (255, 255, 255), 1
    )

    h1, w1 = orig_overlay.shape[:2]
    h2, w2 = aug_overlay.shape[:2]
    if h1 != h2:
        scale = h1 / h2
        aug_overlay = cv2.resize(aug_overlay, (int(w2 * scale), h1))

    combined = np.concatenate([orig_overlay, aug_overlay], axis=1)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), combined_bgr)


def save_supplemental_visualization(
    output_path: Path,
    original_img: np.ndarray,
    original_mask: np.ndarray,
    aug_img: np.ndarray,
    aug_mask: np.ndarray,
    new_objects_mask: np.ndarray,
    case_id: str,
    aug_idx: int,
):
    """Save visualization for supplemental augmentation (Phase B)."""
    orig_overlay = create_mask_overlay(original_img, None, original_mask)
    aug_overlay = create_mask_overlay(aug_img, new_objects_mask, original_mask)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        orig_overlay, f"Original: {case_id}", (10, 30), font, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        orig_overlay, "Red=Original Forgery", (10, 60), font, 0.5, (255, 255, 255), 1
    )
    cv2.putText(
        aug_overlay, f"Augmented: aug{aug_idx}", (10, 30), font, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        aug_overlay, "Red=Original, Green=New", (10, 60), font, 0.5, (255, 255, 255), 1
    )

    h1, w1 = orig_overlay.shape[:2]
    h2, w2 = aug_overlay.shape[:2]
    if h1 != h2:
        scale = h1 / h2
        aug_overlay = cv2.resize(aug_overlay, (int(w2 * scale), h1))

    combined = np.concatenate([orig_overlay, aug_overlay], axis=1)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), combined_bgr)


class CopyMoveAugmentor:
    """
    Generates synthetic copy-move forgeries for data augmentation.

    Core Philosophy:
    1. Internal Consistency: All forgeries are created within the same image
    2. Smart Source Recovery: Identifies original objects using MAD comparison
    3. Anti-Artifact Engineering: Probabilistic soft/hard edge rendering
    """

    def __init__(
        self,
        soft_edge_prob: float = 0.7,
        blur_kernels: list = None,
        min_pastes: int = 1,
        max_pastes: int = 3,
        min_object_area: int = 100,
        max_overlap_ratio: float = 0.3,
        random_seed: int = 42,
    ):
        self.soft_edge_prob = soft_edge_prob
        self.blur_kernels = blur_kernels or [3, 5, 7]
        self.min_pastes = min_pastes
        self.max_pastes = max_pastes
        self.min_object_area = min_object_area
        self.max_overlap_ratio = max_overlap_ratio

        random.seed(random_seed)
        np.random.seed(random_seed)

        self._geo_transform = self._build_geometric_transform()

    def _build_geometric_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=50, sigma=10, p=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomScale(scale_limit=(-0.3, 0.3), p=0.5),
            ]
        )

    def identify_source_object(
        self, forged_img: np.ndarray, authentic_img: np.ndarray, mask: np.ndarray
    ) -> tuple:
        """Identify the source object using Mean Absolute Difference."""
        if mask.ndim == 3:
            mask_2d = np.any(mask, axis=0).astype(np.uint8)
        else:
            mask_2d = mask.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_2d, connectivity=8
        )

        if num_labels <= 1:
            return None, -1, []

        all_components = []
        mad_scores = []

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.min_object_area:
                continue

            component_mask = (labels == label_id).astype(np.uint8)
            all_components.append(component_mask)

            forged_pixels = forged_img[component_mask == 1]
            authentic_pixels = authentic_img[component_mask == 1]
            mad = np.mean(
                np.abs(forged_pixels.astype(float) - authentic_pixels.astype(float))
            )
            mad_scores.append(mad)

        if len(all_components) == 0:
            return None, -1, []

        source_idx = np.argmin(mad_scores)
        source_mask = all_components[source_idx]

        return source_mask, source_idx, all_components

    def extract_object(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        """Extract object from image using mask."""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None, None, None

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        pad = 5
        y_min = max(0, y_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(image.shape[1], x_max + pad)

        obj_img = image[y_min:y_max, x_min:x_max].copy()
        obj_mask = mask[y_min:y_max, x_min:x_max].copy()
        obj_img = obj_img * obj_mask[:, :, np.newaxis]

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return obj_img, obj_mask, bbox

    def apply_transforms(self, obj_img: np.ndarray, obj_mask: np.ndarray) -> tuple:
        """Apply geometric transforms to object with padding to prevent clipping."""
        h, w = obj_mask.shape[:2]

        # Calculate padding needed for worst-case rotation (diagonal of bounding box)
        # When rotated 45 degrees, the diagonal becomes the new width/height
        diagonal = int(np.ceil(np.sqrt(h * h + w * w)))
        pad_h = (diagonal - h) // 2 + 10  # Extra margin
        pad_w = (diagonal - w) // 2 + 10

        # Pad the image and mask with zeros (black/transparent)
        padded_img = np.pad(
            obj_img,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        padded_mask = np.pad(
            obj_mask,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )

        # Apply transforms on padded image/mask
        transformed = self._geo_transform(image=padded_img, mask=padded_mask)
        trans_img = transformed["image"]
        trans_mask = transformed["mask"]

        # Crop back to content bounding box (remove excess padding)
        coords = np.where(trans_mask > 0)
        if len(coords[0]) == 0:
            # No valid mask after transform, return original
            return obj_img, obj_mask

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        # Add small padding around the cropped content
        margin = 2
        y_min = max(0, y_min - margin)
        y_max = min(trans_mask.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(trans_mask.shape[1], x_max + margin)

        cropped_img = trans_img[y_min:y_max, x_min:x_max]
        cropped_mask = trans_mask[y_min:y_max, x_min:x_max]

        return cropped_img, cropped_mask

    def find_valid_position(
        self,
        img_shape: tuple,
        obj_shape: tuple,
        occupied_mask: np.ndarray,
        num_attempts: int = 100,
    ) -> tuple:
        """Find a valid position to paste the object."""
        img_h, img_w = img_shape[:2]
        obj_h, obj_w = obj_shape[:2]

        if obj_h >= img_h or obj_w >= img_w:
            return None

        for _ in range(num_attempts):
            y = random.randint(0, img_h - obj_h)
            x = random.randint(0, img_w - obj_w)

            region = occupied_mask[y : y + obj_h, x : x + obj_w]
            overlap_ratio = np.mean(region > 0)

            if overlap_ratio <= self.max_overlap_ratio:
                return (y, x)

        return None

    def paste_object(
        self,
        background: np.ndarray,
        obj_img: np.ndarray,
        obj_mask: np.ndarray,
        position: tuple,
        use_soft_edge: bool = True,
    ) -> tuple:
        """Paste object onto background with blending."""
        result = background.copy()
        y, x = position
        h, w = obj_mask.shape[:2]

        end_y = min(y + h, background.shape[0])
        end_x = min(x + w, background.shape[1])
        crop_h = end_y - y
        crop_w = end_x - x

        obj_img = obj_img[:crop_h, :crop_w]
        obj_mask = obj_mask[:crop_h, :crop_w]

        if use_soft_edge:
            kernel_size = random.choice(self.blur_kernels)
            alpha = cv2.GaussianBlur(
                obj_mask.astype(np.float32), (kernel_size, kernel_size), 0
            )
            if alpha.max() > 0:
                alpha = alpha / alpha.max()
        else:
            alpha = obj_mask.astype(np.float32)

        alpha_3ch = alpha[:, :, np.newaxis]
        blended_region = (
            result[y:end_y, x:end_x].astype(np.float32) * (1 - alpha_3ch)
            + obj_img.astype(np.float32) * alpha_3ch
        )
        result[y:end_y, x:end_x] = blended_region.astype(np.uint8)

        paste_mask = np.zeros(
            (background.shape[0], background.shape[1]), dtype=np.uint8
        )
        binary_alpha = (alpha > 0.3).astype(np.uint8)
        paste_mask[y:end_y, x:end_x] = binary_alpha

        return result, paste_mask

    def prepare_source_cache(
        self, authentic_img: np.ndarray, forged_img: np.ndarray, mask: np.ndarray
    ) -> dict:
        """Pre-compute and cache source object info for multiple augmentations."""
        source_mask, source_idx, all_components = self.identify_source_object(
            forged_img, authentic_img, mask
        )

        if source_mask is None:
            return {
                "source_mask": None,
                "source_idx": -1,
                "all_components": [],
                "obj_img": None,
                "obj_mask": None,
                "target_masks": None,
            }

        target_masks = np.zeros_like(source_mask)
        for idx, comp in enumerate(all_components):
            if idx != source_idx:
                target_masks = np.maximum(target_masks, comp)

        obj_img, obj_mask, bbox = self.extract_object(authentic_img, source_mask)

        return {
            "source_mask": source_mask,
            "source_idx": source_idx,
            "all_components": all_components,
            "obj_img": obj_img,
            "obj_mask": obj_mask,
            "target_masks": target_masks,
        }

    def generate_from_triplet(
        self,
        authentic_img: np.ndarray,
        forged_img: np.ndarray,
        mask: np.ndarray,
        return_viz_info: bool = False,
        cached_source_info: dict = None,
    ) -> tuple:
        """Phase A: High-Fidelity Generation from (Authentic, Forged, Mask) triplet."""
        if cached_source_info is not None:
            source_mask = cached_source_info["source_mask"]
            target_masks = cached_source_info["target_masks"]
            obj_img = cached_source_info["obj_img"]
            obj_mask = cached_source_info["obj_mask"]

            if source_mask is None or obj_img is None:
                result, combined_mask = self.generate_from_single(
                    forged_img, mask, return_viz_info=False
                )
                if return_viz_info:
                    return (
                        result,
                        combined_mask,
                        {"source_mask": None, "target_masks": None, "fallback": True},
                    )
                return result, combined_mask
        else:
            source_mask, source_idx, all_components = self.identify_source_object(
                forged_img, authentic_img, mask
            )

            if source_mask is None:
                result, combined_mask = self.generate_from_single(
                    forged_img, mask, return_viz_info=False
                )
                if return_viz_info:
                    return (
                        result,
                        combined_mask,
                        {"source_mask": None, "target_masks": None, "fallback": True},
                    )
                return result, combined_mask

            target_masks = np.zeros_like(source_mask)
            for idx, comp in enumerate(all_components):
                if idx != source_idx:
                    target_masks = np.maximum(target_masks, comp)

            obj_img, obj_mask, bbox = self.extract_object(authentic_img, source_mask)

            if obj_img is None:
                result, combined_mask = self.generate_from_single(
                    forged_img, mask, return_viz_info=False
                )
                if return_viz_info:
                    return (
                        result,
                        combined_mask,
                        {"source_mask": None, "target_masks": None, "fallback": True},
                    )
                return result, combined_mask

        result = authentic_img.copy()
        combined_mask = np.zeros(
            (authentic_img.shape[0], authentic_img.shape[1]), dtype=np.uint8
        )

        num_pastes = random.randint(self.min_pastes, self.max_pastes)

        for _ in range(num_pastes):
            trans_img, trans_mask = self.apply_transforms(obj_img, obj_mask)
            pos = self.find_valid_position(
                result.shape[:2], trans_mask.shape[:2], combined_mask
            )

            if pos is None:
                continue

            use_soft = random.random() < self.soft_edge_prob
            result, paste_mask = self.paste_object(
                result, trans_img, trans_mask, pos, use_soft
            )
            combined_mask = np.maximum(combined_mask, paste_mask)

        if return_viz_info:
            return (
                result,
                combined_mask,
                {
                    "source_mask": source_mask,
                    "target_masks": target_masks,
                    "fallback": False,
                },
            )

        return result, combined_mask

    def generate_from_single(
        self, image: np.ndarray, mask: np.ndarray, return_viz_info: bool = False
    ) -> tuple:
        """Phase B: Blind Generation from single image (no authentic pair)."""
        if mask.ndim == 3:
            mask_2d = np.any(mask, axis=0).astype(np.uint8)
        else:
            mask_2d = mask.astype(np.uint8)

        original_mask = mask_2d.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_2d, connectivity=8
        )

        obj_img, obj_mask = None, None

        if num_labels > 1:
            valid_components = []
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= self.min_object_area:
                    component_mask = (labels == label_id).astype(np.uint8)
                    valid_components.append(component_mask)

            if valid_components:
                selected_mask = random.choice(valid_components)
                obj_img, obj_mask, _ = self.extract_object(image, selected_mask)

        if obj_img is None or obj_mask is None:
            obj_img, obj_mask = self._random_crop(image)

        if obj_img is None:
            if return_viz_info:
                return (
                    image.copy(),
                    mask_2d,
                    {"original_mask": original_mask, "new_objects_mask": None},
                )
            return image.copy(), mask_2d

        result = image.copy()
        combined_mask = mask_2d.copy()
        new_objects_mask = np.zeros_like(mask_2d)

        num_pastes = random.randint(self.min_pastes, self.max_pastes)

        for _ in range(num_pastes):
            trans_img, trans_mask = self.apply_transforms(
                obj_img.copy(), obj_mask.copy()
            )
            pos = self.find_valid_position(
                result.shape[:2], trans_mask.shape[:2], combined_mask
            )

            if pos is None:
                continue

            use_soft = random.random() < self.soft_edge_prob
            result, paste_mask = self.paste_object(
                result, trans_img, trans_mask, pos, use_soft
            )
            combined_mask = np.maximum(combined_mask, paste_mask)
            new_objects_mask = np.maximum(new_objects_mask, paste_mask)

        if return_viz_info:
            return (
                result,
                combined_mask,
                {"original_mask": original_mask, "new_objects_mask": new_objects_mask},
            )

        return result, combined_mask

    def _random_crop(
        self, image: np.ndarray, min_size: int = 50, max_size: int = 150
    ) -> tuple:
        """Extract a random crop from image as fallback."""
        h, w = image.shape[:2]
        crop_h = random.randint(min_size, min(max_size, h // 2))
        crop_w = random.randint(min_size, min(max_size, w // 2))
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        crop_img = image[y : y + crop_h, x : x + crop_w].copy()
        crop_mask = np.ones((crop_h, crop_w), dtype=np.uint8)
        return crop_img, crop_mask


# =============================================================================
# Dataset Mode Functions (New Flow)
# =============================================================================


def copy_split_to_output(src_dir: Path, dst_dir: Path, split_name: str) -> list:
    """
    Copy all files from a split directory to output, preserving structure.

    Returns list of sample dicts for the copied files.
    """
    src_split = src_dir / split_name
    dst_split = dst_dir / split_name

    if not src_split.exists():
        print(f"Warning: Source split not found: {src_split}")
        return []

    samples = []

    # Copy forged images and masks
    src_forged = src_split / "images" / "forged"
    dst_forged = dst_split / "images" / "forged"
    src_masks = src_split / "masks"
    dst_masks = dst_split / "masks"

    dst_forged.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    if src_forged.exists():
        for img_path in sorted(src_forged.glob("*.png")):
            case_id = img_path.stem
            dst_img = dst_forged / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy mask if exists
            src_mask = src_masks / f"{case_id}.npy"
            dst_mask = dst_masks / f"{case_id}.npy" if src_mask.exists() else None
            if src_mask.exists():
                shutil.copy2(src_mask, dst_mask)

            samples.append(
                {
                    "image_path": f"{split_name}/images/forged/{case_id}.png",
                    "mask_path": f"{split_name}/masks/{case_id}.npy"
                    if dst_mask
                    else None,
                    "is_forged": True,
                    "case_id": case_id,
                    "is_augmented": False,
                }
            )

    # Copy authentic images
    src_authentic = src_split / "images" / "authentic"
    dst_authentic = dst_split / "images" / "authentic"
    dst_authentic.mkdir(parents=True, exist_ok=True)

    if src_authentic.exists():
        for img_path in sorted(src_authentic.glob("*.png")):
            case_id = img_path.stem
            dst_img = dst_authentic / img_path.name
            shutil.copy2(img_path, dst_img)

            samples.append(
                {
                    "image_path": f"{split_name}/images/authentic/{case_id}.png",
                    "mask_path": None,
                    "is_forged": False,
                    "case_id": case_id,
                    "is_augmented": False,
                }
            )

    return samples


def augment_split(
    src_dir: Path,
    dst_dir: Path,
    split_name: str,
    augmentor: CopyMoveAugmentor,
    num_augmentations: int,
    save_visualizations: bool = False,
    max_viz_samples: int = 0,
) -> list:
    """
    Augment forged samples in a split and save to output directory.

    Args:
        src_dir: Source dataset directory
        dst_dir: Destination dataset directory
        split_name: Name of the split (train, val, test)
        augmentor: CopyMoveAugmentor instance
        num_augmentations: Number of augmentations per sample
        save_visualizations: Whether to save visualizations
        max_viz_samples: Max samples to save viz for (0 = all)

    Returns list of augmented sample dicts.
    """
    src_split = src_dir / split_name
    dst_split = dst_dir / split_name

    src_forged = src_split / "images" / "forged"
    src_authentic = src_split / "images" / "authentic"
    src_masks = src_split / "masks"

    dst_forged = dst_split / "images" / "forged"
    dst_masks = dst_split / "masks"

    if not src_forged.exists():
        return []

    # Create visualization directory
    viz_dir = None
    if save_visualizations:
        viz_dir = dst_dir / "visualizations" / split_name
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Collect forged samples
    forged_samples = []
    for img_path in sorted(src_forged.glob("*.png")):
        case_id = img_path.stem
        mask_path = src_masks / f"{case_id}.npy"
        authentic_path = src_authentic / f"{case_id}.png"

        if mask_path.exists():
            forged_samples.append(
                {
                    "forged_path": img_path,
                    "mask_path": mask_path,
                    "authentic_path": authentic_path
                    if authentic_path.exists()
                    else None,
                    "case_id": case_id,
                }
            )

    if not forged_samples:
        return []

    print(
        f"\nAugmenting {split_name} split: {len(forged_samples)} forged samples x {num_augmentations} = {len(forged_samples) * num_augmentations} augmented"
    )
    phase_a = sum(1 for s in forged_samples if s["authentic_path"] is not None)
    phase_b = len(forged_samples) - phase_a
    print(f"  Phase A (with authentic): {phase_a}, Phase B (blind): {phase_b}")

    augmented_samples = []
    viz_count = 0  # Track number of visualizations saved

    for sample in tqdm(forged_samples, desc=f"Augmenting {split_name}"):
        case_id = sample["case_id"]

        # Load forged image and mask
        forged_img = cv2.imread(str(sample["forged_path"]))
        forged_img = cv2.cvtColor(forged_img, cv2.COLOR_BGR2RGB)
        mask = np.load(sample["mask_path"])

        has_authentic = sample["authentic_path"] is not None

        if has_authentic:
            # Phase A
            authentic_img = cv2.imread(str(sample["authentic_path"]))
            authentic_img = cv2.cvtColor(authentic_img, cv2.COLOR_BGR2RGB)
            cached_source_info = augmentor.prepare_source_cache(
                authentic_img, forged_img, mask
            )

            for aug_idx in range(num_augmentations):
                aug_img, aug_mask, viz_info = augmentor.generate_from_triplet(
                    authentic_img,
                    forged_img,
                    mask,
                    return_viz_info=True,
                    cached_source_info=cached_source_info,
                )

                aug_case_id = f"{case_id}_aug{aug_idx}"
                out_img_path = dst_forged / f"{aug_case_id}.png"
                out_mask_path = dst_masks / f"{aug_case_id}.npy"

                cv2.imwrite(str(out_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                np.save(out_mask_path, aug_mask)

                # Save visualization if enabled and within limit
                should_save_viz = (
                    viz_dir
                    and not viz_info.get("fallback", False)
                    and (max_viz_samples == 0 or viz_count < max_viz_samples)
                )
                if should_save_viz:
                    save_train_visualization(
                        viz_dir / f"{aug_case_id}_viz.png",
                        forged_img,
                        viz_info["source_mask"],
                        viz_info["target_masks"],
                        aug_img,
                        aug_mask,
                        case_id,
                        aug_idx,
                    )
                    viz_count += 1

                augmented_samples.append(
                    {
                        "image_path": f"{split_name}/images/forged/{aug_case_id}.png",
                        "mask_path": f"{split_name}/masks/{aug_case_id}.npy",
                        "is_forged": True,
                        "case_id": aug_case_id,
                        "is_augmented": True,
                        "source_case_id": case_id,
                        "phase": "A",
                    }
                )
        else:
            # Phase B
            for aug_idx in range(num_augmentations):
                aug_img, aug_mask, viz_info = augmentor.generate_from_single(
                    forged_img, mask, return_viz_info=True
                )

                aug_case_id = f"{case_id}_aug{aug_idx}"
                out_img_path = dst_forged / f"{aug_case_id}.png"
                out_mask_path = dst_masks / f"{aug_case_id}.npy"

                cv2.imwrite(str(out_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                np.save(out_mask_path, aug_mask)

                # Save visualization if enabled and within limit
                should_save_viz = viz_dir and (
                    max_viz_samples == 0 or viz_count < max_viz_samples
                )
                if should_save_viz:
                    save_supplemental_visualization(
                        viz_dir / f"{aug_case_id}_viz.png",
                        forged_img,
                        viz_info["original_mask"],
                        aug_img,
                        aug_mask,
                        viz_info["new_objects_mask"],
                        case_id,
                        aug_idx,
                    )
                    viz_count += 1

                augmented_samples.append(
                    {
                        "image_path": f"{split_name}/images/forged/{aug_case_id}.png",
                        "mask_path": f"{split_name}/masks/{aug_case_id}.npy",
                        "is_forged": True,
                        "case_id": aug_case_id,
                        "is_augmented": True,
                        "source_case_id": case_id,
                        "phase": "B",
                    }
                )

    print(f"  Generated {len(augmented_samples)} augmented samples")
    if save_visualizations:
        print(f"  Saved {viz_count} visualizations")
    return augmented_samples


def process_dataset_mode(args):
    """
    Process dataset in dataset mode: copy original + generate augmented.

    Output has same structure as input, can be used as drop-in replacement.
    """
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Augmentation Pipeline - Dataset Mode")
    print("=" * 60)
    print(f"\nInput dataset: {dataset_dir}")
    print(f"Output dataset: {output_dir}")
    print(f"Augmentations per sample: {args.num_augmentations}")
    print(f"Save visualizations: {args.save_viz}")
    if args.save_viz:
        viz_limit_str = "all" if args.viz_samples == 0 else f"max {args.viz_samples}"
        print(f"  Visualization samples: {viz_limit_str}")
    print(f"Random seed: {args.seed}")

    # Create augmentor
    augmentor = CopyMoveAugmentor(
        soft_edge_prob=args.soft_edge_prob,
        min_pastes=args.min_pastes,
        max_pastes=args.max_pastes,
        random_seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = {"train": [], "val": [], "test": []}
    aug_counts = {"train": 0, "val": 0, "test": 0}

    # Process train split: copy original + augment
    print("\n--- Processing TRAIN split ---")
    train_original = copy_split_to_output(dataset_dir, output_dir, "train")
    all_samples["train"].extend(train_original)
    print(f"Copied {len(train_original)} original train samples")

    train_augmented = augment_split(
        dataset_dir,
        output_dir,
        "train",
        augmentor,
        args.num_augmentations,
        args.save_viz,
        args.viz_samples,
    )
    all_samples["train"].extend(train_augmented)
    aug_counts["train"] = len(train_augmented)

    # Process val split: copy original + augment
    print("\n--- Processing VAL split ---")
    val_original = copy_split_to_output(dataset_dir, output_dir, "val")
    all_samples["val"].extend(val_original)
    print(f"Copied {len(val_original)} original val samples")

    val_augmented = augment_split(
        dataset_dir,
        output_dir,
        "val",
        augmentor,
        args.num_augmentations,
        args.save_viz,
        args.viz_samples,
    )
    all_samples["val"].extend(val_augmented)
    aug_counts["val"] = len(val_augmented)

    # Process test split: copy original only (no augmentation)
    print("\n--- Processing TEST split ---")
    test_original = copy_split_to_output(dataset_dir, output_dir, "test")
    all_samples["test"].extend(test_original)
    print(f"Copied {len(test_original)} original test samples (no augmentation)")

    # Save sample JSON files
    for split_name, samples in all_samples.items():
        json_path = output_dir / f"{split_name}_samples.json"
        with open(json_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} samples to {json_path}")

    # Load and update metadata
    src_metadata_path = dataset_dir / "metadata.json"
    if src_metadata_path.exists():
        with open(src_metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update metadata with augmentation info
    metadata["augmentation"] = {
        "num_augmentations": args.num_augmentations,
        "soft_edge_prob": args.soft_edge_prob,
        "min_pastes": args.min_pastes,
        "max_pastes": args.max_pastes,
        "seed": args.seed,
        "source_dataset": str(dataset_dir),
    }
    metadata["splits"] = {
        "train": {
            "total": len(all_samples["train"]),
            "original": len(train_original),
            "augmented": aug_counts["train"],
            "forged": sum(1 for s in all_samples["train"] if s["is_forged"]),
            "authentic": sum(1 for s in all_samples["train"] if not s["is_forged"]),
        },
        "val": {
            "total": len(all_samples["val"]),
            "original": len(val_original),
            "augmented": aug_counts["val"],
            "forged": sum(1 for s in all_samples["val"] if s["is_forged"]),
            "authentic": sum(1 for s in all_samples["val"] if not s["is_forged"]),
        },
        "test": {
            "total": len(all_samples["test"]),
            "original": len(test_original),
            "augmented": 0,
            "forged": sum(1 for s in all_samples["test"] if s["is_forged"]),
            "authentic": sum(1 for s in all_samples["test"] if not s["is_forged"]),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Augmentation Complete!")
    print("=" * 60)
    print(f"\nOutput structure (same as input, drop-in replacement):")
    print(f"  {output_dir}/")
    for split_name in ["train", "val", "test"]:
        split_info = metadata["splits"][split_name]
        aug_str = (
            f" + {split_info['augmented']} aug" if split_info["augmented"] > 0 else ""
        )
        print(f"    ├── {split_name}/")
        print(
            f"    │   ├── images/forged/  ({split_info['forged']} total: {split_info['original'] - split_info['authentic']} orig{aug_str})"
        )
        print(f"    │   ├── images/authentic/  ({split_info['authentic']})")
        print(f"    │   └── masks/")
    print(f"    ├── train_samples.json")
    print(f"    ├── val_samples.json")
    print(f"    ├── test_samples.json")
    print(f"    └── metadata.json")
    if args.save_viz:
        print(f"    └── visualizations/")

    total_orig = len(train_original) + len(val_original) + len(test_original)
    total_aug = aug_counts["train"] + aug_counts["val"]
    print(f"\nSummary:")
    print(f"  Original samples: {total_orig}")
    print(f"  Augmented samples: {total_aug}")
    print(f"  Total samples: {total_orig + total_aug}")
    print(f"\nUse this dataset for training:")
    print(f"  python train.py --dataset-dir {output_dir}")


def process_legacy_mode(args):
    """Process in legacy mode with raw directories."""
    print("=" * 60)
    print("Augmentation Pipeline - Legacy Mode")
    print("=" * 60)
    print(f"\nTrain images: {args.train_images}")
    print(f"Train masks: {args.train_masks}")
    print(f"Output: {args.output_dir}")

    augmentor = CopyMoveAugmentor(
        soft_edge_prob=args.soft_edge_prob,
        min_pastes=args.min_pastes,
        max_pastes=args.max_pastes,
        random_seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process using original functions (simplified)
    train_images_dir = Path(args.train_images)
    train_masks_dir = Path(args.train_masks)

    forged_dir = train_images_dir / "forged"
    authentic_dir = train_images_dir / "authentic"

    out_images = output_dir / "aug_train_images" / "forged"
    out_masks = output_dir / "aug_train_masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    generated = []

    if forged_dir.exists() and authentic_dir.exists():
        triplets = []
        for fp in sorted(forged_dir.glob("*.png")):
            case_id = fp.stem
            ap = authentic_dir / f"{case_id}.png"
            mp = train_masks_dir / f"{case_id}.npy"
            if ap.exists() and mp.exists():
                triplets.append((fp, ap, mp, case_id))

        print(f"\nProcessing {len(triplets)} triplets...")
        for fp, ap, mp, case_id in tqdm(triplets):
            forged = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
            authentic = cv2.cvtColor(cv2.imread(str(ap)), cv2.COLOR_BGR2RGB)
            mask = np.load(mp)

            cache = augmentor.prepare_source_cache(authentic, forged, mask)

            for aug_idx in range(args.num_augmentations):
                aug_img, aug_mask, _ = augmentor.generate_from_triplet(
                    authentic, forged, mask, cached_source_info=cache
                )

                aug_id = f"{case_id}_aug{aug_idx}"
                cv2.imwrite(
                    str(out_images / f"{aug_id}.png"),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                )
                np.save(out_masks / f"{aug_id}.npy", aug_mask)
                generated.append(aug_id)

    print(f"\nGenerated {len(generated)} augmented samples")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic copy-move forgeries for data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dataset mode (recommended) - creates drop-in replacement dataset
  python aug_dataset.py --dataset-dir datasets/processed \\
                        --output-dir datasets/augmented \\
                        --num-augmentations 2 \\
                        --save-viz --viz-samples 10

  # Legacy mode - raw directories
  python aug_dataset.py --train-images datasets/train_images \\
                        --train-masks datasets/train_masks \\
                        --output-dir aug_datasets
        """,
    )

    # Dataset mode
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to preprocessed dataset from create_dataset.py (recommended)",
    )

    # Legacy mode
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--train-masks", type=str, default=None)
    parser.add_argument("--supplemental-images", type=str, default=None)
    parser.add_argument("--supplemental-masks", type=str, default=None)

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/augmented",
        help="Output directory (default: datasets/augmented)",
    )
    parser.add_argument("--save-viz", action="store_true", help="Save visualizations")
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=0,
        help="Max number of samples to save visualizations for (0 = all, default: 0)",
    )

    # Augmentation settings
    parser.add_argument("--num-augmentations", type=int, default=1)
    parser.add_argument("--soft-edge-prob", type=float, default=0.7)
    parser.add_argument("--min-pastes", type=int, default=1)
    parser.add_argument("--max-pastes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Determine mode
    if args.dataset_dir:
        if not Path(args.dataset_dir).exists():
            print(f"Error: Dataset directory not found: {args.dataset_dir}")
            sys.exit(1)
        process_dataset_mode(args)
    elif args.train_images and args.train_masks:
        if not Path(args.train_images).exists():
            print(f"Error: Train images not found: {args.train_images}")
            sys.exit(1)
        process_legacy_mode(args)
    else:
        print(
            "Error: Must provide either --dataset-dir or (--train-images and --train-masks)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
