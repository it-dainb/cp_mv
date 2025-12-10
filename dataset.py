"""
Dataset loader for forgery detection competition.
Handles both forged and authentic images with instance masks.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import random
import json


def load_samples(input_path, base_dir=None):
    """
    Load sample list from JSON file.

    Args:
        input_path: Path to JSON file
        base_dir: Optional base directory to resolve relative paths.
                  If provided, relative paths in JSON will be resolved relative to this directory.
                  If None, paths are used as-is (relative or absolute).

    Returns:
        List of sample dictionaries with Path objects
    """
    with open(input_path, "r") as f:
        samples = json.load(f)

    # Convert strings back to Path objects
    for sample in samples:
        image_path = Path(sample["image_path"])
        mask_path = (
            Path(sample["mask_path"]) if sample["mask_path"] is not None else None
        )

        # If base_dir provided and path is relative, resolve it relative to base_dir
        if base_dir is not None:
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            if mask_path is not None and not mask_path.is_absolute():
                mask_path = base_dir / mask_path

        sample["image_path"] = image_path
        sample["mask_path"] = mask_path

    return samples

class ForgeryDetectionDataset(Dataset):
    """
    Dataset for forgery detection with instance segmentation.

    Structure:
        train_images/
            forged/*.png - Images with duplicated regions
            authentic/*.png - Clean images without duplication
        train_masks/
            *.npy - Binary masks (shape: [num_instances, H, W])
        supplemental_images/
            *.png - Additional forged images (flat structure)
        supplemental_masks/
            *.npy - Corresponding masks for supplemental images
    """

    def __init__(
        self,
        samples,
        imgsz=512,
        split="train",
        transform=None,
        return_aux_targets=False,
        grayscale=False,
    ):
        """
        Args:
            samples: List of sample dictionaries (from create_balanced_splits)
            imgsz: Target image size (height, width)
            split: 'train' or 'val'
            transform: Albumentations transform pipeline
            return_aux_targets: If True, also return position offsets and boundaries for multi-task learning
            grayscale: If True, convert images to grayscale (replicated to 3 channels for model compatibility)
        """
        self.samples = samples
        self.imgsz = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.split = split
        self.transform = transform
        self.return_aux_targets = return_aux_targets
        self.grayscale = grayscale

        forged_count = sum(1 for s in self.samples if s["is_forged"])
        print(f"Loaded {len(self.samples)} images ({split} set)")
        print(
            f"  - Forged: {forged_count}, Authentic: {len(self.samples) - forged_count}"
        )
        if grayscale:
            print(f"  - Grayscale mode: ON (color info discarded)")
        if return_aux_targets:
            print(f"  - Auxiliary targets enabled (position offsets + boundaries)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(str(sample["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale if requested
        # This prevents the model from learning color-based shortcuts
        if self.grayscale:
            # Convert to grayscale using luminosity method (matches human perception)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Keep as single channel - model now supports 1-channel input
            image = gray[:, :, np.newaxis]  # Shape: [H, W, 1]

        # Load or create mask
        if sample["is_forged"] and sample["mask_path"] is not None:
            # Load instance masks and combine into single binary mask
            instance_masks = np.load(
                sample["mask_path"]
            )  # Shape: [num_instances, H, W]

            if instance_masks.ndim == 3:
                # Combine all instances into single binary mask for semantic segmentation
                mask = np.any(instance_masks, axis=0).astype(np.uint8)
            elif instance_masks.ndim == 2:
                # Already a single mask
                mask = instance_masks.astype(np.uint8)
            else:
                raise ValueError(f"Unexpected mask shape: {instance_masks.shape}")
        else:
            # Authentic image - no forgery, all zeros
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize if needed
        if image.shape[:2] != self.imgsz:
            image = cv2.resize(
                image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR
            )
            mask = cv2.resize(
                mask, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_NEAREST
            )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            # Add channel dimension if not present (Albumentations returns [H, W])
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # Ensure mask is float type for loss computation
            mask = mask.float()
        else:
            # Default: normalize and convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Generate auxiliary targets if requested
        if self.return_aux_targets:
            position_offsets, boundary_mask = generate_position_offset_gt(mask)
            return image, mask, position_offsets, boundary_mask, sample["case_id"]
        else:
            return image, mask, sample["case_id"]


def get_train_transforms(imgsz=512, aug_level=0, grayscale=False):
    """
    Training transforms for forgery detection with configurable augmentation levels.

    Designed for SEM (Scanning Electron Microscopy) imagery and copy-move forgery detection:
    - Preserves structural details needed for artifact detection
    - Avoids transforms that destroy copy-paste edge artifacts
    - Uses geometric augmentations that maintain physical meaning

    Args:
        imgsz: Image size (not used, kept for API compatibility)
        aug_level: Augmentation intensity level
            0 = None (only normalize)
            1 = Light (flips + small intensity changes)
            2 = Medium (+ rotation + noise + light blur)
            3 = Strong (+ elastic + coarse dropout)
        grayscale: If True, use grayscale normalization (same value for all 3 channels)
                   instead of ImageNet RGB normalization

    Returns:
        Albumentations Compose object
    """
    transforms_list = []

    if aug_level >= 1:
        # Level 1: Light augmentations (always safe for SEM imagery)
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Small brightness/contrast adjustment
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
            ]
        )

    if aug_level >= 2:
        # Level 2: Medium augmentations (geometric + noise)
        transforms_list.extend(
            [
                # Affine transform (replaces deprecated ShiftScaleRotate)
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15),
                    mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                # Noise augmentation - helps model focus on structure, not noise patterns
                A.OneOf(
                    [
                        A.GaussNoise(
                            std_range=(0.05, 0.15), p=1.0
                        ),  # Updated API: std as fraction of max value
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                    ],
                    p=0.3,
                ),
                # CLAHE - enhances local contrast, good for SEM
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            ]
        )

    if aug_level >= 3:
        # Level 3: Strong augmentations (use with caution)
        transforms_list.extend(
            [
                # Light blur - very small kernel to avoid destroying edge artifacts
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.2,
                ),
                # Subtle elastic transform - don't distort particle shapes too much
                A.ElasticTransform(
                    alpha=30,  # Lower than default for subtle effect
                    sigma=10,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.2,
                ),
                # Coarse dropout - forces model to use context (updated API)
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0,
                    fill_mask=0,  # Critical: dropped regions have no mask
                    p=0.2,
                ),
            ]
        )

    # Always normalize and convert to tensor at the end
    # Use grayscale normalization if grayscale mode is enabled
    if grayscale:
        # Single channel normalization for grayscale
        # ImageNet grayscale equivalent: 0.449 = 0.485*0.299 + 0.456*0.587 + 0.406*0.114 (approx)
        mean = [0.449]
        std = [0.226]
    else:
        # Standard ImageNet RGB normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transforms_list.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


def get_val_transforms(imgsz=512, grayscale=False):
    """
    Validation transforms.
    No augmentation - only normalization and tensor conversion.

    Args:
        imgsz: Image size (not used, kept for API compatibility)
        grayscale: If True, use grayscale normalization (same value for all channels)
    """
    # Use grayscale normalization if grayscale mode is enabled
    if grayscale:
        mean = [0.449]  # Single channel for grayscale
        std = [0.226]
    else:
        # Standard ImageNet RGB normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def extract_instances_from_mask(mask, min_area=50):
    """
    Extract individual instance masks from a binary semantic mask using connected components.

    Args:
        mask: Binary mask [H, W] or [1, H, W] (numpy array or torch tensor)
        min_area: Minimum area (in pixels) for an instance to be kept

    Returns:
        List of individual instance masks (numpy arrays of shape [H, W])
    """
    # Convert to numpy if tensor
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    # Squeeze if needed
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    # Ensure binary
    mask = (mask > 0.5).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    instances = []
    for label_id in range(1, num_labels):  # Skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            instance_mask = (labels == label_id).astype(np.uint8)
            instances.append(instance_mask)

    return instances


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test dataset loading
    dataset = ForgeryDetectionDataset(
        image_dir="datasets/train_images",
        mask_dir="datasets/train_masks",
        imgsz=512,
        split="train",
        transform=get_train_transforms(),
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Visualize first sample
        image, mask, case_id = dataset[0]
        print(f"Case ID: {case_id}")
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")

        # Test instance extraction
        instances = extract_instances_from_mask(mask)
        print(f"Number of instances extracted: {len(instances)}")

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Denormalize image for visualization
        img_vis = image.permute(1, 2, 0).numpy()
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )
        img_vis = np.clip(img_vis, 0, 1)

        axes[0].imshow(img_vis)
        axes[0].set_title(f"Image (ID: {case_id})")
        axes[0].axis("off")

        axes[1].imshow(mask.squeeze(), cmap="gray")
        axes[1].set_title(f"Semantic Mask")
        axes[1].axis("off")

        # Show instances
        if len(instances) > 0:
            instance_vis = np.zeros_like(instances[0])
            for i, inst in enumerate(instances):
                instance_vis += inst * (i + 1)
            axes[2].imshow(instance_vis, cmap="tab20")
            axes[2].set_title(f"Instances ({len(instances)})")
        else:
            axes[2].imshow(np.zeros((512, 512)), cmap="gray")
            axes[2].set_title("No instances")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("dataset_visualization.png", dpi=150, bbox_inches="tight")
        print("Visualization saved to dataset_visualization.png")


def generate_position_offset_gt(mask):
    """
    Generate position offset ground truth for multi-task learning.

    For each manipulated pixel, computes the offset vector (dy, dx) pointing
    to the center of the manipulated region. This auxiliary task helps the model
    learn better feature representations by predicting region centers.

    Args:
        mask: torch.Tensor [B, 1, H, W] or [1, H, W] binary mask (0=authentic, 1=forged)

    Returns:
        offsets: torch.Tensor [B, 2, H, W] or [2, H, W] position offsets (dy, dx)
        boundary: torch.Tensor [B, 1, H, W] or [1, H, W] dilated boundary mask

    Example:
        >>> mask = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> offsets, boundary = generate_position_offset_gt(mask)
        >>> print(offsets.shape)  # [2, 2, 256, 256]
        >>> print(boundary.shape)  # [2, 1, 256, 256]
    """
    # Handle both batched and unbatched inputs
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, _, H, W = mask.shape
    device = mask.device
    dtype = mask.dtype

    # Create coordinate grids
    y_coords = (
        torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    )
    x_coords = (
        torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    )

    # Compute region centers (weighted average by mask)
    masked_y = y_coords * mask
    masked_x = x_coords * mask

    sum_mask = mask.sum(dim=[2, 3], keepdim=True).clamp(min=1e-5)
    center_y = masked_y.sum(dim=[2, 3], keepdim=True) / sum_mask
    center_x = masked_x.sum(dim=[2, 3], keepdim=True) / sum_mask

    # Compute ground truth offsets (center - pixel position)
    # Note: Offsets point FROM pixel TO center
    gt_offset_y = center_y - y_coords
    gt_offset_x = center_x - x_coords
    offsets = torch.cat([gt_offset_y, gt_offset_x], dim=1)  # [B, 2, H, W]

    # Generate boundary ground truth (dilated boundaries)
    # Use morphological gradient: dilation - erosion
    kernel_size = 3
    max_pool = torch.nn.functional.max_pool2d(
        mask, kernel_size, stride=1, padding=kernel_size // 2
    )
    min_pool = -torch.nn.functional.max_pool2d(
        -mask, kernel_size, stride=1, padding=kernel_size // 2
    )
    boundary = max_pool - min_pool

    # Dilate boundaries for stronger training signal (dilation radius = 4)
    dilation_kernel = 2 * 4 + 1  # radius 4 -> kernel 9
    boundary_dilated = torch.nn.functional.max_pool2d(
        boundary, dilation_kernel, stride=1, padding=dilation_kernel // 2
    )

    # Ensure output matches input size
    if boundary_dilated.shape[-2:] != (H, W):
        boundary_dilated = torch.nn.functional.interpolate(
            boundary_dilated, size=(H, W), mode="nearest"
        )

    if squeeze_output:
        offsets = offsets.squeeze(0)
        boundary_dilated = boundary_dilated.squeeze(0)

    return offsets, boundary_dilated
