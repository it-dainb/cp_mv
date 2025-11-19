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


def create_balanced_splits(image_dir, mask_dir, supplemental_image_dir=None, 
                          supplemental_mask_dir=None, val_split=0.2, random_seed=42):
    """
    Create balanced train/val splits from train and supplemental data.
    
    Strategy:
    1. Split train data (has both forged and authentic) into train_1 and val_1
    2. Split supplemental data (only forged) into train_2 and val_2
    3. Combine: train = train_1 + train_2, val = val_1 + val_2
    4. Balance classes by sampling to match minority class in each split
    
    Args:
        image_dir: Path to train_images directory
        mask_dir: Path to train_masks directory
        supplemental_image_dir: Optional path to supplemental_images
        supplemental_mask_dir: Optional path to supplemental_masks
        val_split: Fraction of data to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_samples, val_samples: Lists of sample dictionaries
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Collect train data (forged + authentic)
    train_forged = []
    train_authentic = []
    
    forged_dir = image_dir / 'forged'
    authentic_dir = image_dir / 'authentic'
    
    if forged_dir.exists():
        for img_path in sorted(forged_dir.glob('*.png')):
            case_id = img_path.stem
            mask_path = mask_dir / f'{case_id}.npy'
            if mask_path.exists():
                train_forged.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'is_forged': True,
                    'case_id': case_id
                })
    
    if authentic_dir.exists():
        for img_path in sorted(authentic_dir.glob('*.png')):
            case_id = img_path.stem
            train_authentic.append({
                'image_path': img_path,
                'mask_path': None,
                'is_forged': False,
                'case_id': case_id
            })
    
    # Collect supplemental data (only forged)
    supplemental_forged = []
    if supplemental_image_dir is not None and supplemental_mask_dir is not None:
        supp_img_dir = Path(supplemental_image_dir)
        supp_mask_dir = Path(supplemental_mask_dir)
        
        if supp_img_dir.exists():
            for img_path in sorted(supp_img_dir.glob('*.png')):
                case_id = img_path.stem
                mask_path = supp_mask_dir / f'{case_id}.npy'
                if mask_path.exists():
                    supplemental_forged.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'is_forged': True,
                        'case_id': f'supp_{case_id}'
                    })
    
    # Split train data (stratified by class)
    if len(train_forged) > 0 and len(train_authentic) > 0:
        train_forged_train, train_forged_val = train_test_split(
            train_forged, test_size=val_split, random_state=random_seed
        )
        train_auth_train, train_auth_val = train_test_split(
            train_authentic, test_size=val_split, random_state=random_seed
        )
    else:
        train_forged_train, train_forged_val = train_forged, []
        train_auth_train, train_auth_val = train_authentic, []
    
    # Split supplemental data
    if len(supplemental_forged) > 0:
        supp_train, supp_val = train_test_split(
            supplemental_forged, test_size=val_split, random_state=random_seed
        )
    else:
        supp_train, supp_val = [], []
    
    # Combine splits
    train_forged_combined = train_forged_train + supp_train
    val_forged_combined = train_forged_val + supp_val
    
    # Balance classes in train set
    train_n_forged = len(train_forged_combined)
    train_n_auth = len(train_auth_train)
    train_min_count = min(train_n_forged, train_n_auth)
    
    if train_min_count > 0:
        train_forged_balanced = random.sample(train_forged_combined, min(train_min_count, train_n_forged))
        train_auth_balanced = random.sample(train_auth_train, min(train_min_count, train_n_auth))
    else:
        train_forged_balanced = train_forged_combined
        train_auth_balanced = train_auth_train
    
    train_samples = train_forged_balanced + train_auth_balanced
    
    # Balance classes in val set
    val_n_forged = len(val_forged_combined)
    val_n_auth = len(train_auth_val)
    val_min_count = min(val_n_forged, val_n_auth)
    
    if val_min_count > 0:
        val_forged_balanced = random.sample(val_forged_combined, min(val_min_count, val_n_forged))
        val_auth_balanced = random.sample(train_auth_val, min(val_min_count, val_n_auth))
    else:
        val_forged_balanced = val_forged_combined
        val_auth_balanced = train_auth_val
    
    val_samples = val_forged_balanced + val_auth_balanced
    
    # Shuffle
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    print("\n=== Balanced Split Summary ===")
    print(f"Train: {len(train_samples)} total")
    print(f"  - Forged: {len(train_forged_balanced)}")
    print(f"  - Authentic: {len(train_auth_balanced)}")
    print(f"Val: {len(val_samples)} total")
    print(f"  - Forged: {len(val_forged_balanced)}")
    print(f"  - Authentic: {len(val_auth_balanced)}")
    print("=" * 30)
    
    return train_samples, val_samples


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
    
    def __init__(self, samples, imgsz=512, split='train', transform=None):
        """
        Args:
            samples: List of sample dictionaries (from create_balanced_splits)
            imgsz: Target image size (height, width)
            split: 'train' or 'val'
            transform: Albumentations transform pipeline
        """
        self.samples = samples
        self.imgsz = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.split = split
        self.transform = transform
        
        forged_count = sum(1 for s in self.samples if s['is_forged'])
        print(f"Loaded {len(self.samples)} images ({split} set)")
        print(f"  - Forged: {forged_count}, Authentic: {len(self.samples) - forged_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or create mask
        if sample['is_forged'] and sample['mask_path'] is not None:
            # Load instance masks and combine into single binary mask
            instance_masks = np.load(sample['mask_path'])  # Shape: [num_instances, H, W]
            
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
            image = cv2.resize(image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Add channel dimension if not present (Albumentations returns [H, W])
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            # Default: normalize and convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask, sample['case_id']


def get_train_transforms(imgsz=512):
    """
    Training transforms for forgery detection.
    No augmentation - only normalization and tensor conversion.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(imgsz=512):
    """
    Validation transforms.
    No augmentation - only normalization and tensor conversion.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    instances = []
    for label_id in range(1, num_labels):  # Skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            instance_mask = (labels == label_id).astype(np.uint8)
            instances.append(instance_mask)
    
    return instances


# Example usage and testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Test dataset loading
    dataset = ForgeryDetectionDataset(
        image_dir='datasets/train_images',
        mask_dir='datasets/train_masks',
        imgsz=512,
        split='train',
        transform=get_train_transforms()
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
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis, 0, 1)
        
        axes[0].imshow(img_vis)
        axes[0].set_title(f'Image (ID: {case_id})')
        axes[0].axis('off')
        
        axes[1].imshow(mask.squeeze(), cmap='gray')
        axes[1].set_title(f'Semantic Mask')
        axes[1].axis('off')
        
        # Show instances
        if len(instances) > 0:
            instance_vis = np.zeros_like(instances[0])
            for i, inst in enumerate(instances):
                instance_vis += inst * (i + 1)
            axes[2].imshow(instance_vis, cmap='tab20')
            axes[2].set_title(f'Instances ({len(instances)})')
        else:
            axes[2].imshow(np.zeros((512, 512)), cmap='gray')
            axes[2].set_title('No instances')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to dataset_visualization.png")
