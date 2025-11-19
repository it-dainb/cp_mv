"""
Inference script for forgery detection competition.
Generates submission file with RLE-encoded predictions.
"""

import os
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from src.model import CMSegNet
from dataset import get_val_transforms, extract_instances_from_mask
from competition_metrics import rle_encode


class TestDataset(Dataset):
    """Dataset for test images"""
    
    def __init__(self, image_dir, imgsz=512, transform=None):
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.transform = transform
        
        # Collect all test images
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + 
                                   list(self.image_dir.glob('*.jpg')))
        
        print(f"Loaded {len(self.image_files)} test images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        case_id = img_path.stem
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)
        
        # Resize if needed
        if image.shape[:2] != self.imgsz:
            image = cv2.resize(image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default normalization
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, case_id, original_size


def predict_instances(model, dataloader, device, threshold=0.5, min_area=50):
    """
    Generate instance predictions for all test images.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run inference on
        threshold: Threshold for binarizing predictions
        min_area: Minimum area for instance to be kept
    
    Returns:
        dict: Dictionary mapping case_id to list of instance masks
    """
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for images, case_ids, original_sizes in tqdm(dataloader, desc='Inference'):
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            
            # Process each image in batch
            for i in range(images.size(0)):
                pred_mask = pred_probs[i].cpu()
                case_id = case_ids[i]
                original_size = (original_sizes[0][i].item(), original_sizes[1][i].item())
                
                # Resize mask back to original size
                pred_mask_np = pred_mask.squeeze().numpy()
                if pred_mask_np.shape != original_size:
                    pred_mask_np = cv2.resize(
                        pred_mask_np, 
                        (original_size[1], original_size[0]),  # (W, H) for cv2
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Binarize
                pred_mask_binary = (pred_mask_np > threshold).astype(np.uint8)
                
                # Extract instances using connected components
                instances = extract_instances_from_mask(pred_mask_binary, min_area=min_area)
                
                predictions[case_id] = instances
    
    return predictions


def create_submission(predictions, output_path='submission.csv'):
    """
    Create submission CSV file with RLE-encoded predictions.
    
    Format:
        case_id, annotation
        - If authentic (no instances): case_id, authentic
        - If forged (with instances): case_id, RLE1;RLE2;...
    
    Args:
        predictions: Dictionary mapping case_id to list of instance masks
        output_path: Path to save submission file
    """
    submission_data = []
    
    for case_id, instances in tqdm(predictions.items(), desc='Creating submission'):
        if len(instances) == 0:
            # No forgery detected - mark as authentic
            submission_data.append({
                'case_id': case_id,
                'annotation': 'authentic'
            })
        else:
            # Forgery detected - encode instances with RLE
            rle_encoded = rle_encode(instances, fg_val=1)
            submission_data.append({
                'case_id': case_id,
                'annotation': rle_encoded
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"  - Authentic: {(df['annotation'] == 'authentic').sum()}")
    print(f"  - Forged: {(df['annotation'] != 'authentic').sum()}")
    
    return df


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = CMSegNet()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {args.checkpoint}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    # Optional: Use torch.compile for faster inference
    if hasattr(torch, 'compile') and args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(
        image_dir=args.test_images,
        imgsz=args.imgsz,
        transform=get_val_transforms(args.imgsz)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict_instances(
        model, 
        test_loader, 
        device,
        threshold=args.threshold,
        min_area=args.min_area
    )
    
    # Create submission file
    submission_df = create_submission(predictions, output_path=args.output)
    
    # Optionally save predictions as numpy files for visualization
    if args.save_masks:
        mask_dir = Path(args.output).parent / 'predicted_masks'
        mask_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving predicted masks to {mask_dir}...")
        for case_id, instances in tqdm(predictions.items()):
            if len(instances) > 0:
                # Stack instances into array
                instances_array = np.stack(instances, axis=0)
                np.save(mask_dir / f'{case_id}.npy', instances_array)
        
        print(f"Saved {len([f for f in mask_dir.glob('*.npy')])} mask files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions for forgery detection')
    
    # Data parameters
    parser.add_argument('--test-images', type=str, required=True, 
                        help='Path to test images directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file path')
    parser.add_argument('--imgsz', type=int, default=512,
                        help='Input image size (should match training)')
    
    # Inference parameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing predictions')
    parser.add_argument('--min-area', type=int, default=50,
                        help='Minimum area (pixels) for instance to be kept')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Optimization parameters
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster inference')
    
    # Output options
    parser.add_argument('--save-masks', action='store_true',
                        help='Save predicted masks as numpy files for visualization')
    
    args = parser.parse_args()
    
    main(args)
