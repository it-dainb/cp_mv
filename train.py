import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

from src.model import CMSegNet
from src.loss import MixedLoss
from dataset import ForgeryDetectionDataset, create_balanced_splits, get_train_transforms, get_val_transforms, extract_instances_from_mask
from competition_metrics import oF1_score, calculate_instance_metrics_from_masks


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    
    The learning rate linearly increases from 0 to base_lr during warmup,
    then follows cosine annealing decay.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, 
                 eta_min=1e-6, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize learning rate
        if last_epoch == -1:
            for group, lr in zip(optimizer.param_groups, [warmup_start_lr] * len(self.base_lrs)):
                group['lr'] = lr
    
    def get_lr(self, epoch):
        """Calculate learning rate for current epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            lrs = [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lrs = [self.eta_min + (base_lr - self.eta_min) * 0.5 * 
                   (1 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]
        return lrs
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
    
    def step_update(self, num_updates):
        """Update learning rate per iteration (for fine-grained warmup)"""
        # Convert iteration to fractional epoch
        # This is called from train_one_epoch if needed for per-iteration updates
        pass


# Dataset is now imported from dataset.py
# See dataset.py for ForgeryDetectionDataset implementation


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate comprehensive segmentation metrics.
    
    Args:
        pred: Predicted probabilities [B, C, H, W]
        target: Ground truth masks [B, C, H, W]
        threshold: Threshold for binarization
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Binarize predictions and targets
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    fp = (pred_binary * (1 - target_binary)).sum(dim=(1, 2, 3))
    tn = ((1 - pred_binary) * (1 - target_binary)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_binary) * target_binary).sum(dim=(1, 2, 3))
    
    # Add epsilon for numerical stability
    eps = 1e-7
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + eps) / (union + eps)
    
    # Precision
    precision = (tp + eps) / (tp + fp + eps)
    
    # Recall (Sensitivity)
    recall = (tp + eps) / (tp + fn + eps)
    
    # Specificity
    specificity = (tn + eps) / (tn + fp + eps)
    
    # F1 Score (Dice coefficient)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    
    # Return mean values across batch
    return {
        'iou': iou.mean().item(),
        'f1': f1.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'specificity': specificity.mean().item()
    }


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_dice_loss = 0.0
    running_bce_loss = 0.0
    running_metrics = {
        'iou': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'specificity': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, masks, case_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=scaler.is_enabled()):
            outputs = model(images)
            total_loss, dice_loss, bce_loss = criterion(outputs, masks)
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler if it's a per-iteration scheduler
        if scheduler is not None and hasattr(scheduler, 'step_update'):
            scheduler.step_update(epoch * len(dataloader) + batch_idx)
        
        # Calculate metrics
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            batch_metrics = calculate_metrics(pred_probs, masks)
        
        # Update running metrics
        running_loss += total_loss.item()
        running_dice_loss += dice_loss.item()
        running_bce_loss += bce_loss.item()
        for key in running_metrics:
            running_metrics[key] += batch_metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'iou': running_metrics['iou'] / (batch_idx + 1),
            'f1': running_metrics['f1'] / (batch_idx + 1),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    result = {
        'loss': running_loss / num_batches,
        'dice_loss': running_dice_loss / num_batches,
        'bce_loss': running_bce_loss / num_batches,
    }
    for key in running_metrics:
        result[key] = running_metrics[key] / num_batches
    
    return result


@torch.no_grad()
def validate(model, dataloader, criterion, device, compute_of1=True):
    """
    Validate the model with both semantic and instance segmentation metrics.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss criterion
        device: Device to run on
        compute_of1: Whether to compute competition oF1 score (slower but accurate)
    """
    model.eval()
    
    running_loss = 0.0
    running_dice_loss = 0.0
    running_bce_loss = 0.0
    running_metrics = {
        'iou': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'specificity': 0.0
    }
    
    # For oF1 computation
    of1_scores = []
    
    pbar = tqdm(dataloader, desc='Validation')
    
    for batch_idx, (images, masks, case_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        outputs = model(images)
        total_loss, dice_loss, bce_loss = criterion(outputs, masks)
        
        pred_probs = torch.sigmoid(outputs)
        batch_metrics = calculate_metrics(pred_probs, masks)
        
        running_loss += total_loss.item()
        running_dice_loss += dice_loss.item()
        running_bce_loss += bce_loss.item()
        for key in running_metrics:
            running_metrics[key] += batch_metrics[key]
        
        # Compute oF1 for each sample in batch
        if compute_of1:
            for i in range(images.size(0)):
                pred_mask = pred_probs[i].cpu()
                gt_mask = masks[i].cpu()
                
                # Extract instances from predictions and ground truth
                pred_instances = extract_instances_from_mask(pred_mask, min_area=50)
                gt_instances = extract_instances_from_mask(gt_mask, min_area=50)
                
                # Calculate oF1
                if len(gt_instances) == 0:
                    # Authentic image
                    score = 1.0 if len(pred_instances) == 0 else 0.0
                elif len(pred_instances) == 0:
                    # Missed all instances
                    score = 0.0
                else:
                    metrics = calculate_instance_metrics_from_masks(pred_instances, gt_instances)
                    score = metrics['oF1']
                
                of1_scores.append(score)
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'iou': running_metrics['iou'] / (batch_idx + 1),
            'f1': running_metrics['f1'] / (batch_idx + 1),
            'oF1': np.mean(of1_scores) if of1_scores else 0.0
        })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    result = {
        'loss': running_loss / num_batches,
        'dice_loss': running_dice_loss / num_batches,
        'bce_loss': running_bce_loss / num_batches,
    }
    for key in running_metrics:
        result[key] = running_metrics[key] / num_batches
    
    # Add oF1 score if computed
    if compute_of1 and of1_scores:
        result['oF1'] = np.mean(of1_scores)
    
    return result


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, metrics


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    model = CMSegNet()
    model = model.to(device)
    
    # Optional: Use torch.compile for PyTorch 2.0+ (significant speedup)
    if hasattr(torch, 'compile') and args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Initialize loss function
    criterion = MixedLoss(dice_weight=args.dice_weight, eps=1.0)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        warmup_start_lr=args.warmup_start_lr,
        eta_min=args.min_lr
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.amp)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    # Create balanced train/val splits
    print("\nCreating balanced train/val splits...")
    train_samples, val_samples = create_balanced_splits(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        supplemental_image_dir=args.supplemental_images,
        supplemental_mask_dir=args.supplemental_masks,
        val_split=args.val_split,
        random_seed=args.seed
    )
    
    # Initialize datasets and dataloaders
    train_dataset = ForgeryDetectionDataset(
        samples=train_samples,
        imgsz=args.imgsz,
        split='train',
        transform=get_train_transforms(args.imgsz)
    )
    
    val_dataset = ForgeryDetectionDataset(
        samples=val_samples,
        imgsz=args.imgsz,
        split='val',
        transform=get_val_transforms(args.imgsz)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Training loop
    best_iou = 0.0
    
    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs-1}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice_loss']:.4f}, "
              f"BCE: {train_metrics['bce_loss']:.4f}")
        print(f"        mIoU: {train_metrics['iou']:.4f}, "
              f"mF1: {train_metrics['f1']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"Spec: {train_metrics['specificity']:.4f}")
        
        # Validate - compute oF1 every N epochs or on last epoch
        compute_of1 = (epoch % args.of1_freq == 0) or (epoch == args.epochs - 1)
        val_metrics = validate(model, val_loader, criterion, device, compute_of1=compute_of1)
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice_loss']:.4f}, "
              f"BCE: {val_metrics['bce_loss']:.4f}")
        print_str = (f"        mIoU: {val_metrics['iou']:.4f}, "
                     f"mF1: {val_metrics['f1']:.4f}, "
                     f"Prec: {val_metrics['precision']:.4f}, "
                     f"Recall: {val_metrics['recall']:.4f}, "
                     f"Spec: {val_metrics['specificity']:.4f}")
        if 'oF1' in val_metrics:
            print_str += f", oF1: {val_metrics['oF1']:.4f}"
        print(print_str)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
        
        # Save best model based on oF1 score (if available) or F1 score
        metric_key = 'oF1' if 'oF1' in val_metrics else 'f1'
        if val_metrics[metric_key] > best_iou:
            best_iou = val_metrics[metric_key]
            best_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            print(f"New best model saved! {metric_key}: {best_iou:.4f}, IoU: {val_metrics['iou']:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation score: {best_iou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CMSegNet for segmentation')
    
    # Data parameters
    parser.add_argument('--train-images', type=str, required=True, help='Path to training images')
    parser.add_argument('--train-masks', type=str, required=True, help='Path to training masks')
    parser.add_argument('--supplemental-images', type=str, default=None, 
                        help='Path to supplemental images (optional, merged with training data)')
    parser.add_argument('--supplemental-masks', type=str, default=None,
                        help='Path to supplemental masks (optional, merged with training data)')
    parser.add_argument('--val-split', type=float, default=0.2, 
                        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--imgsz', type=int, default=512, help='Input image size (height and width)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dice-weight', type=float, default=0.5, help='Dice loss weight')
    
    # Scheduler parameters
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-6, help='Starting learning rate for warmup')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # Optimization parameters
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() for speedup')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--of1-freq', type=int, default=5, help='Compute oF1 metric every N epochs (expensive)')
    
    args = parser.parse_args()
    
    main(args)
