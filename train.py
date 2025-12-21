import os
import gc
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import wandb


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    This ensures that:
    - Data shuffling is reproducible
    - Data augmentation randomness is reproducible
    - Model weight initialization is reproducible
    - Dropout and other stochastic layers behave consistently

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For full determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Worker init function for DataLoader to ensure reproducibility.

    Each worker needs its own seed derived from the base seed to ensure
    reproducible data loading across multiple workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


from src.model import CMSegNet, CMFreqSegNet
from src.loss import LossV2, LossV1
from dataset import (
    ForgeryDetectionDataset,
    load_samples,
    get_train_transforms,
    get_val_transforms,
    extract_instances_from_mask,
)
from competition_metrics import calculate_instance_metrics_from_masks


# =============================================================================
# Distributed Training Utilities
# =============================================================================


def is_distributed():
    """Check if distributed training is enabled."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get the rank of the current process."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the total number of processes."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed():
    """
    Initialize distributed training environment.

    This function is called when using torchrun/torch.distributed.launch.
    Environment variables RANK, LOCAL_RANK, and WORLD_SIZE are set by the launcher.
    """
    if "RANK" not in os.environ:
        # Not running in distributed mode
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get distributed info from environment (set by torchrun)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU training (fastest)
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Synchronize all processes
    dist.barrier()

    return True, local_rank, device


def cleanup_distributed():
    """Clean up distributed training resources."""
    if is_distributed():
        dist.destroy_process_group()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing.

    The learning rate linearly increases from 0 to base_lr during warmup,
    then follows cosine annealing decay.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=1e-6,
        eta_min=1e-6,
        last_epoch=-1,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Initialize learning rate
        if last_epoch == -1:
            for group, lr in zip(
                optimizer.param_groups, [warmup_start_lr] * len(self.base_lrs)
            ):
                group["lr"] = lr

    def get_lr(self, epoch):
        """Calculate learning rate for current epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            lrs = [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            lrs = [
                self.eta_min
                + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]
        return lrs

    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        lrs = self.get_lr(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def step_update(self, num_updates):
        """Update learning rate per iteration (for fine-grained warmup)"""
        # Convert iteration to fractional epoch
        # This is called from train_one_epoch if needed for per-iteration updates
        pass


class CosineAnnealingWarmRestartsDecay:
    """
    Cosine annealing scheduler with warm restarts and optional LR decay.

    Similar to PyTorch's CosineAnnealingWarmRestarts but with decay factor
    that reduces the max LR after each restart. This is useful for progressive
    augmentation where early phases need high LR and later phases benefit from
    more stable, lower LR.

    When total_epochs is provided, the scheduler keeps the same number of cycles
    (total_epochs // T_0) but distributes any remainder epochs across the later
    cycles. This ensures each cycle completes a full smooth cosine curve and
    training ends exactly at eta_min.

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of epochs for the first restart cycle
        T_mult: Factor to increase cycle length after each restart (default: 1)
        eta_min: Minimum learning rate (default: 1e-6)
        decay: Factor to multiply max LR after each restart (default: 1.0, no decay)
               e.g., decay=0.8 means max LR becomes 80% after each restart
        total_epochs: Total number of training epochs (required for smooth ending)

    Example with T_0=12, decay=0.8, base_lr=1e-4, total_epochs=50:
        num_cycles = 50 // 12 = 4, remainder = 50 % 12 = 2
        Cycle lengths: [12, 12, 13, 13] (remainder distributed to later cycles)
        Each cycle completes a full smooth cosine from max_lr to eta_min
    """

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=1e-6,
        decay=1.0,
        last_epoch=-1,
        total_epochs=None,
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Track current cycle
        self.T_cur = 0  # Current position in cycle
        self.T_i = T_0  # Current cycle length
        self.cycle = 0  # Current cycle number

        # Pre-compute cycle boundaries for smooth ending
        self._cycle_starts = None  # Start epoch for each cycle
        self._cycle_lengths = None  # Length of each cycle
        self._num_cycles = None
        if total_epochs is not None and T_mult == 1:
            self._compute_cycle_boundaries()

    def _compute_cycle_boundaries(self):
        """
        Compute cycle start epochs and lengths to fit total_epochs exactly.
        Distributes remainder epochs to the later cycles.
        """
        num_cycles = self.total_epochs // self.T_0
        if num_cycles == 0:
            num_cycles = 1  # At least one cycle

        remainder = self.total_epochs - (num_cycles * self.T_0)

        # Distribute remainder to later cycles
        # e.g., 50 epochs, T_0=12, num_cycles=4, remainder=2
        # lengths = [12, 12, 13, 13]
        self._cycle_lengths = []
        for i in range(num_cycles):
            length = self.T_0
            # Add 1 to later cycles to distribute remainder
            if i >= num_cycles - remainder:
                length += 1
            self._cycle_lengths.append(length)

        # Compute start epochs for each cycle
        self._cycle_starts = [0]
        for i in range(num_cycles - 1):
            self._cycle_starts.append(self._cycle_starts[-1] + self._cycle_lengths[i])

        self._num_cycles = num_cycles

    def _get_cycle_info(self, epoch):
        """Get cycle number, position within cycle, and cycle length for given epoch."""
        if self._cycle_starts is None:
            # Fallback to original behavior
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    return epoch // self.T_0, epoch % self.T_0, self.T_0
                else:
                    n = int(
                        np.log((epoch / self.T_0 * (self.T_mult - 1) + 1))
                        / np.log(self.T_mult)
                    )
                    T_cur = epoch - self.T_0 * (self.T_mult**n - 1) // (self.T_mult - 1)
                    T_i = self.T_0 * self.T_mult**n
                    return n, T_cur, T_i
            else:
                return 0, epoch, self.T_0

        # Find which cycle this epoch belongs to
        for i in range(self._num_cycles - 1, -1, -1):
            if epoch >= self._cycle_starts[i]:
                T_cur = epoch - self._cycle_starts[i]
                T_i = self._cycle_lengths[i]
                return i, T_cur, T_i

        return 0, epoch, self._cycle_lengths[0]

    def get_lr(self):
        """Calculate learning rate for current epoch"""
        # Calculate decay factor based on cycle number
        decay_factor = self.decay**self.cycle

        # Cosine annealing within current cycle
        # Use (T_i - 1) as denominator so that the last epoch (T_cur = T_i - 1)
        # reaches exactly eta_min (cos(π) = -1)
        # This ensures smooth ending at minimum LR
        T_denom = max(self.T_i - 1, 1)  # Avoid division by zero for single-epoch cycles
        lrs = [
            self.eta_min
            + (base_lr * decay_factor - self.eta_min)
            * 0.5
            * (1 + np.cos(np.pi * self.T_cur / T_denom))
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Get cycle info (handles both old and new behavior)
        self.cycle, self.T_cur, self.T_i = self._get_cycle_info(epoch)

        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


def get_progressive_aug_level(epoch: int, total_epochs: int, max_level: int = 3) -> int:
    """
    Calculate augmentation level for progressive/curriculum learning.

    Divides training into 2 phases:
    - First half: light augmentation (level 1)
    - Second half: strong augmentation (level 3)

    This simplified approach helps the model learn basic features first with
    light augmentation, then adapt to harder augmentations in the second half.

    Uses integer division (total_epochs // 2) to align with LR scheduler
    cycle boundaries when using cosine_restarts with T_0 = total_epochs // 2.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        max_level: Maximum augmentation level (default: 3, used for strong phase)

    Returns:
        Augmentation level: 1 (light) for first half, 3 (strong) for second half

    Example with 100 epochs:
        Epochs 0-49:   level 1 (light augmentation)
        Epochs 50-99:  level 3 (strong augmentation)
    """
    # 2 phases: light (level 1) -> strong (level 3)
    half_epochs = total_epochs // 2
    if half_epochs == 0:
        half_epochs = 1  # Safety for very short training

    if epoch < half_epochs:
        return 1  # Light augmentation
    else:
        return max_level  # Strong augmentation (default: 3)


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
        "iou": iou.mean().item(),
        "f1": f1.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "specificity": specificity.mean().item(),
    }


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    epoch,
    log_wandb=True,
    loss_version=1,
):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    running_loss_component1 = 0.0  # dice_loss or focal_loss
    running_loss_component2 = 0.0  # bce_loss or dice_loss
    running_loss_component3 = 0.0  # None or boundary_loss
    running_metrics = {
        "iou": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())

    for batch_idx, (images, masks, case_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, enabled=scaler.is_enabled()):
            outputs = model(images)
            loss_output = criterion(outputs, masks)

            # Handle different loss function outputs
            if loss_version == 1:
                total_loss, loss_comp1, loss_comp2 = loss_output
                loss_comp3 = torch.tensor(0.0).to(device)
            else:
                total_loss, loss_comp1, loss_comp2, loss_comp3 = loss_output

        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler if it's a per-iteration scheduler
        if scheduler is not None and hasattr(scheduler, "step_update"):
            scheduler.step_update(epoch * len(dataloader) + batch_idx)

        # Calculate metrics
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            batch_metrics = calculate_metrics(pred_probs, masks)

        # Update running metrics
        running_loss += total_loss.item()
        running_loss_component1 += loss_comp1.item()
        running_loss_component2 += loss_comp2.item()
        running_loss_component3 += loss_comp3.item()
        for key in running_metrics:
            running_metrics[key] += batch_metrics[key]

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": running_loss / (batch_idx + 1),
                "iou": running_metrics["iou"] / (batch_idx + 1),
                "f1": running_metrics["f1"] / (batch_idx + 1),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    # Calculate average metrics
    num_batches = len(dataloader)
    result = {
        "loss": running_loss / num_batches,
        "loss_component1": running_loss_component1 / num_batches,
        "loss_component2": running_loss_component2 / num_batches,
        "loss_component3": running_loss_component3 / num_batches,
    }
    for key in running_metrics:
        result[key] = running_metrics[key] / num_batches

    # Log to wandb
    if log_wandb:
        log_dict = {
            "train/loss": result["loss"],
            "train/iou": result["iou"],
            "train/f1": result["f1"],
            "train/precision": result["precision"],
            "train/recall": result["recall"],
            "train/specificity": result["specificity"],
            "epoch": epoch,
        }

        # Add loss components with appropriate names
        if loss_version == 1:
            log_dict["train/dice_loss"] = result["loss_component1"]
            log_dict["train/bce_loss"] = result["loss_component2"]
        else:
            log_dict["train/focal_loss"] = result["loss_component1"]
            log_dict["train/dice_loss"] = result["loss_component2"]
            log_dict["train/boundary_loss"] = result["loss_component3"]

        wandb.log(log_dict)

    return result


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
    compute_of1=True,
    log_wandb=True,
    epoch=0,
    loss_version=1,
    log_prefix="val",
):
    """
    Validate the model with both semantic and instance segmentation metrics.

    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss criterion
        device: Device to run on
        compute_of1: Whether to compute competition oF1 score (slower but accurate)
        log_wandb: Whether to log to wandb
        epoch: Current epoch number for logging
        loss_version: Loss function version (1 for LossV1, 2 for LossV2)
        log_prefix: Prefix for wandb logging (e.g., "val" or "test")
    """
    model.eval()

    running_loss = 0.0
    running_loss_component1 = 0.0  # dice_loss or focal_loss
    running_loss_component2 = 0.0  # bce_loss or dice_loss
    running_loss_component3 = 0.0  # None or boundary_loss
    running_metrics = {
        "iou": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }

    # For oF1 computation
    of1_scores = []

    pbar = tqdm(
        dataloader, desc=f"{log_prefix.capitalize()}", disable=not is_main_process()
    )

    for batch_idx, (images, masks, case_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss_output = criterion(outputs, masks)

        # Handle different loss function outputs
        if loss_version == 1:
            total_loss, loss_comp1, loss_comp2 = loss_output
            loss_comp3 = torch.tensor(0.0).to(device)
        else:
            total_loss, loss_comp1, loss_comp2, loss_comp3 = loss_output

        pred_probs = torch.sigmoid(outputs)
        batch_metrics = calculate_metrics(pred_probs, masks)

        running_loss += total_loss.item()
        running_loss_component1 += loss_comp1.item()
        running_loss_component2 += loss_comp2.item()
        running_loss_component3 += loss_comp3.item()
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

                # Calculate oF1 using the competition metric function
                # It handles all edge cases (no GT, no pred, etc.)
                metrics = calculate_instance_metrics_from_masks(
                    pred_instances, gt_instances
                )
                of1_scores.append(metrics["oF1"])

        pbar.set_postfix(
            {
                "loss": running_loss / (batch_idx + 1),
                "iou": running_metrics["iou"] / (batch_idx + 1),
                "f1": running_metrics["f1"] / (batch_idx + 1),
                "oF1": np.mean(of1_scores) if of1_scores else 0.0,
            }
        )

    # Calculate average metrics
    num_batches = len(dataloader)
    result = {
        "loss": running_loss / num_batches,
        "loss_component1": running_loss_component1 / num_batches,
        "loss_component2": running_loss_component2 / num_batches,
        "loss_component3": running_loss_component3 / num_batches,
    }
    for key in running_metrics:
        result[key] = running_metrics[key] / num_batches

    # Add oF1 score if computed
    if compute_of1 and of1_scores:
        result["oF1"] = np.mean(of1_scores)

    # Log to wandb
    if log_wandb:
        log_dict = {
            f"{log_prefix}/loss": result["loss"],
            f"{log_prefix}/iou": result["iou"],
            f"{log_prefix}/f1": result["f1"],
            f"{log_prefix}/precision": result["precision"],
            f"{log_prefix}/recall": result["recall"],
            f"{log_prefix}/specificity": result["specificity"],
            "epoch": epoch,
        }

        # Add loss components with appropriate names
        if loss_version == 1:
            log_dict[f"{log_prefix}/dice_loss"] = result["loss_component1"]
            log_dict[f"{log_prefix}/bce_loss"] = result["loss_component2"]
        else:
            log_dict[f"{log_prefix}/focal_loss"] = result["loss_component1"]
            log_dict[f"{log_prefix}/dice_loss"] = result["loss_component2"]
            log_dict[f"{log_prefix}/boundary_loss"] = result["loss_component3"]

        if "oF1" in result:
            log_dict[f"{log_prefix}/oF1"] = result["oF1"]
        wandb.log(log_dict)

    return result


def save_checkpoint(model, optimizer, epoch, metrics, path, distributed=False):
    """Save model checkpoint (handles DDP wrapped models)."""
    # Get the underlying model if wrapped with DDP
    model_to_save = model.module if distributed and hasattr(model, "module") else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, distributed=False):
    """Load model checkpoint (handles DDP wrapped models)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Get the underlying model if wrapped with DDP
    model_to_load = model.module if distributed and hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    print_rank0(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, metrics


def main(args):
    # Set random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed} for reproducibility")

    # Setup distributed training (if using torchrun)
    distributed, local_rank, device = setup_distributed()

    if distributed:
        print_rank0(f"Distributed training enabled: {get_world_size()} GPUs")
        print_rank0(f"Process rank: {get_rank()}, Local rank: {local_rank}")
    else:
        print(f"Using device: {device}")

    # Create output directory
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    # Synchronize before proceeding (ensure output dir exists)
    if distributed:
        dist.barrier()

    # Initialize wandb (only on main process)
    if not args.no_wandb and is_main_process():
        # Need to define dataset_path early for config
        dataset_path = Path(args.dataset_path)
        config_dict = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * get_world_size(),
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "imgsz": args.imgsz,
            "progressive_aug": args.progressive_aug,
            "grayscale": args.grayscale,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "seed": args.seed,
            "amp": args.amp,
            "compile": args.compile,
            "model_mode": args.model_mode,
            "encoder_attention_type": args.encoder_attention_type,
            "decoder_attention_type": args.decoder_attention_type,
            "loss_version": args.loss_version,
            "dice_weight": args.dice_weight,
            "dataset_path": args.dataset_path,
            "has_test_split": (dataset_path / "test_samples.json").exists(),
            "distributed": distributed,
            "world_size": get_world_size(),
        }

        # Add scheduler-specific parameters
        if args.scheduler == "warmup_cosine":
            config_dict.update(
                {
                    "warmup_epochs": args.warmup_epochs,
                    "warmup_start_lr": args.warmup_start_lr,
                }
            )
        elif args.scheduler == "cosine_restarts":
            config_dict.update(
                {
                    "t0": args.t0,
                    "t_mult": args.t_mult,
                    "lr_decay": args.lr_decay,
                }
            )

        # Add LossV2 specific parameters
        if args.loss_version == 2:
            config_dict.update(
                {
                    "focal_weight": args.focal_weight,
                    "boundary_weight": args.boundary_weight,
                    "focal_alpha": args.focal_alpha,
                    "focal_gamma": args.focal_gamma,
                    "boundary_theta": args.boundary_theta,
                }
            )

        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=config_dict
        )

    # Initialize model
    print_rank0("Initializing model...")
    print_rank0(f"Using model mode: {args.model_mode}")
    print_rank0(f"Using encoder attention type: {args.encoder_attention_type}")
    print_rank0(f"Using decoder attention type: {args.decoder_attention_type}")

    # Determine input channels based on grayscale flag
    in_channels = 1 if args.grayscale else 3
    print_rank0(
        f"Using input channels: {in_channels} ({'grayscale' if args.grayscale else 'RGB'})"
    )

    if args.model_mode == "img":
        model = CMSegNet(
            encoder_attention_type=args.encoder_attention_type,
            decoder_attention_type=args.decoder_attention_type,
            in_channels=in_channels,
        )
    elif args.model_mode == "freq":
        model = CMFreqSegNet(
            encoder_attention_type=args.encoder_attention_type,
            decoder_attention_type=args.decoder_attention_type,
            in_channels=in_channels,
        )
    else:
        raise ValueError(f"Unknown model mode: {args.model_mode}")

    model = model.to(device)

    # Wrap model with DistributedDataParallel if using distributed training
    # find_unused_parameters=True is needed because:
    # - img mode doesn't use frequency branch parameters
    # - freq mode may have unused parameters in certain paths
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        print_rank0(
            "Model wrapped with DistributedDataParallel (find_unused_parameters=True)"
        )

    # Optional: Use torch.compile for PyTorch 2.0+ (significant speedup)
    if hasattr(torch, "compile") and args.compile:
        print_rank0("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Watch model with wandb (only on main process)
    if not args.no_wandb and is_main_process():
        wandb.watch(model, log="all", log_freq=100)

    # Initialize loss function
    if args.loss_version == 1:
        print_rank0("Using LossV1 (BCE + Dice)")
        criterion = LossV1(dice_weight=args.dice_weight, eps=1.0)
    elif args.loss_version == 2:
        print_rank0("Using LossV2 (Focal + Dice + Boundary)")
        criterion = LossV2(
            focal_weight=args.focal_weight,
            dice_weight=args.dice_weight,
            boundary_weight=args.boundary_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            boundary_theta=args.boundary_theta,
            dice_eps=1.0,
        )
    else:
        raise ValueError(f"Unknown loss version: {args.loss_version}")

    # Scale learning rate for distributed training (linear scaling rule)
    # When batch size increases by N, LR should also increase by N (or sqrt(N) for stability)
    base_lr = args.lr
    scaled_lr = base_lr * get_world_size()  # Linear scaling
    scaled_min_lr = args.min_lr * get_world_size()
    scaled_warmup_start_lr = args.warmup_start_lr * get_world_size()

    if distributed:
        print_rank0(
            f"Learning rate scaling: {base_lr:.2e} x {get_world_size()} GPUs = {scaled_lr:.2e}"
        )

    # Initialize optimizer with scaled learning rate
    optimizer = optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler (use scaled values)
    if args.scheduler == "warmup_cosine":
        # Warn if using warmup_cosine with progressive augmentation
        if args.progressive_aug:
            print_rank0("\n" + "=" * 60)
            print_rank0("WARNING: Using warmup_cosine with progressive augmentation")
            print_rank0("  Late epochs have hard augmentations but low learning rate.")
            print_rank0(
                "  Consider using --scheduler cosine_restarts for better results."
            )
            print_rank0("=" * 60 + "\n")
        print_rank0(
            f"Using WarmupCosineScheduler (warmup: {args.warmup_epochs} epochs)"
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            warmup_start_lr=scaled_warmup_start_lr,
            eta_min=scaled_min_lr,
        )
    elif args.scheduler == "cosine_restarts":
        # Auto-set t0 to align with progressive augmentation phases
        t0 = args.t0
        t_mult = args.t_mult
        if args.progressive_aug:
            t0 = args.epochs // 2
            t_mult = 1  # Force t_mult=1 for equal cycle lengths with progressive aug
            print_rank0(
                f"\nProgressive aug detected: auto-setting T_0 = epochs/2 = {t0}, T_mult = 1"
            )

        # Auto-set decay for progressive augmentation if not specified
        lr_decay = args.lr_decay
        print_rank0(
            f"Using CosineAnnealingWarmRestartsDecay (T_0: {t0}, T_mult: {t_mult}, decay: {lr_decay}, total_epochs: {args.epochs})"
        )
        scheduler = CosineAnnealingWarmRestartsDecay(
            optimizer,
            T_0=t0,
            T_mult=t_mult,
            eta_min=scaled_min_lr,
            decay=lr_decay,
            total_epochs=args.epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler}")

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.amp)

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, distributed)
        start_epoch += 1

    # Load preprocessed splits from dataset directory
    print_rank0("\nLoading preprocessed dataset...")
    dataset_path = Path(args.dataset_path)
    train_split_path = dataset_path / "train_samples.json"
    val_split_path = dataset_path / "val_samples.json"
    test_split_path = dataset_path / "test_samples.json"

    if not train_split_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_split_path}")
    if not val_split_path.exists():
        raise FileNotFoundError(f"Val split not found: {val_split_path}")

    print_rank0(f"  Dataset path: {dataset_path}")

    # Pass dataset_path as base_dir to resolve relative paths
    train_samples = load_samples(train_split_path, base_dir=dataset_path)
    val_samples = load_samples(val_split_path, base_dir=dataset_path)

    # Load test split if exists (optional for backward compatibility)
    test_samples = None
    if test_split_path.exists():
        print_rank0(
            f"  Loading: {train_split_path.name}, {val_split_path.name}, {test_split_path.name}"
        )
        test_samples = load_samples(test_split_path, base_dir=dataset_path)
    else:
        print_rank0(
            f"  Loading: {train_split_path.name}, {val_split_path.name} (no test split found)"
        )

    forged_count_train = sum(1 for s in train_samples if s["is_forged"])
    forged_count_val = sum(1 for s in val_samples if s["is_forged"])
    print_rank0(f"\n=== Loaded Split Summary ===")
    print_rank0(f"Train: {len(train_samples)} total")
    print_rank0(f"  - Forged: {forged_count_train}")
    print_rank0(f"  - Authentic: {len(train_samples) - forged_count_train}")
    print_rank0(f"Val: {len(val_samples)} total")
    print_rank0(f"  - Forged: {forged_count_val}")
    print_rank0(f"  - Authentic: {len(val_samples) - forged_count_val}")
    if test_samples is not None:
        forged_count_test = sum(1 for s in test_samples if s["is_forged"])
        print_rank0(f"Test: {len(test_samples)} total")
        print_rank0(f"  - Forged: {forged_count_test}")
        print_rank0(f"  - Authentic: {len(test_samples) - forged_count_test}")
    print_rank0("=" * 30)

    # Update wandb config with split sizes (only on main process)
    if not args.no_wandb and is_main_process():
        wandb.config.update(
            {
                "train_samples": len(train_samples),
                "train_forged": forged_count_train,
                "val_samples": len(val_samples),
                "val_forged": forged_count_val,
                "test_samples": len(test_samples) if test_samples else 0,
                "test_forged": forged_count_test if test_samples else 0,
            }
        )

    # Initialize datasets and dataloaders
    # Determine initial augmentation level
    if args.progressive_aug:
        current_aug_level = get_progressive_aug_level(start_epoch, args.epochs)
        print_rank0(f"\nProgressive augmentation enabled!")
        print_rank0(f"  Training will progress through 2 phases: light (1) → strong (3)")
        print_rank0(f"  Starting at level {current_aug_level} (epoch {start_epoch})")
    else:
        current_aug_level = 0
        print_rank0(f"\nUsing fixed augmentation level: {current_aug_level}")

    # Print aug level description
    aug_descriptions = {
        0: "No augmentation (normalize only)",
        1: "Light: flips + small brightness/contrast",
        2: "Medium: + shift/scale/rotate + noise + CLAHE",
        3: "Strong: + blur + elastic + coarse dropout",
    }
    print_rank0(f"  → {aug_descriptions.get(current_aug_level, 'Unknown')}")

    if args.grayscale:
        print_rank0("  → Grayscale mode: ON (1 channel input)")

    # Helper function to create train dataloader with specific aug level
    def create_train_dataloader(aug_level):
        train_dataset = ForgeryDetectionDataset(
            samples=train_samples,
            imgsz=args.imgsz,
            split="train",
            transform=get_train_transforms(
                args.imgsz, aug_level=aug_level, grayscale=args.grayscale
            ),
            grayscale=args.grayscale,
        )

        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                seed=args.seed,
            )

        # Create a seeded generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=True if distributed else False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return train_loader, train_sampler

    # Create initial train dataloader
    train_loader, train_sampler = create_train_dataloader(current_aug_level)

    # Create validation dataset and dataloader
    val_dataset = ForgeryDetectionDataset(
        samples=val_samples,
        imgsz=args.imgsz,
        split="val",
        transform=get_val_transforms(args.imgsz, grayscale=args.grayscale),
        grayscale=args.grayscale,
    )

    val_sampler = None
    if distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            seed=args.seed,
        )

    # Create a seeded generator for reproducible data loading
    g_val = torch.Generator()
    g_val.manual_seed(args.seed)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=g_val,
    )

    # Create test dataloader if test split exists
    test_loader = None
    test_sampler = None
    if test_samples is not None:
        test_dataset = ForgeryDetectionDataset(
            samples=test_samples,
            imgsz=args.imgsz,
            split="test",
            transform=get_val_transforms(args.imgsz, grayscale=args.grayscale),
            grayscale=args.grayscale,
        )

        if distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                seed=args.seed,
            )

        # Create a seeded generator for reproducible data loading
        g_test = torch.Generator()
        g_test.manual_seed(args.seed)

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            worker_init_fn=seed_worker,
            generator=g_test,
        )
        print_rank0(f"Test dataloader: {len(test_loader)} batches")

    # Training loop
    best_iou = 0.0

    print_rank0(f"\nStarting training from epoch {start_epoch}...")
    if distributed:
        print_rank0(
            f"Effective batch size: {args.batch_size * get_world_size()} ({args.batch_size} x {get_world_size()} GPUs)"
        )

    for epoch in range(start_epoch, args.epochs):
        # Progressive augmentation: check if we need to update aug level
        if args.progressive_aug:
            new_aug_level = get_progressive_aug_level(epoch, args.epochs)
            if new_aug_level != current_aug_level:
                current_aug_level = new_aug_level
                print_rank0(f"\n{'=' * 50}")
                print_rank0(f"Progressive Aug: Switching to level {current_aug_level}")
                print_rank0(f"  → {aug_descriptions.get(current_aug_level, 'Unknown')}")
                print_rank0(f"{'=' * 50}")

                # Recreate train dataloader with new augmentation level
                train_loader, train_sampler = create_train_dataloader(current_aug_level)

        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Update learning rate at START of epoch (before training)
        # This ensures the scheduler computes LR for the current epoch
        if args.scheduler == "warmup_cosine":
            scheduler.step(epoch)
        elif args.scheduler == "cosine_restarts":
            scheduler.step(epoch)

        print_rank0(f"\nEpoch {epoch}/{args.epochs - 1}")
        print_rank0(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        if args.progressive_aug:
            print_rank0(f"Aug level: {current_aug_level}")

        # Log aug level to wandb
        if not args.no_wandb and is_main_process():
            wandb.log({"aug_level": current_aug_level, "epoch": epoch})

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            scheduler,
            device,
            epoch,
            log_wandb=(not args.no_wandb and is_main_process()),
            loss_version=args.loss_version,
        )

        # Print loss components
        if args.loss_version == 1:
            print_rank0(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Dice: {train_metrics['loss_component1']:.4f}, "
                f"BCE: {train_metrics['loss_component2']:.4f}"
            )
        else:
            print_rank0(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Focal: {train_metrics['loss_component1']:.4f}, "
                f"Dice: {train_metrics['loss_component2']:.4f}, "
                f"Boundary: {train_metrics['loss_component3']:.4f}"
            )

        print_rank0(
            f"        mIoU: {train_metrics['iou']:.4f}, "
            f"mF1: {train_metrics['f1']:.4f}, "
            f"Prec: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"Spec: {train_metrics['specificity']:.4f}"
        )

        # Validate - compute oF1 every N epochs or on last epoch
        compute_of1 = (epoch % args.of1_freq == 0) or (epoch == args.epochs - 1)
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            compute_of1=compute_of1,
            log_wandb=(not args.no_wandb and is_main_process()),
            epoch=epoch,
            loss_version=args.loss_version,
        )

        # Print loss components
        if args.loss_version == 1:
            print_rank0(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Dice: {val_metrics['loss_component1']:.4f}, "
                f"BCE: {val_metrics['loss_component2']:.4f}"
            )
        else:
            print_rank0(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Focal: {val_metrics['loss_component1']:.4f}, "
                f"Dice: {val_metrics['loss_component2']:.4f}, "
                f"Boundary: {val_metrics['loss_component3']:.4f}"
            )

        print_str = (
            f"        mIoU: {val_metrics['iou']:.4f}, "
            f"mF1: {val_metrics['f1']:.4f}, "
            f"Prec: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, "
            f"Spec: {val_metrics['specificity']:.4f}"
        )
        if "oF1" in val_metrics:
            print_str += f", oF1: {val_metrics['oF1']:.4f}"
        print_rank0(print_str)

        # Log learning rate to wandb (the LR used for training this epoch)
        if not args.no_wandb and is_main_process():
            wandb.log(
                {"learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch}
            )

        # Save checkpoint (only on main process)
        if is_main_process() and (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            save_checkpoint(
                model, optimizer, epoch, val_metrics, checkpoint_path, distributed
            )

        # Save best model based on oF1 score (if available) or F1 score
        metric_key = "oF1" if "oF1" in val_metrics else "f1"
        if val_metrics[metric_key] > best_iou:
            best_iou = val_metrics[metric_key]
            best_path = os.path.join(args.output_dir, "best_model.pth")

            # Only save on main process
            if is_main_process():
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, best_path, distributed
                )
                print(
                    f"New best model saved! {metric_key}: {best_iou:.4f}, IoU: {val_metrics['iou']:.4f}"
                )

                # Log best model to wandb
                if not args.no_wandb:
                    wandb.run.summary[f"best_{metric_key}"] = best_iou
                    wandb.run.summary["best_iou"] = val_metrics["iou"]
                    # Save model artifact
                    artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)

        # Synchronize all processes before next epoch
        if distributed:
            dist.barrier()

    print_rank0("\nTraining completed!")
    print_rank0(f"Best validation score: {best_iou:.4f}")

    # Check if we need to run test evaluation
    has_test = test_loader is not None
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    has_best_model = os.path.exists(best_model_path)

    # Cleanup distributed training and free memory BEFORE test evaluation
    # This prevents OOM issues on multi-GPU setups
    if distributed:
        dist.barrier()  # Ensure all processes are done with training

    # Delete training objects to free GPU memory
    del model
    del optimizer
    del train_loader
    del val_loader
    if scheduler is not None:
        del scheduler
    torch.cuda.empty_cache()
    gc.collect()

    # Cleanup distributed process group
    cleanup_distributed()

    # Final test evaluation - run only on main process with single GPU
    if has_test and is_main_process() and has_best_model:
        print("\n" + "=" * 50)
        print("Running final evaluation on TEST set...")
        print("=" * 50)

        # Use cuda:0 for test evaluation (single GPU)
        test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

        # Create fresh model for test evaluation (same config as training)
        print("Creating fresh model for test evaluation...")
        test_in_channels = 1 if args.grayscale else 3
        if args.model_mode == "img":
            test_model = CMSegNet(
                encoder_attention_type=args.encoder_attention_type,
                decoder_attention_type=args.decoder_attention_type,
                in_channels=test_in_channels,
            )
        else:
            test_model = CMFreqSegNet(
                encoder_attention_type=args.encoder_attention_type,
                decoder_attention_type=args.decoder_attention_type,
                in_channels=test_in_channels,
            )

        # Load best checkpoint
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(
            best_model_path, map_location=test_device, weights_only=False
        )
        test_model.load_state_dict(checkpoint["model_state_dict"])
        test_model = test_model.to(test_device)
        test_model.eval()

        # Re-create test criterion
        if args.loss_version == 1:
            test_criterion = LossV1(dice_weight=args.dice_weight, eps=1.0)
        else:
            test_criterion = LossV2(
                focal_weight=args.focal_weight,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                boundary_theta=args.boundary_theta,
                dice_eps=1.0,
            )

        test_metrics = validate(
            test_model,
            test_loader,
            test_criterion,
            test_device,
            compute_of1=True,  # Always compute oF1 for final test
            log_wandb=(not args.no_wandb),
            epoch=args.epochs,  # Use final epoch for logging
            loss_version=args.loss_version,
            log_prefix="test",
        )

        # Print test results
        print("\n=== FINAL TEST RESULTS ===")
        if args.loss_version == 1:
            print(
                f"Test  - Loss: {test_metrics['loss']:.4f}, "
                f"Dice: {test_metrics['loss_component1']:.4f}, "
                f"BCE: {test_metrics['loss_component2']:.4f}"
            )
        else:
            print(
                f"Test  - Loss: {test_metrics['loss']:.4f}, "
                f"Focal: {test_metrics['loss_component1']:.4f}, "
                f"Dice: {test_metrics['loss_component2']:.4f}, "
                f"Boundary: {test_metrics['loss_component3']:.4f}"
            )

        print(
            f"        mIoU: {test_metrics['iou']:.4f}, "
            f"mF1: {test_metrics['f1']:.4f}, "
            f"Prec: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, "
            f"Spec: {test_metrics['specificity']:.4f}"
        )
        if "oF1" in test_metrics:
            print(f"        oF1 (competition metric): {test_metrics['oF1']:.4f}")
        print("=" * 30)

        # Store test metrics in wandb summary
        if not args.no_wandb:
            wandb.run.summary["test/loss"] = test_metrics["loss"]
            wandb.run.summary["test/iou"] = test_metrics["iou"]
            wandb.run.summary["test/f1"] = test_metrics["f1"]
            wandb.run.summary["test/precision"] = test_metrics["precision"]
            wandb.run.summary["test/recall"] = test_metrics["recall"]
            wandb.run.summary["test/specificity"] = test_metrics["specificity"]
            if "oF1" in test_metrics:
                wandb.run.summary["test/oF1"] = test_metrics["oF1"]

        # Cleanup test model
        del test_model
        del test_criterion
        torch.cuda.empty_cache()

    elif has_test and is_main_process() and not has_best_model:
        print("Best model not found, skipping test evaluation")

    # Finish wandb run (only on main process)
    if not args.no_wandb and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CMSegNet for segmentation")

    # Data parameters - preprocessed dataset directory
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to preprocessed dataset directory (from create_dataset.py)",
    )

    # Common data parameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512, help="Input image size (height and width)"
    )

    # Augmentation parameters
    parser.add_argument(
        "--progressive-aug",
        action="store_true",
        help="Enable progressive augmentation: start with level 0 and gradually increase to level 3 during training (curriculum learning)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert images to grayscale (1 channel). Prevents model from learning color-based shortcuts.",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Model parameters
    parser.add_argument(
        "--model-mode",
        type=str,
        default="img",
        choices=["img", "freq"],
        help="Model mode: img (standard CMSegNet) or freq (frequency-enhanced)",
    )
    parser.add_argument(
        "--encoder-attention-type",
        type=str,
        default="sa",
        choices=["sa", "cara", "ela"],
        help="Attention mechanism type: sa (SpatialAttention) or cara (CARA) or ela (ELAAttention)",
    )
    parser.add_argument(
        "--decoder-attention-type",
        type=str,
        default="sa",
        choices=["sa", "cara"],
        help="Attention mechanism type for decoder: sa (SpatialAttention) or cara (CARA)",
    )

    # Loss function parameters
    parser.add_argument(
        "--loss-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="Loss function version: 1 (LossV1: BCE + Dice) or 2 (LossV2: Focal + Dice + Boundary)",
    )
    parser.add_argument(
        "--dice-weight", type=float, default=0.5, help="Dice loss weight"
    )
    parser.add_argument(
        "--focal-weight",
        type=float,
        default=0.5,
        help="Focal loss weight (LossV2 only)",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.3,
        help="Boundary loss weight (LossV2 only)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha parameter (class balance, LossV2 only)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
        help="Focal loss gamma parameter (focus on hard examples, 1.0 optimal for combined losses, LossV2 only)",
    )
    parser.add_argument(
        "--boundary-theta",
        type=float,
        default=5.0,
        help="Boundary loss theta parameter (boundary emphasis strength, LossV2 only)",
    )

    # Scheduler parameters
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmup_cosine",
        choices=["warmup_cosine", "cosine_restarts"],
        help="Learning rate scheduler type (warmup_cosine or cosine_restarts)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs (warmup_cosine only)",
    )
    parser.add_argument(
        "--warmup-start-lr",
        type=float,
        default=1e-6,
        help="Starting learning rate for warmup (warmup_cosine only)",
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument(
        "--t0",
        type=int,
        default=10,
        help="Number of epochs for first restart cycle (cosine_restarts only)",
    )
    parser.add_argument(
        "--t-mult",
        type=int,
        default=2,
        help="Factor to increase cycle length after restart (cosine_restarts only)",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=1.0,
        help="LR decay factor after each restart (cosine_restarts only). e.g., 0.8 means max LR becomes 80%% after each restart. Default 1.0 (no decay).",
    )

    # Optimization parameters
    parser.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile() for speedup"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of dataloader workers"
    )

    # Checkpoint parameters
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--save-freq", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--of1-freq",
        type=int,
        default=5,
        help="Compute oF1 metric every N epochs (expensive)",
    )

    # Wandb parameters
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="forgery-detection",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )

    args = parser.parse_args()

    # Validate that dataset directory exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        parser.error(f"Dataset directory not found: {args.dataset_path}")
    if not dataset_path.is_dir():
        parser.error(f"Dataset path is not a directory: {args.dataset_path}")

    # Validate that required split files exist
    train_split_path = dataset_path / "train_samples.json"
    val_split_path = dataset_path / "val_samples.json"
    if not train_split_path.exists():
        parser.error(f"Train split file not found: {train_split_path}")
    if not val_split_path.exists():
        parser.error(f"Val split file not found: {val_split_path}")

    main(args)
