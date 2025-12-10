"""
Standalone test evaluation script with wandb logging.

Usage:
    python eval_test.py --checkpoint checkpoints/best_model.pth \
                        --dataset-path datasets/processed \
                        --wandb-run-id <run_id> \
                        --wandb-project <project_name>

To resume logging to an existing wandb run:
    python eval_test.py --checkpoint checkpoints/best_model.pth \
                        --dataset-path datasets/processed \
                        --wandb-run-id 1szpc749 \
                        --wandb-project recod_cp_mv
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.model import CMSegNet, CMFreqSegNet
from src.loss import LossV2, LossV1
from dataset import ForgeryDetectionDataset, load_samples, extract_instances_from_mask
from competition_metrics import calculate_instance_metrics_from_masks


# Combined loss wrapper
class CombinedLoss:
    def __init__(self, version=1):
        if version == 1:
            self.loss_fn = LossV1()
        else:
            self.loss_fn = LossV2()
        self.version = version

    def __call__(self, pred, target):
        return self.loss_fn(pred, target)


def compute_metrics(pred, target, threshold=0.5):
    """Compute segmentation metrics."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, loss_version=1):
    """Evaluate model on test set."""
    model.eval()

    running_loss = 0.0
    running_loss_component1 = 0.0
    running_loss_component2 = 0.0
    running_loss_component3 = 0.0
    running_metrics = {
        "iou": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }

    of1_scores = []

    pbar = tqdm(dataloader, desc="Test Evaluation")

    for batch_idx, (images, masks, case_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss_result = criterion(outputs, masks)

        if isinstance(loss_result, tuple):
            if len(loss_result) == 3:
                loss, loss_c1, loss_c2 = loss_result
                loss_c3 = torch.tensor(0.0)
            else:
                loss, loss_c1, loss_c2, loss_c3 = loss_result
        else:
            loss = loss_result
            loss_c1 = loss_c2 = loss_c3 = torch.tensor(0.0)

        running_loss += loss.item()
        running_loss_component1 += loss_c1.item()
        running_loss_component2 += loss_c2.item()
        running_loss_component3 += (
            loss_c3.item() if isinstance(loss_c3, torch.Tensor) else loss_c3
        )

        # Compute metrics
        probs = torch.sigmoid(outputs)
        batch_metrics = compute_metrics(probs, masks)
        for key in running_metrics:
            running_metrics[key] += batch_metrics[key]

        # Compute oF1 for each sample in batch
        for i in range(images.size(0)):
            pred_mask = (probs[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
            gt_mask = masks[i, 0].cpu().numpy().astype(np.uint8)

            # Extract instances using connected components
            pred_instances = extract_instances_from_mask(pred_mask, min_area=50)
            gt_instances = extract_instances_from_mask(gt_mask, min_area=50)

            # Calculate oF1 using the competition metric function
            metrics = calculate_instance_metrics_from_masks(
                pred_instances, gt_instances
            )
            of1_scores.append(metrics["oF1"])

        pbar.set_postfix(
            {
                "loss": running_loss / (batch_idx + 1),
                "iou": running_metrics["iou"] / (batch_idx + 1),
                "oF1": np.mean(of1_scores) if of1_scores else 0.0,
            }
        )

    # Calculate averages
    num_batches = len(dataloader)
    result = {
        "loss": running_loss / num_batches,
        "loss_component1": running_loss_component1 / num_batches,
        "loss_component2": running_loss_component2 / num_batches,
        "loss_component3": running_loss_component3 / num_batches,
    }
    for key in running_metrics:
        result[key] = running_metrics[key] / num_batches

    result["oF1"] = np.mean(of1_scores) if of1_scores else 0.0

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on test set with wandb logging"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")

    # Model config
    parser.add_argument(
        "--model-mode", type=str, default="img", choices=["img", "freq"]
    )
    parser.add_argument("--encoder-attention-type", type=str, default="eca")
    parser.add_argument("--decoder-attention-type", type=str, default="eca")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale input")
    parser.add_argument("--loss-version", type=int, default=2, choices=[1, 2])

    # Wandb config
    parser.add_argument(
        "--wandb-run-id", type=str, default=None, help="Wandb run ID to resume"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="recod_cp_mv", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Wandb entity/team name"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    dataset_path = Path(args.dataset_path)
    test_samples = load_samples(dataset_path, "test")

    if not test_samples:
        print("No test samples found!")
        return

    print(f"Found {len(test_samples)} test samples")

    test_dataset = ForgeryDetectionDataset(
        samples=test_samples,
        imgsz=args.imgsz,
        split="test",
        transform=None,
        grayscale=args.grayscale,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    in_channels = 1 if args.grayscale else 3
    if args.model_mode == "img":
        model = CMSegNet(
            encoder_attention_type=args.encoder_attention_type,
            decoder_attention_type=args.decoder_attention_type,
            in_channels=in_channels,
        )
    else:
        model = CMFreqSegNet(
            encoder_attention_type=args.encoder_attention_type,
            decoder_attention_type=args.decoder_attention_type,
            in_channels=in_channels,
        )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    # Create criterion
    criterion = CombinedLoss(version=args.loss_version)

    # Initialize wandb
    if not args.no_wandb:
        if args.wandb_run_id:
            # Resume existing run
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="must",
            )
            print(f"Resumed wandb run: {args.wandb_run_id}")
        else:
            # Create new run
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="test_evaluation",
                config=vars(args),
            )
            print(f"Created new wandb run: {wandb.run.id}")

    # Run evaluation
    print("\n" + "=" * 50)
    print("Running TEST evaluation...")
    print("=" * 50)

    test_metrics = evaluate(model, test_loader, criterion, device, args.loss_version)

    # Print results
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
    print(f"        oF1 (competition metric): {test_metrics['oF1']:.4f}")
    print("=" * 30)

    # Log to wandb
    if not args.no_wandb:
        wandb.run.summary["test/loss"] = test_metrics["loss"]
        wandb.run.summary["test/iou"] = test_metrics["iou"]
        wandb.run.summary["test/f1"] = test_metrics["f1"]
        wandb.run.summary["test/precision"] = test_metrics["precision"]
        wandb.run.summary["test/recall"] = test_metrics["recall"]
        wandb.run.summary["test/specificity"] = test_metrics["specificity"]
        wandb.run.summary["test/oF1"] = test_metrics["oF1"]

        # Also log as a step for visibility
        wandb.log(
            {
                "test/loss": test_metrics["loss"],
                "test/iou": test_metrics["iou"],
                "test/f1": test_metrics["f1"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/specificity": test_metrics["specificity"],
                "test/oF1": test_metrics["oF1"],
            }
        )

        wandb.finish()
        print("\nResults logged to wandb!")


if __name__ == "__main__":
    main()
