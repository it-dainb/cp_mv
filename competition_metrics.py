"""
Competition metrics for instance segmentation evaluation.
Based on the optimal F1 (oF1) score with Hungarian matching.
"""

import json
import numba
import numpy as np
from numba import types
import numpy.typing as npt
import scipy.optimize
import torch


class ParticipantVisibleError(Exception):
    pass


@numba.jit(nopython=True)
def _rle_encode_jit(x: npt.NDArray, fg_val: int = 1) -> list[int]:
    """Numba-jitted RLE encoder."""
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    """
    Adapted from contrails RLE https://www.kaggle.com/code/inversion/contrails-rle-submission
    
    Args:
        masks: list of numpy array of shape (height, width), 1 - mask, 0 - background
    
    Returns: 
        run length encodings as a string, with each RLE JSON-encoded and separated by a semicolon.
    """
    return ';'.join([json.dumps(_rle_encode_jit(x, fg_val)) for x in masks])


@numba.njit
def _rle_decode_jit(mask_rle: npt.NDArray, height: int, width: int) -> npt.NDArray:
    """
    s: numpy array of run-length encoding pairs (start, length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if len(mask_rle) % 2 != 0:
        # Numba requires raising a standard exception.
        raise ValueError('One or more rows has an odd number of values.')
    
    starts, lengths = mask_rle[0::2], mask_rle[1::2]
    starts -= 1
    ends = starts + lengths
    
    for i in range(len(starts) - 1):
        if ends[i] > starts[i + 1]:
            raise ValueError('Pixels must not be overlapping.')
    
    img = np.zeros(height * width, dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> npt.NDArray:
    """
    mask_rle: run-length as string formatted (start length)
              empty predictions need to be encoded with '-'
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    
    starts = mask_rle[0::2]
    if sorted(starts) != list(starts):
        raise ParticipantVisibleError('Submitted values must be in ascending order.')
    
    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise ParticipantVisibleError(str(e)) from e


def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray):
    """Calculate F1 score between two binary masks."""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def calculate_f1_matrix(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """
    Calculate F1 score matrix for all pairs of predicted and ground truth masks.
    
    Parameters:
        pred_masks: List of predicted binary masks, each of shape (height, width)
        gt_masks: List of ground truth binary masks, each of shape (height, width)
    
    Returns:
        F1 score matrix of shape (num_pred, num_gt)
    """
    num_instances_pred = len(pred_masks)
    num_instances_gt = len(gt_masks)
    
    f1_matrix = np.zeros((num_instances_pred, num_instances_gt))
    
    # Calculate F1 scores for each pair of predicted and ground truth masks
    for i in range(num_instances_pred):
        for j in range(num_instances_gt):
            pred_flat = pred_masks[i].flatten()
            gt_flat = gt_masks[j].flatten()
            f1_matrix[i, j] = calculate_f1_score(pred_mask=pred_flat, gt_mask=gt_flat)
    
    if f1_matrix.shape[0] < len(gt_masks):
        # Add rows of zeros if fewer predictions than ground truth instances
        f1_matrix = np.vstack((
            f1_matrix, 
            np.zeros((len(gt_masks) - len(f1_matrix), num_instances_gt))
        ))
    
    return f1_matrix


def oF1_score(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """
    Calculate the optimal F1 score for a set of predicted masks against
    ground truth masks using Hungarian algorithm for optimal matching.
    
    This function finds the optimal assignment of predicted masks to ground truth 
    masks based on the F1 score matrix and applies a penalty for excess predictions.
    
    Parameters:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
    
    Returns:
        float: Optimal F1 score with excess prediction penalty
    """
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    
    # Find the best matching between predicted and ground truth masks
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    
    # The linear_sum_assignment discards excess predictions so we need a separate penalty
    excess_predictions_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    
    return np.mean(f1_matrix[row_ind, col_ind]) * excess_predictions_penalty


def calculate_semantic_metrics(pred, target, threshold=0.5):
    """
    Calculate semantic segmentation metrics (for single-mask predictions).
    This is what the current train.py uses.
    
    Args:
        pred: Predicted probabilities [B, C, H, W] (torch tensor)
        target: Ground truth masks [B, C, H, W] (torch tensor)
        threshold: Threshold for binarization
    
    Returns:
        dict: Dictionary containing semantic segmentation metrics
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


def calculate_instance_metrics_from_masks(pred_instances: list[npt.NDArray], 
                                          gt_instances: list[npt.NDArray]):
    """
    Calculate instance segmentation metrics (competition format).
    
    Args:
        pred_instances: List of predicted instance masks (numpy arrays)
        gt_instances: List of ground truth instance masks (numpy arrays)
    
    Returns:
        dict: Dictionary containing instance segmentation metrics including oF1
    """
    if len(gt_instances) == 0:
        return {'oF1': 1.0 if len(pred_instances) == 0 else 0.0}
    
    if len(pred_instances) == 0:
        return {'oF1': 0.0}
    
    # Calculate optimal F1 score
    of1 = oF1_score(pred_instances, gt_instances)
    
    return {'oF1': of1}


# Example usage
if __name__ == '__main__':
    # Test semantic metrics (current implementation)
    print("Testing semantic segmentation metrics:")
    pred = torch.rand(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    semantic_metrics = calculate_semantic_metrics(pred, target)
    print(f"Semantic metrics: {semantic_metrics}")
    
    # Test instance metrics (competition format)
    print("\nTesting instance segmentation metrics (competition format):")
    pred_instances = [
        np.random.randint(0, 2, (256, 256)).astype(bool),
        np.random.randint(0, 2, (256, 256)).astype(bool)
    ]
    gt_instances = [
        np.random.randint(0, 2, (256, 256)).astype(bool),
        np.random.randint(0, 2, (256, 256)).astype(bool)
    ]
    instance_metrics = calculate_instance_metrics_from_masks(pred_instances, gt_instances)
    print(f"Instance metrics (oF1): {instance_metrics}")
