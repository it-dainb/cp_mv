import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance and focus on hard-to-classify pixels
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal modulation factor
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing factor
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss.mean()


class SoftDiceLoss(nn.Module):
    """
    Optimized Dice Loss for region overlap
    """
    def __init__(self, eps=1.0):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = logits.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        union = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1) + self.eps
        
        dice_score = (2.0 * intersection + self.eps) / union
        loss = 1.0 - dice_score.mean()
        
        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for precise edge detection of forged regions
    """
    def __init__(self, theta=5.0):
        super(BoundaryLoss, self).__init__()
        self.theta = theta  # Controls boundary emphasis strength
    
    def forward(self, logits, targets):
        # Compute boundary using morphological gradient approximation
        # Using max pooling and erosion approximation
        kernel_size = 3
        max_pool = F.max_pool2d(targets, kernel_size, stride=1, padding=kernel_size//2)
        min_pool = -F.max_pool2d(-targets, kernel_size, stride=1, padding=kernel_size//2)
        boundary_mask = (max_pool - min_pool).detach()  # Don't backprop through boundary detection
        
        # Weight boundary pixels more heavily
        boundary_weight = 1.0 + self.theta * boundary_mask
        
        # BCE with logits on boundary-weighted regions (safe for autocast)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_loss = bce_loss * boundary_weight
        
        return weighted_loss.mean()

class LossV2(nn.Module):
    """
    Final combined loss for copy-move forgery detection
    
    Combines:
    1. Focal Loss (handles class imbalance, focuses on hard examples)
    2. Dice Loss (optimizes region overlap, robust to imbalance)
    3. Boundary Loss (ensures precise forgery edges)
    """
    def __init__(self,
                 focal_weight=0.5,
                 dice_weight=0.5,
                 boundary_weight=0.3,
                 focal_alpha=0.25,
                 focal_gamma=1.0,
                 boundary_theta=5.0,
                 dice_eps=1.0):
        super(LossV2, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = SoftDiceLoss(eps=dice_eps)
        self.boundary_loss = BoundaryLoss(theta=boundary_theta)
    
    def forward(self, y_pred, y_true):
        # Compute individual loss components
        focal = self.focal_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        boundary = self.boundary_loss(y_pred, y_true)
        
        # Weighted combination
        total_loss = (self.focal_weight * focal +
                      self.dice_weight * dice +
                      self.boundary_weight * boundary)
        
        return total_loss, focal, dice, boundary


class LossV1(nn.Module):
    def __init__(self, dice_weight=0.5, eps=1.0):
        super(LossV1, self).__init__()
        self.dice_weight = dice_weight
        self.eps = eps
        # BCEWithLogitsLoss already uses fused sigmoid + BCE
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(eps=eps)

    def forward(self, y_pred, y_true):
        # Compute BCE (already optimized with fused sigmoid)
        bce_loss = self.bce(y_pred, y_true)

        # Compute Dice Loss
        dice_loss = self.dice(y_pred, y_true)

        # Weighted sum of losses
        total_loss = (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss
        
        return total_loss, dice_loss, bce_loss


# Backward compatibility aliases
MixedLoss = LossV1
CopyMoveForgeryLoss = LossV2