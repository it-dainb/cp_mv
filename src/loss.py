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


class PositionLoss(nn.Module):
    """
    Position offset loss from FastForensics
    Predicts offset of each manipulated pixel to the center of the manipulated region
    """
    def __init__(self):
        super(PositionLoss, self).__init__()
    
    def compute_region_centers(self, mask):
        """
        Compute the center coordinates of manipulated regions
        
        Args:
            mask: [B, 1, H, W] binary mask
        Returns:
            center_y, center_x: [B, 1, 1, 1] center coordinates
        """
        B, _, H, W = mask.shape
        device = mask.device
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=mask.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.arange(W, device=device, dtype=mask.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Weighted average by mask to find center
        masked_y = y_coords * mask
        masked_x = x_coords * mask
        
        sum_mask = mask.sum(dim=[2, 3], keepdim=True).clamp(min=1e-5)
        center_y = masked_y.sum(dim=[2, 3], keepdim=True) / sum_mask
        center_x = masked_x.sum(dim=[2, 3], keepdim=True) / sum_mask
        
        return center_y, center_x
    
    def forward(self, pred_offsets, mask):
        """
        Args:
            pred_offsets: [B, 2, H, W] predicted (dy, dx) offsets
            mask: [B, 1, H, W] ground truth binary mask
        Returns:
            L1 loss between predicted and true offsets
        """
        B, _, H, W = mask.shape
        device = mask.device
        
        # Get region centers
        center_y, center_x = self.compute_region_centers(mask)
        
        # Create pixel coordinate grids
        y_coords = torch.arange(H, device=device, dtype=mask.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.arange(W, device=device, dtype=mask.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Compute ground truth offsets (pixel position - center)
        gt_offset_y = y_coords - center_y
        gt_offset_x = x_coords - center_x
        gt_offsets = torch.cat([gt_offset_y, gt_offset_x], dim=1)  # [B, 2, H, W]
        
        # Only compute loss on manipulated pixels
        # Expand mask to match offset channels
        mask_expanded = mask.expand_as(gt_offsets)
        
        # L1 loss on masked regions
        loss = F.l1_loss(pred_offsets * mask_expanded, gt_offsets * mask_expanded, reduction='sum')
        loss = loss / (mask_expanded.sum() + 1e-5)  # Normalize by number of manipulated pixels
        
        return loss


class EnhancedBoundaryLoss(nn.Module):
    """
    Enhanced boundary loss from FastForensics
    Explicitly predicts and supervises boundary regions with Canny-style edge detection
    """
    def __init__(self, dilation_kernel=4, theta=5.0):
        super(EnhancedBoundaryLoss, self).__init__()
        self.dilation_kernel = dilation_kernel
        self.theta = theta
    
    def extract_boundaries(self, mask):
        """
        Extract boundaries using morphological operations similar to Canny filter
        Then dilate the boundaries for better training signal
        
        Args:
            mask: [B, 1, H, W] binary mask
        Returns:
            boundary_mask: [B, 1, H, W] dilated boundary mask
        """
        # Compute morphological gradient (dilation - erosion)
        kernel_size = 3
        max_pool = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
        min_pool = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size//2)
        boundary = max_pool - min_pool
        
        # Dilate boundaries for stronger training signal
        boundary_dilated = F.max_pool2d(
            boundary, 
            self.dilation_kernel, 
            stride=1, 
            padding=self.dilation_kernel//2
        )
        
        return boundary_dilated.detach()
    
    def forward(self, pred_boundary, mask):
        """
        Args:
            pred_boundary: [B, 1, H, W] predicted boundary logits
            mask: [B, 1, H, W] ground truth binary mask
        Returns:
            BCE loss on boundary regions
        """
        # Extract ground truth boundaries
        gt_boundary = self.extract_boundaries(mask)
        
        # Standard BCE loss for boundary prediction
        loss = F.binary_cross_entropy_with_logits(pred_boundary, gt_boundary)
        
        return loss


class LossV3(nn.Module):
    """
    FastForensics-inspired multi-task loss combining:
    1. Focal Loss (pixel-wise classification, handles imbalance)
    2. Dice Loss (region overlap optimization)
    3. Boundary Loss (precise edge detection)
    4. Enhanced Boundary Loss (explicit boundary supervision from FastForensics)
    5. Position Loss (offset prediction from FastForensics)
    
    This is designed for models that output:
    - Main prediction: manipulation mask
    - Auxiliary prediction 1: boundary map
    - Auxiliary prediction 2: position offsets (dy, dx)
    """
    def __init__(self,
                 # Main task weights (pixel-wise classification)
                 focal_weight=0.5,
                 dice_weight=0.5,
                 boundary_weight=0.3,
                 
                 # FastForensics auxiliary task weights (λ values from paper)
                 enhanced_boundary_weight=2.0,  # λ_bry from paper
                 position_weight=5.0,  # λ_pos from paper
                 
                 # Hyperparameters
                 focal_alpha=0.25,
                 focal_gamma=1.0,
                 boundary_theta=5.0,
                 dice_eps=1.0,
                 boundary_dilation=4):
        super(LossV3, self).__init__()
        
        # Main task weights
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        # Auxiliary task weights (FastForensics)
        self.enhanced_boundary_weight = enhanced_boundary_weight
        self.position_weight = position_weight
        
        # Loss components
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = SoftDiceLoss(eps=dice_eps)
        self.boundary_loss = BoundaryLoss(theta=boundary_theta)
        
        # FastForensics auxiliary losses
        self.enhanced_boundary_loss = EnhancedBoundaryLoss(
            dilation_kernel=boundary_dilation, 
            theta=boundary_theta
        )
        self.position_loss = PositionLoss()
    
    def forward(self, outputs, targets, aux_outputs=None):
        """
        Args:
            outputs: [B, 1, H, W] main manipulation prediction (logits)
            targets: [B, 1, H, W] ground truth binary mask
            aux_outputs: dict with optional auxiliary predictions:
                - 'boundary': [B, 1, H, W] boundary prediction (logits)
                - 'position': [B, 2, H, W] position offset prediction (dy, dx)
        
        Returns:
            total_loss: weighted combination of all losses
            loss_dict: dictionary with individual loss components for logging
        """
        # === Main Task Losses (Pixel-wise Classification) ===
        focal = self.focal_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        boundary = self.boundary_loss(outputs, targets)
        
        # Main task total
        main_loss = (self.focal_weight * focal +
                    self.dice_weight * dice +
                    self.boundary_weight * boundary)
        
        total_loss = main_loss
        
        loss_dict = {
            'focal': focal.item(),
            'dice': dice.item(),
            'boundary': boundary.item(),
            'main_total': main_loss.item(),
        }
        
        # === Auxiliary Task Losses (FastForensics) ===
        if aux_outputs is not None:
            # Enhanced boundary loss (explicit boundary supervision)
            if 'boundary' in aux_outputs:
                enhanced_bry = self.enhanced_boundary_loss(aux_outputs['boundary'], targets)
                total_loss += self.enhanced_boundary_weight * enhanced_bry
                loss_dict['enhanced_boundary'] = enhanced_bry.item()
            
            # Position offset loss
            if 'position' in aux_outputs:
                pos = self.position_loss(aux_outputs['position'], targets)
                total_loss += self.position_weight * pos
                loss_dict['position'] = pos.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


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