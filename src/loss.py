import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1.0):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # Use torch.sigmoid instead of separate sigmoid call for potential fusion
        probs = torch.sigmoid(logits)
        
        # Flatten tensors without intermediate views (more memory efficient)
        batch_size = logits.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # Fused multiply and sum operations
        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        union = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1) + self.eps

        # Compute dice score and loss in one expression for better fusion
        dice_score = (2.0 * intersection + self.eps) / union
        loss = 1.0 - dice_score.mean()

        return loss

class MixedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, eps=1.0):
        super(MixedLoss, self).__init__()
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

        # Fused weighted sum
        total_loss = torch.addcmul(bce_loss, self.dice_weight, dice_loss)
        
        return total_loss, dice_loss, bce_loss