"""
Test script to verify Model V3 + LossV3 with proper auxiliary targets.
Checks gradient flow to all auxiliary heads.
"""

import torch
import torch.nn as nn
from src.model import CMSegNetV2WithAux
from src.loss import LossV3
from dataset import generate_position_offset_gt

def count_parameters_with_gradients(model):
    """Count how many model parameters have gradients"""
    total_params = sum(1 for _ in model.parameters())
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    params_with_nonzero_grad = sum(1 for p in model.parameters() 
                                   if p.grad is not None and p.grad.abs().sum() > 0)
    return total_params, params_with_grad, params_with_nonzero_grad

def check_module_gradients(model, module_name):
    """Check if a specific module has gradients"""
    module = dict(model.named_modules())[module_name]
    total = sum(1 for _ in module.parameters())
    with_grad = sum(1 for p in module.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    return total, with_grad

def test_model_v3_with_aux_targets():
    """Test Model V3 + LossV3 with proper auxiliary targets"""
    print("="*80)
    print("Testing Model V3 + LossV3 with Auxiliary Targets")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model and loss
    model = CMSegNetV2WithAux(aux_from_decoder_idx=1).to(device)
    criterion = LossV3(
        focal_weight=0.5,
        dice_weight=0.5,
        boundary_weight=0.3,
        enhanced_boundary_weight=2.0,
        position_weight=5.0,
        boundary_dilation=4
    ).to(device)
    
    print(f"Model: CMSegNetV2WithAux (aux from decoder idx 1)")
    print(f"Loss: LossV3 (multi-task with auxiliary supervision)\n")
    
    # Create dummy data
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    
    # Generate auxiliary targets
    print("Generating auxiliary targets...")
    position_offsets, boundary_masks = generate_position_offset_gt(masks)
    aux_targets = {
        'position_offsets': position_offsets.to(device),
        'boundary_masks': boundary_masks.to(device)
    }
    
    print(f"  - Position offsets shape: {position_offsets.shape}")
    print(f"  - Boundary masks shape: {boundary_masks.shape}")
    
    # Forward pass
    print("\nForward pass...")
    model.train()
    outputs = model(images)
    
    print(f"  - Main output shape: {outputs['main'].shape}")
    print(f"  - Boundary output shape: {outputs['boundary'].shape}")
    print(f"  - Offset output shape: {outputs['offset'].shape}")
    
    # Prepare aux_outputs for loss
    aux_outputs = {
        'boundary': outputs['boundary'],
        'offset': outputs['offset']
    }
    
    # Compute loss
    print("\nComputing loss with auxiliary targets...")
    total_loss, loss_dict = criterion(outputs['main'], masks, aux_outputs=aux_outputs, aux_targets=aux_targets)
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print("  Loss components:")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"    - {key}: {value:.4f}")
    
    # Backward pass
    print("\nBackward pass...")
    model.zero_grad()
    total_loss.backward()
    
    # Check gradients
    print("\nGradient Analysis:")
    print("-" * 80)
    
    total, with_grad, nonzero_grad = count_parameters_with_gradients(model)
    print(f"Overall gradient coverage: {nonzero_grad}/{total} ({100*nonzero_grad/total:.1f}%)")
    
    # Check specific modules
    modules_to_check = [
        'boundary_head',
        'offset_head',
        'decoder.0',
        'decoder.1',
        'decoder.2',
        'final_conv'
    ]
    
    print("\nPer-module gradient coverage:")
    for module_name in modules_to_check:
        try:
            total_params, with_grad_params = check_module_gradients(model, module_name)
            status = "✓" if with_grad_params > 0 else "✗"
            print(f"  {status} {module_name}: {with_grad_params}/{total_params} parameters with gradients")
        except KeyError:
            print(f"  ? {module_name}: module not found")
    
    # Verify that offset head now receives gradients (this was the issue before)
    offset_total, offset_with_grad = check_module_gradients(model, 'offset_head')
    
    print("\n" + "="*80)
    if offset_with_grad > 0:
        print("SUCCESS! Offset head is receiving gradients with proper auxiliary targets.")
        print(f"Offset head: {offset_with_grad}/{offset_total} parameters with gradients")
    else:
        print("WARNING: Offset head still not receiving gradients!")
        print("This suggests an issue with the loss computation or gradient flow.")
    print("="*80)
    
    return offset_with_grad > 0

if __name__ == '__main__':
    success = test_model_v3_with_aux_targets()
    exit(0 if success else 1)
