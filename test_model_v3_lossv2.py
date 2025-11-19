"""
Test Model V3 + LossV2 compatibility (backward compatible mode).
Verifies that aux heads are ignored when using LossV2.
"""

import torch
from src.model import CMSegNetV2WithAux
from src.loss import LossV2

def test_model_v3_with_lossv2():
    """Test Model V3 + LossV2 backward compatibility"""
    print("="*80)
    print("Testing Model V3 + LossV2 (Backward Compatible Mode)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model and loss
    model = CMSegNetV2WithAux(aux_from_decoder_idx=1).to(device)
    criterion = LossV2(
        focal_weight=0.5,
        dice_weight=0.5,
        boundary_weight=0.3
    ).to(device)
    
    print("Model: CMSegNetV2WithAux (with auxiliary heads)")
    print("Loss: LossV2 (does NOT use auxiliary heads)")
    print("Expected: Main head trained, aux heads ignored\n")
    
    # Create dummy data
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    
    # Forward pass
    print("Forward pass...")
    model.train()
    outputs = model(images)
    
    print(f"  - Main output shape: {outputs['main'].shape}")
    print(f"  - Boundary output shape: {outputs['boundary'].shape}")
    print(f"  - Offset output shape: {outputs['offset'].shape}")
    
    # Compute loss (should NOT pass aux_outputs to LossV2)
    print("\nComputing loss (without aux_outputs)...")
    try:
        # This is what the fixed train.py does - only pass main output
        total_loss, focal, dice, boundary = criterion(outputs['main'], masks)
        print(f"  ✓ Loss computed successfully")
        print(f"    Total loss: {total_loss.item():.4f}")
        print(f"    Focal: {focal.item():.4f}, Dice: {dice.item():.4f}, Boundary: {boundary.item():.4f}")
    except TypeError as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Backward pass
    print("\nBackward pass...")
    model.zero_grad()
    total_loss.backward()
    
    # Check gradients
    def check_module_gradients(model, module_name):
        """Check if a specific module has gradients"""
        module = dict(model.named_modules())[module_name]
        total = sum(1 for _ in module.parameters())
        with_grad = sum(1 for p in module.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        return total, with_grad
    
    print("\nGradient Analysis:")
    print("-" * 80)
    
    main_total, main_with_grad = check_module_gradients(model, 'final_conv')
    boundary_total, boundary_with_grad = check_module_gradients(model, 'boundary_head')
    offset_total, offset_with_grad = check_module_gradients(model, 'offset_head')
    
    print(f"  Main head (final_conv): {main_with_grad}/{main_total} with gradients")
    print(f"  Boundary head: {boundary_with_grad}/{boundary_total} with gradients (expected: 0)")
    print(f"  Offset head: {offset_with_grad}/{offset_total} with gradients (expected: 0)")
    
    print("\n" + "="*80)
    if main_with_grad > 0 and boundary_with_grad == 0 and offset_with_grad == 0:
        print("SUCCESS! Model V3 + LossV2 backward compatibility working correctly.")
        print("Main head trained, auxiliary heads ignored as expected.")
    else:
        print("WARNING: Unexpected gradient flow pattern!")
    print("="*80)
    
    return main_with_grad > 0 and boundary_with_grad == 0 and offset_with_grad == 0

if __name__ == '__main__':
    success = test_model_v3_with_lossv2()
    exit(0 if success else 1)
