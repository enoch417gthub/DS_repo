"""
sanity_check.py
Run this BEFORE any real training to verify everything works.
"""

import torch
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, set_seed
from model import BloodCellCNN
from dataset import get_dataloaders


def check_gradient_flow(model):
    """
    Check that gradients are flowing to early layers.
    Call this after loss.backward().
    """
    print("\n" + "="*50)
    print("GRADIENT FLOW INSPECTION")
    print("="*50)
    
    has_dead = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            status = "OK" if grad_norm > 1e-7 else "⚠️ DEAD (vanishing)"
            if grad_norm <= 1e-7:
                has_dead = True
            print(f"{name:45s} norm={grad_norm:.2e} {status}")
        else:
            print(f"{name:45s} ❌ NO GRADIENT")
            has_dead = True
    
    return not has_dead


def overfit_one_batch_test(model, train_loader, device, num_steps=50):
    """
    The single most useful debugging technique.
    If your model cannot overfit a single batch to near-zero loss,
    there is a fundamental bug in your code.
    """
    print("\n" + "="*70)
    print("OVERFIT ONE BATCH TEST")
    print("="*70)
    print("Testing if model can memorize a single batch...")
    
    # Get one batch
    imgs, labels = next(iter(train_loader))
    imgs = imgs.to(device)
    labels = labels.to(device)
    
    print(f"Batch shape: {imgs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")
    
    # Re-initialize model and optimizer for this test
    set_seed(42)
    model_test = BloodCellCNN(
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG["base_filters"],
        dropout_rate=0.0,  # No dropout for overfitting test
    ).to(device)
    
    optimizer = torch.optim.AdamW(model_test.parameters(), lr=1e-2)  # Higher LR for quick overfitting
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model_test(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 10 == 0 or step == num_steps - 1:
            accuracy = (logits.argmax(1) == labels).float().mean().item()
            print(f"  Step {step:3d}: loss={loss.item():.6f}, acc={accuracy:.4f}")
    
    final_loss = losses[-1]
    success = final_loss < 0.1
    
    if success:
        print("\n✅ OVERFIT TEST PASSED! Model can memorize a batch.")
    else:
        print(f"\n❌ OVERFIT TEST FAILED! Final loss: {final_loss:.4f} (should be < 0.1)")
        print("   Possible issues:")
        print("   - Labels are incorrect")
        print("   - Data not normalized correctly")
        print("   - Model architecture has a bug")
        print("   - Optimizer not working")
    
    return success, losses


def sanity_check():
    """
    Run all sanity checks before training.
    """
    print("\n" + "🔍"*35)
    print("SANITY CHECK - Run this before training!")
    print("🔍"*35)
    
    device = CONFIG["device"]
    print(f"\nDevice: {device}")
    
    # 1. Load data
    print("\n" + "-"*50)
    print("1. Checking data loading...")
    try:
        loaders, class_names = get_dataloaders(CONFIG)
        train_loader = loaders["train"]
        print(f"   ✅ Data loaded successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Classes: {class_names}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # 2. Get one batch and check shapes
    print("\n" + "-"*50)
    print("2. Checking tensor shapes...")
    try:
        imgs, labels = next(iter(train_loader))
        print(f"   Images shape: {imgs.shape} (expected: [batch, 3, {CONFIG['img_size']}, {CONFIG['img_size']}])")
        print(f"   Labels shape: {labels.shape} (expected: [batch])")
        
        if imgs.shape[1] != 3:
            print(f"   ❌ Wrong number of channels: {imgs.shape[1]}")
            return False
        if imgs.shape[2] != CONFIG["img_size"] or imgs.shape[3] != CONFIG["img_size"]:
            print(f"   ❌ Wrong image size: {imgs.shape[2]}x{imgs.shape[3]}")
            return False
        print("   ✅ Shapes correct")
    except Exception as e:
        print(f"   ❌ Shape check failed: {e}")
        return False
    
    # 3. Check pixel range (should be roughly [-2, 2] after normalisation)
    print("\n" + "-"*50)
    print("3. Checking pixel range...")
    img_min = imgs.min().item()
    img_max = imgs.max().item()
    print(f"   Pixel min: {img_min:.3f}")
    print(f"   Pixel max: {img_max:.3f}")
    
    if img_min < -3 or img_max > 3:
        print(f"   ⚠️ Pixel range looks unusual. Did you normalise correctly?")
    else:
        print("   ✅ Pixel range looks normal")
    
    # 4. Check initial loss (should be approx log(num_classes))
    print("\n" + "-"*50)
    print("4. Checking initial loss...")
    model = BloodCellCNN(
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG["base_filters"],
        dropout_rate=0.0,
    ).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(imgs.to(device))
    
    criterion = torch.nn.CrossEntropyLoss()
    initial_loss = criterion(logits, labels.to(device)).item()
    expected_loss = math.log(CONFIG["num_classes"])
    
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Expected loss (random): {expected_loss:.4f} = log({CONFIG['num_classes']})")
    
    # Loss should be close to expected (within ~0.5)
    if abs(initial_loss - expected_loss) > 0.5:
        print(f"   ⚠️ Initial loss is far from expected. Check weight initialisation.")
    else:
        print("   ✅ Initial loss looks reasonable")
    
    # 5. Check output shape
    print("\n" + "-"*50)
    print("5. Checking output shape...")
    print(f"   Output shape: {logits.shape} (expected: [{imgs.shape[0]}, {CONFIG['num_classes']}])")
    
    if logits.shape[1] != CONFIG["num_classes"]:
        print(f"   ❌ Wrong output dimension")
        return False
    print("   ✅ Output shape correct")
    
    # 6. Overfit one batch test (most important!)
    success, losses = overfit_one_batch_test(model, train_loader, device)
    
    if not success:
        return False
    
    # 7. Check gradient flow
    print("\n" + "-"*50)
    print("7. Checking gradient flow...")
    
    # Do one forward-backward pass
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    imgs, labels = next(iter(train_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()
    
    grad_ok = check_gradient_flow(model)
    
    # 8. Count parameters
    print("\n" + "-"*50)
    print("8. Model parameter count...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    if total_params > 5_000_000:
        print(f"   ⚠️ Model is large ({total_params/1e6:.1f}M). Consider reducing base_filters.")
    else:
        print("   ✅ Model size reasonable")
    
    # Final verdict
    print("\n" + "="*70)
    if success and grad_ok:
        print("✅✅✅ ALL SANITY CHECKS PASSED! You can start training. ✅✅✅")
        print("="*70)
        return True
    else:
        print("❌❌❌ SANITY CHECKS FAILED! Fix issues before training. ❌❌❌")
        print("="*70)
        return False


def lr_range_test(model, loader, device, start_lr=1e-7, end_lr=10.0, num_iters=100):
    """
    Learning Rate Range Test.
    Find optimal LR where loss drops fastest.
    """
    print("\n" + "="*70)
    print("LEARNING RATE RANGE TEST")
    print("="*70)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Exponential factor to go from start_lr to end_lr
    factor = (end_lr / start_lr) ** (1 / num_iters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor)
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    data_iter = iter(loader)
    
    for i in range(num_iters):
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, labels = next(data_iter)
        
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if loss.item() > 4 * best_loss and i > 10:
            print(f"   Stopping early at step {i} - loss diverging")
            break
    
    # Smooth losses for cleaner plot
    def smooth(y, window=10):
        return [np.mean(y[max(0, i-window):i+1]) for i in range(len(y))]
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, smooth(losses), 'b-', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('LR Range Test - Pick LR at steepest downward slope', fontsize=14)
    plt.grid(which='both', alpha=0.3)
    
    # Find the steepest slope
    smoothed_losses = smooth(losses)
    gradients = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(gradients[:len(lrs)//2])  # Look at first half
    recommended_lr = lrs[steepest_idx]
    
    plt.axvline(recommended_lr, color='red', linestyle='--', 
                label=f'Recommended LR: {recommended_lr:.2e}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lr_range_test.png', dpi=120)
    plt.show()
    
    print(f"\nRecommended learning rate: {recommended_lr:.2e}")
    print(f"Range to try: [{recommended_lr/3:.2e}, {recommended_lr*3:.2e}]")
    
    return lrs, losses, recommended_lr


if __name__ == "__main__":
    import numpy as np
    
    # Run sanity check
    passed = sanity_check()
    
    if passed:
        print("\n" + "="*70)
        print("Next steps:")
        print("1. Run LR range test to find optimal learning rate")
        print("2. Update config.py with the recommended LR")
        print("3. Run: python train.py")
        print("="*70)
        
        # Optional: Run LR range test
        run_lr_test = input("\nRun LR range test now? (y/n): ").lower()
        if run_lr_test == 'y':
            loaders, _ = get_dataloaders(CONFIG)
            model = BloodCellCNN(
                num_classes=CONFIG["num_classes"],
                base_filters=CONFIG["base_filters"],
                dropout_rate=0.0,
            ).to(CONFIG["device"])
            lr_range_test(model, loaders['train'], CONFIG['device'])