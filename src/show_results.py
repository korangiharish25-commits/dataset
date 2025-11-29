"""Script to display training and evaluation results."""
import pandas as pd
import torch
import os

print("\n" + "="*60)
print("ASSESSMENT PROJECT - RESULTS SUMMARY")
print("="*60)

# Dataset statistics
print("\n📊 Dataset Statistics:")
print("-" * 60)
try:
    train = pd.read_csv('data/splits/train.csv')
    val = pd.read_csv('data/splits/val.csv')
    test = pd.read_csv('data/splits/test.csv')
    print(f"  Train samples: {len(train)}")
    print(f"  Val samples:   {len(val)}")
    print(f"  Test samples:  {len(test)}")
    print(f"  Total samples: {len(train) + len(val) + len(test)}")
except Exception as e:
    print(f"  Error loading CSV files: {e}")

# Model checkpoint info
print("\n🎯 Model Checkpoint:")
print("-" * 60)
try:
    if os.path.exists('models/best.pth'):
        ckpt = torch.load('models/best.pth', map_location='cpu', weights_only=False)
        print(f"  Status: ✓ Model checkpoint found")
        print(f"  Best validation Dice: {ckpt.get('best_dice', 'N/A'):.4f}")
        print(f"  Trained for: {ckpt.get('epoch', 'N/A')} epochs")
    else:
        print("  Status: ✗ No checkpoint found (run training first)")
except Exception as e:
    print(f"  Error loading checkpoint: {e}")

# Results directory
print("\n📁 Generated Results:")
print("-" * 60)
if os.path.exists('results/preds'):
    pred_files = [f for f in os.listdir('results/preds') if f.endswith('.png')]
    print(f"  Prediction images: {len(pred_files)} files in results/preds/")
    if pred_files:
        print(f"  Sample files: {', '.join(pred_files[:3])}")
else:
    print("  No predictions found (run evaluation first)")

print("\n" + "="*60)
print("✅ All systems operational!")
print("="*60 + "\n")

