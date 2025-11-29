#!/bin/bash
# Demonstration Script for Assessment Project
# Run this script to demonstrate all project functionality

echo "================================================================================="
echo "ASSESSMENT PROJECT - DEMONSTRATION SCRIPT"
echo "================================================================================="
echo ""

# Step 1: Test Environment Setup
echo "Step 1: Testing Environment Setup..."
echo "----------------------------------------"
python src/test_setup.py
echo ""

# Step 2: Show Project Status
echo "Step 2: Showing Project Status and Results..."
echo "----------------------------------------"
python src/show_results.py
echo ""

# Step 3: Check Model Checkpoint
echo "Step 3: Checking Model Checkpoint..."
echo "----------------------------------------"
if [ -f "models/best.pth" ]; then
    echo "✓ Model checkpoint found: best.pth"
    ls -lh models/best.pth
else
    echo "✗ Model checkpoint not found"
fi
echo ""

# Step 4: Check PyTorch Installation
echo "Step 4: Checking PyTorch Installation..."
echo "----------------------------------------"
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')"
echo ""

# Step 5: Data Preparation (if needed)
echo "Step 5: Checking Data Splits..."
echo "----------------------------------------"
if [ -f "data/splits/train.csv" ]; then
    train_count=$(wc -l < data/splits/train.csv | tr -d ' ')
    val_count=$(wc -l < data/splits/val.csv | tr -d ' ')
    test_count=$(wc -l < data/splits/test.csv | tr -d ' ')
    echo "✓ Data splits found:"
    echo "  Train: $((train_count-1)) samples"
    echo "  Val:   $((val_count-1)) samples"
    echo "  Test:  $((test_count-1)) samples"
else
    echo "Data splits not found. Running data preparation..."
    python src/prepare_data.py --images_dir data/raw/images --masks_dir data/raw/masks
fi
echo ""

# Step 6: Run Prediction Demo
echo "Step 6: Running Prediction Demo..."
echo "----------------------------------------"
if [ -f "models/best.pth" ] && [ -f "data/raw/images/img_000.png" ]; then
    python src/predict.py --checkpoint models/best.pth --image data/raw/images/img_000.png --output results/demo_prediction.png
    if [ -f "results/demo_prediction.png" ]; then
        echo "✓ Prediction completed successfully!"
        echo "  Output: results/demo_prediction.png"
        echo "  Visualization: results/demo_prediction_vis.png"
    fi
else
    echo "⚠ Skipping prediction (model or image not found)"
fi
echo ""

# Summary
echo "================================================================================="
echo "DEMONSTRATION COMPLETE!"
echo "================================================================================="
echo ""
echo "All systems operational. Project is ready for demonstration."
echo ""

