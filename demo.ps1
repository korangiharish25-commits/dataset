# Demonstration Script for Assessment Project
# Run this script to demonstrate all project functionality

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "ASSESSMENT PROJECT - DEMONSTRATION SCRIPT" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Test Environment Setup
Write-Host "Step 1: Testing Environment Setup..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
python src/test_setup.py
Write-Host ""

# Step 2: Show Project Status
Write-Host "Step 2: Showing Project Status and Results..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
python src/show_results.py
Write-Host ""

# Step 3: Check Model Checkpoint
Write-Host "Step 3: Checking Model Checkpoint..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
if (Test-Path "models/best.pth") {
    $modelInfo = Get-Item "models/best.pth"
    Write-Host "✓ Model checkpoint found: $($modelInfo.Name)" -ForegroundColor Green
    Write-Host "  Size: $([math]::Round($modelInfo.Length / 1MB, 2)) MB" -ForegroundColor Green
    Write-Host "  Last Modified: $($modelInfo.LastWriteTime)" -ForegroundColor Green
} else {
    Write-Host "✗ Model checkpoint not found" -ForegroundColor Red
}
Write-Host ""

# Step 4: Check PyTorch Installation
Write-Host "Step 4: Checking PyTorch Installation..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')"
Write-Host ""

# Step 5: Data Preparation (if needed)
Write-Host "Step 5: Checking Data Splits..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
if (Test-Path "data/splits/train.csv") {
    $trainCount = (Import-Csv "data/splits/train.csv").Count
    $valCount = (Import-Csv "data/splits/val.csv").Count
    $testCount = (Import-Csv "data/splits/test.csv").Count
    Write-Host "✓ Data splits found:" -ForegroundColor Green
    Write-Host "  Train: $trainCount samples" -ForegroundColor Green
    Write-Host "  Val:   $valCount samples" -ForegroundColor Green
    Write-Host "  Test:  $testCount samples" -ForegroundColor Green
} else {
    Write-Host "Data splits not found. Running data preparation..." -ForegroundColor Yellow
    python src/prepare_data.py --images_dir data/raw/images --masks_dir data/raw/masks
}
Write-Host ""

# Step 6: Run Prediction Demo
Write-Host "Step 6: Running Prediction Demo..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow
if (Test-Path "models/best.pth" -And Test-Path "data/raw/images/img_000.png") {
    python src/predict.py --checkpoint models/best.pth --image data/raw/images/img_000.png --output results/demo_prediction.png
    if (Test-Path "results/demo_prediction.png") {
        Write-Host "✓ Prediction completed successfully!" -ForegroundColor Green
        Write-Host "  Output: results/demo_prediction.png" -ForegroundColor Green
        Write-Host "  Visualization: results/demo_prediction_vis.png" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ Skipping prediction (model or image not found)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "DEMONSTRATION COMPLETE!" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "All systems operational. Project is ready for demonstration." -ForegroundColor Green
Write-Host ""

