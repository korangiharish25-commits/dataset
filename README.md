# Assessment Project - Segmentation (UNet) - Ready to run

This project contains a beginner-friendly, ready-to-run PyTorch U-Net implementation for image segmentation.
It is designed to run locally on your machine (CPU or GPU).

## Folder structure
```
assessment_project/
  src/
    dataset.py
    model.py
    losses.py
    train.py
    evaluate.py
    predict.py
    prepare_data.py
    utils.py
    test_setup.py
  data/
    raw/
      images/
      masks/
    splits/
  models/
  results/
  notebooks/
  docs/
  requirements.txt
```

## Quick setup (Windows / Linux / macOS)
1. Open VS Code and open this folder (`assessment_project`).
2. Create and activate a Python virtual environment:
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Install requirements:
```
pip install -r requirements.txt
```

## How to run (examples)
1. Test environment:
```
python src/test_setup.py
```

2. Prepare data splits (if you have raw images and masks):
```
python src/prepare_data.py --images_dir data/raw/images --masks_dir data/raw/masks
```
This will automatically create train/val/test CSV files in `data/splits/`.

3. Train:
```
python src/train.py --data_csv data/splits/train.csv --epochs 5 --batch_size 4 --lr 1e-3
```

4. Evaluate on test set:
```
python src/evaluate.py --checkpoint models/best.pth --data_csv data/splits/test.csv --out_dir results/preds
```

5. Predict on a single image:
```
python src/predict.py --checkpoint models/best.pth --image path/to/image.png --output results/prediction.png
```

## Notes
- This implementation uses a self-contained U-Net (no heavy external segmentation libraries).
- If you want to use pretrained encoders (EfficientNet), install `timm` and modify `model.py`.
- Put your raw images and masks under `data/raw/images/` and `data/raw/masks/` or follow the CSV format.

If anything breaks, paste the full error here and I'll help fix it.
