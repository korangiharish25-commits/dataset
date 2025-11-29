"""
Script to prepare data splits from raw images and masks.
Creates CSV files for train/val/test splits.
"""
import argparse
import os
import pandas as pd
from pathlib import Path

def find_matching_pairs(images_dir, masks_dir):
    """Find matching image-mask pairs."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    pairs = []
    for img_path in image_files:
        # Try to find matching mask
        # Common patterns: img_001.jpg -> img_001_mask.png or img_001.png
        img_stem = img_path.stem
        mask_candidates = [
            masks_dir / f"{img_stem}_mask.png",
            masks_dir / f"{img_stem}_mask.jpg",
            masks_dir / f"{img_stem}.png",
            masks_dir / f"{img_stem}.jpg",
            masks_dir / img_path.name,  # Same filename
        ]
        
        mask_path = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path:
            pairs.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path)
            })
        else:
            print(f"Warning: No mask found for {img_path.name}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description='Prepare data splits from raw images and masks')
    parser.add_argument('--images_dir', type=str, default='data/raw/images',
                        help='Directory containing images')
    parser.add_argument('--masks_dir', type=str, default='data/raw/masks',
                        help='Directory containing masks')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Output directory for CSV files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation data (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of test data (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # Find matching pairs
    print(f"Scanning images in {args.images_dir}...")
    print(f"Scanning masks in {args.masks_dir}...")
    pairs = find_matching_pairs(args.images_dir, args.masks_dir)
    
    if len(pairs) == 0:
        print("Error: No matching image-mask pairs found!")
        return
    
    print(f"Found {len(pairs)} matching pairs")
    
    # Create DataFrame
    df = pd.DataFrame(pairs)
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val  # Remaining goes to test
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test_df = df.iloc[n_train+n_val:].reset_index(drop=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save CSV files
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSplits created:")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Val:   {len(val_df)} samples -> {val_path}")
    print(f"  Test:  {len(test_df)} samples -> {test_path}")

if __name__ == '__main__':
    main()

