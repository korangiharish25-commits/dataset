"""
Script to predict segmentation mask for a single image.
"""
import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from model import UNet
from utils import plot_preds

def main():
    parser = argparse.ArgumentParser(description='Predict segmentation mask for a single image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization (default: same as image with _pred suffix)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask (default: 0.5)')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        img_path = Path(args.image)
        args.output = str(img_path.parent / f"{img_path.stem}_pred{img_path.suffix}")
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape[:2]
    
    # Preprocess image
    img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized.astype('float32') / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW, add batch dim
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, base_c=32).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Predict
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        output = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Create binary mask
    pred_mask = (output > args.threshold).astype('uint8') * 255
    
    # Resize prediction back to original size if needed
    if original_shape != (args.img_size, args.img_size):
        pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Save visualization
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create visualization with original image and prediction
    fig_path = args.output.replace('.png', '_vis.png').replace('.jpg', '_vis.jpg')
    if not fig_path.endswith('_vis.png') and not fig_path.endswith('_vis.jpg'):
        fig_path = args.output + '_vis.png'
    
    plot_preds(img_rgb, None, pred_mask, fig_path)
    
    # Also save just the mask
    cv2.imwrite(args.output, pred_mask)
    
    print(f"Prediction saved to: {args.output}")
    print(f"Visualization saved to: {fig_path}")

if __name__ == '__main__':
    main()

