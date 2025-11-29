import argparse, os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset, get_transforms
from model import UNet
from utils import load_checkpoint, plot_preds

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_csv', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results/preds')
    p.add_argument('--img_size', type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.data_csv)
    ds = SegmentationDataset(df, img_size=(args.img_size,args.img_size), transforms=get_transforms(False,(args.img_size,args.img_size)))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, base_c=32).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(loader)):
            img_cuda = img.to(device).float()
            out = model(img_cuda)
            out = torch.sigmoid(out).cpu().numpy()[0,0]
            pred_mask = (out > 0.5).astype('uint8')*255
            # original image from dataset (resized)
            img_np = (img.numpy()[0].transpose(1,2,0)*255).astype('uint8')
            mask_np = (mask.numpy()[0,0]*255).astype('uint8')
            save_path = os.path.join(args.out_dir, f'pred_{i:03d}.png')
            plot_preds(img_np, mask_np, pred_mask, save_path)

if __name__ == '__main__':
    main()
