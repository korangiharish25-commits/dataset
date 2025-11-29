import argparse, os, time
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import SegmentationDataset, get_transforms
from model import UNet
from losses import dice_loss, FocalTverskyLoss, dice_coeff
from utils import save_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_csv', type=str, required=True, help='CSV with image_path,mask_path')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--save_dir', type=str, default='models')
    return p.parse_args()

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device).float()
        masks = masks.to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()
            outputs = model(imgs)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()
            batch_dice = []
            for p, t in zip(preds, masks):
                batch_dice.append(dice_coeff(p, t).item())
            dices.extend(batch_dice)
    return np.mean(dices)

def main():
    args = parse_args()
    df = pd.read_csv(args.data_csv)
    # simple split if CSV contains all and 'split' column exists
    if 'split' in df.columns:
        train_df = df[df['split']=='train'].reset_index(drop=True)
        val_df = df[df['split']=='val'].reset_index(drop=True)
    else:
        # assume provided CSV is for training only, we will split 80/20
        train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df = df.drop(train_df.index).reset_index(drop=True)

    train_ds = SegmentationDataset(train_df, img_size=(args.img_size,args.img_size), transforms=get_transforms(True, (args.img_size,args.img_size)))
    val_ds = SegmentationDataset(val_df, img_size=(args.img_size,args.img_size), transforms=get_transforms(False, (args.img_size,args.img_size)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, base_c=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # composite loss: Dice + FocalTversky
    ft_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    def composite_loss(preds, targets):
        return dice_loss(preds, targets) + 0.5 * ft_loss(preds, targets)

    best_dice = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    print('Starting training on device:', device)
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, composite_loss)
        val_dice = eval_epoch(model, val_loader, device)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} val_dice: {val_dice:.4f} time: {t1-t0:.1f}s")
        # save best
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'epoch': epoch, 'best_dice': best_dice},
                            os.path.join(args.save_dir, 'best.pth'))
    print('Training finished. Best val dice:', best_dice)

if __name__ == '__main__':
    main()
