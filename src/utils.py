import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_checkpoint(state, filename='models/checkpoint.pth'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state'])
    return checkpoint

def plot_preds(img, mask, pred, out_path):
    # img: HxWx3 RGB, mask/pred: HxW (0/1 or 0/255)
    # mask can be None for single image prediction
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if mask is not None:
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(img); axes[0].set_title('Image'); axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray'); axes[1].set_title('Mask'); axes[1].axis('off')
        axes[2].imshow(pred, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1,2, figsize=(10,4))
        axes[0].imshow(img); axes[0].set_title('Image'); axes[0].axis('off')
        axes[1].imshow(pred, cmap='gray'); axes[1].set_title('Prediction'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
