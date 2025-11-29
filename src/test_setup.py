import torch, numpy as np, cv2, os
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
# write test image
os.makedirs('results', exist_ok=True)
img = (np.random.rand(256,256,3)*255).astype('uint8')
cv2.imwrite("results/test_image.png", img)
print("Wrote results/test_image.png")
print("Setup looks good.")
