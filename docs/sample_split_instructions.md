# How to prepare CSV splits (train/val/test)

Create a folder structure:
```
data/raw/images/   # original RGB images (png/jpg)
data/raw/masks/    # binary masks (png) same filenames as images
```

Then create CSV files with two columns: `image_path,mask_path`
Example row:
```
data/raw/images/img_001.jpg,data/raw/masks/img_001_mask.png
```

Save them as:
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv

You can generate splits using a small Python script or manually.
