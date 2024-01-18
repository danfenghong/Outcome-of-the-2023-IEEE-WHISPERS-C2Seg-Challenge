import os
import numpy as np
from torchvision import datasets, transforms
import tifffile as tiff

root = 'data/c2seg_ab/img_dir/trainval/sar'

mean_accumulator = np.zeros(2)
std_accumulator = np.zeros(2)

list_dir = os.listdir(root)
for file in list_dir:
    img_name = os.path.join(root, file)
    img = tiff.imread(img_name)
    mean_accumulator += np.mean(img, axis=(0, 1))
    std_accumulator += np.std(img, axis=(0, 1))

mean_values = mean_accumulator / len(list_dir)
std_values = std_accumulator / len(list_dir)

print("Mean values:", [mean_values])
print("Std values:", [std_values])
