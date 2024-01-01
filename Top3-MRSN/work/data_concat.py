import os
import numpy as np
from skimage import io
from tqdm import tqdm

def img_concat(img1_dir, img2_dir, img3_dir):
    if not os.path.exists(img3_dir):
        os.makedirs(img3_dir)
    imgs_list = os.listdir(img1_dir)
    # imgs1_list = [os.path.join(img1_dir, file_name) for file_name in imgs_list]
    # imgs2_list = [os.path.join(img2_dir, file_name) for file_name in imgs_list]
    # imgs3_list = [os.path.join(img3_dir, file_name) for file_name in imgs_list]

    for file_name in tqdm(imgs_list):
        img1_path = os.path.join(img1_dir, file_name)
        img2_path = os.path.join(img2_dir, file_name)
        img3_path = os.path.join(img3_dir, file_name)
        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)
        img3 = np.concatenate([img1, img2], axis=2)
        io.imsave(img3_path, img3)

if __name__ == '__main__':
    pass
    img1_dir = r'/home/aistudio/data/src/C2Seg_AB/train/msi'
    img2_dir = r'/home/aistudio/data/src/C2Seg_AB/train/sar'
    img3_dir = r'/home/aistudio/data/src/C2Seg_AB/train/msisar'
    img_concat(img1_dir, img2_dir, img3_dir)

    img1_dir = r'/home/aistudio/data/src/C2Seg_AB/test/msi'
    img2_dir = r'/home/aistudio/data/src/C2Seg_AB/test/sar'
    img3_dir = r'/home/aistudio/data/src/C2Seg_AB/test/msisar'
    img_concat(img1_dir, img2_dir, img3_dir)


    img1_dir = r'/home/aistudio/data/src/C2Seg_BW/train/msi'
    img2_dir = r'/home/aistudio/data/src/C2Seg_BW/train/sar'
    img3_dir = r'/home/aistudio/data/src/C2Seg_BW/train/msisar'
    img_concat(img1_dir, img2_dir, img3_dir)

    img1_dir = r'/home/aistudio/data/src/C2Seg_BW/test/msi'
    img2_dir = r'/home/aistudio/data/src/C2Seg_BW/test/sar'
    img3_dir = r'/home/aistudio/data/src/C2Seg_BW/test/msisar'
    img_concat(img1_dir, img2_dir, img3_dir)