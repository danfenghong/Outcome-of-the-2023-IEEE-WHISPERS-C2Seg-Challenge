import random
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from PIL import Image
import math
from matplotlib import pyplot as plt
from augment import OnlineImageAugmentor
from torchvision import transforms as T

import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError, RasterioIOError


class RoadDataset(IterableDataset):
    def __init__(self, file_path, aug_cfg_filepath=None, use_aug=True):
        temp = pd.read_csv(file_path, header=0)
        self.length = temp.shape[0]
        del temp
        if aug_cfg_filepath is not None:
            self.use_aug = use_aug
            self.augmentor = OnlineImageAugmentor(aug_cfg_filepath)
            if self.augmentor.use_mosaic and self.use_aug:
                self.data_iter = pd.read_csv(file_path, iterator=True, header=0, chunksize=self.augmentor.mosaic_num)
            else:
                self.data_iter = pd.read_csv(file_path, iterator=True, header=0, chunksize=1)
        else:
            self.use_aug = False
            self.data_iter = pd.read_csv(file_path, iterator=True, header=0, chunksize=1)
        if isinstance(self.augmentor.crop_size, int):
            self.chip_size = self.augmentor.crop_size
        else:
            self.chip_size = self.augmentor.crop_size[0]
    
    # def stream_data_fns(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
    #         worker_id = 0
    #         num_workers = 1
    #     else:
    #         worker_id = worker_info.id
    #         num_workers = worker_info.num_workers
        
    #     for data in self.data_iter:
    #         num_files_per_worker = int(math.ceil(N / num_workers))
    
    def open_image_and_mask(self, img_path, mask_path):
        print(img_path)
        print(mask_path)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return img, mask

    def __iter__(self):
        for data in self.data_iter:
            data_list = data.values.tolist()
            data_len = len(data_list)
            if self.use_aug:
                if self.augmentor.use_mosaic:
                    images = []
                    masks = []
                    for i in range(data_len):
                        img_path = data_list[i][0]
                        mask_path = data_list[i][1]
                        img, mask = self.open_image_and_mask(img_path, mask_path)
                        images.append(img)
                        masks.append(mask)
                    if self.augmentor.compose_before_mosaic:
                        for i in range(len(images)):
                            images[i], masks[i] = self.augmentor.compose_aug(images[i], masks[i])
                        m_image, m_mask = self.augmentor.mosaic(images, masks, self.chip_size // 2, self.augmentor.mosaic_num)
                    else:
                        m_image, m_mask = self.augmentor.mosaic(images, masks, self.chip_size // 2, self.augmentor.mosaic_num)
                        m_image, m_mask = self.augmentor.compose_aug(m_image, m_mask)
                else:
                    img_path = data_list[0][0]
                    mask_path = data_list[0][1]
                    img, mask = self.open_image_and_mask(img_path, mask_path)
                    m_image, m_mask = self.augmentor.compose_aug(img, mask)
            else:
                img_path = data_list[0][0]
                mask_path = data_list[0][1]
                m_image, m_mask = self.open_image_and_mask(img_path, mask_path)
            Transform = T.ToTensor()
            m_image = Transform(m_image).squeeze()
            m_mask = Transform(m_mask).squeeze()

            yield m_image, m_mask

    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = dataset.length
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    start = overall_start + worker_id * per_worker
    end = min(start + per_worker, overall_end)


