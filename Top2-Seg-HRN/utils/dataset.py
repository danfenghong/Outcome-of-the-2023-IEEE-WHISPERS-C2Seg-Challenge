import random
import torch
from torch.utils import data
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import math
import numpy as np

from .augment import OnlineImageAugmentor
from torchvision import transforms as T
import yaml
import os

import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError, RasterioIOError

import warnings

warnings.filterwarnings('ignore')


class RoadDataset(Dataset):
    def __init__(self, file_path, aug_cfg_filepath=None, stat_filepath=None, use_aug=True, mode='train'):

        self.data_fns = pd.read_csv(file_path, header=0)
        self.data_list = self.data_fns.values.tolist()
        self.length = self.data_fns.shape[0]
        # print(self.data_list)
        # print(self.length)
        self.mode = mode
        self.stat_filepath = stat_filepath
        assert self.mode in ['train', 'val', 'test'], "Invalid indicator for 'mode', which should be selected from {'train', 'val', 'test'}"
        if aug_cfg_filepath is not None:
            self.use_aug = use_aug
            self.augmentor = OnlineImageAugmentor(aug_cfg_filepath)
            if isinstance(self.augmentor.crop_size, int):
                self.chip_size = [self.augmentor.crop_size, self.augmentor.crop_size]
            elif isinstance(self.augmentor.crop_size, list):
                self.chip_size = self.augmentor.crop_size
            elif self.augmentor.crop_size is None:
                if str.find(file_path, 'massachusetts'):
                    self.chip_size = [1500, 1500]
                elif str.find(file_path, 'deepglobe'):
                    self.chip_size = [1024, 1024]
                elif str.find(file_path, 'Nantes'):
                    # print("Nantes")
                    self.chip_size = [1024, 1024]
                elif str.find(file_path, 'roadtracer-dataset'):
                    self.chip_size = [4096, 4096]
                elif str.find(file_path, 'spacenet'):
                    self.chip_size = [1300, 1300]
                elif str.find(file_path, 'EPFML17'):
                    self.chip_size = [400, 400]
                else:
                    raise ValueError("Unregistered dataset!")
            else:
                raise TypeError("Invalid type of chip_size, which should be int or list or 'None'!")
        else:
            self.use_aug = False
    
    def parse_stat(self, cfg_filename):
        cfg_file = open(cfg_filename, 'r')
        stat = yaml.load(cfg_file, Loader=yaml.FullLoader)
        R_Aver = stat['INFO']['IMAGES']['R_AVER']
        G_Aver = stat['INFO']['IMAGES']['G_AVER']
        B_Aver = stat['INFO']['IMAGES']['B_AVER']
        R_Stdv = stat['INFO']['IMAGES']['R_STDV']
        G_Stdv = stat['INFO']['IMAGES']['G_STDV']
        B_Stdv = stat['INFO']['IMAGES']['B_STDV']
        return R_Aver, G_Aver, B_Aver, R_Stdv, G_Stdv, B_Stdv
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index, backend='rasterio'):
        if self.mode == 'train':
            if self.use_aug:
                if self.augmentor.use_mosaic:
                    random.seed(self.augmentor.seed + index)
                    mosaic_idxs = []
                    images = []
                    masks = []
                    for _ in range(self.augmentor.mosaic_num - 1):
                        x = random.randint(0, self.length - 1)
                        if x != index:
                            mosaic_idxs.append(x)
                    mosaic_idxs.append(index)
                    for i in range(self.augmentor.mosaic_num):
                        img_path = self.data_list[i][0]
                        mask_path = self.data_list[i][1]
                        img, mask = self.open_image_and_mask(img_path, mask_path, backend=backend)
                        images.append(img)
                        masks.append(mask)
                    if self.augmentor.compose_before_mosaic:
                        for i in range(len(images)):
                            images[i], masks[i] = self.augmentor.compose_aug(images[i], masks[i])
                        m_image, m_mask = self.augmentor.mosaic(images, masks, self.chip_size[0] // 2, self.augmentor.mosaic_num)
                    else:
                        m_image, m_mask = self.augmentor.mosaic(images, masks, self.chip_size[0] // 2, self.augmentor.mosaic_num)
                        m_image, m_mask = self.augmentor.compose_aug(m_image, m_mask)
                else:
                    img_path = self.data_list[index][0]
                    mask_path = self.data_list[index][1]
                    img, mask = self.open_image_and_mask(img_path, mask_path, backend=backend)
                    m_image, m_mask = self.augmentor.compose_aug(img, mask)
            else:
                img_path = self.data_list[index][0]
                mask_path = self.data_list[index][1]
                m_image, m_mask = self.open_image_and_mask(img_path, mask_path, backend=backend)
                # print(index)
                # print("here, "*10)
        else:
            img_path = self.data_list[index][0]
            mask_path = self.data_list[index][1]
            m_image, m_mask = self.open_image_and_mask(img_path, mask_path, backend=backend, window_sample=False)
        
        # if self.stat_filepath is not None:
        #     R_Aver, G_Aver, B_Aver, R_Stdv, G_Stdv, B_Stdv = self.parse_stat(self.stat_filepath)
        #     Image_Transform = T.Compose([
        #         T.ToTensor(),
        #         T.Normalize([R_Aver / 255.0, G_Aver / 255.0, B_Aver / 255.0], [R_Stdv / 255.0, G_Stdv / 255.0, B_Stdv / 255.0])
        #     ])
        # else:
        #     Image_Transform = T.Compose([
        #         T.ToTensor(),
        #         T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #     ])
        Image_Transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
        ])
        Mask_Transform = T.ToTensor()
        # print(np.asarray(m_mask),11111)
        # print(m_image.dtype,  22222) #[1024, 1024])
        # print(np.min(m_image),np.max(m_image))
        m_image = Image_Transform(m_image).squeeze()
        m_mask = Mask_Transform(m_mask).squeeze()*255

        if self.mode == 'test':
            return m_image, m_mask, img_path
        else:
            return m_image, m_mask

    def is_data_valid(self, image, mask, fg_rate=0.00001, pad_value=0, pad_rate=0.1):
        image_array = np.array(image, dtype=np.int)
        mask_array = np.array(mask, dtype=np.int)
        h, w = mask_array.shape
        num_fg = len(np.where(mask_array != 0)[0])
        q = image_array[:, :, 0] + image_array[:, :, 1] + image_array[:, :, 2]
        num_zero = len(np.where(q == pad_value)[0])
        r_fg = num_fg / float(h * w)
        r_zero = num_zero / float(h * w)
        if r_fg < fg_rate:
            return False
        # if r_zero > pad_rate:
        #     return False
        return True
    
    def open_image_and_mask(self, img_path, mask_path, backend='rasterio', window_sample=True):
        if backend == 'pillow':
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        elif backend == 'rasterio':
            img_fp = rasterio.open(img_path, "r")
            mask_fp = rasterio.open(mask_path, "r")
            # print(img_fp.read().shape,"img") #(4, 2000, 1999)
            # print(np.max(np.unique(mask_fp.read()[0])),"mask")
            # print(img_fp.read()[0].shape,"img  ",mask_fp.read()[0].shape)
            # if (img_fp.read()[0].shape[0] != mask_fp.read()[0].shape[0]) or (img_fp.read()[0].shape[1] != mask_fp.read()[0].shape[1]):
            #     print("img shape ! = gt shape")
            #     print(img_fp.read()[0].shape,"img")
            #     print(mask_path)
            # if (len(np.unique(mask_fp.read()[0])) == 1):
            #     print(mask_path,111111111)    
            if (len(np.unique(mask_fp.read()[0])) == 1) and (0 in np.unique(mask_fp.read()[0])):
                print(mask_path,222222222)
            # print(mask_path)
            height, width = img_fp.shape
            # random.seed()
            if not window_sample:
                img = np.rollaxis(img_fp.read(), 0, 3)
                mask = mask_fp.read()[0]
            else: 
                x = random.randint(0, width-self.chip_size[0])
                y = random.randint(0, height-self.chip_size[1])
                img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size[0], self.chip_size[1])), 0, 3)
                mask = mask_fp.read(window=Window(x, y, self.chip_size[0], self.chip_size[1]))[0]
                while not self.is_data_valid(img, mask):
                    x = random.randint(0, width-self.chip_size[0])
                    y = random.randint(0, height-self.chip_size[1])
                    img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size[0], self.chip_size[1])), 0, 3)
                    mask = mask_fp.read(window=Window(x, y, self.chip_size[0], self.chip_size[1]))[0]
    
            img = Image.fromarray(img).convert('RGB')
            mask = Image.fromarray(mask).convert('L')
        else:
            raise ValueError("Invalid indicator for backend, which should be 'pillow' or 'rasterio'.")
        return img, mask

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

