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
    def __init__(self, file_path, aug_cfg_filepath=None, use_aug=True, mode='train'):
        self.data_fns = pd.read_csv(file_path, header=0)
        self.data_list = self.data_fns.values.tolist()
        self.length = self.data_fns.shape[0]
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "Invalid indicator for 'mode', which should be selected from {'train', 'val', 'test'}"
        if aug_cfg_filepath is not None:
            self.use_aug = use_aug
            self.augmentor = OnlineImageAugmentor(aug_cfg_filepath)
            self.chip_size = [256,256] #[128, 128] ###
        else:
            self.use_aug = False

    def __len__(self):
        return self.length
    
    def __getitem__(self, index, backend='rasterio'):
        if self.mode == 'train':
            if self.use_aug:
                msi_path = self.data_list[index][0]
                sar_path = self.data_list[index][1]
                hsi_path = self.data_list[index][2]
                mask_path = self.data_list[index][3]
                m_image, m_mask = self.open_image_and_mask(msi_path, sar_path, hsi_path, mask_path)
                m_image, m_mask = self.augmentor.compose_aug(m_image, m_mask) ### 待修改
            else:
                msi_path = self.data_list[index][0]
                sar_path = self.data_list[index][1]
                hsi_path = self.data_list[index][2]
                mask_path = self.data_list[index][3]
                m_image, m_mask = self.open_image_and_mask(msi_path, sar_path, hsi_path, mask_path)
        else:
            msi_path = self.data_list[index][0]
            sar_path = self.data_list[index][1]
            hsi_path = self.data_list[index][2]
            mask_path = self.data_list[index][3]
            m_image, m_mask = self.open_image_and_mask(msi_path, sar_path, hsi_path, mask_path)
        
        # Image_Transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #     # T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
        # ])
        # Mask_Transform = T.ToTensor()
        # m_image = Image_Transform(m_image).squeeze()
        # m_mask = Mask_Transform(m_mask).squeeze()*255

        m_image = torch.from_numpy(m_image).float()
        m_mask = torch.from_numpy(m_mask).long()

        if self.mode == 'test':
            return m_image, m_mask, msi_path
        else:
            return m_image, m_mask

    def open_image_and_mask(self, msi_path, sar_path, hsi_path, mask_path):
        msi_fp = rasterio.open(msi_path, "r").read().astype(np.float32)
        # sar_fp = rasterio.open(sar_path, "r").read().astype(np.float32)
        # hsi_fp = rasterio.open(hsi_path, "r").read().astype(np.float32)
        mask_fp = rasterio.open(mask_path, "r").read().astype(np.int32)
        # print(msi_fp.shape,mask_fp.shape,"msi_fp.read().shape,mask_fp.read().shape")
        ##(4, 128, 128) (1, 128, 128) msi_fp.read().shape,mask_fp.read().shape 

        img = msi_fp
        mask = mask_fp[0]
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
