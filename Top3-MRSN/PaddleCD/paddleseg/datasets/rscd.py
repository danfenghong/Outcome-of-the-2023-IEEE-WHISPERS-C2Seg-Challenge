# -*- coding: utf-8 -*- 
# @File             : rscd.py 
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/9/11 9:28 
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/9/11 9:28 
'''
@File Description:

'''
from __future__ import print_function
from __future__ import division
import os

import paddle
import numpy as np
from PIL import Image
from skimage import io
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F
from paddle.io import Dataset

@manager.DATASETS.add_component
class RSCD(paddle.io.Dataset):
    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes=2,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):

        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError(
                    '`train_path` is not found: {}'.format(train_path))
            else:
                file_path = train_path
        elif mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 3:
                    if mode == 'train' or mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = os.path.join(self.dataset_root, items[1])
                    label_path = None
                else:
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = os.path.join(self.dataset_root, items[1])
                    label_path = os.path.join(self.dataset_root, items[2])
                self.file_list.append([image1_path, image2_path, label_path])

    def __getitem__(self, idx):
        image1_path, image2_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im1, im2, _ = self.transforms(im1_path=image1_path, im2_path=image2_path)
            im1 = im1[np.newaxis, ...]
            im2 = im2[np.newaxis, ...]
            return im1, im2, image1_path, image2_path
        elif self.mode == 'val':
            im1, im2, _ = self.transforms(im1_path=image1_path, im2_path=image2_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            return im1, im2, label
        else:
            im1, im2, label = self.transforms(im1_path=image1_path, im2_path=image2_path, label_path=label_path)

            
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im1, im2, label, edge_mask
            else:
                return im1, im2, label

    def __len__(self):
        return len(self.file_list)

@manager.DATASETS.add_component
class RS_MD2B(paddle.io.Dataset):
    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes=2,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):

        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError(
                    '`train_path` is not found: {}'.format(train_path))
            else:
                file_path = train_path
        elif mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 3:
                    if mode == 'train' or mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = os.path.join(self.dataset_root, items[1])
                    label_path = None
                else:
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = os.path.join(self.dataset_root, items[1])
                    label_path = os.path.join(self.dataset_root, items[2])
                self.file_list.append([image1_path, image2_path, label_path])

    def __getitem__(self, idx):
        image1_path, image2_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im1, im2, _ = self.transforms(im1_path=image1_path, im2_path=image2_path)
            im1 = im1[np.newaxis, ...]
            im2 = im2[np.newaxis, ...]
            im2 = paddle.concat([im1[:, 3:, ...], im2], axis=1)
            im1 = im1[:, :3]
            return im1, im2, image1_path, image2_path
        elif self.mode == 'val':
            im1, im2, label = self.transforms(im1_path=image1_path, im2_path=image2_path, label_path=label_path)
            im2 = np.concatenate([im1[3:, ...], im2], axis=0)
            im1 = im1[:3]
            return im1, im2, label
        else:
            im1, im2, label = self.transforms(im1_path=image1_path, im2_path=image2_path, label_path=label_path)
            im2 = np.concatenate([im1[3:, ...], im2], axis=0)
            im1 = im1[:3]
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im1, im2, label, edge_mask
            else:
                return im1, im2, label

    def __len__(self):
        return len(self.file_list)

@manager.DATASETS.add_component
class RS_MD3B(paddle.io.Dataset):
    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes=2,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):

        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError(
                    '`train_path` is not found: {}'.format(train_path))
            else:
                file_path = train_path
        elif mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 3:
                    if mode == 'train' or mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image1_path = os.path.join(self.dataset_root, items[0].replace('msi','msisar'))
                    image2_path = os.path.join(self.dataset_root, items[1].replace('sar','hsi'))
                    label_path = None
                else:
                    image1_path = os.path.join(self.dataset_root, items[0].replace('msi','msisar'))
                    image2_path = os.path.join(self.dataset_root, items[1].replace('sar','hsi'))
                    label_path = os.path.join(self.dataset_root, items[2])
                self.file_list.append([image1_path, image2_path, label_path])

    def __getitem__(self, idx):
        image1_path, image2_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im1, im2, _ = self.transforms(im1_path=image1_path, im2_path=image2_path)
            im1 = im1[np.newaxis, ...]
            im2 = im2[np.newaxis, ...]
            return im1, im2, image1_path, image2_path
        elif self.mode == 'val':
            im1, im2, label = self.transforms(im1_path=image1_path, im2_path=image2_path, label_path=label_path)
            return im1, im2, label
        else:
            im1, im2, label = self.transforms(im1_path=image1_path, im2_path=image2_path, label_path=label_path)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im1, im2, label, edge_mask
            else:
                return im1, im2, label

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    pass
