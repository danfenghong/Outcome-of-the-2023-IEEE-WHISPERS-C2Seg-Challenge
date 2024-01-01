# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def read_imgs(img_list, img1_dir, img2_dir, ids, transforms):
    imgs1 = []
    imgs2 = []
    imgs_name = []
    for index in ids:
        if index < len(img_list):
            im1_path = img_list[index]
            im2_path = im1_path.replace(img1_dir, img2_dir)

            # im1 = functional.im_read(im1_path)
            # im2 = functional.im_read(im2_path)

            im1 = im1_path
            im2 = im2_path

            im1, im2, _ = transforms(im1, im2)
            imgs1.append(im1)
            imgs2.append(im2)

            imgs_name.append(os.path.basename(img_list[index]))

    imgs1 = paddle.to_tensor(imgs1)
    imgs2 = paddle.to_tensor(imgs2)
    return imgs1, imgs2, imgs_name

def save_imgs(preds, save_dir, imgs_name):
    preds = preds.numpy().astype('uint8')
    for index, pred in enumerate(preds):
        pred = np.squeeze(pred)*1
        pred_saved_path = os.path.join(save_dir, imgs_name[index].replace('.tiff', '.png').replace('.tif', '.png'))
        cv2.imwrite(pred_saved_path, pred)

def save_npys(preds, save_dir, imgs_name):
    preds = preds.numpy().astype('float16')
    for index, pred in enumerate(preds):
        pred = np.squeeze(pred)*1
        pred_saved_path = os.path.join(save_dir, imgs_name[index].replace('.tif', '.npy'))
        np.save(pred_saved_path, pred)

def predict(model,
            model_path,
            transforms,
            image_list,
            batch_size=1,
            patch_size=1,
            image_dir=None,
            image_dir2=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            swap=0,
            save_logit = False,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)

    with paddle.no_grad():
        for i in range(0, len(img_lists[local_rank]), batch_size):
            ids = [i+index for index in range(batch_size)]
            ims1, ims2, imgs_name = read_imgs(img_lists[local_rank], image_dir, image_dir2, ids, transforms)
            if save_logit:
                ori_shape = None
            else:
                ori_shape = ims1.shape[2:]

            if is_slide and aug_pred:
                preds = infer.slide_infer(
                    model,
                    ims1,
                    ims2,
                    patch_size=patch_size,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    stride=stride,
                    crop_size=crop_size)
            elif aug_pred:
                preds = infer.aug_inference(
                    model,
                    ims1,
                    ims2,
                    swap=swap,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    stride=stride,
                    crop_size=crop_size)
            else:
                preds = infer.inference(
                    model,
                    ims1,
                    ims2,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    stride=stride,
                    crop_size=crop_size)
            if save_logit:
                save_npys(preds, save_dir, imgs_name)
            else:
                save_imgs(preds, save_dir, imgs_name)
            progbar_pred.update(i + len(imgs_name))
