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

import cv2, imghdr
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage.morphology import distance_transform_edt
from skimage import io


def im_read(img_path):
    img_format = imghdr.what(img_path)

    if img_format == 'tiff':
        img = io.imread(img_path).astype('float32')
    elif img_format in ['jpeg', 'bmp', 'png', 'jpg']:
        img =  cv2.imread(img_path)
        if img is None:
            try:
                img = io.imread(img_path).astype('float32')
            except:
                print(img_path)
    elif img_path.endwith('.npy'):
        img = np.load(img_path)
    else:
        raise Exception('Image format {} is not supported!'.format(img_path.split[-1]))
        img = None
    return img

def gt_read(gt_path):
    gt = io.imread(gt_path).astype('float32')
    # gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # if gt is None:
    #     gt = io.imread(gt_path).astype('float32')
    return gt



def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False)
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im

# region flip
def img_flip(im, method=0):
    """
    flip image in different ways, this function provides 5 method to filp
    this function can be applied to 2D or 3D images

    Args:
        im(array): image array
        method(int or string): choose the flip method, it must be one of [
                                0, 1, 2, 3, 4, 'h', 'v', 'hv', 'rt2lb', 'lt2rb', 'dia', 'adia']
        0 or 'h': flipped in horizontal direction, which is the most frequently used method
        1 or 'v': flipped in vertical direction
        2 or 'hv': flipped in both horizontal diction and vertical direction
        3 or 'rt2lb' or 'dia': flipped around the diagonal,
                                which also can be thought as changing the RightTop part with LeftBottom part,
                                so it is called 'rt2lb' as well.
        4 or 'lt2rb' or 'adia': flipped around the anti-diagonal
                                    which also can be thought as changing the LeftTop part with RightBottom part,
                                    so it is called 'lt2rb' as well.

    Returns:
        flipped image(array)

    Raises:
        ValueError: Shape of image should 2d, 3d or more.

    Examples:
        --assume an image is like this:

        img:
        / + +
        - / *
        - * /

        --we can flip it in following code:

        img_h = im_flip(img, 'h')
        img_v = im_flip(img, 'v')
        img_vh = im_flip(img, 2)
        img_rt2lb = im_flip(img, 3)
        img_lt2rb = im_flip(img, 4)

        --we can get flipped image:

        img_h, flipped in horizontal direction
        + + \
        * \ -
        \ * -

        img_v, flipped in vertical direction
        - * \
        - \ *
        \ + +

        img_vh, flipped in both horizontal diction and vertical direction
        / * -
        * / -
        + + /

        img_rt2lb, flipped around the diagonal
        / | |
        + / *
        + * /

        img_lt2rb, flipped around the anti-diagonal
        / * +
        * / +
        | | /

    """
    if not len(im.shape) >= 2:
        raise ValueError("Shape of image should 2d, 3d or more")
    if method==0 or method=='h':
        return horizontal_flip(im)
    elif method==1 or method=='v':
        return vertical_flip(im)
    elif method==2 or method=='hv':
        return hv_flip(im)
    elif method==3 or method=='rt2lb' or method=='dia':
        return rt2lb_flip(im)
    elif method==4 or method=='lt2rb' or method=='adia':
        return lt2rb_flip(im)
    else:
        return im

def horizontal_flip(im):
    im = im[:, ::-1, ...]
    return im

def vertical_flip(im):
    im = im[::-1, :, ...]
    return im

def hv_flip(im):
    im = im[::-1, ::-1, ...]
    return im

def rt2lb_flip(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im.transpose(axs_list)
    return im

def lt2rb_flip(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im[::-1, ::-1, ...].transpose(axs_list)
    return im

# endregion

# region rotation
def img_simple_rotate(im, method=0):
    """
    rotate image in simple ways, this function provides 3 method to rotate
    this function can be applied to 2D or 3D images

    Args:
        im(array): image array
        method(int or string): choose the flip method, it must be one of [
                                0, 1, 2, 90, 180, 270
                                ]
        0 or 90 : rotated in 90 degree, clockwise
        1 or 180: rotated in 180 degree, clockwise
        2 or 270: rotated in 270 degree, clockwise

    Returns:
        flipped image(array)


    Raises:
        ValueError: Shape of image should 2d, 3d or more.


    Examples:
        --assume an image is like this:

        img:
        / + +
        - / *
        - * /

        --we can rotate it in following code:

        img_r90 = img_simple_rotate(img, 90)
        img_r180 = img_simple_rotate(img, 1)
        img_r270 = img_simple_rotate(img, 2)

        --we can get rotated image:

        img_r90, rotated in 90 degree
        | | \
        * \ +
        \ * +

        img_r180, rotated in 180 degree
        / * -
        * / -
        + + /

        img_r270, rotated in 270 degree
        + * \
        + \ *
        \ | |


    """
    if not len(im.shape) >= 2:
        raise ValueError("Shape of image should 2d, 3d or more")
    if method==0 or method==90:
        return rot_90(im)
    elif method==1 or method==180:
        return rot_180(im)
    elif method==2 or method==270:
        return rot_270(im)
    else:
        return im

def rot_90(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im[::-1, :, ...].transpose(axs_list)
    return im

def rot_180(im):
    im = im[::-1, ::-1, ...]
    return im

def rot_270(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im[:, ::-1, ...].transpose(axs_list)
    return im
# endregion



def brightness(im, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def sharpness(im, sharpness_lower, sharpness_upper):
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def rotate(im, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def mask_to_onehot(mask, num_classes):
    """
    Convert a mask (H, W) to onehot (K, H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Onehot mask with shape(K, H, W).
    """
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):
    """
    Convert a onehot mask (K, H, W) to a edge mask.

    Args:
        mask (np.ndarray): Onehot mask with shape (K, H, W)
        radius (int|float): Radius of edge.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    if radius < 1:
        raise ValueError('`radius` should be greater than or equal to 1')
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(
        mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(
            mask[i, :]) + distance_transform_edt(1.0 - mask[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):
    """
    Convert a segmentic segmentation mask (H, W) to a binary edge mask(H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        radius (int|float): Radius of edge.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge
