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

import collections.abc
from itertools import combinations

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F

# region aug tools
def bi_argmax(x,y):
    xy = paddle.concat([paddle.unsqueeze(x,axis=0),paddle.unsqueeze(y,axis=0)])
    mxy = paddle.max(xy,axis=0)
    return mxy
def bi_argmin(x,y):
    xy = paddle.concat([paddle.unsqueeze(x,axis=0),paddle.unsqueeze(y,axis=0)])
    mxy = paddle.min(xy,axis=0)
    return mxy

def get_reverse_list(ori_shape, transforms):
    """
    get reverse list of transform.

    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.

    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    h, w = ori_shape[0], ori_shape[1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            short_edge = int(round(short_edge * op.long_size / long_edge))
            long_edge = op.long_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['PaddingByAspectRatio']:
            reverse_list.append(('padding', (h, w)))
            ratio = w / h
            if ratio == op.aspect_ratio:
                pass
            elif ratio > op.aspect_ratio:
                h = int(w / op.aspect_ratio)
            else:
                w = int(h * op.aspect_ratio)
        if op.__class__.__name__ in ['LimitLong']:
            long_edge = max(h, w)
            short_edge = min(h, w)
            if ((op.max_long is not None) and (long_edge > op.max_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.max_long
                short_edge = int(round(short_edge * op.max_long / long_edge))
            elif ((op.min_long is not None) and (long_edge < op.min_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.min_long
                short_edge = int(round(short_edge * op.min_long / long_edge))
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
    return reverse_list


def reverse_transform(pred, ori_shape, transforms, mode='nearest'):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu':
                pred = paddle.cast(pred, 'uint8')
                pred = F.interpolate(pred, (h, w), mode=mode)
                pred = paddle.cast(pred, 'int32')
            else:
                pred = F.interpolate(pred, (h, w), mode=mode)
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def tensor_rot(x, rot_angle):
    """Flip tensor according directions"""
    if rot_angle == 90:
        x = x[:, :, ::-1, :].transpose((0, 1, 3, 2))
    elif rot_angle == 180:
        x = x[:, :, ::-1, ::-1]
    elif rot_angle == 270:
        x = x[:, :, :, ::-1].transpose((0, 1, 3, 2))
    return x


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def patch_init():
    patch_count = 0
    patch_ims1 = []
    patch_ims2 = []
    patch_pos = []
    return patch_count, patch_ims1, patch_ims2, patch_pos


def patch_merge(final_logits, patch_logits, patch_pos, count):
    assert patch_logits.shape[0] == len(patch_pos), print(patch_logits.shape[0], len(patch_pos))
    for i in range(len(patch_pos)):
        h1, h2, w1, w2 = patch_pos[i]
        final_logits[:, :, h1:h2, w1:w2] += patch_logits[i:i + 1, :, :h2 - h1, :w2 - w1]
        count[:, :, h1:h2, w1:w2] += 1
    return final_logits, count

# endregion



#region infer related function
def slide_infer(model,
                im1,
                im2,
                ori_shape,
                transforms,
                patch_size=1.0,
                scales=1.0,
                rot_list=[],
                scales_weight=1.0,
                class_weight=1.0,
                flip_horizontal=False,
                flip_vertical=False,
                stride=None,
                crop_size=None):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """

    if isinstance(scales, float):
        scales = [scales]
    h_im, w_im = im1.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = np.int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = np.int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logits = None
    count = np.zeros([1, 1, h_im, w_im])
    flip_comb = flip_combination(flip_horizontal, flip_vertical)

    patch_count, patch_ims1, patch_ims2, patch_pos = patch_init()

    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop1 = im1[:, :, h1:h2, w1:w2]
            im_crop2 = im2[:, :, h1:h2, w1:w2]

            patch_ims1.append(im_crop1)
            patch_ims2.append(im_crop2)
            patch_count += 1
            patch_pos.append([h1, h2, w1, w2])

            if patch_count == patch_size or (cols - 1) * (rows - 1) == r * c:
                patch_ims1 = paddle.to_tensor(patch_ims1).squeeze(1)
                patch_ims2 = paddle.to_tensor(patch_ims2).squeeze(1)

                patch_logits = patch_infer(model, patch_ims1, patch_ims2, scales, h_crop, w_crop, flip_comb, rot_list)
                if final_logits is None:
                    final_logits = np.zeros([1, patch_logits.shape[1], h_im, w_im])
                final_logits, count = patch_merge(final_logits, patch_logits, patch_pos, count)
                patch_count, patch_ims1, patch_ims2, patch_pos = patch_init()
            else:
                continue
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logits = final_logits / count
    final_logits = paddle.to_tensor(final_logits)

    if ori_shape is not None:
        pred = reverse_transform(final_logits, ori_shape, transforms, mode='bilinear')
        pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
        return pred
    else:
        return final_logits




def patch_infer(model, im_crop1, im_crop2, scales, h_input, w_input, flip_comb, rot_list):
    final_logit = 0
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im_crop1 = F.interpolate(im_crop1, (h, w), mode='bilinear')
        im_crop2 = F.interpolate(im_crop2, (h, w), mode='bilinear')
        ims_flip1 = []
        ims_flip2 = []
        for flip in flip_comb:
            im_flip1 = tensor_flip(im_crop1, flip)
            im_flip2 = tensor_flip(im_crop2, flip)
            ims_flip1.append(im_flip1)
            ims_flip2.append(im_flip2)
        images_flip1 = paddle.concat(ims_flip1, axis=0)
        images_flip2 = paddle.concat(ims_flip2, axis=0)
        logit = inference(
            model,
            images_flip1,
            images_flip2)
        logits = paddle.split(logit, num_or_sections=len(ims_flip1), axis=0)
        for index, flip in enumerate(flip_comb):
            logit = tensor_flip(logits[index], flip)
            logit = F.interpolate(logit, (h_input, w_input), mode='bilinear')

            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit

        # for rot_angle in rot_list:
        #     im_rot = tensor_rot(im_crop, rot_angle)
        #     logit = inference(
        #         model,
        #         im_rot)
        #     logit = tensor_rot(logit, 360 - rot_angle)
        #     logit = F.interpolate(logit, (h_input, w_input), mode='bilinear')
        #     logit = F.softmax(logit, axis=1)
        #     final_logit += logit * 1
    return final_logit.numpy()





def aug_inference(model,
                  im1,
                  im2,
                  ori_shape,
                  transforms,
                  scales=1.0,
                  swap=False,
                  rot_list = [],
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=False,
                  stride=None,
                  crop_size=None):
    """
    Infer with augmentation.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) is returned.
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    
    final_logit = 0
    count = 0

    h_input, w_input = im1.shape[-2], im1.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    

    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im1 = F.interpolate(im1, (h, w), mode='bilinear')
        im2 = F.interpolate(im2, (h, w), mode='bilinear')


        ims_flip1,ims_flip2 = [],[]
        for flip in flip_comb:
            im_flip1 = tensor_flip(im1, flip)
            im_flip2 = tensor_flip(im2, flip)
            ims_flip1.append(im_flip1)
            ims_flip2.append(im_flip2)
        images_flip1 = paddle.concat(ims_flip1, axis=0)
        images_flip2 = paddle.concat(ims_flip2, axis=0)

        logit = inference(model, images_flip1, images_flip2)
        if swap:
            swap_logits = inference(model, images_flip2, images_flip1)
            if swap==1:
                logit = logit + swap_logits
            elif swap==2:
                logit = bi_argmax(logit, swap_logits)
            elif swap==3:
                logit[:, 0, ...] = bi_argmax(logit[:, 0, ...], swap_logits[:, 0, ...])
            elif swap==4:
                logit[:, 1, ...] = bi_argmax(logit[:, 1, ...], swap_logits[:, 1, ...])
            elif swap==5:
                logit[:, 0, ...] = bi_argmin(logit[:, 0, ...], swap_logits[:, 0, ...])
            elif swap==6:
                logit[:, 1, ...] = bi_argmin(logit[:, 1, ...], swap_logits[:, 1, ...])
            elif swap==7:
                logit[:, 0, ...] = bi_argmax(logit[:, 0, ...], swap_logits[:, 0, ...])
                logit[:, 1, ...] = bi_argmin(logit[:, 1, ...], swap_logits[:, 1, ...])
            elif swap==8:
                logit[:, 0, ...] = bi_argmin(logit[:, 0, ...], swap_logits[:, 0, ...])
                logit[:, 1, ...] = bi_argmax(logit[:, 1, ...], swap_logits[:, 1, ...])
        logits = paddle.split(logit, num_or_sections=len(ims_flip1), axis=0)


        for index, flip in enumerate(flip_comb):
            logit = tensor_flip(logits[index], flip)
            logit = F.interpolate(logit, (h_input, w_input), mode='bilinear')
            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit
        

    # final_logit /= count
    if ori_shape is not None:
        pred = reverse_transform(final_logit, ori_shape, transforms, mode='bilinear')
        pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
        return pred
    else:
        return final_logit


def inference(model,
              im1,
              im2,
              ori_shape=None,
              transforms=None,
              stride=None,
              crop_size=None):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im1 = im1.transpose((0, 2, 3, 1))
        im2 = im2.transpose((0, 2, 3, 1))

    logits = model(im1, im2)

    if len(logits) == 1:
        logit = logits[0]
    elif len(logits) == 2:
        logit = logits[0]+logits[1]*0.4
    else:
        logit = logits[0]

    if len(logit.shape) == 3:
        logit = logit.unsqueeze(1)

    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        logit = logit.transpose((0, 3, 1, 2))
    if ori_shape is not None:
        pred = reverse_transform(logit, ori_shape, transforms, mode='bilinear')

        if len(pred.shape)==3 or pred.shape[1] == 1:
            pred = paddle.round(pred).astype('int32')
        else:
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
        return pred
    else:
        return logit

# endregion
