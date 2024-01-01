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

import random
import math

import cv2
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import functional


@manager.TRANSFORMS.add_component
class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms


    def __call__(self, im1_path, im2_path, label_path=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im1_path, str) and isinstance(im2_path, str) :
            im1 = functional.im_read(im1_path)
            im2 = functional.im_read(im2_path)
            if im1 is None:
                raise ValueError('Can\'t read The image file {}!'.format(im1_path))
            if im2 is None:
                raise ValueError('Can\'t read The image file {}!'.format(im2_path))
        else:
            raise TypeError('Can\'t read The image file {} or {}!'.format(im1_path, im2_path))
        if isinstance(label_path, str):
            label = functional.gt_read(label_path)
            if label is None:
                raise ValueError('Can\'t read The label file {}!'.format(label_path))
        else:
            label = None

        for op in self.transforms:
            outputs = op(im1, im2, label)
            im1, im2 = outputs[0], outputs[1]
            if len(outputs) == 3:
                label = outputs[2]
        im1 = np.transpose(im1, (2, 0, 1))
        im2 = np.transpose(im2, (2, 0, 1))
        return (im1, im2, label)


@manager.TRANSFORMS.add_component
class Normalize2:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean1=(0.5, 0.5, 0.5), std1=(0.5, 0.5, 0.5), mean2=(0.5, 0.5, 0.5), std2=(0.5, 0.5, 0.5)):
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
        if not (isinstance(self.mean1, (list, tuple)) and isinstance(self.std1, (list, tuple))) or \
            not (isinstance(self.mean2, (list, tuple)) and isinstance(self.std2, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std1) == 0 or reduce(lambda x, y: x * y, self.std2) == 0:
            raise ValueError('{}: std is invalid!'.format(self))
        
        self.mean1 = np.array(self.mean1)[np.newaxis, np.newaxis, :]
        self.std1 = np.array(self.std1)[np.newaxis, np.newaxis, :]
        self.mean2 = np.array(self.mean2)[np.newaxis, np.newaxis, :]
        self.std2 = np.array(self.std2)[np.newaxis, np.newaxis, :]

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im1 = functional.normalize(im1, self.mean1, self.std1)
        im2 = functional.normalize(im2, self.mean2, self.std2)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


class Normalize2B:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean1=(0.5, 0.5, 0.5), std1=(0.5, 0.5, 0.5), mean2=(0.5, 0.5, 0.5), std2=(0.5, 0.5, 0.5)):
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
        if not (isinstance(self.mean1, (list, tuple)) and isinstance(self.std1, (list, tuple))) or \
            not (isinstance(self.mean2, (list, tuple)) and isinstance(self.std2, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std1) == 0 or reduce(lambda x, y: x * y, self.std2) == 0:
            raise ValueError('{}: std is invalid!'.format(self))
        
        self.mean1 = np.array(self.mean1)[np.newaxis, np.newaxis, :]
        self.std1 = np.array(self.std1)[np.newaxis, np.newaxis, :]
        self.mean2 = np.array(self.mean2)[np.newaxis, np.newaxis, :]
        self.std2 = np.array(self.std2)[np.newaxis, np.newaxis, :]

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im1 = functional.normalize(im1, self.mean1, self.std1)
        im2 = functional.normalize(im2, self.mean2, self.std2)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple)) and isinstance(self.std, (list, tuple))) :
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))
        
        self.mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        self.std = np.array(self.std)[np.newaxis, np.newaxis, :]


    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im1 = functional.normalize(im1, self.mean, self.std)
        im2 = functional.normalize(im2, self.mean, self.std)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomFlipOrRotation:
    """
    Flip or Rotate an image in different ways with a certain probability.

    Args:
        probs (list of float): Probabilities of flipping and rotation. Default: [0.35,0.25].
        probsf (list of float): Probabilities of 5 flipping mode
                                (horizontal, vertical, both horizontal diction and vertical, diagonal, anti-diagonal).
                                Default: [0.3, 0.3, 0.2, 0.1, 0.1].
        probsr (list of float): Probabilities of 3 rotation mode(90°, 180°, 270° clockwise). Default: [0.25,0.5,0.25].

    Examples:

        from paddlers import transforms as T

        # 定义数据增强
        train_transforms = T.Compose([
            T.RandomFlipOrRotation(
                probs  = [0.3, 0.2]             # 进行flip增强的概率是0.3，进行rotate增强的概率是0.2，不变的概率是0.5
                probsf = [0.3, 0.25, 0, 0, 0]   # flip增强时，使用水平flip、垂直flip的概率分别是0.3、0.25，水平且垂直flip、对角线flip、反对角线flip概率均为0，不变的概率是0.45
                probsr = [0, 0.65, 0]),         # rotate增强时，顺时针旋转90度的概率是0，顺时针旋转180度的概率是0.65，顺时针旋转90度的概率是0，不变的概率是0.35
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    """

    def __init__(self, probs=[0.35, 0.25], probsf=[0.3, 0.3, 0.2, 0.1, 0.1], probsr=[0.25,0.5,0.25]):
        self.probs = [probs[0], probs[0]+probs[1]]
        self.probsf = self.get_probs_range(probsf)
        self.probsr = self.get_probs_range(probsr)

    def get_probs_range(self, probs):
        '''
        Change various probabilities into cumulative probabilities

        Args:
            probs(list of float): probabilities of different mode, shape:[n]

        Returns:
            probability intervals(list of binary list): shape:[n, 2]
        '''
        ps = []
        last_prob = 0
        for prob in probs:
            p_s = last_prob
            cur_prob = prob / sum(probs) if sum(probs)!=0 else prob / (sum(probs)+1e-10)
            last_prob += cur_prob
            p_e = last_prob
            ps.append([p_s, p_e])
        return ps

    def judge_probs_range(self, p, probs):
        '''
        Judge whether a probability value falls within the given probability interval

        Args:
            p(float): probability
            probs(list of binary list): probability intervals, shape:[n, 2]

        Returns:
            mode id(int):the probability interval number where the input probability falls,
                         if return -1, the image will remain as it is and will not be processed
        '''
        for id, id_range in enumerate(probs):
            if p> id_range[0] and p<id_range[1]:
                return id
        return -1

    def __call__(self, im1, im2, label=None):
        p_m = random.random()
        if p_m < self.probs[0]:
            mode_p = random.random()
            mode_id = self.judge_probs_range(mode_p, self.probsf)

            im1 = functional.img_flip(im1, mode_id)
            im2 = functional.img_flip(im2, mode_id)
            if label is not None:
                label = functional.img_flip(label, mode_id)

        elif p_m < self.probs[1]:
            mode_p = random.random()
            mode_id = self.judge_probs_range(mode_p, self.probsr)

            im1 = functional.img_simple_rotate(im1, mode_id)
            im2 = functional.img_simple_rotate(im2, mode_id)
            if label is not None:
                label = functional.img_simple_rotate(label, mode_id)


        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

# class RandomFlipOrRotation(Transform):
#     """
#     Flip or Rotate an image in different ways with a certain probability.

#     Args:
#         probs (list of float): Probabilities of flipping and rotation. Default: [0.35,0.25].
#         probsf (list of float): Probabilities of 5 flipping mode
#                                 (horizontal, vertical, both horizontal diction and vertical, diagonal, anti-diagonal).
#                                 Default: [0.3, 0.3, 0.2, 0.1, 0.1].
#         probsr (list of float): Probabilities of 3 rotation mode(90°, 180°, 270° clockwise). Default: [0.25,0.5,0.25].

#     Examples:

#         from paddlers import transforms as T

#         # 定义数据增强
#         train_transforms = T.Compose([
#             T.RandomFlipOrRotation(
#                 probs  = [0.3, 0.2]             # 进行flip增强的概率是0.3，进行rotate增强的概率是0.2，不变的概率是0.5
#                 probsf = [0.3, 0.25, 0, 0, 0]   # flip增强时，使用水平flip、垂直flip的概率分别是0.3、0.25，水平且垂直flip、对角线flip、反对角线flip概率均为0，不变的概率是0.45
#                 probsr = [0, 0.65, 0]),         # rotate增强时，顺时针旋转90度的概率是0，顺时针旋转180度的概率是0.65，顺时针旋转90度的概率是0，不变的概率是0.35
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     """

#     def __init__(self, probs=[0.35, 0.25], probsf=[0.3, 0.3, 0.2, 0.1, 0.1], probsr=[0.25,0.5,0.25]):
#         super(RandomFlipOrRotation, self).__init__()
#         # Change various probabilities into probability intervals, to judge in which mode to flip or rotate
#         self.probs = [probs[0], probs[0]+probs[1]]
#         self.probsf = self.get_probs_range(probsf)
#         self.probsr = self.get_probs_range(probsr)

#     def apply_im(self, image, mode_id, flip_mode=True):
#         if flip_mode:
#             image = img_flip(image, mode_id)
#         else:
#             image = img_simple_rotate(image, mode_id)
#         return image

#     def apply_mask(self, mask, mode_id, flip_mode=True):
#         if flip_mode:
#             mask = img_flip(mask, mode_id)
#         else:
#             mask = img_simple_rotate(mask, mode_id)
#         return mask

#     def get_probs_range(self, probs):
#         '''
#         Change various probabilities into cumulative probabilities

#         Args:
#             probs(list of float): probabilities of different mode, shape:[n]

#         Returns:
#             probability intervals(list of binary list): shape:[n, 2]
#         '''
#         ps = []
#         last_prob = 0
#         for prob in probs:
#             p_s = last_prob
#             cur_prob = prob / sum(probs)
#             last_prob += cur_prob
#             p_e = last_prob
#             ps.append([p_s, p_e])
#         return ps

#     def judge_probs_range(self, p, probs):
#         '''
#         Judge whether a probability value falls within the given probability interval

#         Args:
#             p(float): probability
#             probs(list of binary list): probability intervals, shape:[n, 2]

#         Returns:
#             mode id(int):the probability interval number where the input probability falls,
#                          if return -1, the image will remain as it is and will not be processed
#         '''
#         for id, id_range in enumerate(probs):
#             if p> id_range[0] and p<id_range[1]:
#                 return id
#         return -1

#     def apply(self, sample):
#         p_m = random.random()
#         if p_m < self.probs[0]:
#             mode_p = random.random()
#             mode_id = self.judge_probs_range(mode_p, self.probsf)
#             sample['image'] = self.apply_im(sample['image'], mode_id, True)
#             if 'image2' in sample:
#                 sample['image2'] = self.apply_im(sample['image2'], mode_id, True)
#             if 'mask' in sample:
#                 sample['mask'] = self.apply_mask(sample['mask'], mode_id, True)
#         elif p_m < self.probs[1]:
#             mode_p = random.random()
#             mode_id = self.judge_probs_range(mode_p, self.probsr)
#             sample['image'] = self.apply_im(sample['image'], mode_id, False)
#             if 'image2' in sample:
#                 sample['image2'] = self.apply_im(sample['image2'], mode_id, False)
#             if 'mask' in sample:
#                 sample['mask'] = self.apply_mask(sample['mask'], mode_id, False)
#         return sample


@manager.TRANSFORMS.add_component
class RandomFlip:
    """
    Flip an image horizontally with a certain probability.

    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """

    def __init__(self, prob=0.5, flip4=False, probs=[0.5,0.5]):
        self.prob = prob
        self.flip4 = flip4
        self.probs = self.get_probs_range(probs)
        if (len(probs)==4 and self.flip4) or (len(probs)==2 and not self.flip4):
            pass
        else:
            raise ValueError('probs length {} not match filp param {}'.format(self.probs, self.flip4))

    def get_probs_range(self, probs):
        ps = []
        last_prob = 0
        for prob in probs:
            p_s = last_prob
            cur_prob = prob / sum(probs)
            last_prob += cur_prob
            p_e = last_prob
            ps.append([p_s, p_e])
        return ps
    def judge_probs_range(self, p, probs):
        for id, id_range in enumerate(probs):
            if p> id_range[0] and p<id_range[1]:
                return id
        return 0

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            p_m = random.random()
            id_m = self.judge_probs_range(p_m, self.probs)
            im1 = functional.im_flip(im1, id_m)
            im2 = functional.im_flip(im2, id_m)
            if label is not None:
                label = functional.im_flip(label, id_m)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class RandomVerticalFlip:
    """
    Flip an image vertically with a certain probability.

    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            im1 = functional.vertical_flip(im1)
            im2 = functional.vertical_flip(im2)
            if label is not None:
                label = functional.vertical_flip(label)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class RandomSwap:
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            im1, im2 = im2, im1
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class RandomHorizontalFlip:
    """
    Flip an image horizontally with a certain probability.

    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            im1 = functional.horizontal_flip(im1)
            im2 = functional.horizontal_flip(im2)
            if label is not None:
                label = functional.horizontal_flip(label)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class RandomLT2RBFlip:
    """
    Flip an image vertically with a certain probability.

    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            im1 = functional.lt2rb_flip(im1)
            im2 = functional.lt2rb_flip(im2)
            if label is not None:
                label = functional.lt2rb_flip(label)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class RandomRT2LBFlip:
    """
    Flip an image vertically with a certain probability.

    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            im1 = functional.rt2lb_flip(im1)
            im2 = functional.rt2lb_flip(im2)
            if label is not None:
                label = functional.rt2lb_flip(label)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomSimpleRotate:
    """
    Flip an image horizontally with a certain probability.

    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """

    def __init__(self, prob=0.5, probs=[0.3,0.4,0.3]):
        self.prob = prob
        self.probs = self.get_probs_range(probs)

    def get_probs_range(self, probs):
        ps = []
        last_prob = 0
        for prob in probs:
            p_s = last_prob
            cur_prob = prob / sum(probs)
            last_prob += cur_prob
            p_e = last_prob
            ps.append([p_s, p_e])
        return ps
    def judge_probs_range(self, p, probs):
        for id, id_range in enumerate(probs):
            if p> id_range[0] and p<id_range[1]:
                return id
        return 0

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            p_m = random.random()
            id_m = self.judge_probs_range(p_m, self.probs)

            im1 = functional.im_simple_rotate(im1, id_m)
            im2 = functional.im_simple_rotate(im2, id_m)
            if label is not None:
                label = functional.im_simple_rotate(label, id_m)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)



@manager.TRANSFORMS.add_component
class RandomRotation:
    """
    Rotate an image randomly with padding.

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 max_rotation=15,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.max_rotation > 0:
            (h, w) = im1.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            im1 = cv2.warpAffine(
                im1,
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            im2 = cv2.warpAffine(
                im2,
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(
                    label,
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)

@manager.TRANSFORMS.add_component
class Resize:
    """
    Resize an image.

    Args:
        target_size (list|tuple, optional): The target size of image. Default: (512, 512).
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".

    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label),

        Raises:
            TypeError: When the 'img' type is not numpy.
            ValueError: When the length of "im" shape is not 3.
        """

        if not isinstance(im1, np.ndarray) or not isinstance(im2, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im1.shape) != 3 or len(im2.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im1 = functional.resize(im1, self.target_size, self.interp_dict[interp])
        im2 = functional.resize(im2, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = functional.resize(label, self.target_size,
                                      cv2.INTER_NEAREST)

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class ResizeByLong:
    """
    Resize the long side of an image to given size, and then scale the other side proportionally.

    Args:
        long_size (int): The target size of long side.
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im1 = functional.resize_long(im1, self.long_size)
        im2 = functional.resize_long(im2, self.long_size)
        if label is not None:
            label = functional.resize_long(label, self.long_size,
                                           cv2.INTER_NEAREST)

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class LimitLong:
    """
    Limit the long edge of image.

    If the long edge is larger than max_long, resize the long edge
    to max_long, while scale the short edge proportionally.

    If the long edge is smaller than min_long, resize the long edge
    to min_long, while scale the short edge proportionally.

    Args:
        max_long (int, optional): If the long edge of image is larger than max_long,
            it will be resize to max_long. Default: None.
        min_long (int, optional): If the long edge of image is smaller than min_long,
            it will be resize to min_long. Default: None.
    """

    def __init__(self, max_long=None, min_long=None):
        if max_long is not None:
            if not isinstance(max_long, int):
                raise TypeError(
                    "Type of `max_long` is invalid. It should be int, but it is {}"
                    .format(type(max_long)))
        if min_long is not None:
            if not isinstance(min_long, int):
                raise TypeError(
                    "Type of `min_long` is invalid. It should be int, but it is {}"
                    .format(type(min_long)))
        if (max_long is not None) and (min_long is not None):
            if min_long > max_long:
                raise ValueError(
                    '`max_long should not smaller than min_long, but they are {} and {}'
                    .format(max_long, min_long))
        self.max_long = max_long
        self.min_long = min_long

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        h, w = im1.shape[0], im2.shape[1]
        long_edge = max(h, w)
        target = long_edge
        if (self.max_long is not None) and (long_edge > self.max_long):
            target = self.max_long
        elif (self.min_long is not None) and (long_edge < self.min_long):
            target = self.min_long

        if target != long_edge:
            im1 = functional.resize_long(im1, target)
            im2 = functional.resize_long(im2, target)
            if label is not None:
                label = functional.resize_long(label, target, cv2.INTER_NEAREST)

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class ResizeRangeScaling:
    """
    Resize the long side of an image into a range, and then scale the other side proportionally.

    Args:
        min_value (int, optional): The minimum value of long side after resize. Default: 400.
        max_value (int, optional): The maximum value of long side after resize. Default: 600.
    """

    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(
                                 min_value, max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        im1 = functional.resize_long(im1, random_size, cv2.INTER_LINEAR)
        im2 = functional.resize_long(im2, random_size, cv2.INTER_LINEAR)
        if label is not None:
            label = functional.resize_long(label, random_size,
                                           cv2.INTER_NEAREST)

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class ResizeStepScaling:
    """
    Scale an image proportionally within a range.

    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25.

    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * im1.shape[1]))
        h = int(round(scale_factor * im2.shape[0]))

        im1 = functional.resize(im1, (w, h), cv2.INTER_LINEAR)
        im2 = functional.resize(im2, (w, h), cv2.INTER_LINEAR)
        if label is not None:
            label = functional.resize(label, (w, h), cv2.INTER_NEAREST)

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class Padding:
    """
    Add bottom-right padding to a raw image or annotation image.

    Args:
        target_size (list|tuple): The target size after padding.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When target_size is neither list nor tuple.
        ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of target_size is invalid. It should be list or tuple, now is {}"
                .format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im_height, im_width = im1.shape[0], im1.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'The size of image should be less than `target_size`, but the size of image ({}, {}) is larger than `target_size` ({}, {})'
                .format(im_width, im_height, target_width, target_height))
        else:
            im1 = cv2.copyMakeBorder(
                im1,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=self.im_padding_value)
            im2 = cv2.copyMakeBorder(
                im2,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=self.im_padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(
                    label,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.label_padding_value)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomCrop:
    """

    Args:
        aspect_ratio (int|float, optional): The aspect ratio = width / height. Default: 1.
    """

    def __init__(self,
                 crop_size=None,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False
                 ):
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def _generate_crop_info(self, im1):
        im_h, im_w = im1.shape[:2]
        crop_box = self._get_crop_box(im_h, im_w)
        return crop_box, None, None

    def _get_crop_box(self, im_h, im_w):
        scale = np.random.uniform(*self.scaling)
        if self.aspect_ratio is not None:
            min_ar, max_ar = self.aspect_ratio
            aspect_ratio = np.random.uniform(
                max(min_ar, scale**2), min(max_ar, scale**-2))
            h_scale = scale / np.sqrt(aspect_ratio)
            w_scale = scale * np.sqrt(aspect_ratio)
        else:
            h_scale = np.random.uniform(*self.scaling)
            w_scale = np.random.uniform(*self.scaling)
        crop_h = im_h * h_scale
        crop_w = im_w * w_scale
        if self.aspect_ratio is None:
            if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                return None
        crop_h = int(crop_h)
        crop_w = int(crop_w)
        # print(im_h, crop_h, im_w, crop_w)
        crop_y = np.random.randint(0, im_h - crop_h)
        crop_x = np.random.randint(0, im_w - crop_w)
        return [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
    
    def apply_im(self, image, crop):
        x1, y1, x2, y2 = crop
        return image[y1:y2, x1:x2, :]

    def apply_mask(self, mask, crop):
        x1, y1, x2, y2 = crop
        return mask[y1:y2, x1:x2, ...]
        
    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        crop_info = self._generate_crop_info(im1)
        if crop_info is not None:
            crop_box, cropped_box, valid_ids = crop_info
            im_h, im_w = im1.shape[:2]
            im1 = self.apply_im(im1, crop_box)
            im2 = self.apply_im(im2, crop_box)
            label = self.apply_mask(label, crop_box)

        if self.crop_size is not None:
            im1, im2, label = Resize(self.crop_size)(im1, im2, label)
        return im1, im2, label


@manager.TRANSFORMS.add_component
class PaddingByAspectRatio:
    """

    Args:
        aspect_ratio (int|float, optional): The aspect ratio = width / height. Default: 1.
    """

    def __init__(self,
                 aspect_ratio=1,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.aspect_ratio = aspect_ratio
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        img_height = im1.shape[0]
        img_width = im1.shape[1]
        ratio = img_width / img_height
        if ratio == self.aspect_ratio:
            if label is None:
                return (im1, im2, )
            else:
                return (im1, im2, label)
        elif ratio > self.aspect_ratio:
            img_height = int(img_width / self.aspect_ratio)
        else:
            img_width = int(img_height * self.aspect_ratio)
        padding = Padding((img_width, img_height),
                          im_padding_value=self.im_padding_value,
                          label_padding_value=self.label_padding_value)
        return padding(im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomPaddingCrop:
    """
    Crop a sub-image from a raw image and annotation image randomly. If the target cropping size
    is larger than original image, then the bottom-right padding will be added.

    Args:
        crop_size (tuple, optional): The target cropping size. Default: (512, 512).
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When crop_size is neither list nor tuple.
        ValueError: When the length of crop_size is not 2.
    """

    def __init__(self,
                 crop_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'Type of `crop_size` is list or tuple. It should include 2 elements, but it is {}'
                    .format(crop_size))
        else:
            raise TypeError(
                "The type of `crop_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im1.shape[0]
        img_width = im1.shape[1]

        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return (im1, im2, )
            else:
                return (im1, im2, label)
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im1 = cv2.copyMakeBorder(
                    im1,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.im_padding_value)
                im2 = cv2.copyMakeBorder(
                    im2,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.im_padding_value)
                if label is not None:
                    label = cv2.copyMakeBorder(
                        label,
                        0,
                        pad_height,
                        0,
                        pad_width,
                        cv2.BORDER_CONSTANT,
                        value=self.label_padding_value)
                img_height = im1.shape[0]
                img_width = im1.shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im1 = im1[h_off:(crop_height + h_off), w_off:(
                    w_off + crop_width), :]
                im2 = im2[h_off:(crop_height + h_off), w_off:(
                    w_off + crop_width), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width)]
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class ScalePadding:
    """
        Add center padding to a raw image or annotation image,then scale the
        image to target size.

        Args:
            target_size (list|tuple, optional): The target size of image. Default: (512, 512).
            im_padding_value (list, optional): The padding value of raw image.
                Default: [127.5, 127.5, 127.5].
            label_padding_value (int, optional): The padding value of annotation image. Default: 255.

        Raises:
            TypeError: When target_size is neither list nor tuple.
            ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        height = im1.shape[0]
        width = im1.shape[1]

        new_im1 = np.zeros(
            (max(height, width), max(height, width), 3)) + self.im_padding_value
        new_im2 = np.zeros(
            (max(height, width), max(height, width), 3)) + self.im_padding_value
        if label is not None:
            new_label = np.zeros((max(height, width), max(
                height, width))) + self.label_padding_value

        if height > width:
            padding = int((height - width) / 2)
            new_im1[:, padding:padding + width, :] = im1
            new_im2[:, padding:padding + width, :] = im2
            if label is not None:
                new_label[:, padding:padding + width] = label
        else:
            padding = int((width - height) / 2)
            new_im1[padding:padding + height, :, :] = im1
            new_im2[padding:padding + height, :, :] = im2
            if label is not None:
                new_label[padding:padding + height, :] = label

        im1 = np.uint8(new_im1)
        im2 = np.uint8(new_im2)
        im1 = functional.resize(im1, self.target_size, interp=cv2.INTER_CUBIC)
        im2 = functional.resize(im2, self.target_size, interp=cv2.INTER_CUBIC)
        if label is not None:
            label = np.uint8(new_label)
            label = functional.resize(
                label, self.target_size, interp=cv2.INTER_CUBIC)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomNoise:
    """
    Superimposing noise on an image with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.5.
        max_sigma(float, optional): The maximum value of standard deviation of the distribution.
            Default: 10.0.
    """

    def __init__(self, prob=0.5, max_sigma=10.0, same_noise=False):
        self.prob = prob
        self.max_sigma = max_sigma
        self.same_nosie = same_noise

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        if random.random() < self.prob:
            mu = 0
            sigma = random.random() * self.max_sigma
            im1 = np.array(im1, dtype=np.float32)
            im2 = np.array(im2, dtype=np.float32)
            if self.same_noise:
                ns = np.random.normal(mu, sigma, im1.shape)
                im1 += ns
                im2 +=ns
            else:
                im1 += np.random.normal(mu, sigma, im1.shape)
                im2 += np.random.normal(mu, sigma, im2.shape)
            im1[im1 > 255] = 255
            im1[im2 < 0] = 0

        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
        blur_type(str, optional): A type of blurring an image,
            gaussian stands for cv2.GaussianBlur,
            median stands for cv2.medianBlur,
            blur stands for cv2.blur,
            random represents randomly selected from above.
            Default: gaussian.
    """

    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im1 = np.array(im1, dtype='uint8')
                im2 = np.array(im2, dtype='uint8')
                if self.blur_type == "gaussian":
                    im1 = cv2.GaussianBlur(im1, (radius, radius), 0, 0)
                    im2 = cv2.GaussianBlur(im2, (radius, radius), 0, 0)
                elif self.blur_type == "median":
                    im1 = cv2.medianBlur(im1, radius)
                    im1 = cv2.medianBlur(im2, radius)
                elif self.blur_type == "blur":
                    im1 = cv2.blur(im1, (radius, radius))
                    im2 = cv2.blur(im2, (radius, radius))
                elif self.blur_type == "random":
                    select = random.random()
                    if select < 0.3:
                        im1 = cv2.GaussianBlur(im1, (radius, radius), 0)
                        im2 = cv2.GaussianBlur(im2, (radius, radius), 0)
                    elif select < 0.6:
                        im1 = cv2.medianBlur(im1, radius)
                        im2 = cv2.medianBlur(im2, radius)
                    else:
                        im1 = cv2.blur(im1, (radius, radius))
                        im2 = cv2.blur(im2, (radius, radius))
                else:
                    im1 = cv2.GaussianBlur(im1, (radius, radius), 0, 0)
                    im2 = cv2.GaussianBlur(im2, (radius, radius), 0, 0)
        im1 = np.array(im1, dtype='float32')
        im2 = np.array(im2, dtype='float32')
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomScaleAspect:
    """
    Crop a sub-image from an original image with a range of area ratio and aspect and
    then scale the sub-image back to the size of the original image.

    Args:
        min_scale (float, optional): The minimum area ratio of cropped image to the original image. Default: 0.5.
        aspect_ratio (float, optional): The minimum aspect ratio. Default: 0.33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im1.shape[0]
            img_width = im2.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im1 = im1[h1:(h1 + dh), w1:(w1 + dw), :]
                    im2 = im2[h1:(h1 + dh), w1:(w1 + dw), :]
                    im1 = cv2.resize(
                        im1, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    im2= cv2.resize(
                        im2, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    if label is not None:
                        label = label[h1:(h1 + dh), w1:(w1 + dw)]
                        label = cv2.resize(
                            label, (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomDistort:
    """
    Distort an image with random configurations.

    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [
            functional.brightness, functional.contrast, functional.saturation,
            functional.hue, functional.sharpness
        ]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        im1 = im1.astype('uint8')
        im2 = im2.astype('uint8')
        im1 = Image.fromarray(im1)
        im2 = Image.fromarray(im2)
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]

            if np.random.uniform(0, 1) < prob:
                p = np.random.uniform(0, 1)
                if p<0.5:
                    params['im'] = im1
                    im1 = ops[id](**params)
                    params['im'] = im2
                    im2 = ops[id](**params)
                elif p<0.75:
                    params['im'] = im1
                    im1 = ops[id](**params)
                elif p<1:
                    params['im'] = im2
                    im2 = ops[id](**params)              
        im1 = np.asarray(im1).astype('float32')
        im2 = np.asarray(im2).astype('float32')
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)


@manager.TRANSFORMS.add_component
class RandomAffine:
    """
    Affine transform an image with random configurations.

    Args:
        size (tuple, optional): The target size after affine transformation. Default: (224, 224).
        translation_offset (float, optional): The maximum translation offset. Default: 0.
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        im_padding_value (float, optional): The padding value of raw image. Default: (128, 128, 128).
        label_padding_value (int, optional): The padding value of annotation image. Default: (255, 255, 255).
    """

    def __init__(self,
                 size=(224, 224),
                 translation_offset=0,
                 max_rotation=15,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 im_padding_value=(128, 128, 128),
                 label_padding_value=(255, 255, 255)):
        self.size = size
        self.translation_offset = translation_offset
        self.max_rotation = max_rotation
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        w, h = self.size
        bbox = [0, 0, im1.shape[1] - 1, im1.shape[0] - 1]
        x_offset = (random.random() - 0.5) * 2 * self.translation_offset
        y_offset = (random.random() - 0.5) * 2 * self.translation_offset
        dx = (w - (bbox[2] + bbox[0])) / 2.0
        dy = (h - (bbox[3] + bbox[1])) / 2.0

        matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

        angle = random.random() * 2 * self.max_rotation - self.max_rotation
        scale = random.random() * (self.max_scale_factor - self.min_scale_factor
                                   ) + self.min_scale_factor
        scale *= np.mean(
            [float(w) / (bbox[2] - bbox[0]),
             float(h) / (bbox[3] - bbox[1])])
        alpha = scale * math.cos(angle / 180.0 * math.pi)
        beta = scale * math.sin(angle / 180.0 * math.pi)

        centerx = w / 2.0 + x_offset
        centery = h / 2.0 + y_offset
        matrix = np.array(
            [[alpha, beta, (1 - alpha) * centerx - beta * centery],
             [-beta, alpha, beta * centerx + (1 - alpha) * centery],
             [0, 0, 1.0]])

        matrix = matrix.dot(matrix_trans)[0:2, :]
        im1 = cv2.warpAffine(
            np.uint8(im1),
            matrix,
            tuple(self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.im_padding_value)
        im2 = cv2.warpAffine(
            np.uint8(im2),
            matrix,
            tuple(self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.im_padding_value)
        if label is not None:
            label = cv2.warpAffine(
                np.uint8(label),
                matrix,
                tuple(self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT)
        if label is None:
            return (im1, im2, )
        else:
            return (im1, im2, label)
