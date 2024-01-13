from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import yaml
import math

class OnlineImageAugmentor:
    def __init__(self, cfg_filename):
        cfg_file = open(cfg_filename, 'r')
        self.cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        cfg_file.close()
        if isinstance(self.cfg['AUGMENTATION']['CROP_SIZE'], int):
            self.crop_size = (self.cfg['AUGMENTATION']['CROP_SIZE'], self.cfg['AUGMENTATION']['CROP_SIZE'])
        elif isinstance(self.cfg['AUGMENTATION']['CROP_SIZE'], list):
            assert len(self.cfg['AUGMENTATION']['CROP_SIZE']) == 2
            self.crop_size = self.cfg['AUGMENTATION']['CROP_SIZE']
        elif self.cfg['AUGMENTATION']['CROP_SIZE'] == 'None':
            self.crop_size = None
        else:
            raise TypeError("Invalid type of chip_size, which should be int or list or 'None'!")
        self.resize_scales = self.cfg['AUGMENTATION']['RESIZE_SCALES']
        self.resize_mode = self.cfg['AUGMENTATION']['RESIZE_MODE']
        self.rotate_angles = self.cfg['AUGMENTATION']['ROTATE_ANGLES']
        self.hflip_prob = self.cfg['AUGMENTATION']['HFIP_PROB']
        self.vflip_prob = self.cfg['AUGMENTATION']['VFIP_PROB']
        self.shift_ratio = self.cfg['AUGMENTATION']['SHIFTING']
        self.cropped_rotate_max_angle = self.cfg['AUGMENTATION']['CROP_ROTATE_MAX_ANGLE']
        self.cropped_rotate_max_center_distance = self.cfg['AUGMENTATION']['CROP_ROTATE_MAX_CENTER_DISTANCE']
        self.brightness_factor_range = self.cfg['AUGMENTATION']['BRIGHTNESS']
        self.color_factor_range = self.cfg['AUGMENTATION']['COLOR']
        self.contrast_factor_range = self.cfg['AUGMENTATION']['CONTRAST']
        self.sharpness_factor_range = self.cfg['AUGMENTATION']['SHARPNESS']
        self.use_mosaic = self.cfg['AUGMENTATION']['MOSAIC']
        self.mosaic_num = self.cfg['AUGMENTATION']['MOSAIC_NUM']
        self.composed_augments = self.cfg['COMPOSE']['AUGS']
        self.compose_before_mosaic = self.cfg['COMPOSE']['BEFORE_MOSAIC']
        self.seed = self.cfg['AUGMENTATION']['SEED']
        self.norm_means = [0.5, 0.5, 0.5]
        self.norm_stds = [0.5, 0.5, 0.5]
        assert isinstance(self.seed, int) or self.seed is None
        if self.seed is not None:
            random.seed(self.seed)
        else:
            random.seed()
    
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

    def check_size_matched(self, image, mask):
        w1, h1 = image.size
        w2, h2 = mask.size
        assert (h1 == h2 and w1 == w2)
    
    def crop(self, image, mask, size, x, y):
        # crop size is not the same as chip size!!!!!!
        # if size is tuple, then it should be (crop_w, crop_h)
        self.check_size_matched(image, mask)
        if isinstance(size, int):
            crop_size = (size, size)
        elif isinstance(size, tuple):
            crop_size = size
        else:
            raise TypeError("Invalid type of crop_size, which should be int or tuple!")
        w, h = image.size
        box = (max(x, 0), max(y, 0), min(x + crop_size[0], w), min(y + crop_size[1], h))
        croped_image = image.crop(box)
        croped_mask = mask.crop(box)
        return croped_image, croped_mask
    
    def random_crop(self, image, mask, pad_value=0):
        self.check_size_matched(image, mask)
        crop_size = tuple(self.crop_size)
        w, h = image.size
        if w < crop_size[0]:
            left = int(0.5 * (crop_size[0] - w))
            right = crop_size[0] - w - left
            image, mask = self.pad(image, mask, (left, 0, right, 0), pad_value=pad_value)
        if h < crop_size[1]:
            top = int(0.5 * (crop_size[1] - h))
            bottom = crop_size[1] - h - top
            image, mask = self.pad(image, mask, (0, top, 0, bottom), pad_value=pad_value)
        rw, rh = image.size
        y = random.randint(0, rh - crop_size[1])
        x = random.randint(0, rw - crop_size[0])
        croped_image, croped_mask = self.crop(image, mask, crop_size, x, y)
        while not self.is_data_valid(croped_image, croped_mask, pad_value=pad_value):
            random.seed()
            y = random.randint(0, rh - crop_size[1])
            x = random.randint(0, rw - crop_size[0])
            croped_image, croped_mask = self.crop(image, mask, crop_size, x, y)
        return croped_image, croped_mask
    
    def rotate(self, image, mask, angle, expand=True):
        self.check_size_matched(image, mask)
        assert isinstance(angle, int)
        if angle == 0:
            rotated_image = image
            rotated_mask = mask
        elif angle == 90:
            rotated_image = image.transpose(Image.ROTATE_90)
            rotated_mask = mask.transpose(Image.ROTATE_90)
        elif angle == 180:
            rotated_image = image.transpose(Image.ROTATE_180)
            rotated_mask = mask.transpose(Image.ROTATE_180)
        elif angle == 270:
            rotated_image = image.transpose(Image.ROTATE_270)
            rotated_mask = mask.transpose(Image.ROTATE_270)
        else:
            rotated_image = image.rotate(angle, resample=Image.NEAREST, expand=expand)
            rotated_mask = mask.rotate(angle, resample=Image.NEAREST, expand=expand)
        return rotated_image, rotated_mask
    
    def random_rotate(self, image, mask):
        self.check_size_matched(image, mask)
        angles = self.rotate_angles
        angle_idx = random.randint(0, len(angles) - 1)
        return self.rotate(image, mask, angles[angle_idx])
    
    def flip(self, image, mask, mode='vertical'):
        self.check_size_matched(image, mask)
        if mode == 'vertical':
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        elif mode == 'horizontal':
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raise ValueError("Invalid mode for flip operation, you must choose from {'vertical', 'horizontal'}")
        return flipped_image, flipped_mask
    
    def random_vflip(self, image, mask):
        prob = self.vflip_prob
        if random.random() < prob:
            return self.flip(image, mask, mode='vertical')
        return image, mask

    def random_hflip(self, image, mask):
        prob = self.hflip_prob
        if random.random() < prob:
            return self.flip(image, mask, mode='horizontal')
        return image, mask
    
    def brightness_change(self, image, brightness_factor):
        enhance_brightness = ImageEnhance.Brightness(image)
        brightness_enhanced_image = enhance_brightness.enhance(factor=brightness_factor)
        return brightness_enhanced_image
    
    def random_brightness_change(self, image):
        brightness_factor_range = self.brightness_factor_range
        brightness_factor = random.random() * (brightness_factor_range[1] - brightness_factor_range[0]) + brightness_factor_range[0]
        return self.brightness_change(image, brightness_factor)

    def color_change(self, image, color_factor):
        enhance_color = ImageEnhance.Color(image)
        color_enhanced_image = enhance_color.enhance(factor=color_factor)
        return color_enhanced_image

    def random_color_change(self, image):
        color_factor_range = self.color_factor_range
        color_factor = random.random() * (color_factor_range[1] - color_factor_range[0]) + color_factor_range[0]
        return self.color_change(image, color_factor)
    
    def contrast_change(self, image, contrast_factor):
        enhance_contrast = ImageEnhance.Contrast(image)
        contrast_enhanced_image = enhance_contrast.enhance(factor=contrast_factor)
        return contrast_enhanced_image

    def random_contrast_change(self, image):
        contrast_factor_range = self.contrast_factor_range
        contrast_factor = random.random() * (contrast_factor_range[1] - contrast_factor_range[0]) + contrast_factor_range[0]
        return self.contrast_change(image, contrast_factor)
    
    def sharpness_change(self, image, sharpness_factor):
        enhance_sharpness = ImageEnhance.Sharpness(image)
        sharpness_enhanced_image = enhance_sharpness.enhance(factor=sharpness_factor)
        return sharpness_enhanced_image

    def random_sharpness_change(self, image):
        sharpness_factor_range = self.sharpness_factor_range
        sharpness_factor = random.random() * (sharpness_factor_range[1] - sharpness_factor_range[0]) + sharpness_factor_range[0]
        return self.sharpness_change(image, sharpness_factor)
    
    def pad(self, image, mask, padding, pad_value=0):
        # (left_padding, top_padding, right_padding, bottom_padding)
        if isinstance(padding, int):
            paddings = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            paddings = padding
        else:
            raise TypeError("Invalid type of padding, which should be int or tuple")
        padded_image = ImageOps.expand(image, paddings, fill=pad_value)
        padded_mask = ImageOps.expand(mask, paddings, fill=pad_value)
        return padded_image, padded_mask
    
    def resize(self, image, mask, param, mode='min_size'):
        # param indicates the resize parameter, and mode determines the resize mode (size or scale)
        self.check_size_matched(image, mask)
        w, h = image.size
        if mode == 'min_size':
            if not isinstance(param, int):
                raise TypeError("Invalid type of size, which should be int, if you want to resize the image with scale factor, please set mode='scale'!")
            ratio = param / float(min(h, w))
            rh = int(h * ratio)
            rw = int(w * ratio)
            size = (rw, rh)
        elif mode == 'max_size':
            if not isinstance(param, int):
                raise TypeError("Invalid type of size, which should be int, if you want to resize the image with scale factor, please set mode='scale'!")
            ratio = param / float(max(h, w))
            rh = int(h * ratio)
            rw = int(w * ratio)
            size = (rw, rh)
        elif mode == 'scale':
            rh = int(h * param)
            rw = int(w * param)
            size = (rw, rh)
        else:
            raise ValueError("Invalid mode for resize operation, you must choose from {'min_size', 'max_size', 'scale}")

        resized_image = image.resize(size, resample=Image.NEAREST)
        resized_mask = mask.resize(size, resample=Image.NEAREST)
        return resized_image, resized_mask
    
    def random_resize(self, image, mask):
        self.check_size_matched(image, mask)
        params = self.resize_scales
        mode = self.resize_mode
        param_idx = random.randint(0, len(params) - 1)
        return self.resize(image, mask, params[param_idx], mode=mode)

    def random_image_shifting(self, image, mask, pad_value=0):
        self.check_size_matched(image, mask)
        w, h = image.size
        offset_w = int(self.shift_ratio * w)
        offset_h = int(self.shift_ratio * h)
        offset = (offset_w, offset_h, offset_w, offset_h)
        padded_image, padded_mask = self.pad(image, mask, offset, pad_value=pad_value)
        y = random.randint(0, offset_w * 2)
        x = random.randint(0, offset_h * 2)
        shifted_image, shifted_mask = self.crop(padded_image, padded_mask, (w, h), x, y)
        return shifted_image, shifted_mask
    
    def random_image_cropped_rotating(self, image, mask, pad_value=0):
        self.check_size_matched(image, mask)
        w, h = image.size
        angle = random.randint(0, self.cropped_rotate_max_angle)

        # expand = True
        # r = math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        # theta = math.atan2(h, w) * 180.0 / math.pi
        # gamma1 = (theta + angle) * math.pi / 180.0
        # gamma2 = (-theta + angle) * math.pi / 180.0
        # h_new = math.ceil(r * math.sin(gamma1))
        # w_new = math.ceil(r * math.cos(gamma2))
        # padding_w = int(w_new - w / 2)
        # padding_h = int(h_new - h / 2)
        # padded_image, padded_mask = self.pad(image, mask, (padding_w, padding_h, padding_w, padding_h), pad_value=pad_value)

        # padding before rotate
        temp_image = image.convert('RGBA')
        temp_mask = mask.convert('RGBA')

        rotated_image, rotated_mask = self.rotate(temp_image, temp_mask, angle)
        padding_background = Image.new('RGBA', rotated_image.size, (pad_value,)*4)
        processed_image = Image.composite(rotated_image, padding_background, rotated_image).convert(image.mode)
        processed_mask = Image.composite(rotated_mask, padding_background, rotated_mask).convert(mask.mode)
        rw, rh  = processed_image.size
        center_w = int(0.5 * (rw - w))
        center_h = int(0.5 * (rh - h))
        d_w = int(self.cropped_rotate_max_center_distance * (rw - w))
        d_h = int(self.cropped_rotate_max_center_distance * (rh - h))
        y = random.randint(center_w - d_w, center_w + d_w)
        x = random.randint(center_h - d_h, center_h + d_h)
        cropped_image, cropped_mask = self.crop(processed_image, processed_mask, (w, h), x, y)
        return cropped_image, cropped_mask
    
    def adjust_size(self, image, mask, size):
        # adjust the size of the image so that its length and width are consistent with the set value--size
        # i.e. make the image become a square with size of (size, size)
        assert isinstance(size, int)
        self.check_size_matched(image, mask)
        o_image, o_mask = self.resize(image, mask, size, mode='max_size')
        ow, oh = o_image.size
        left_padding = (size - ow) // 2
        right_padding = size - ow - left_padding
        top_padding = (size - oh) // 2
        bottom_padding = size - oh - top_padding
        padding = (left_padding, top_padding, right_padding, bottom_padding)
        adjusted_image, adjusted_mask = self.pad(o_image, o_mask, padding)
        return adjusted_image, adjusted_mask
    
    def mosaic(self, images, masks, size, mosaic_num):
        assert len(images) == len(masks) and len(images) == mosaic_num
        assert isinstance(size, int) and isinstance(mosaic_num, int)
        assert mosaic_num == 4 or mosaic_num == 9, "Number of mosaic images can only be 4 or 9"
        idx = [i for i in range(mosaic_num)]
        random.shuffle(idx)
        num_per_size = int(math.sqrt(mosaic_num))
        output_image = Image.new('RGB', (num_per_size * size, num_per_size * size), (0, 0, 0))
        output_mask = Image.new('L', (num_per_size * size, num_per_size * size), 0)
        count = 0
        for i in idx:
            i_y, i_x = count // num_per_size, count % num_per_size
            adjusted_image, adjusted_mask = self.adjust_size(images[i], masks[i], size)
            output_image.paste(adjusted_image, (i_x * size, i_y * size, i_x * size + size, i_y * size + size))
            output_mask.paste(adjusted_mask, (i_x * size, i_y * size, i_x * size + size, i_y * size + size))
            count += 1
        return output_image, output_mask
    
    def normalize(self, image):
        image_array = np.array(image).astype(np.float)
        if len(image_array.shape) == 2:
            assert len(self.norm_means) == 1 and len(self.norm_stds) == 1
        else:
            assert len(self.norm_means) == image_array.shape[-1] and len(self.norm_stds) == image_array.shape[-1]
        ones = np.ones_like(image_array).astype(np.float)
        norm_means = np.expand_dims(np.expand_dims(np.array(self.norm_means), 0), 0).astype(np.float) * ones * 255
        norm_stds = np.expand_dims(np.expand_dims(np.array(self.norm_stds), 0), 0).astype(np.float) * ones * 255
        normed_image_array = ((image_array - norm_means) / norm_stds * 255).astype(np.uint8)
        normed_image = Image.fromarray(normed_image_array)
        return normed_image
    
    def compose_aug(self, image, mask):
        # combine a set of augments
        output_image, output_mask = image, mask
        augment_list = self.composed_augments
        for augment in augment_list:
            assert augment in ['random_crop', 'random_rotate', 'random_resize',
                                  'random_vflip', 'random_hflip', 'random_image_shifting', 'random_image_cropped_rotating',
                                  'random_brightness_change', 'random_color_change', 
                                  'random_contrast_change', 'random_sharpness_change', 'normalize']
            if augment in [ 'random_brightness_change', 'random_color_change', 
                               'random_contrast_change', 'random_sharpness_change', 'normalize']:
                output_image = eval('self.' + augment + '(output_image)')
            else:
                output_image, output_mask = eval('self.' + augment + '(output_image, output_mask)')
        
        return output_image, output_mask

if __name__ == '__main__':
    aug = OnlineImageAugmentor('cfg.yaml')