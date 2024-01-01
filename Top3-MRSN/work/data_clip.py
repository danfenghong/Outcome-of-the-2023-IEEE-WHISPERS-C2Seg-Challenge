import glob, os
from skimage import io
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def img_read(img_path):
    return io.imread(img_path)
def img_save(img_path, img):
    io.imsave(img_path, img)
    # print('save to', img_path)

def imgs_clip(divide_num, files_path, files_divide_path, dst_size):
    '''
    将单张图片裁剪出多个不同图片
    :param divide_num: 1个输入图片裁剪出的样本图片个数
    :param files_path: 输入图片的文件夹路径
    :param files_divide_path: 输出样本图片的文件夹路径
    :param dst_size: 输出样本/裁剪尺寸
    :return:
    '''
 
    def img_divide(img, img_name, divide_num, save_path, out_size):
        '''
        裁剪图片，原始尺寸4N*4N，目标尺寸N*N，本文有3种方案：
        按照4*4，无重复裁剪，得到16个样本
        按照4*4+3*3，在横纵方向各偏移0.5N距离，得到16+9个样本
        按照4*4+3*3+3*4+4*3，在横纵方向各偏移0.5N距离，在横向偏移0.5N距离纵向不偏移，在纵向偏移0.5N距离横向不偏移，得到16+9+12+12个样本
        :param img:
        :param img_name:
        :param divide_num:
        :param save_path:
        :param out_size:
        :return:
        '''
        if divide_num == 16:
            # 4*4
            for i in range(4):
                for j in range(4):
                    dst = np.array(img[out_size * j:out_size * (j + 1), out_size * i:out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 4 + j).zfill(2))
                    img_save(os.path.join(save_path, dst_name), dst)
        elif divide_num == 4:
            # 2*2
            for i in range(2):
                for j in range(2):
                    dst = np.array(img[out_size * j:out_size * (j + 1), out_size * i:out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 2 + j).zfill(2))
                    img_save(os.path.join(save_path, dst_name), dst)
        elif divide_num == 25:
            diff = int(out_size * 0.5)
            # 4*4
            for i in range(4):
                for j in range(4):
                    dst = np.array(img[out_size * j:out_size * (j + 1), out_size * i:out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 4 + j).zfill(2))
                    img_save(os.path.join(save_path, dst_name), dst)
            # 3*3
            for i in range(3):
                for j in range(3):
                    dst = np.array(
                        img[diff + out_size * j:diff + out_size * (j + 1),
                        diff + out_size * i:diff + out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 3 + j + 16).zfill(2))  # 保存序号增大
                    img_save(os.path.join(save_path, dst_name), dst)
        elif divide_num == 49:
            diff = int(out_size * 0.5)
            # 4*4
            for i in range(4):
                for j in range(4):
                    dst = np.array(img[out_size * j:out_size * (j + 1), out_size * i:out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 4 + j).zfill(2))
                    img_save(os.path.join(save_path, dst_name), dst)
            # 3*3
            for i in range(3):
                for j in range(3):
                    dst = np.array(
                        img[diff + out_size * j:diff + out_size * (j + 1),
                        diff + out_size * i:diff + out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 3 + j + 16).zfill(2))  # 保存序号增大
                    img_save(os.path.join(save_path, dst_name), dst)
            # 4*3
            for i in range(4):
                for j in range(3):
                    dst = np.array(
                        img[diff + out_size * j:diff + out_size * (j + 1), out_size * i:out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 3 + j + 25).zfill(2))  # 保存序号增大
                    img_save(os.path.join(save_path, dst_name), dst)
            # 3*4
            for i in range(3):
                for j in range(4):
                    dst = np.array(
                        img[out_size * j:out_size * (j + 1), diff + out_size * i:diff + out_size * (i + 1)])
                    dst_name = nameAdd(img_name, '_'+str(i * 4 + j + 37).zfill(2))  # 保存序号增大
                    img_save(os.path.join(save_path, dst_name), dst)
 
    files = glob.glob(os.path.join(files_path, '*.png'))+glob.glob(os.path.join(files_path, '*.tif'))
    if not os.path.exists(files_divide_path):
        os.makedirs(files_divide_path)
    with tqdm(files) as pbar:
        for file_path in pbar:
            img = img_read(file_path)
            img_name = os.path.basename(file_path)
            img_divide(img, img_name, divide_num, files_divide_path, dst_size)
 
 
def nameAdd(src_name, suf_fix, pre_fix=''):
    filename, extension = os.path.splitext(src_name)
    return pre_fix + filename + suf_fix + extension
 
if __name__ == '__main__':
    pass
    img1_dir = r'/home/aistudio/data/src/stage1/train/Image1'
    img2_dir = r'/home/aistudio/data/src/stage1/train/Image2'
    gt_dir = r'/home/aistudio/data/src/stage1/train/label1'
    img1_dir_clip = r'/home/aistudio/data/src/stage1/train_clip/Image1'
    img2_dir_clip = r'/home/aistudio/data/src/stage1/train_clip/Image2'
    gt_dir_clip = r'/home/aistudio/data/src/stage1/train_clip/label1'
    imgs_clip(4, img1_dir, img1_dir_clip, 256)
    imgs_clip(4, img2_dir, img2_dir_clip, 256)
    imgs_clip(4, gt_dir, gt_dir_clip, 256)

