import os, shutil
import pandas as pd
import numpy as np
# from skimage import io
from tqdm import tqdm
Img_chs = 3
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
Cls = namedtuple('cls', ['name', 'id', 'color'])
Clss = [
    Cls('bg', 0, (0, 0, 0)),
    Cls('change', 1, (0, 200, 0)),
]
def img_read(img_path):
    # img = io.imread(img_path)
    img = Image.open(img_path)
    img = np.array(img)
    return img
def gt_read(img_path):
    # img = io.imread(img_path)
    img = Image.open(img_path)
    img = np.array(img)
    return img
def dir_ms_sta(imgs_path, save_path, img_chs=Img_chs):
    '''
    统计影像各通道mean、std统计值，以文件夹方式进行
    :param imgs_path: 影像文件夹路径
    :param save_path: csv结果路径
    :return:
    '''
    print('images mean std sta:')
    imgs_list = [os.path.join(imgs_path, file_name) for file_name in os.listdir(imgs_path)]
    list_ms_sta(imgs_list, save_path, img_chs=img_chs)
    print('finish\n')


def list_ms_sta(imgs_list, save_path, img_chs=Img_chs):
    '''
    统计影像各通道mean、std统计值，以list方式进行
    :param imgs_list: 影像文件路径list
    :param save_path: csv结果路径
    :param img_chs: 影像通道数
    :return:
    '''
    ms_df = pd.DataFrame(None, columns=['mean' + str(i + 1) for i in range(img_chs)] +
                                       ['std' + str(i + 1) for i in range(img_chs)])
    with tqdm(imgs_list) as pbar:
        for index, img_path in enumerate(pbar):
            img = img_read(img_path)
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            ms_df.loc[index] = np.append(mean, std)
            pbar.set_description('mean std sta ')

    ms_df.loc['Col_avg'] = ms_df.apply(lambda x: x.mean())
    ms_df.loc['Col_avgTensor'] = ms_df.loc['Col_avg'].apply(lambda x: x / 255)
    ms_df.to_csv(save_path)
    print(ms_df[-2:])
    print('save to %s' % save_path)

def dir_class_sta(gts_path, save_path, clss=Clss):
    '''
    统计gt各类数量分布，以文件夹方式进行
    :param gts_path: gt文件夹路径
    :param save_path: csv结果保存路径
    :param clss: 类别映射表
    :return:
    '''
    print('gts class sta:')
    gts_list = [os.path.join(gts_path, file_name) for file_name in os.listdir(gts_path)]
    list_class_sta(gts_list, save_path, clss=clss)
    print('finish\n')


def list_class_sta(gts_list, save_path, clss=Clss):
    '''
    统计gt各类数量分布，以list方式进行
    :param gts_list: gt文件路径list
    :param save_path: csv结果保存路径
    :param clss: 类别映射表
    :return:
    '''
    class_df = pd.DataFrame(None, columns=[cls.name for cls in clss])
    with tqdm(gts_list) as pbar:
        for index, gt_path in enumerate(pbar):
            gt = gt_read(gt_path)
            cls_sta = []
            for cls in clss:
                cls_sta.append(np.sum(gt == cls.id))
            class_df.loc[index] = cls_sta
            pbar.set_description('class sta ')

    class_df.loc['Pixel_sum'] = class_df.apply(lambda x: x.sum())
    class_df.loc['Pixel_pct'] = class_df.loc['Pixel_sum'] / class_df.loc['Pixel_sum'].sum()
    class_df.loc['Pixel_pct'].plot(kind='bar', title='Pixel_pct')
    plt.savefig(save_path.replace('.csv', '.png'))
    plt.close()

    class_df.loc['Sample_sum'] = class_df.apply(lambda x: np.sum(x > 0) - 1)
    class_df.loc['Sample_pct'] = class_df.loc['Sample_sum'] / class_df.loc['Sample_sum'].sum()
    class_df.loc['Sample_pct'].plot(kind='bar', title='Sample_pct')
    plt.savefig(save_path.replace('.csv', '_sample.png'))
    plt.close()

    class_df['filename'] = [os.path.basename(gt_path) for gt_path in gts_list] + ['', '', '', '']
    class_df.to_csv(save_path)
    class_df.pop('filename')
    print(class_df[-4:])
    print('save to %s' % save_path)


def img_merge(img1_dir, img2_dir, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    with tqdm(os.listdir(img1_dir)) as pbar:
        for file_name in pbar:
            src_path = os.path.join(img1_dir, file_name)
            dst_path = os.path.join(img_dir, file_name)
            shutil.copy(src_path, dst_path)
    with tqdm(os.listdir(img2_dir)) as pbar:
        for file_name in pbar:
            src_path = os.path.join(img2_dir, file_name)
            dst_path = os.path.join(img_dir, file_name)
            shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    pass
    # img1_dir = r'/home/aistudio/data/src/train_png/A'
    # img2_dir = r'/home/aistudio/data/src/train_png/B'
    # gt_dir = r'/home/aistudio/data/src/train_png/label'
    img1_dir = r'/home/aistudio/data/src/all/A'
    img2_dir = r'/home/aistudio/data/src/all/B'
    img1_dir_train = r'/home/aistudio/data/src/train/A'
    img2_dir_train = r'/home/aistudio/data/src/train/B'
    img1_dir_test = r'/home/aistudio/data/src/test/A'
    img2_dir_test = r'/home/aistudio/data/src/test/B'
    img_merge(img1_dir_train, img1_dir_test, img1_dir)
    img_merge(img2_dir_train, img2_dir_test, img2_dir)
    # gt_dir = r'/home/aistudio/data/src/train/label'
    dir_ms_sta(img1_dir, img1_dir+'.csv')
    dir_ms_sta(img2_dir, img2_dir+'.csv')
    # gt_dir = r'/home/aistudio/data/src/train/label'
    # dir_class_sta(gt_dir, gt_dir+'.csv')


