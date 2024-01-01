import os, glob, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
np.random.seed(0)
import warnings
 
warnings.filterwarnings("ignore")

def nameAdd(src_path, suf_fix='', pre_fix=''):
    '''
    路径名添加前后缀，支持文件路径、文件夹路径，支持绝对路径、相对路径
    :param src_path: 路径名
    :param suf_fix: 后缀
    :param pre_fix: 前缀
    :return:
    '''
    if os.path.isabs(src_path):
        src_name = os.path.basename(src_path)
        src_path = os.path.dirname(src_path)
        filename, extension = os.path.splitext(src_name)
        dst_name = pre_fix + filename + suf_fix + extension
        dst_path = os.path.join(src_path, dst_name)
    else:
        file_name, extension = os.path.splitext(src_path)
        dst_path = pre_fix + file_name + suf_fix + extension
    return dst_path


def name_img2gt(img_name):
    gt_name = img_name.replace('.tif', '.png')
    return gt_name

def dataset_write(dataset_dict, csv_path):
    '''
    写入数据集split.csv
    :param dataset_dict: 数据集字典
    :param csv_path: csv路径
    :return:
    '''
    df = pd.DataFrame(dataset_dict,)
    df.to_csv(csv_path, header=None, index=None, sep=' ')

def dataset_split_kfold(imgs_path1, imgs_path2, save_path, gts_path=None, k=5):
    '''
    根据img文件列表，随机划分为不同数据集，并写入csv中
    :param imgs_path1: img1路径
    :param imgs_path2: img2路径
    :param gts_path: gt路径
    :param save_path: csv保存路径
    :param train_rate: 训练集比例
    :return:
    '''
    print('dataset split:')
    files_list = os.listdir(imgs_path1)

    # region 普通的随机打乱，val样本可能会与train样本有部分图像块上的重叠
    np.random.shuffle(files_list)
    num = len(files_list)//k


    for i in range(k):
        if i == k-1:
            files_list_train = files_list[:-num]
            files_list_val = files_list[-num:]  
        else:
            files_list_train = files_list[:num*i] + files_list[num*i+num:]
            files_list_val = files_list[num*i:num*i+num] 

        img_train_list1 = [os.path.join(os.path.basename(imgs_path1), file_name) for file_name in files_list_train]
        img_val_list1 = [os.path.join(os.path.basename(imgs_path1), file_name) for file_name in files_list_val]
        img_train_list2 = [os.path.join(os.path.basename(imgs_path2), file_name) for file_name in files_list_train]
        img_val_list2 = [os.path.join(os.path.basename(imgs_path2), file_name) for file_name in files_list_val]


        gt_train_list = [os.path.join(os.path.basename(gts_path), name_img2gt(file_name)) for file_name in files_list_train]
        gt_val_list = [os.path.join(os.path.basename(gts_path), name_img2gt(file_name)) for file_name in files_list_val]

        info_dict = {'img_path1': img_train_list1,
                'img_path2': img_train_list2,
                'gt_path': gt_train_list,
                }
        save_path_i = nameAdd(save_path, '_%d'%i)
        dataset_write(info_dict, save_path_i)
        info_dict = {'img_path1': img_val_list1,
        'img_path2': img_val_list2,
        'gt_path': gt_val_list,
        }
        dataset_write(info_dict, save_path_i.replace('train_%d.txt'%i, 'val_%d.txt'%i))
        print('split fold  %d\ntrain %d, val %d\nresult save to %s' % (k, len(files_list)-num, num,  nameAdd(save_path, '_%d'%i)))
        print('finish\n')


if __name__ == '__main__':
    img1_dir = r'/home/aistudio/data/src/stage1/train_clip/Image1'
    img2_dir =  r'/home/aistudio/data/src/stage1/train_clip/Image2'
    gt_dir = r'/home/aistudio/data/src/stage1/train_clip/label1'
    dst_path = r'/home/aistudio/work/clip_train.txt'
    dataset_split_kfold(img1_dir, img2_dir, dst_path, gt_dir)

