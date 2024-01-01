import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
np.random.seed(0)
import warnings
 
warnings.filterwarnings("ignore")

def name_img2gt(img_name):
    gt_name = img_name
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

def dataset_split(imgs_path1, imgs_path2, save_path, gts_path=None, train_rate=0.9):
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
    train_num = int(train_rate * len(files_list))
    val_num = len(files_list) - train_num
    files_list_train = files_list[:train_num]
    files_list_val = files_list[train_num:]
    # endregion

    img_train_list1 = [os.path.join(os.path.basename(imgs_path1), file_name) for file_name in files_list_train]
    img_val_list1 = [os.path.join(os.path.basename(imgs_path1), file_name) for file_name in files_list_val]
    img_train_list2 = [os.path.join(os.path.basename(imgs_path2), file_name) for file_name in files_list_train]
    img_val_list2 = [os.path.join(os.path.basename(imgs_path2), file_name) for file_name in files_list_val]

    if gts_path is not None:
        gt_train_list = [os.path.join(os.path.basename(gts_path), name_img2gt(file_name)) for file_name in files_list_train]
        gt_val_list = [os.path.join(os.path.basename(gts_path), name_img2gt(file_name)) for file_name in files_list_val]

        info_dict = {'img_path1': img_train_list1,
                'img_path2': img_train_list2,
                'gt_path': gt_train_list,
                }
        dataset_write(info_dict, save_path)
        info_dict = {'img_path1': img_val_list1,
        'img_path2': img_val_list2,
        'gt_path': gt_val_list,
        }
        dataset_write(info_dict, save_path.replace('train.txt', 'val.txt'))
        print('train rate %.2f\ntrain %d, val %d\nresult save to %s' % (train_rate, train_num, val_num, save_path))
        print('finish\n')
    else:
        info_dict = {'img_path1': img_train_list1 + img_val_list1,
                'img_path2': img_train_list2 + img_val_list2,
                }
        dataset_write(info_dict, save_path)
        print('train rate %.2f\ntrain %d, val %d\nresult save to %s' % (train_rate, train_num, val_num, save_path))
        print('finish\n')


def pre_1to255(pres_dir):
    with tqdm(glob.glob(os.path.join(pres_dir, '*.png'))) as pbar:
        for gt_path in pbar:
            img = io.imread(gt_path)
            img[img!=0] = 255
            io.imsave(gt_path, img)

if __name__ == '__main__':
    pres_dir = r'/home/aistudio/data/src/result'
    pre_1to255(pres_dir)

