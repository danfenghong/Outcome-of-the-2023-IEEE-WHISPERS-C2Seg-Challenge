import numpy as np
from skimage import io
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# def soft_ensemble(logits_dir, save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     files_list = os.listdir(logits_dir[0])
#     for file_name in tqdm(files_list):
#         logit_src = np.zeros_like(np.load(os.path.join(logits_dir[0], file_name), allow_pickle=True))
#         for logit_dir in logits_dir:
#             file_path = os.path.join(logit_dir, file_name)
#             logit = np.load(file_path, allow_pickle=True)
#             logit_src += logit
#         pred = np.argmax(logit_src, axis=0)
#         pred = np.array(pred, dtype=np.uint8)
#         save_path = os.path.join(save_dir, file_name.replace('.npy', '.png'))
#         io.imsave(save_path, pred)


def hard_ensemble(logits_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files_list = os.listdir(logits_dir[0])
    for file_name in tqdm(files_list):
        logit_src = np.zeros_like(np.load(os.path.join(logits_dir[0], file_name), allow_pickle=True))
        for logit_dir in logits_dir:
            file_path = os.path.join(logit_dir, file_name)
            logit = np.load(file_path, allow_pickle=True)
            logit_src += logit
        pred = logit_src/len(logits_dir)
        pred[pred>=0.5]=1
        pred[pred<=0.5]=0
        pred = np.array(pred, dtype=np.uint8)
        save_path = os.path.join(save_dir, file_name.replace('.npy', '.png'))
        io.imsave(save_path, pred)


if __name__ == '__main__':
    pass
    logits_dir = [
        r'/home/aistudio/data/src/result_logit0',
        r'/home/aistudio/data/src/result_logit1',
        r'/home/aistudio/data/src/result_logit2',
        # r'/home/aistudio/data/src/result_logit3',
        r'/home/aistudio/data/src/result_logit4',
    ]
    save_dir = r'/home/aistudio/data/src/result_ensemble'
    # soft_ensemble(logits_dir, save_dir)
    hard_ensemble(logits_dir, save_dir)
