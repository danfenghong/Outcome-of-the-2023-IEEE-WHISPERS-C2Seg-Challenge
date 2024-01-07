import os, time
import argparse
import torch
from torch.backends import cudnn
from utils.hsicropsolver import Solver
from utils.visualize import Visualizer

import torch.distributed as dist

RUN_ROOT = '/Top2-Seg-HRN/run'

def main(opt):
    cudnn.benchmark = True
    # torch.cuda.set_device(opt.local_rank)
    # dist.init_process_group(backend='nccl')

    if opt.mode == 'train':
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path = os.path.join(RUN_ROOT, 'train', time_stamp)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            torch.cuda.set_device(opt.local_rank)
            dist.init_process_group(backend='nccl')
            if dist.get_rank() in [-1, 0]:
                os.mkdir(path)
                dist.barrier()
            else:
                dist.barrier()
        else:
            os.mkdir(path)
        solver = Solver(opt, path)
        solver.train()
    elif opt.mode == 'test':
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            raise ValueError("test mode does not support ddp!")
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path = os.path.join(RUN_ROOT, 'test', time_stamp)
        os.mkdir(path)
        solver = Solver(opt, path)
        solver.inference()
    elif opt.mode == 'vis':
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            raise ValueError("vis mode does not support ddp!")
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path = os.path.join(RUN_ROOT, 'vis', time_stamp)
        os.mkdir(path)
        visualizer = Visualizer(opt, path)
        visualizer.vis_features()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='', help='the config file used to set training process')
    parser.add_argument('--resume_path', type=str, default='',help='the checkpoint file for loading')
    parser.add_argument('--mode', type=str, default='train', help='mode of process')

    # distributed training parameters
    parser.add_argument("--local_rank", default=-1, type=int)

    opt = parser.parse_args()
    main(opt)