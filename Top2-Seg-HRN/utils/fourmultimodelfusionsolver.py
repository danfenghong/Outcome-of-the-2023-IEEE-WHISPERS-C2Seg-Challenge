from functools import cmp_to_key
import os, time, sys, yaml, math
from numpy.lib.npyio import savetxt
from typing import IO
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .cropdataset import RoadDataset, SequentialDistributedSampler
from .criterion import ConfusionMatrixBasedMetric
from .function import BinaryComposedLoss, BinaryDiceLoss, CELoss, OhemCELoss, FocalLoss, MulticlassDiceLoss, ComposedLoss, ComposedFocalLoss, BCELoss
from .scheduler import GradualWarmupScheduler
import models
from shutil import copyfile

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


LOG_ROOT = '/Top2-Seg-HRN/run'

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    # print(x.shape,"x.shape")
    x = x.contiguous().view(-1, *xsize[dim:])
    x = x.contiguous().view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
    -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.contiguous().view(xsize)
def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x
class Solver:
    def __init__(self, args, log_path):
        self.opt = args
        ### config 1
        cfg_file = open(self.opt.config, 'r')
        self.config = yaml.load(cfg_file, Loader=yaml.FullLoader)
        cfg_file.close()
        self.log_path = log_path
        self.writter = SummaryWriter(os.path.join(self.log_path, 'log'))
        self.dst_cfg_fn = os.path.join(self.log_path, os.path.basename(self.opt.config))
        copyfile(self.opt.config, self.dst_cfg_fn)

        # environment
        # os.environ["CUDA_VISIBLE_DEVICES"] = config['ENV']['DEVICES']
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.local_rank = self.opt.local_rank
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.use_ddp = True
                device_count = torch.cuda.device_count()
                import logging
                logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
            else:
                self.device = torch.device("cuda")
                self.use_ddp = False
                device_count = 1
        else:
            self.device = torch.device("cpu")
            self.use_ddp = False
            device_count = 1

        print(self.opt.config)

        aug_config_path = self.config['DATASET']['AUG_CONFIG_PATH']
        train_dataset = RoadDataset(self.config['DATASET']['TRAIN_DATASET_CSV'], aug_config_path, 
                                    use_aug=self.config['DATASET']['USE_AUG'], mode='train')
        print(train_dataset[0][0].shape,"train_dataset[0].shape")
        self.train_size = train_dataset.chip_size
        # print(self.train_size,"self.train_size")
        val_dataset = RoadDataset(self.config['DATASET']['VAL_DATASET_CSV'], aug_config_path, use_aug=False, mode='val')
        test_dataset = RoadDataset(self.config['DATASET']['TEST_DATASET_CSV'], mode='test')
        val_batchsize = 1
        self.num_batches_per_epoch_for_training = len(train_dataset) // self.config['TRAIN']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_validating = len(val_dataset) // val_batchsize // device_count
        self.num_batches_per_epoch_for_testing = len(test_dataset)
        
        if self.use_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = SequentialDistributedSampler(val_dataset, batch_size=1)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(train_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(val_dataset)

        self.train_loader = DataLoader(dataset=train_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=self.config['TRAIN']['BATCH_SIZE'], sampler=self.train_sampler, drop_last=True)
        self.val_loader = DataLoader(dataset=val_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=val_batchsize, sampler=self.val_sampler)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        # model
        # print("self.opt.resume_path",self.opt.resume_path)
        # print("self.config['MODEL']['IMG_CH']",self.config['MODEL']['IMG_CH'])
        # whether to load checkpoint
        if self.opt.resume_path == '':
            print('model was not created')
            self.model = models.build_model(self.config['MODEL']['IMG_CH'], self.config['MODEL']['N_CLASSES'], model_key=self.config['MODEL']['NAME'], 
                                            backbone=self.config['MODEL']['BACKBONE'], pretrained_flag=self.config['MODEL']['PRETRAINED'], 
                                            resume_path=self.config['MODEL']['RESUME_PATH']).to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.config['TRAIN']['LR']['BACKBONE_RATE'],
                },
            ]
            if self.use_ddp:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.config['TRAIN']['OPTIMIZER'] == 'SGD':
                self.optimizer = optim.SGD(param_dicts, self.config['TRAIN']['LR']['RATE'], 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config['TRAIN']['OPTIMIZER'] == 'Adam':
                self.optimizer = optim.Adam(param_dicts, self.config['TRAIN']['LR']['RATE'] * 0.1, 
                                            [self.config['TRAIN']['LR']['BETA1'], self.config['TRAIN']['LR']['BETA2']], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config['TRAIN']['OPTIMIZER'] == 'AdamW':
                self.optimizer = optim.AdamW(param_dicts, self.config['TRAIN']['LR']['RATE'] * 0.1, weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            else:
                raise ValueError("wrong indicator for optimizer, which should be selected from {'SGD', 'Adam', 'AdamW'}")
            self.current_epoch = 0
            self.last_acc = -999.0
            self.best_acc = -999.0

        elif os.path.exists(self.opt.resume_path):
            print('%s was existed' % (self.opt.resume_path))
            self.model = models.build_model(self.config['MODEL']['IMG_CH'], self.config['MODEL']['N_CLASSES'], 
                                            backbone=self.config['MODEL']['BACKBONE'], model_key=self.config['MODEL']['NAME']).to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.config['TRAIN']['LR']['BACKBONE_RATE'],
                },
            ]
            if self.config['TRAIN']['OPTIMIZER'] == 'SGD':
                self.optimizer = optim.SGD(param_dicts, self.config['TRAIN']['LR']['RATE'], 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config['TRAIN']['OPTIMIZER'] == 'Adam':
                self.optimizer = optim.Adam(param_dicts, self.config['TRAIN']['LR']['RATE'] * 0.1, 
                                            [self.config['TRAIN']['LR']['BETA1'], self.config['TRAIN']['LR']['BETA2']], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config['TRAIN']['OPTIMIZER'] == 'AdamW':
                self.optimizer = optim.AdamW(param_dicts, self.config['TRAIN']['LR']['RATE'] * 0.1, weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            else:
                raise ValueError("wrong indicator for optimizer, which should be selected from {'SGD', 'Adam', 'AdamW'}")
            # if self.use_ddp:
            #     self.load_checkpoint(self.opt.resume_path, rank=self.local_rank)
            #     self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            #     self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            # else:
            # self.load_checkpoint(self.opt.resume_path,self.opt.resume_path1)
        else:
            raise FileNotFoundError(self.opt.resume_path + " not existed")
        
        # if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
        #     gpu_ids = [i for i in range(torch.cuda.device_count())]
        #     self.model = nn.DataParallel(self.model, device_ids=gpu_ids)  # 多gpu训练,自动选择gpu
        #     self.optimizer = nn.DataParallel(self.optimizer, device_ids=gpu_ids)
        if self.config['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
            self.base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "max", patience=self.config['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
        elif self.config['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
            milestones = [int(self.config['TRAIN']['EPOCHS'] * self.config['TRAIN']['SCHEDULER']['MILESTONES'][i])
                          for i in range(len(self.config['TRAIN']['SCHEDULER']['MILESTONES']))]
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)

        if self.config['TRAIN']['SCHEDULER']['WARMUP']:
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=int(self.config['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler

        if self.config['TRAIN']['LOSS']['WEIGHT'] == 'None':
            self.class_weight = None
        else:
            self.class_weight = self.config['TRAIN']['LOSS']['WEIGHT']

        if self.config['TRAIN']['LOSS']['NAME'] == 'CELoss':
            self.loss = CELoss(reduction='mean', ignore_index=0, weight=self.class_weight).to(self.device)
            # self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'OhemCELoss':
            self.loss = OhemCELoss(self.config['TRAIN']['LOSS']['THRESH'], self.config['TRAIN']['BATCH_SIZE'] // 5, reduction='mean', ignore_index=0, weight=self.class_weight).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'FocalLoss':
            self.loss = FocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'MulticlassDiceLoss':
            self.loss = MulticlassDiceLoss(reduction='mean', ignore_index=0, weight=self.class_weight).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'ComposedLoss':
            self.loss = ComposedLoss(reduction='mean', ignore_index=0, weight=self.class_weight).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'ComposedFocalLoss':
            self.loss = ComposedFocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'BCELoss':
            self.loss = BCELoss(reduction='mean').to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'BinaryDiceLoss':
            self.loss = BinaryDiceLoss(reduction='mean').to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'BinaryComposedLoss':
            self.loss = BinaryComposedLoss(reduction='mean').to(self.device)
        else:
            raise ValueError("wrong indicator for loss function, which should be selected from {'CELoss', 'OhemCELoss', 'FocalLoss', 'MulticlassDiceLoss', 'BCELoss', 'BinaryDiceLoss', 'BinaryComposedLoss'}")

        ######################## config 1 ########################
        cfg_file1 = open(self.opt.config1, 'r')
        self.config1 = yaml.load(cfg_file1, Loader=yaml.FullLoader)
        cfg_file1.close()
        self.log_path1 = log_path
        self.writter = SummaryWriter(os.path.join(self.log_path1, 'log'))
        self.dst_cfg_fn1 = os.path.join(self.log_path1, os.path.basename(self.opt.config1))
        copyfile(self.opt.config1, self.dst_cfg_fn1)

        # environment
        # os.environ["CUDA_VISIBLE_DEVICES"] = config['ENV']['DEVICES']
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.local_rank = self.opt.local_rank
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.use_ddp = True
                device_count = torch.cuda.device_count()
                import logging
                logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
            else:
                self.device = torch.device("cuda")
                self.use_ddp = False
                device_count = 1
        else:
            self.device = torch.device("cpu")
            self.use_ddp = False
            device_count = 1

        print(self.opt.config1)

        aug_config_path1 = self.config1['DATASET']['AUG_CONFIG_PATH']
        train_dataset1 = RoadDataset(self.config1['DATASET']['TRAIN_DATASET_CSV'], aug_config_path1, 
                                    use_aug=self.config1['DATASET']['USE_AUG'], mode='train')
        self.train_size1 = train_dataset1.chip_size
        val_dataset1 = RoadDataset(self.config1['DATASET']['VAL_DATASET_CSV'], aug_config_path1, use_aug=False, mode='val')
        test_dataset1 = RoadDataset(self.config1['DATASET']['TEST_DATASET_CSV'], mode='test')
        val_batchsize = 1
        self.num_batches_per_epoch_for_training = len(train_dataset1) // self.config1['TRAIN']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_validating = len(val_dataset1) // val_batchsize // device_count
        self.num_batches_per_epoch_for_testing = len(test_dataset1)
        
        if self.use_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = SequentialDistributedSampler(val_dataset, batch_size=1)
        else:
            self.train_sampler1 = torch.utils.data.RandomSampler(train_dataset1)
            self.val_sampler1 = torch.utils.data.SequentialSampler(val_dataset1)

        self.train_loader1 = DataLoader(dataset=train_dataset1, num_workers=self.config1['TRAIN']['NUM_WORKERS'], batch_size=self.config1['TRAIN']['BATCH_SIZE'], sampler=self.train_sampler1, drop_last=True)
        self.val_loader1 = DataLoader(dataset=val_dataset1, num_workers=self.config1['TRAIN']['NUM_WORKERS'], batch_size=val_batchsize, sampler=self.val_sampler1)
        self.test_loader1 = DataLoader(dataset=test_dataset1, batch_size=1)

        # model
        print('self.opt.resume_path1',self.opt.resume_path1)
        # exit()
        if os.path.exists(self.opt.resume_path1):
            print('%s was existed' % (self.opt.resume_path1))
            self.model1 = models.build_model(self.config1['MODEL']['IMG_CH'], self.config1['MODEL']['N_CLASSES'], 
                                            backbone=self.config1['MODEL']['BACKBONE'], model_key=self.config1['MODEL']['NAME']).to(self.device)
            param_dicts1 = [
                {"params": [p for n, p in self.model1.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.model1.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.config1['TRAIN']['LR']['BACKBONE_RATE'],
                },
            ]
            if self.config1['TRAIN']['OPTIMIZER'] == 'SGD':
                self.optimizer1 = optim.SGD(param_dicts1, self.config1['TRAIN']['LR']['RATE'], 
                                           momentum=self.config1['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config1['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config1['TRAIN']['OPTIMIZER'] == 'Adam':
                self.optimizer1 = optim.Adam(param_dicts1, self.config1['TRAIN']['LR']['RATE'] * 0.1, 
                                            [self.config1['TRAIN']['LR']['BETA1'], self.config1['TRAIN']['LR']['BETA2']], weight_decay=self.config1['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config1['TRAIN']['OPTIMIZER'] == 'AdamW':
                self.optimizer1 = optim.AdamW(param_dicts1, self.config1['TRAIN']['LR']['RATE'] * 0.1, weight_decay=self.config1['TRAIN']['LR']['WEIGHT_DECAY'])
            else:
                raise ValueError("wrong indicator for optimizer, which should be selected from {'SGD', 'Adam', 'AdamW'}")
            # if self.use_ddp:
            #     self.load_checkpoint(self.opt.resume_path1, rank=self.local_rank)
            #     self.model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model1).to(self.device)
            #     self.model1 = DDP(self.model1, device_ids=[self.local_rank], output_device=self.local_rank)
            # else:
            # self.load_checkpoint(self.opt.resume_path,self.opt.resume_path1)
        else:
            raise FileNotFoundError(self.opt.resume_path + " not existed")


        if self.config1['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
            self.base_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, "max", patience=self.config1['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config1['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config1['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
        elif self.config1['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
            milestones1 = [int(self.config1['TRAIN']['EPOCHS'] * self.config1['TRAIN']['SCHEDULER']['MILESTONES'][i])
                          for i in range(len(self.config1['TRAIN']['SCHEDULER']['MILESTONES']))]
            self.base_scheduler1 = optim.lr_scheduler.MultiStepLR(self.optimizer1, milestones=milestones1, gamma=0.2)

        if self.config['TRAIN']['SCHEDULER']['WARMUP']:
            self.scheduler1 = GradualWarmupScheduler(self.optimizer1, multiplier=1, total_epoch=int(self.config1['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler1)
        else:
            self.scheduler1 = self.base_scheduler1

        if self.config1['TRAIN']['LOSS']['WEIGHT'] == 'None':
            self.class_weight1 = None
        else:
            self.class_weight1 = self.config1['TRAIN']['LOSS']['WEIGHT']

        if self.config1['TRAIN']['LOSS']['NAME'] == 'CELoss':
            self.loss1 = CELoss(reduction='mean', ignore_index=0, weight=self.class_weight1).to(self.device)
            # self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0).to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'OhemCELoss':
            self.loss1 = OhemCELoss(self.config['TRAIN']['LOSS']['THRESH'], self.config1['TRAIN']['BATCH_SIZE'] // 5, reduction='mean', ignore_index=0, weight=self.class_weight1).to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'FocalLoss':
            self.loss1 = FocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight1).to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'MulticlassDiceLoss':
            self.loss1 = MulticlassDiceLoss(reduction='mean', ignore_index=0, weight=self.class_weight1).to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'ComposedLoss':
            self.loss1 = ComposedLoss(reduction='mean', ignore_index=0, weight=self.class_weight1).to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'ComposedFocalLoss':
            self.loss1 = ComposedFocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight1).to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'BCELoss':
            self.loss1 = BCELoss(reduction='mean').to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'BinaryDiceLoss':
            self.loss1 = BinaryDiceLoss(reduction='mean').to(self.device)
        elif self.config1['TRAIN']['LOSS']['NAME'] == 'BinaryComposedLoss':
            self.loss1 = BinaryComposedLoss(reduction='mean').to(self.device)
        else:
            raise ValueError("wrong indicator for loss function, which should be selected from {'CELoss', 'OhemCELoss', 'FocalLoss', 'MulticlassDiceLoss', 'BCELoss', 'BinaryDiceLoss', 'BinaryComposedLoss'}")

        ######################## config 2 ########################
        cfg_file2 = open(self.opt.config2, 'r')
        self.config2 = yaml.load(cfg_file2, Loader=yaml.FullLoader)
        cfg_file2.close()
        self.log_path2 = log_path
        self.writter = SummaryWriter(os.path.join(self.log_path2, 'log'))
        self.dst_cfg_fn2 = os.path.join(self.log_path2, os.path.basename(self.opt.config2))
        copyfile(self.opt.config2, self.dst_cfg_fn2)

        # environment
        # os.environ["CUDA_VISIBLE_DEVICES"] = config['ENV']['DEVICES']
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.local_rank = self.opt.local_rank
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.use_ddp = True
                device_count = torch.cuda.device_count()
                import logging
                logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
            else:
                self.device = torch.device("cuda")
                self.use_ddp = False
                device_count = 1
        else:
            self.device = torch.device("cpu")
            self.use_ddp = False
            device_count = 1

        print(self.opt.config2)

        aug_config_path2 = self.config2['DATASET']['AUG_CONFIG_PATH']
        train_dataset2 = RoadDataset(self.config2['DATASET']['TRAIN_DATASET_CSV'], aug_config_path2, 
                                    use_aug=self.config2['DATASET']['USE_AUG'], mode='train')
        self.train_size2 = train_dataset2.chip_size
        val_dataset2 = RoadDataset(self.config2['DATASET']['VAL_DATASET_CSV'], aug_config_path2, use_aug=False, mode='val')
        test_dataset2 = RoadDataset(self.config2['DATASET']['TEST_DATASET_CSV'], mode='test')
        val_batchsize = 1
        self.num_batches_per_epoch_for_training = len(train_dataset2) // self.config2['TRAIN']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_validating = len(val_dataset2) // val_batchsize // device_count
        self.num_batches_per_epoch_for_testing = len(test_dataset2)
        
        if self.use_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = SequentialDistributedSampler(val_dataset, batch_size=1)
        else:
            self.train_sampler2 = torch.utils.data.RandomSampler(train_dataset2)
            self.val_sampler2 = torch.utils.data.SequentialSampler(val_dataset2)

        self.train_loader2 = DataLoader(dataset=train_dataset2, num_workers=self.config2['TRAIN']['NUM_WORKERS'], batch_size=self.config2['TRAIN']['BATCH_SIZE'], sampler=self.train_sampler2)
        self.val_loader2 = DataLoader(dataset=val_dataset2, num_workers=self.config2['TRAIN']['NUM_WORKERS'], batch_size=val_batchsize, sampler=self.val_sampler2)
        self.test_loader2 = DataLoader(dataset=test_dataset2, batch_size=1)

        # model
        if os.path.exists(self.opt.resume_path2):
            print('%s was existed' % (self.opt.resume_path2))
            self.model2 = models.build_model(self.config2['MODEL']['IMG_CH'], self.config2['MODEL']['N_CLASSES'], 
                                            backbone=self.config2['MODEL']['BACKBONE'], model_key=self.config2['MODEL']['NAME']).to(self.device)
            param_dicts2 = [
                {"params": [p for n, p in self.model2.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.model2.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.config2['TRAIN']['LR']['BACKBONE_RATE'],
                },
            ]
            if self.config2['TRAIN']['OPTIMIZER'] == 'SGD':
                self.optimizer2 = optim.SGD(param_dicts2, self.config2['TRAIN']['LR']['RATE'], 
                                           momentum=self.config2['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config2['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config2['TRAIN']['OPTIMIZER'] == 'Adam':
                self.optimizer2 = optim.Adam(param_dicts2, self.config2['TRAIN']['LR']['RATE'] * 0.1, 
                                            [self.config2['TRAIN']['LR']['BETA1'], self.config2['TRAIN']['LR']['BETA2']], weight_decay=self.config2['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config2['TRAIN']['OPTIMIZER'] == 'AdamW':
                self.optimizer2 = optim.AdamW(param_dicts2, self.config2['TRAIN']['LR']['RATE'] * 0.1, weight_decay=self.config2['TRAIN']['LR']['WEIGHT_DECAY'])
            else:
                raise ValueError("wrong indicator for optimizer, which should be selected from {'SGD', 'Adam', 'AdamW'}")
            # if self.use_ddp:
            #     self.load_checkpoint(self.opt.resume_path2, rank=self.local_rank)
            #     self.model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model2).to(self.device)
            #     self.model2 = DDP(self.model2, device_ids=[self.local_rank], output_device=self.local_rank)
            # else:
            #     self.load_checkpoint(self.opt.resume_path,self.opt.resume_path1,self.opt.resume_path2)
        else:
            raise FileNotFoundError(self.opt.resume_path + " not existed")


        if self.config2['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
            self.base_scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer2, "max", patience=self.config2['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config2['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config2['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
        elif self.config2['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
            milestones2 = [int(self.config2['TRAIN']['EPOCHS'] * self.config2['TRAIN']['SCHEDULER']['MILESTONES'][i])
                          for i in range(len(self.config2['TRAIN']['SCHEDULER']['MILESTONES']))]
            self.base_scheduler2 = optim.lr_scheduler.MultiStepLR(self.optimizer2, milestones=milestones2, gamma=0.2)

        if self.config['TRAIN']['SCHEDULER']['WARMUP']:
            self.scheduler2 = GradualWarmupScheduler(self.optimizer2, multiplier=1, total_epoch=int(self.config2['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler2)
        else:
            self.scheduler2 = self.base_scheduler2

        if self.config2['TRAIN']['LOSS']['WEIGHT'] == 'None':
            self.class_weight2 = None
        else:
            self.class_weight2 = self.config2['TRAIN']['LOSS']['WEIGHT']

        if self.config2['TRAIN']['LOSS']['NAME'] == 'CELoss':
            self.loss2 = CELoss(reduction='mean', ignore_index=0, weight=self.class_weight2).to(self.device)
            # self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0).to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'OhemCELoss':
            self.loss2 = OhemCELoss(self.config['TRAIN']['LOSS']['THRESH'], self.config2['TRAIN']['BATCH_SIZE'] // 5, reduction='mean', ignore_index=0, weight=self.class_weight2).to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'FocalLoss':
            self.loss2 = FocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight2).to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'MulticlassDiceLoss':
            self.loss2 = MulticlassDiceLoss(reduction='mean', ignore_index=0, weight=self.class_weight2).to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'ComposedLoss':
            self.loss2 = ComposedLoss(reduction='mean', ignore_index=0, weight=self.class_weight2).to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'ComposedFocalLoss':
            self.loss2 = ComposedFocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight2).to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'BCELoss':
            self.loss2 = BCELoss(reduction='mean').to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'BinaryDiceLoss':
            self.loss2 = BinaryDiceLoss(reduction='mean').to(self.device)
        elif self.config2['TRAIN']['LOSS']['NAME'] == 'BinaryComposedLoss':
            self.loss2 = BinaryComposedLoss(reduction='mean').to(self.device)
        else:
            raise ValueError("wrong indicator for loss function, which should be selected from {'CELoss', 'OhemCELoss', 'FocalLoss', 'MulticlassDiceLoss', 'BCELoss', 'BinaryDiceLoss', 'BinaryComposedLoss'}")

        ######################## config 3 ########################
        cfg_file3 = open(self.opt.config3, 'r')
        self.config3 = yaml.load(cfg_file3, Loader=yaml.FullLoader)
        cfg_file3.close()
        self.log_path3 = log_path
        self.writter = SummaryWriter(os.path.join(self.log_path3, 'log'))
        self.dst_cfg_fn3 = os.path.join(self.log_path3, os.path.basename(self.opt.config3))
        copyfile(self.opt.config3, self.dst_cfg_fn3)

        # environment
        # os.environ["CUDA_VISIBLE_DEVICES"] = config['ENV']['DEVICES']
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.local_rank = self.opt.local_rank
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.use_ddp = True
                device_count = torch.cuda.device_count()
                import logging
                logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
            else:
                self.device = torch.device("cuda")
                self.use_ddp = False
                device_count = 1
        else:
            self.device = torch.device("cpu")
            self.use_ddp = False
            device_count = 1

        print(self.opt.config3)

        aug_config_path3 = self.config3['DATASET']['AUG_CONFIG_PATH']
        train_dataset3 = RoadDataset(self.config3['DATASET']['TRAIN_DATASET_CSV'], aug_config_path3, 
                                    use_aug=self.config3['DATASET']['USE_AUG'], mode='train')
        self.train_size3 = train_dataset3.chip_size
        val_dataset3 = RoadDataset(self.config3['DATASET']['VAL_DATASET_CSV'], aug_config_path3, use_aug=False, mode='val')
        test_dataset3 = RoadDataset(self.config3['DATASET']['TEST_DATASET_CSV'], mode='test')
        val_batchsize = 1
        self.num_batches_per_epoch_for_training = len(train_dataset3) // self.config3['TRAIN']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_validating = len(val_dataset3) // val_batchsize // device_count
        self.num_batches_per_epoch_for_testing = len(test_dataset3)
        
        if self.use_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = SequentialDistributedSampler(val_dataset, batch_size=1)
        else:
            self.train_sampler3 = torch.utils.data.RandomSampler(train_dataset3)
            self.val_sampler3 = torch.utils.data.SequentialSampler(val_dataset3)

        self.train_loader3 = DataLoader(dataset=train_dataset3, num_workers=self.config3['TRAIN']['NUM_WORKERS'], batch_size=self.config3['TRAIN']['BATCH_SIZE'], sampler=self.train_sampler3)
        self.val_loader3 = DataLoader(dataset=val_dataset3, num_workers=self.config3['TRAIN']['NUM_WORKERS'], batch_size=val_batchsize, sampler=self.val_sampler3)
        self.test_loader3 = DataLoader(dataset=test_dataset3, batch_size=1)

        # model
        if os.path.exists(self.opt.resume_path3):
            print('%s was existed' % (self.opt.resume_path3))
            self.model3 = models.build_model(self.config3['MODEL']['IMG_CH'], self.config3['MODEL']['N_CLASSES'], 
                                            backbone=self.config3['MODEL']['BACKBONE'], model_key=self.config3['MODEL']['NAME']).to(self.device)
            param_dicts3 = [
                {"params": [p for n, p in self.model3.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.model3.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.config3['TRAIN']['LR']['BACKBONE_RATE'],
                },
            ]
            if self.config3['TRAIN']['OPTIMIZER'] == 'SGD':
                self.optimizer3 = optim.SGD(param_dicts3, self.config3['TRAIN']['LR']['RATE'], 
                                           momentum=self.config3['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config3['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config3['TRAIN']['OPTIMIZER'] == 'Adam':
                self.optimizer3 = optim.Adam(param_dicts3, self.config3['TRAIN']['LR']['RATE'] * 0.1, 
                                            [self.config3['TRAIN']['LR']['BETA1'], self.config3['TRAIN']['LR']['BETA3']], weight_decay=self.config3['TRAIN']['LR']['WEIGHT_DECAY'])
            elif self.config3['TRAIN']['OPTIMIZER'] == 'AdamW':
                self.optimizer3 = optim.AdamW(param_dicts3, self.config3['TRAIN']['LR']['RATE'] * 0.1, weight_decay=self.config3['TRAIN']['LR']['WEIGHT_DECAY'])
            else:
                raise ValueError("wrong indicator for optimizer, which should be selected from {'SGD', 'Adam', 'AdamW'}")
            if self.use_ddp:
                self.load_checkpoint(self.opt.resume_path3, rank=self.local_rank)
                self.model3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model3).to(self.device)
                self.model3 = DDP(self.model3, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.load_checkpoint(self.opt.resume_path,self.opt.resume_path1,self.opt.resume_path2,self.opt.resume_path3)
        else:
            raise FileNotFoundError(self.opt.resume_path + " not existed")


        if self.config3['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
            self.base_scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer3, "max", patience=self.config3['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config3['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config3['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
        elif self.config3['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
            milestones3 = [int(self.config3['TRAIN']['EPOCHS'] * self.config3['TRAIN']['SCHEDULER']['MILESTONES'][i])
                          for i in range(len(self.config3['TRAIN']['SCHEDULER']['MILESTONES']))]
            self.base_scheduler3 = optim.lr_scheduler.MultiStepLR(self.optimizer3, milestones=milestones3, gamma=0.2)

        if self.config3['TRAIN']['SCHEDULER']['WARMUP']:
            self.scheduler3 = GradualWarmupScheduler(self.optimizer3, multiplier=1, total_epoch=int(self.config3['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler3)
        else:
            self.scheduler3 = self.base_scheduler3

        if self.config3['TRAIN']['LOSS']['WEIGHT'] == 'None':
            self.class_weight3 = None
        else:
            self.class_weight3 = self.config3['TRAIN']['LOSS']['WEIGHT']

        if self.config3['TRAIN']['LOSS']['NAME'] == 'CELoss':
            self.loss3 = CELoss(reduction='mean', ignore_index=0, weight=self.class_weight3).to(self.device)
            # self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0).to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'OhemCELoss':
            self.loss3 = OhemCELoss(self.config['TRAIN']['LOSS']['THRESH'], self.config3['TRAIN']['BATCH_SIZE'] // 5, reduction='mean', ignore_index=0, weight=self.class_weight3).to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'FocalLoss':
            self.loss3 = FocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight3).to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'MulticlassDiceLoss':
            self.loss3 = MulticlassDiceLoss(reduction='mean', ignore_index=0, weight=self.class_weight3).to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'ComposedLoss':
            self.loss3 = ComposedLoss(reduction='mean', ignore_index=0, weight=self.class_weight3).to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'ComposedFocalLoss':
            self.loss3 = ComposedFocalLoss(reduction='mean', ignore_index=0, weight=self.class_weight3).to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'BCELoss':
            self.loss3 = BCELoss(reduction='mean').to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'BinaryDiceLoss':
            self.loss3 = BinaryDiceLoss(reduction='mean').to(self.device)
        elif self.config3['TRAIN']['LOSS']['NAME'] == 'BinaryComposedLoss':
            self.loss3 = BinaryComposedLoss(reduction='mean').to(self.device)
        else:
            raise ValueError("wrong indicator for loss function, which should be selected from {'CELoss', 'OhemCELoss', 'FocalLoss', 'MulticlassDiceLoss', 'BCELoss', 'BinaryDiceLoss', 'BinaryComposedLoss'}")

        self.epochs = self.config['TRAIN']['EPOCHS']
        self.save_interval = self.config['TRAIN']['SAVE_INTERVAL']

    
    # def load_checkpoint(self, resume_path, resume_path1, rank=None, train_mode=True):
    #     if os.path.exists(resume_path):
    #         checkpoint = torch.load(resume_path)
    #         if rank is not None:
    #             if rank in [-1, 0]:
    #                 self.model.load_state_dict(checkpoint['net'])
    #                 if train_mode:
    #                     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #         else:
    #             self.model.load_state_dict(checkpoint['net'])
    #             if train_mode:
    #                 self.optimizer.load_state_dict(checkpoint['optimizer'])

    #         self.current_epoch = checkpoint['epoch']
    #         self.last_acc = checkpoint['last_acc']
    #         self.best_acc = checkpoint['best_acc']
    #         self.config['TRAIN']['SCHEDULER']['WARMUP'] = False
    #         print('load checkpoint successfully!')

    #         checkpoint1 = torch.load(resume_path1)
    #         if rank is not None:
    #             if rank in [-1, 0]:
    #                 self.model1.load_state_dict(checkpoint1['net'])
    #                 if train_mode:
    #                     self.optimizer1.load_state_dict(checkpoint1['optimizer'])
    #         else:
    #             self.model1.load_state_dict(checkpoint1['net'])
    #             if train_mode:
    #                 self.optimizer1.load_state_dict(checkpoint1['optimizer'])

    #         self.current_epoch1 = checkpoint1['epoch']
    #         self.last_acc1 = checkpoint1['last_acc']
    #         self.best_acc1 = checkpoint1['best_acc']
    #         self.config1['TRAIN']['SCHEDULER']['WARMUP'] = False

    #         # checkpoint2 = torch.load(resume_path2)
    #         # if rank is not None:
    #         #     if rank in [-1, 0]:
    #         #         self.model2.load_state_dict(checkpoint2['net'])
    #         #         if train_mode:
    #         #             self.optimizer2.load_state_dict(checkpoint2['optimizer'])
    #         # else:
    #         #     self.model2.load_state_dict(checkpoint2['net'])
    #         #     if train_mode:
    #         #         self.optimizer2.load_state_dict(checkpoint2['optimizer'])

    #         # self.current_epoch2 = checkpoint2['epoch']
    #         # self.last_acc2 = checkpoint2['last_acc']
    #         # self.best_acc2 = checkpoint2['best_acc']
    #         # self.config2['TRAIN']['SCHEDULER']['WARMUP'] = False

    def load_checkpoint(self, resume_path, resume_path1, resume_path2, resume_path3, rank=None, train_mode=True):
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path)
            if rank is not None:
                if rank in [-1, 0]:
                    self.model.load_state_dict(checkpoint['net'])
                    if train_mode:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.model.load_state_dict(checkpoint['net'])
                if train_mode:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.current_epoch = checkpoint['epoch']
            self.last_acc = checkpoint['last_acc']
            self.best_acc = checkpoint['best_acc']
            self.config['TRAIN']['SCHEDULER']['WARMUP'] = False
            print('load checkpoint successfully!')

            checkpoint1 = torch.load(resume_path1)
            if rank is not None:
                if rank in [-1, 0]:
                    self.model1.load_state_dict(checkpoint1['net'])
                    if train_mode:
                        self.optimizer1.load_state_dict(checkpoint1['optimizer'])
            else:
                self.model1.load_state_dict(checkpoint1['net'])
                if train_mode:
                    self.optimizer1.load_state_dict(checkpoint1['optimizer'])

            self.current_epoch1 = checkpoint1['epoch']
            self.last_acc1 = checkpoint1['last_acc']
            self.best_acc1 = checkpoint1['best_acc']
            self.config1['TRAIN']['SCHEDULER']['WARMUP'] = False

            checkpoint2 = torch.load(resume_path2)
            if rank is not None:
                if rank in [-1, 0]:
                    self.model2.load_state_dict(checkpoint2['net'])
                    if train_mode:
                        self.optimizer2.load_state_dict(checkpoint2['optimizer'])
            else:
                self.model2.load_state_dict(checkpoint2['net'])
                if train_mode:
                    self.optimizer2.load_state_dict(checkpoint2['optimizer'])

            self.current_epoch2 = checkpoint2['epoch']
            self.last_acc2 = checkpoint2['last_acc']
            self.best_acc2 = checkpoint2['best_acc']
            self.config2['TRAIN']['SCHEDULER']['WARMUP'] = False

            checkpoint3 = torch.load(resume_path3)
            if rank is not None:
                if rank in [-1, 0]:
                    self.model3.load_state_dict(checkpoint3['net'])
                    if train_mode:
                        self.optimizer3.load_state_dict(checkpoint3['optimizer'])
            else:
                self.model3.load_state_dict(checkpoint3['net'])
                if train_mode:
                    self.optimizer3.load_state_dict(checkpoint3['optimizer'])

            self.current_epoch3 = checkpoint3['epoch']
            self.last_acc3 = checkpoint3['last_acc']
            self.best_acc3 = checkpoint3['best_acc']
            self.config3['TRAIN']['SCHEDULER']['WARMUP'] = False

    def save_checkpoint(self, resume_path):
        state = {'net': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'epoch': self.current_epoch,
                 'last_acc': self.last_acc,
                 'best_acc': self.last_acc}
        if os.path.exists(resume_path):
            print('resume_path exists, we will cover it!')
        else:
            print('resume_path does not exist, it will be new-built!')
        torch.save(state, resume_path)
    
    def train_one_epoch(self, memo=''):
        self.model.train()
        losses = []
        tic = time.time()

        if not self.use_ddp:
            iter = tqdm(enumerate(self.train_loader), total=self.num_batches_per_epoch_for_training, file=sys.stdout)
            # iter = enumerate(self.train_loader)

        elif self.local_rank in [-1, 0]:
            iter = tqdm(enumerate(self.train_loader), total=self.num_batches_per_epoch_for_training, file=sys.stdout)
        else:
            iter = enumerate(self.train_loader)
        for batch_idx, (data, targets) in iter:
            data = data.to(self.device)
            targets = targets.to(self.device).long()
            self.optimizer.zero_grad()
            outputs = self.model(data)
            # print(torch.argmax(outputs, dim=1))
            # print(targets)
            if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
                loss = 0.4 * self.loss(outputs[0], targets) + self.loss(outputs[1], targets)
            elif str.find(self.config['MODEL']['NAME'], 'BT') != -1:
                loss = self.loss(outputs[0], targets)
                for i in range(1, len(outputs)):
                    loss += self.loss(outputs[i], targets)
            else:
                loss = self.loss(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        avg_loss = np.mean(losses)

        if not self.use_ddp:
            print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
                memo, self.current_epoch, time.time()-tic, avg_loss), end=""
            )
            print("")
        elif self.local_rank in [-1, 0]:
            print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
                memo, self.current_epoch, time.time()-tic, avg_loss), end=""
            )
            print("")
        
        return avg_loss
    
    def patch_based_inference(self, input_data, model, overlap_rate=0.5):
        overlap_size = (int(overlap_rate * self.train_size[0]), int(overlap_rate * self.train_size[1]))
        B, C, H, W = input_data.size()
        N_w = int(math.ceil((W - self.train_size[0]) / (self.train_size[0] - overlap_size[0])) + 1)
        N_h = int(math.ceil((H - self.train_size[1]) / (self.train_size[1] - overlap_size[1])) + 1)
        pad_w = int((N_w - 1) * (self.train_size[0] - overlap_size[0]) + self.train_size[0])
        pad_h = int((N_h - 1) * (self.train_size[1] - overlap_size[1]) + self.train_size[1])
        count = torch.zeros((input_data.size(0), self.config['MODEL']['N_CLASSES'], pad_h, pad_w))
        output = torch.zeros((input_data.size(0), self.config['MODEL']['N_CLASSES'], pad_h, pad_w))
        input_data = F.pad(input_data, (0, pad_h - H, 0, pad_w - W))
        flag = torch.ones((input_data.size(0), self.config['MODEL']['N_CLASSES'], self.train_size[1], self.train_size[0]))
        for i in range(N_w):
            x = i * (self.train_size[0] - overlap_size[0])
            for j in range(N_h):
                y = j * (self.train_size[1] - overlap_size[1])
                temp_input = input_data[:, :, y:y + self.train_size[1], x:x + self.train_size[0]]
                output[:, :, y:y + self.train_size[1], x:x + self.train_size[0]] = model(temp_input)[1]
                count[:, :, y:y + self.train_size[1], x:x + self.train_size[0]] += flag
        output /= count
        output = output[:, :, 0:H, 0:W]
        return output

    def eval_model(self, memo=''):
        self.model.eval()
        tic = time.time()
        cmbm = ConfusionMatrixBasedMetric(self.config['MODEL']['N_CLASSES'])

        if not self.use_ddp:
            iter = tqdm(enumerate(self.val_loader), total=self.num_batches_per_epoch_for_validating, file=sys.stdout)
            # iter = enumerate(self.val_loader)
            
        elif self.local_rank in [-1, 0]:
            iter = tqdm(enumerate(self.val_loader), total=self.num_batches_per_epoch_for_validating, file=sys.stdout)
        else:
            iter = enumerate(self.val_loader)

        with torch.no_grad():
            for batch_idx, (data, targets) in iter:
                data = data.to(self.device)
                targets = targets.to(self.device)
                # print(data.shape,targets.shape,"data,targets")
                outputs = self.model(data)
                # print(outputs.shape,222)
                # outputs = self.patch_based_inference(data, self.model)
                # print(outputs.shape,111)
                
                if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
                    cmbm.add_batch(targets, outputs)#[1]
                elif str.find(self.config['MODEL']['NAME'], 'BT') != -1:
                    cmbm.add_batch(targets, outputs[0])
                else:
                    cmbm.add_batch(targets, outputs)

                # if batch_idx % 3 == 0:
                #     e = ConfusionMatrixBasedMetric(self.config['MODEL']['N_CLASSES'])
                #     e.add_batch(targets, outputs)
                #     iou = e.calculate_Metric(metric='IoU', class_idx=1, reduce=True)
                #     print(iou)
                #     outputs = torch.argmax(outputs, dim=1)
                #     from matplotlib import pyplot as plt
                #     from torchvision import transforms as T
                #     unloader = T.ToPILImage()
                #     img = unloader(data[0].squeeze().float().cpu())
                #     mask = unloader(targets[0].squeeze().float().cpu())
                #     pred = unloader(outputs[0].squeeze().float().cpu())
                #     plt.figure(0)
                #     plt.subplot(131)
                #     plt.imshow(img)
                #     plt.title('image')
                #     plt.subplot(132)
                #     plt.imshow(mask)
                #     plt.title('mask')
                #     plt.subplot(133)
                #     plt.imshow(pred)
                #     plt.title('pred')
                #     plt.show()

        iou = cmbm.calculate_Metric(metric='IoU', reduce=True, binary=False)
        f1 = cmbm.calculate_Metric(metric='F-score', reduce=True, binary=False)
        precision = cmbm.calculate_Metric(metric='Precision', reduce=True, binary=False)
        recall = cmbm.calculate_Metric(metric='Recall', reduce=True, binary=False)

        # iou = cmbm.calculate_Metric(metric='IoU', class_idx=1, reduce=True)
        # f1 = cmbm.calculate_Metric(metric='F-score', class_idx=1, reduce=True)
        # precision = cmbm.calculate_Metric(metric='Precision', class_idx=1, reduce=True)
        # recall = cmbm.calculate_Metric(metric='Recall', class_idx=1, reduce=True)

        if not self.use_ddp:
            print('[{}] Validation Epoch: {}\t Time elapsed: {:.4f} seconds\t MIoU: {:.4f}\t F1: {:.4f}\t Precision: {:.4f}\t Recall: {:.4f}'.format(
                memo, self.current_epoch, time.time()-tic, iou, f1, precision, recall), end=""
            )
            print("")
        elif self.local_rank in [-1, 0]:
            print('[{}] Validation Epoch: {}\t Time elapsed: {:.4f} seconds\t MIoU: {:.4f}\t F1: {:.4f}\t Precision: {:.4f}\t Recall: {:.4f}'.format(
                memo, self.current_epoch, time.time()-tic, iou, f1, precision, recall), end=""
            )
            print("")
    
        return [iou, f1, precision, recall]
    
    def inference(self, memo=''):

        self.model.eval()
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        tic = time.time()

        NLCD_CLASS_COLORMAP = { # Copied from the emebedded color table in the NLCD data files
            0:  (0, 0, 0),
    1:  (0, 112, 255),
    2: (255, 0, 0),
    3: (255, 255, 115),
    4: (255, 170, 0),
    5: (78, 78, 78),
    6: (163, 255, 115),
    7: (255,235,190),
    8: (255,167,127),
    9: (209,255,115),
    10: (38,115,0),
    11: (230,230,0),
    12: (204,204,204),
    13:  (0,168,132)
        }
        if not self.use_ddp:
            iter = tqdm(enumerate(self.test_loader1), total=self.num_batches_per_epoch_for_testing, file=sys.stdout)
        elif self.local_rank in [-1, 0]:
            iter = tqdm(enumerate(self.test_loader1), total=self.num_batches_per_epoch_for_testing, file=sys.stdout)
        else:
            iter = enumerate(self.test_loader1)

        output_dir = os.path.join(self.log_path, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with torch.no_grad():
            for batch_idx, (data, targets, index) in iter:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # ### way 1
                # data_f_x = flip(data,dim=2)
                # data_f_y = flip(data,dim=3)

                # outputs = self.model(data)[1]
                # outputs_f_x = flip(self.model(data_f_x)[1],dim=2)
                # outputs_f_y = flip(self.model(data_f_y)[1],dim=3)
                # outputs = (outputs+outputs_f_x+outputs_f_y)/3

                # outputs1 = self.model1(data)[1]
                # outputs_f_x1 = flip(self.model1(data_f_x)[1],dim=2)
                # outputs_f_y1 = flip(self.model1(data_f_y)[1],dim=3)
                # outputs1 = (outputs1+outputs_f_x1+outputs_f_y1)/3

                # outputs = (outputs+outputs1)/2

                # print(data.shape,"data") #torch.Size([1, 3, 2001, 2000]) data

                ##### filp + rotate
                outputs = self.model(data)
                # print(outputs.shape,11111)
                output_data_f_x = flip(self.model(flip(data,dim=2)),dim=2)
                output_data_f_y = flip(self.model(flip(data,dim=3)),dim=3)
                dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees. # [1, 16, 2004, 2000]
                output_data_r_90 = rot_img(self.model(rotated_90), -np.pi/2, dtype)
                rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                output_data_r_180 = rot_img(self.model(rotated_180), -np.pi, dtype)
                rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                output_data_r_270 = rot_img(self.model(rotated_270), -(np.pi/2)*3, dtype)
                outputs = (outputs+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6

                outputs1 = self.model1(data)
                # print(outputs1.shape,22222)
                output_data_f_x = flip(self.model1(flip(data,dim=2)),dim=2)
                output_data_f_y = flip(self.model1(flip(data,dim=3)),dim=3)
                dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees. # [1, 16, 2004, 2000]
                output_data_r_90 = rot_img(self.model1(rotated_90), -np.pi/2, dtype)
                rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                output_data_r_180 = rot_img(self.model1(rotated_180), -np.pi, dtype)
                rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                output_data_r_270 = rot_img(self.model1(rotated_270), -(np.pi/2)*3, dtype)
                outputs1 = (outputs1+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6
                
                # outputs1 = self.model1(data)[1]
                # # print(outputs1.shape,22222)
                # output_data_f_x = flip(self.model1(flip(data,dim=2))[1],dim=2)
                # output_data_f_y = flip(self.model1(flip(data,dim=3))[1],dim=3)
                # dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                # rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees. # [1, 16, 2004, 2000]
                # output_data_r_90 = rot_img(self.model1(rotated_90)[1], -np.pi/2, dtype)
                # rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                # output_data_r_180 = rot_img(self.model1(rotated_180)[1], -np.pi, dtype)
                # rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                # output_data_r_270 = rot_img(self.model1(rotated_270)[1], -(np.pi/2)*3, dtype)
                # outputs1 = (outputs1+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6

                # outputs2 = self.model2(data)[1]
                # # print(outputs1.shape,22222)
                # output_data_f_x = flip(self.model2(flip(data,dim=2))[1],dim=2)
                # output_data_f_y = flip(self.model2(flip(data,dim=3))[1],dim=3)
                # dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                # rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees. # [1, 16, 2004, 2000]
                # output_data_r_90 = rot_img(self.model2(rotated_90)[1], -np.pi/2, dtype)
                # rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                # output_data_r_180 = rot_img(self.model2(rotated_180)[1], -np.pi, dtype)
                # rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                # output_data_r_270 = rot_img(self.model2(rotated_270)[1], -(np.pi/2)*3, dtype)
                # outputs2 = (outputs2+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6

                outputs2 = self.model2(data)
                # print(outputs1.shape,22222)
                output_data_f_x = flip(self.model2(flip(data,dim=2)),dim=2)
                output_data_f_y = flip(self.model2(flip(data,dim=3)),dim=3)
                dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees. # [1, 16, 2004, 2000]
                output_data_r_90 = rot_img(self.model2(rotated_90), -np.pi/2, dtype)
                rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                output_data_r_180 = rot_img(self.model2(rotated_180), -np.pi, dtype)
                rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                output_data_r_270 = rot_img(self.model2(rotated_270), -(np.pi/2)*3, dtype)
                outputs2 = (outputs2+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6


                outputs3 = self.model3(data)
                # print(outputs1.shape,22222)
                output_data_f_x = flip(self.model3(flip(data,dim=2)),dim=2)
                output_data_f_y = flip(self.model3(flip(data,dim=3)),dim=3)
                dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees. # [1, 16, 2004, 2000]
                output_data_r_90 = rot_img(self.model3(rotated_90), -np.pi/2, dtype)
                rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                output_data_r_180 = rot_img(self.model3(rotated_180), -np.pi, dtype)
                rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                output_data_r_270 = rot_img(self.model3(rotated_270), -(np.pi/2)*3, dtype)
                outputs3 = (outputs3+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6


                outputs = (outputs+outputs1+outputs2+outputs3)/4
                ##### filp + rotate


                # outputs2 = self.model2(data)[1]
                # outputs_f_x2 = flip(self.model2(data_f_x)[1],dim=2)
                # outputs_f_y2 = flip(self.model2(data_f_y)[1],dim=3)
                # outputs2 = (outputs2+outputs_f_x2+outputs_f_y2)/3

                # outputs = (outputs+outputs1+outputs2)/3
                
                ### way 2
                # outputs = self.model(data)
                # outputs = self.patch_based_inference(data, self.model)

                if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
                    # if outputs[1].size(1) >= 2 and len(outputs[1].size()) > 3:
                    #     outputs = torch.argmax(outputs[1], dim=1)
                    # else:
                    #     outputs = torch.sigmoid(outputs[1]).squeeze()
                    #     outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))

                    if outputs.size(1) >= 2 and len(outputs.size()) > 3:
                        outputs = torch.argmax(outputs, dim=1)
                    else:
                        outputs = torch.sigmoid(outputs).squeeze()
                        outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                elif str.find(self.config['MODEL']['NAME'], 'BT') != -1:
                    if outputs[0].size(1) >= 2 and len(outputs[0].size()) > 3:
                        outputs = torch.argmax(outputs[0], dim=1)
                    else:
                        outputs = torch.sigmoid(outputs[0]).squeeze()
                        outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                else:
                    if outputs.size(1) >= 2 and len(outputs.size()) > 3:
                        outputs = torch.argmax(outputs, dim=1)
                    else:
                        outputs = torch.sigmoid(outputs).squeeze()
                        outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                
                # from matplotlib import pyplot as plt
                # from torchvision import transforms as T
                from PIL import Image
                import rasterio
                # unloader = T.ToPILImage()
                # data[0] = data[0] * 0.5 + 0.5
                # img = unloader(data[0].squeeze().float().cpu())
                # mask = unloader(targets[0].squeeze().float().cpu())
                pred = outputs.squeeze().float().cpu().numpy()
                city_name = index[0].split('/')[-3]

                input_fp = rasterio.open(index[0], 'r')
                input_profile = input_fp.profile.copy()
                output_profile = input_profile.copy()
                output_profile["driver"] = "GTiff"
                output_profile["dtype"] = "uint8"
                output_profile["count"] = 1
                output_profile["nodata"] = 0

                dist_dir = os.path.join(self.log_path, 'output/' + city_name)
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)
                basename = os.path.basename(index[0])
                img_name = str(basename.split('.')[0]) + "_prediction.tif"
                with rasterio.open(os.path.join(dist_dir, img_name), "w", **output_profile) as dst:
                    dst.write(pred, 1)
                    dst.write_colormap(1, NLCD_CLASS_COLORMAP)

                # plt.figure(0)
                # plt.subplot(131)
                # plt.imshow(img)
                # plt.title('image')
                # plt.subplot(132)
                # plt.imshow(mask)
                # plt.title('mask')
                # plt.subplot(133)
                # plt.imshow(pred)
                # plt.title('pred: %.4f' % iou)
                # plt.savefig(os.path.join(self.log_path, img_name))
                # plt.show()
                # if batch_idx > 30:
                    # break

        # iou = cmbm.calculate_Metric(metric='IoU', class_idx=1, reduce=True)
        # f1 = cmbm.calculate_Metric(metric='F-score', class_idx=1, reduce=True)
        # precision = cmbm.calculate_Metric(metric='Precision', class_idx=1, reduce=True)
        # recall = cmbm.calculate_Metric(metric='Recall', class_idx=1, reduce=True)
    
        return

    def train(self):

        start_epoch = self.current_epoch
        
        for i in range(start_epoch, self.epochs):
            if self.use_ddp:
                self.train_loader.sampler.set_epoch(i)

            training_loss = self.train_one_epoch()
            eval_res = self.eval_model()

            IoU = eval_res[0]
            F1 = eval_res[1]
            PC = eval_res[2]
            RE = eval_res[3]
            
            self.scheduler.step(metrics=IoU)
            self.writter.add_scalar('loss', training_loss, global_step=i)
            self.writter.add_scalar('IoU', IoU, global_step=i)

            self.current_epoch += 1

            # if (i + 1) % self.save_interval == 0:
            #     current_model_fn = os.path.join(self.log_path, "epoch_%d_model.pt" % (i + 1))
            #     self.save_checkpoint(current_model_fn)
            #     print("epoch %d model saved" % (i + 1))

            self.last_acc = IoU
            current_model_fn = os.path.join(self.log_path, "last_model.pt")
            self.save_checkpoint(current_model_fn)

            if IoU > self.best_acc:
                best_model_fn = os.path.join(self.log_path, "best_model.pt")
                self.best_acc = IoU
                self.save_checkpoint(best_model_fn)
                print("best model saved")
        
        self.writter.close()

