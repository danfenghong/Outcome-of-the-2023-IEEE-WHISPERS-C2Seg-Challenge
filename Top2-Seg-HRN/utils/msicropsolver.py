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
from .msicropdataset import RoadDataset, SequentialDistributedSampler
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
        self.val_loader = DataLoader(dataset=val_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=val_batchsize, sampler=self.val_sampler, drop_last=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        # model

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
            if self.use_ddp:
                self.load_checkpoint(self.opt.resume_path, rank=self.local_rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.load_checkpoint(self.opt.resume_path)
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
            # print(self.class_weight)
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

        self.epochs = self.config['TRAIN']['EPOCHS']
        self.save_interval = self.config['TRAIN']['SAVE_INTERVAL']
    
    def load_checkpoint(self, resume_path, rank=None, train_mode=True):
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
            
            # if self.use_ddp:
            #     self.model.module.load_state_dict(checkpoint['net'])
            #     if train_mode:
            #         self.optimizer.module.load_state_dict(checkpoint['optimizer'])
            # else:
            #     self.model.load_state_dict(checkpoint['net'])
            #     if train_mode:
            #         self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch = checkpoint['epoch']
            self.last_acc = checkpoint['last_acc']
            self.best_acc = checkpoint['best_acc']
            self.config['TRAIN']['SCHEDULER']['WARMUP'] = False
            print('load checkpoint successfully!')
        else:
            print('resume_path does not exist, we will train from scratch!')
            self.current_epoch = 0
            self.last_acc = -999.0
            self.best_acc = -999.0
    
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
                output[:, :, y:y + self.train_size[1], x:x + self.train_size[0]] = model(temp_input)
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
                    cmbm.add_batch(targets, outputs[1])
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
            iter = tqdm(enumerate(self.test_loader), total=self.num_batches_per_epoch_for_testing, file=sys.stdout)
        elif self.local_rank in [-1, 0]:
            iter = tqdm(enumerate(self.test_loader), total=self.num_batches_per_epoch_for_testing, file=sys.stdout)
        else:
            iter = enumerate(self.test_loader)

        output_dir = os.path.join(self.log_path, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with torch.no_grad():
            for batch_idx, (data, targets, index) in iter:
                data = data.to(self.device)
                targets = targets.to(self.device)

                ### way 1
                # outputs = self.model(data)
                # output_data_f_x = flip(self.model(flip(data,dim=2)),dim=2)
                # output_data_f_y = flip(self.model(flip(data,dim=3)),dim=3)
                # outputs = (outputs+output_data_f_x+output_data_f_y)/3

                ##### filp + rotate
                if str.find(self.config['MODEL']['NAME'], 'OCR') != -1: ## include OCR
                    outputs = self.model(data)[1]
                    output_data_f_x = flip(self.model(flip(data,dim=2))[1],dim=2)
                    output_data_f_y = flip(self.model(flip(data,dim=3))[1],dim=3)
                    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                    rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees.
                    output_data_r_90 = rot_img(self.model(rotated_90)[1], -np.pi/2, dtype)
                    rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                    output_data_r_180 = rot_img(self.model(rotated_180)[1], -np.pi, dtype)
                    rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                    output_data_r_270 = rot_img(self.model(rotated_270)[1], -(np.pi/2)*3, dtype)
                else:
                    outputs = self.model(data)
                    output_data_f_x = flip(self.model(flip(data,dim=2)),dim=2)
                    output_data_f_y = flip(self.model(flip(data,dim=3)),dim=3)
                    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                    rotated_90 = rot_img(data, np.pi/2, dtype) # Rotate image by 90 degrees.
                    # print(self.model(rotated_90).shape,"@"*10) # [1, 16, 2004, 2000]
                    output_data_r_90 = rot_img(self.model(rotated_90), -np.pi/2, dtype)
                    rotated_180 = rot_img(rotated_90, np.pi/2, dtype) # Rotate image by 180 degrees.
                    output_data_r_180 = rot_img(self.model(rotated_180), -np.pi, dtype)
                    rotated_270 = rot_img(rotated_180, np.pi/2, dtype) # Rotate image by 270 degrees.
                    output_data_r_270 = rot_img(self.model(rotated_270), -(np.pi/2)*3, dtype)

                outputs = (outputs+output_data_f_x+output_data_f_y+output_data_r_90+output_data_r_180+output_data_r_270)/6
                ##### filp + rotate

                ### way 2
                # outputs = self.model(data)
                # outputs = self.patch_based_inference(data, self.model)

                if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
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

