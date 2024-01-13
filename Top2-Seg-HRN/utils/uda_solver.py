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
from .udacropdataset import RoadDataset, SequentialDistributedSampler
from .criterion import ConfusionMatrixBasedMetric
from .function import CELoss, OhemCELoss, FocalLoss, MulticlassDiceLoss, ComposedLoss, BCELoss, BinaryDiceLoss, BinaryComposedLoss
from .scheduler import GradualWarmupScheduler
import models
from models.discriminator.discriminator import get_fc_discriminator
from shutil import copyfile
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


LOG_ROOT = '/Top2-Seg-HRN/run'

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    # print(prob.shape,prob.size(),"prob.shape,prob.size()")
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def adjust_learning_rate(optimizer, i_iter, learning_rate, epochs):
    lr = lr_poly(learning_rate, i_iter, epochs, 0.9)
    optimizer.param_groups[0]['lr'] = lr

class UDA_Solver:
    def __init__(self, args, log_path):
        self.opt = args
        cfg_file = open(self.opt.config, 'r')
        self.config = yaml.load(cfg_file, Loader=yaml.FullLoader)
        cfg_file.close()
        self.log_path = log_path
        self.writter = SummaryWriter(os.path.join(self.log_path, 'log'))
        self.dst_cfg_fn = os.path.join(self.log_path, os.path.basename(self.opt.config))
        cudnn.benchmark = True
        cudnn.enabled = True
        copyfile(self.opt.config, self.dst_cfg_fn)

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

        if self.config['DATASET']['MAX_ITER'] == 'None':
            max_iter = None
        elif isinstance(self.config['DATASET']['MAX_ITER'], int):
            max_iter = self.config['DATASET']['MAX_ITER']
        else:
            raise TypeError("Invalid type of chip_size, which should be int or 'None'!")

        aug_config_path = self.config['DATASET']['AUG_CONFIG_PATH']
        train_source_dataset = RoadDataset(self.config['DATASET']['TRAIN_SOURCE_DATASET_CSV'], aug_config_path, 
                                    use_aug=self.config['DATASET']['USE_AUG'], mode='train', max_iter=max_iter)
        self.train_size = train_source_dataset.chip_size
        train_target_dataset = RoadDataset(self.config['DATASET']['TRAIN_TARGET_DATASET_CSV'], aug_config_path, 
                                    use_aug=self.config['DATASET']['USE_AUG'], mode='train', max_iter=max_iter)
        val_dataset = RoadDataset(self.config['DATASET']['VAL_DATASET_CSV'], aug_config_path, use_aug=False, mode='val')
        test_dataset = RoadDataset(self.config['DATASET']['TEST_DATASET_CSV'], mode='test')
        val_batchsize = self.config['TRAIN']['BATCH_SIZE'] // 4 if self.config['TRAIN']['BATCH_SIZE'] // 4 > 0 else 1
        self.num_batches_per_epoch_for_training = len(train_source_dataset) // self.config['TRAIN']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_validating = len(val_dataset) // val_batchsize // device_count
        self.num_batches_per_epoch_for_testing = len(test_dataset)
        
        if self.use_ddp:
            self.train_source_sampler = torch.utils.data.distributed.DistributedSampler(train_source_dataset)
            self.train_target_sampler = torch.utils.data.distributed.DistributedSampler(train_target_dataset)
            self.val_sampler = SequentialDistributedSampler(val_dataset, batch_size=self.config['TRAIN']['BATCH_SIZE'])
        else:
            self.train_source_sampler = torch.utils.data.RandomSampler(train_source_dataset)
            self.train_target_sampler = torch.utils.data.RandomSampler(train_target_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(val_dataset)

        self.train_source_loader = DataLoader(dataset=train_source_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=self.config['TRAIN']['BATCH_SIZE'], sampler=self.train_source_sampler)
        self.train_target_loader = DataLoader(dataset=train_target_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=self.config['TRAIN']['BATCH_SIZE'], sampler=self.train_target_sampler)
        self.val_loader = DataLoader(dataset=val_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=val_batchsize, sampler=self.val_sampler)
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

            # feature-level
            self.d_aux = get_fc_discriminator(in_ch=self.config['MODEL']['N_CLASSES'])
            self.d_aux.train()
            self.d_aux.to(self.device)

            # seg maps, i.e. output, level
            self.d_main = get_fc_discriminator(in_ch=self.config['MODEL']['N_CLASSES'])
            self.d_main.train()
            self.d_main.to(self.device)

            if self.use_ddp:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            
            self.optimizer = optim.SGD(param_dicts, self.config['TRAIN']['LR']['RATE'], 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            self.optimizer_d_aux = optim.SGD(self.d_aux.parameters(), self.config['TRAIN']['LR']['RATE'] / 2.5, 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            self.optimizer_d_main = optim.SGD(self.d_main.parameters(), self.config['TRAIN']['LR']['RATE'] / 2.5, 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
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

            # feature-level
            self.d_aux = get_fc_discriminator(in_ch=self.config['MODEL']['N_CLASSES'])
            self.d_aux.train()
            self.d_aux.to(self.device)

            # seg maps, i.e. output, level
            self.d_main = get_fc_discriminator(in_ch=self.config['MODEL']['N_CLASSES'])
            self.d_main.train()
            self.d_main.to(self.device)

            self.optimizer = optim.SGD(param_dicts, self.config['TRAIN']['LR']['RATE'], 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            self.optimizer_d_aux = optim.SGD(self.d_aux.parameters(), self.config['TRAIN']['LR']['RATE'] / 2.5, 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])
            self.optimizer_d_main = optim.SGD(self.d_main.parameters(), self.config['TRAIN']['LR']['RATE'] / 2.5, 
                                           momentum=self.config['TRAIN']['LR']['MOMENTUM'], weight_decay=self.config['TRAIN']['LR']['WEIGHT_DECAY'])

            if self.use_ddp:
                self.load_checkpoint(self.opt.resume_path, rank=self.local_rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.load_checkpoint(self.opt.resume_path)
        else:
            raise FileNotFoundError(self.opt.resume_path + " not existed")
       
        if self.config['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
            self.base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "max", patience=self.config['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
            self.base_scheduler_d_aux = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d_aux, "max", patience=self.config['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
            self.base_scheduler_d_main = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d_main, "max", patience=self.config['TRAIN']['SCHEDULER']['PATIENCE'], 
                                                                  factor=self.config['TRAIN']['SCHEDULER']['FACTOR'], 
                                                                  threshold=self.config['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True, min_lr=1e-6)
        elif self.config['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
            milestones = [int(self.config['TRAIN']['EPOCHS'] * self.config['TRAIN']['SCHEDULER']['MILESTONES'][i])
                          for i in range(len(self.config['TRAIN']['SCHEDULER']['MILESTONES']))]
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)
            self.base_scheduler_d_aux = optim.lr_scheduler.MultiStepLR(self.optimizer_d_aux, milestones=milestones, gamma=0.2)
            self.base_scheduler_d_main = optim.lr_scheduler.MultiStepLR(self.optimizer_d_main, milestones=milestones, gamma=0.2)

        if self.config['TRAIN']['SCHEDULER']['WARMUP']:
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=int(self.config['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler)
            self.scheduler_d_aux = GradualWarmupScheduler(self.optimizer_d_aux, multiplier=1, total_epoch=int(self.config['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler)
            self.scheduler_d_main = GradualWarmupScheduler(self.optimizer_d_main, multiplier=1, total_epoch=int(self.config['TRAIN']['SCHEDULER']['WARMUP_EPOCH']), 
                                                    after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler
            self.scheduler_d_aux = self.base_scheduler_d_aux
            self.scheduler_d_main = self.base_scheduler_d_main

        if self.config['TRAIN']['LOSS']['WEIGHT'] == 'None':
            self.class_weight = None
        else:
            self.class_weight = self.config['TRAIN']['LOSS']['WEIGHT']

        if self.config['TRAIN']['LOSS']['NAME'] == 'CELoss':
            # self.loss = nn.CrossEntropyLoss(reduction='mean', weight=self.class_weight, ignore_index=0).to(self.device)
            self.loss = CELoss(reduction='mean', weight=self.class_weight, ignore_index=0).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'OhemCELoss':
            self.loss = OhemCELoss(self.config['TRAIN']['LOSS']['THRESH'], self.config['TRAIN']['BATCH_SIZE'] * self.train_size[0] * self.train_size[1] // 5, reduction='mean', weight=self.class_weight, ignore_index=0).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'FocalLoss':
            self.loss = FocalLoss(reduction='mean', weight=self.class_weight, ignore_index=0).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'MulticlassDiceLoss':
            self.loss = MulticlassDiceLoss(reduction='mean', weight=self.class_weight, ignore_index=0).to(self.device)
        elif self.config['TRAIN']['LOSS']['NAME'] == 'ComposedLoss':
            self.loss = ComposedLoss(reduction='mean', weight=self.class_weight, ignore_index=0).to(self.device)
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
                    if 'd_aux' in list(checkpoint.keys()):
                        self.d_aux.load_state_dict(checkpoint['d_aux'])
                    if 'd_main' in list(checkpoint.keys()):
                        self.d_main.load_state_dict(checkpoint['d_main'])
                    if train_mode:
                        pass
            else:
                self.model.load_state_dict(checkpoint['net'])
                if 'd_aux' in list(checkpoint.keys()):
                    self.d_aux.load_state_dict(checkpoint['d_aux'])
                if 'd_main' in list(checkpoint.keys()):
                    self.d_main.load_state_dict(checkpoint['d_main'])
                if train_mode:
                    pass
                    
            self.current_epoch = 0
            self.last_acc = -999.0
            self.best_acc = -999.0
            self.config['TRAIN']['SCHEDULER']['WARMUP'] = False
            print('load checkpoint successfully!')
        else:
            print('resume_path does not exist, we will train from scratch!')
            self.current_epoch = 0
            self.last_acc = -999.0
            self.best_acc = -999.0
    
    def save_checkpoint(self, resume_path):
        state = {'net': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
                 'd_aux': self.d_aux.state_dict(),
                 'd_main': self.d_main.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'optimizer_d_aux': self.optimizer_d_aux.state_dict(),
                 'optimizer_d_main': self.optimizer_d_main.state_dict(),
                 'epoch': self.current_epoch,
                 'last_acc': self.last_acc,
                 'best_acc': self.last_acc}
        if os.path.exists(resume_path):
            print('resume_path exists, we will cover it!')
        else:
            print('resume_path does not exist, it will be new-built!')
        torch.save(state, resume_path)
    
    def uda_train_one_epoch(self, memo=''):
        self.model.train()
        source_losses = []
        adv_losses = []
        dis_aux_src_losses = []
        dis_main_src_losses = []
        dis_aux_trg_losses = []
        dis_main_trg_losses = []
        tic = time.time()

        # labels for adversarial training
        source_index = 0
        target_index = 1
        trainloader_iter = enumerate(self.train_source_loader)
        targetloader_iter = enumerate(self.train_target_loader)

        # print(self.num_batches_per_epoch_for_training, "self.num_batches_per_epoch_for_training") #15
        for i_iter in tqdm(range(self.num_batches_per_epoch_for_training)):

            # reset optimizers
            self.optimizer.zero_grad()
            self.optimizer_d_aux.zero_grad()
            self.optimizer_d_main.zero_grad()

            # adapt LR if needed
            adjust_learning_rate(self.optimizer, i_iter, self.config['TRAIN']['LR']['RATE'], self.epochs)
            adjust_learning_rate(self.optimizer_d_aux, i_iter, self.config['TRAIN']['LR']['RATE'] / 2.5, self.epochs)
            adjust_learning_rate(self.optimizer_d_main, i_iter, self.config['TRAIN']['LR']['RATE'] / 2.5, self.epochs)

            # UDA Training
            # only train segnet. Don't accumulate grads in disciminators
            for param in self.d_aux.parameters():
                param.requires_grad = False
            for param in self.d_main.parameters():
                param.requires_grad = False

            # train on source
            _, (source_data, source_label) = trainloader_iter.__next__()
            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)

            # interpolate output segmaps
            interp = nn.Upsample(size=(source_label.size(-2), source_label.size(-1)), mode='bilinear', align_corners=True)
            interp_target = nn.Upsample(size=(source_label.size(-2), source_label.size(-1)), mode='bilinear', align_corners=True)

            pred_src, src_feats = self.model(source_data)
            src_feats = interp(src_feats)
            pred_src_list = [src_feats, pred_src]
            pred_src = interp(pred_src)

            if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
                source_loss = 0.4 * self.loss(pred_src[0], source_label) + self.loss(pred_src[1], source_label)
            else:
                source_loss = self.loss(pred_src, source_label)

            source_losses.append(source_loss.item())
            source_loss.backward()

            # adversarial training to fool the discriminator
            _, (target_data, _) = targetloader_iter.__next__()
            target_data = target_data.to(self.device)
            pred_trg, trg_feats = self.model(target_data) ## torch.Size([2, 14, 32, 32]) torch.Size([2, 14, 4, 4])
            trg_feats = interp_target(trg_feats)
            pred_trg = [trg_feats, pred_trg]

            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg[-2])))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_index)
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg[-1])))
            loss_adv_trg_main = bce_loss(d_out_main, source_index)
            
            adv_loss = 0.0002 * loss_adv_trg_aux + 0.001 * loss_adv_trg_main
            adv_losses.append(adv_loss.item())
            adv_loss.backward()

            # Train discriminator networks
            # enable training mode on discriminator networks
            for param in self.d_aux.parameters():
                param.requires_grad = True
            for param in self.d_main.parameters():
                param.requires_grad = True

            # train with source
            pred_src_list[-2] = pred_src_list[-2].detach()
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_src_list[-2])))
            loss_d_aux = bce_loss(d_out_aux, source_index)
            loss_d_aux = loss_d_aux / 2
            dis_aux_src_losses.append(loss_d_aux.item())
            loss_d_aux.backward()

            pred_src_list[-1]= pred_src_list[-1].detach()
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_list[-1])))
            loss_d_main = bce_loss(d_out_main, source_index)
            loss_d_main = loss_d_main / 2
            dis_main_src_losses.append(loss_d_main.item())
            loss_d_main.backward()

            # train with target
            pred_trg[-2] = pred_trg[-2].detach()
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg[-2])))
            loss_d_aux = bce_loss(d_out_aux, target_index)
            loss_d_aux = loss_d_aux / 2
            dis_aux_trg_losses.append(loss_d_aux.item())
            loss_d_aux.backward()

            pred_trg[-1] = pred_trg[-1].detach()
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg[-1])))
            loss_d_main = bce_loss(d_out_main, target_index)
            loss_d_main = loss_d_main / 2
            dis_main_trg_losses.append(loss_d_main.item())
            loss_d_main.backward()

            self.optimizer.step()
            self.optimizer_d_aux.step()
            self.optimizer_d_main.step()

        avg_source_loss = np.mean(source_losses)
        avg_adv_loss = np.mean(adv_losses)

        avg_dis_aux_src_loss = np.mean(dis_aux_src_losses)
        avg_dis_main_src_loss = np.mean(dis_main_src_losses)
        avg_dis_aux_trg_loss = np.mean(dis_aux_trg_losses)
        avg_dis_main_trg_loss = np.mean(dis_main_trg_losses)
    
        if not self.use_ddp:
            print('[{}] Training Epoch: {}\t Time elapsed: {:.7f} seconds\t source_loss: {:.7f}\t adv_loss: {:.7f}\t dis_aux_src_loss: {:.7f}\t dis_main_src_loss: {:.7f}\t dis_aux_trg_loss: {:.7f}\t dis_main_trg_loss: {:.7f}'.format(
                memo, self.current_epoch, time.time()-tic, avg_source_loss, avg_adv_loss, avg_dis_aux_src_loss, avg_dis_main_src_loss, avg_dis_aux_trg_loss, avg_dis_main_trg_loss), end=""
            )
            print("")
        elif self.local_rank in [-1, 0]:
            print('[{}] Training Epoch: {}\t Time elapsed: {:.7f} seconds\t source_loss: {:.7f}\t adv_loss: {:.7f}\t dis_aux_src_loss: {:.7f}\t dis_main_src_loss: {:.7f}\t dis_aux_trg_loss: {:.7f}\t dis_main_trg_loss: {:.7f}'.format(
                memo, self.current_epoch, time.time()-tic, avg_source_loss, avg_adv_loss, avg_dis_aux_src_loss, avg_dis_main_src_loss, avg_dis_aux_trg_loss, avg_dis_main_trg_loss), end=""
            )
            print("")

        return avg_source_loss, avg_adv_loss, avg_dis_aux_src_loss, avg_dis_main_src_loss, avg_dis_aux_trg_loss, avg_dis_main_trg_loss
    
    def train_one_epoch(self, memo=''):
        self.model.train()
        source_losses = []
        tic = time.time()

        # labels for adversarial training
        trainloader_iter = enumerate(self.train_source_loader)

        for i_iter in tqdm(range(self.num_batches_per_epoch_for_training)):

            # reset optimizers
            self.optimizer.zero_grad()

            # train on source
            _, (source_data, source_label) = trainloader_iter.__next__()
            source_data = source_data.to(self.device)
            source_label = source_label.long().to(self.device)

            # interpolate output segmaps
            interp = nn.Upsample(size=(source_label.size(-2), source_label.size(-1)), mode='bilinear', align_corners=True)

            pred_src, _ = self.model(source_data)
            pred_src = interp(pred_src)

            if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
                source_loss = 0.4 * self.loss(pred_src[0], source_label) + self.loss(pred_src[1], source_label)
            elif str.find(self.config['MODEL']['NAME'], 'PointFlow') != -1:
                source_loss = self.loss(pred_src, source_label)
            else:
                # print(pred_src.shape, source_label.shape,"pred_src.shape, source_label.shape")
                ##torch.Size([32, 14, 128, 128]) torch.Size([32, 128, 128]) pred_src.shape, source_label.shape
                source_loss = self.loss(pred_src, source_label)
            # pred_src = interp(pred_src)

            source_losses.append(source_loss.item())
            source_loss.backward()

            self.optimizer.step()

        avg_source_loss = np.mean(source_losses)
    
        if not self.use_ddp:
            print('[{}] Training Epoch: {}\t Time elapsed: {:.4f} seconds\t source_loss: {:.4f}\t'.format(
                memo, self.current_epoch, time.time()-tic, avg_source_loss), end=""
            )
            print("")
        elif self.local_rank in [-1, 0]:
            print('[{}] Training Epoch: {}\t Time elapsed: {:.4f} seconds\t source_loss: {:.4f}\t'.format(
                memo, self.current_epoch, time.time()-tic, avg_source_loss), end=""
            )
            print("")
        
        return avg_source_loss

    def eval_model(self, memo=''):
        self.model.eval()
        tic = time.time()
        cmbm = ConfusionMatrixBasedMetric(self.config['MODEL']['N_CLASSES'])

        if not self.use_ddp:
            iter = tqdm(enumerate(self.val_loader), total=self.num_batches_per_epoch_for_validating, file=sys.stdout)
        elif self.local_rank in [-1, 0]:
            iter = tqdm(enumerate(self.val_loader), total=self.num_batches_per_epoch_for_validating, file=sys.stdout)
        else:
            iter = enumerate(self.val_loader)

        with torch.no_grad():
            for batch_idx, (data, targets) in iter:
                data = data.to(self.device)
                targets = targets.to(self.device)
                # outputs =self.patch_based_inference(data, self.model)
                outputs, _ = self.model(data)
                interp = nn.Upsample(size=(targets.size(-2), targets.size(-1)), mode='bilinear', align_corners=True)
                outputs = interp(outputs)
                cmbm.add_batch(targets, outputs)

        iou = cmbm.calculate_Metric(metric='IoU', reduce=True, binary=False)
        f1 = cmbm.calculate_Metric(metric='F-score', reduce=True, binary=False)
        precision = cmbm.calculate_Metric(metric='Precision', reduce=True, binary=False)
        recall = cmbm.calculate_Metric(metric='Recall', reduce=True, binary=False)

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

        NLCD_CLASS_COLORMAP = {
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
                outputs, _ = self.model(data)

                if str.find(self.config['MODEL']['NAME'], 'OCR') != -1:
                    outputs = outputs[1]

                ### way 2
                # outputs = self.model(data)
                # outputs = self.patch_based_inference(data, self.model)

                if outputs.size(1) >= 2 and len(outputs.size()) > 3:
                    outputs = torch.argmax(outputs, dim=1)
                else:
                    outputs = torch.sigmoid(outputs).squeeze()
                    outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))

                from PIL import Image
                import rasterio
                pred = outputs.squeeze().float().cpu().numpy().astype('uint8')
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

        return 

    def uda_train(self):
        start_epoch = self.current_epoch        
        for i in range(start_epoch, self.epochs):
            if self.use_ddp:
                self.train_loader.sampler.set_epoch(i)
            training_loss = self.uda_train_one_epoch()
            self.writter.add_scalar('loss', sum(training_loss), global_step=i)
            self.current_epoch += 1
            current_model_fn = os.path.join(self.log_path, "last_model.pt")
            self.save_checkpoint(current_model_fn)
        self.writter.close()
    
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
            self.last_acc = IoU
            current_model_fn = os.path.join(self.log_path, "last_model.pt")
            self.save_checkpoint(current_model_fn)

            if IoU > self.best_acc:
                best_model_fn = os.path.join(self.log_path, "best_model.pt")
                self.best_acc = IoU
                self.save_checkpoint(best_model_fn)
                print("best model saved")
        
        self.writter.close()

