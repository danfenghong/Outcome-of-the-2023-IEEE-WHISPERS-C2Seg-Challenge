from functools import cmp_to_key
import os, time, sys, yaml, math
import numpy as np
import torch
from torch import nn, optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .dataset import RoadDataset, SequentialDistributedSampler
from .criterion import ConfusionMatrixBasedMetric
import models
from shutil import copyfile

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

LOG_ROOT = '/Top2-Seg-HRN/run'

class Visualizer:
    def __init__(self, args, log_path):
        self.opt = args
        cfg_file = open(self.opt.config, 'r')
        self.config = yaml.load(cfg_file, Loader=yaml.FullLoader)
        cfg_file.close()
        self.log_path = log_path
        self.writter = SummaryWriter(os.path.join(self.log_path, 'log'))
        self.dst_cfg_fn = os.path.join(self.log_path, os.path.basename(self.opt.config))
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

        aug_config_path = self.config['DATASET']['AUG_CONFIG_PATH']
        train_dataset = RoadDataset(self.config['DATASET']['TRAIN_DATASET_CSV'], aug_config_path, stat_filepath=self.config['DATASET']['STAT_PATH'], 
                                    use_aug=self.config['DATASET']['USE_AUG'], mode='train')
        self.train_size = train_dataset.chip_size

        test_dataset = RoadDataset(self.config['DATASET']['TEST_DATASET_CSV'], stat_filepath=self.config['DATASET']['STAT_PATH'], mode='test')
        self.num_batches_per_epoch_for_testing = len(test_dataset)

        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        if self.opt.resume_path == '':
            print('model was not created')
            self.model = models.build_model_vis(self.config['MODEL']['IMG_CH'], self.config['MODEL']['N_CLASSES'], model_key=self.config['MODEL']['NAME'], 
                                            pretrained_flag=self.config['MODEL']['PRETRAINED'], resume_path=self.config['MODEL']['RESUME_PATH']).to(self.device)
        elif os.path.exists(self.opt.resume_path):
            print('%s was existed' % (self.opt.resume_path))
            self.model = models.build_model_vis(self.config['MODEL']['IMG_CH'], self.config['MODEL']['N_CLASSES'], model_key=self.config['MODEL']['NAME']).to(self.device)

            if self.use_ddp:
                self.load_checkpoint(self.opt.resume_path, rank=self.local_rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.load_checkpoint(self.opt.resume_path)
        else:
            raise FileNotFoundError(self.opt.resume_path + " not existed")
    
    def load_checkpoint(self, resume_path, rank=None):
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path)
            if rank is not None:
                if rank in [-1, 0]:
                    self.model.load_state_dict(checkpoint['net'])
            else:
                self.model.load_state_dict(checkpoint['net'])
            print('load checkpoint successfully!')
        else:
            print('resume_path does not exist, we will train from scratch!')
    
    def view_params(self):
        if str.find(self.config['MODEL']['NAME'], 'SUA') != -1:
            relative_scale_bias = self.model.SUA.relative_scale_bias_table[self.model.SUA.relative_scale_index.view(-1)].view(
                                                                sum(self.model.SUA.square_size), sum(self.model.SUA.square_size), -1)  # Wh*Ww,Wh*Ww,nH
            relative_scale_bias = relative_scale_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            self.writter.add_image('rpb-scale', relative_scale_bias[0], dataformats='HW')

            relative_spatial_bias = self.model.SUA.relative_spatial_bias_table[self.model.SUA.relative_spatial_index.view(-1)].view(
                self.model.SUA.num_pyramid, self.model.SUA.num_pyramid, -1)  # Wh*Ww,Wh*Ww,nH
            relative_spatial_bias = relative_spatial_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            self.writter.add_image('rpb-spatial', relative_spatial_bias[0], dataformats='HW')

    def patch_based_inference(self, input_data, model, overlap_rate=0.5):
        overlap_size = (int(overlap_rate * self.train_size[0]), int(overlap_rate * self.train_size[1]))
        B, C, H, W = input_data.size()
        count = torch.zeros((input_data.size(0), self.config['MODEL']['N_CLASSES'], H, W))
        N_w = int(math.ceil((W - self.train_size[0]) / (self.train_size[0] - overlap_size[0])) + 1)
        N_h = int(math.ceil((H - self.train_size[1]) / (self.train_size[1] - overlap_size[1])) + 1)
        pad_w = int((N_w - 1) * (self.train_size[0] - overlap_size[0]) + self.train_size[0])
        pad_h = int((N_h - 1) * (self.train_size[1] - overlap_size[1]) + self.train_size[1])
        count = torch.zeros((input_data.size(0), self.config['MODEL']['N_CLASSES'], pad_h, pad_w))
        output = torch.zeros((input_data.size(0), self.config['MODEL']['N_CLASSES'], pad_h, pad_w))
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
    
    def vis_features(self):
        self.model.eval()

        if not self.use_ddp:
            iter = tqdm(enumerate(self.test_loader), total=self.num_batches_per_epoch_for_testing, file=sys.stdout)
        elif self.local_rank in [-1, 0]:
            iter = tqdm(enumerate(self.test_loader), total=self.num_batches_per_epoch_for_testing, file=sys.stdout)
        else:
            iter = enumerate(self.test_loader)
        
        with torch.no_grad():
            for batch_idx, (data, targets, index) in iter:
                data = data[:, :, 0:self.train_size[1], 0:self.train_size[0]].to(self.device)
                targets = targets[:, 0:self.train_size[1], 0:self.train_size[0]].to(self.device)

                if self.config['MODEL']['NAME'] == 'Swin_LinkNet_vis':
                    pyramids, outputs = self.model(data)
                    save_dict = {
                        'data': data,
                        'targets': targets,
                        'pyramids': pyramids,
                        'outputs': outputs
                    }
                    torch.save(save_dict, os.path.join(self.log_path, 'saved_features_%d.pt' % batch_idx))
                    # outputs = torch.argmax(outputs, dim=1)
                    # c1_grid = vutils.make_grid(pyramids[0].detach().transpose(0, 1).cpu(), nrow=5, padding=20, normalize=True, scale_each=True, pad_value=1)
                    # self.writter.add_image('data', data[0], dataformats='CHW')
                    # self.writter.add_image('outputs', outputs[0], dataformats='HW')
                    # self.writter.add_image('c1_%d' % batch_idx, c1_grid)
                    # self.writter.add_image('c2', pyramids[0][0, 0], dataformats='HW')
                    # self.writter.add_image('c3', pyramids[0][0, 0], dataformats='HW')
                    # self.writter.add_image('c4', pyramids[0][0, 0], dataformats='HW')
                elif self.config['MODEL']['NAME'] == 'Swin_LinkNet_SUA_vis':
                    pyramids, out_attn, pyramids_, outputs = self.model(data)
                    save_dict = {
                        'data': data,
                        'targets': targets,
                        'pyramids': pyramids,
                        'outputs': outputs,
                        'out_attn': out_attn,
                        'pyramids_': pyramids_
                    }
                    torch.save(save_dict, os.path.join(self.log_path, 'saved_features_%d.pt' % batch_idx))
                elif self.config['MODEL']['NAME'] == 'Swin_LinkNet_SUA_wo_vis':
                    pyramids, out_attn, pyramids_, outputs = self.model(data)
                    save_dict = {
                        'data': data,
                        'targets': targets,
                        'pyramids': pyramids,
                        'outputs': outputs,
                        'out_attn': out_attn,
                        'pyramids_': pyramids_
                    }
                    torch.save(save_dict, os.path.join(self.log_path, 'saved_features_%d.pt' % batch_idx))

                if batch_idx >= 10:
                    break
                
                # if batch_idx % 3 == 0:
                #     e = ConfusionMatrixBasedMetric(2)
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


                
                # outputs = self.patch_based_inference(data, self.model)
