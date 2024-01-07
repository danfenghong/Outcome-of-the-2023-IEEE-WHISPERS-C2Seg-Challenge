import torch
from torch._C import dtype
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def to_one_hot(tensor, n_classes):
    device = tensor.device
    if len(tensor.size()) >= 4:
        tensor = torch.squeeze(tensor)
    tensor = tensor.long()
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, h, w).to(device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot
 
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def	forward(self, input, target):

        N = target.size(0) # batch size
        smooth = 1.0
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss
 
        return loss

class BinaryDiceLoss(nn.Module):
    def __init__(self, reduction='mean', labelsmooth=None):
        super(BinaryDiceLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def	forward(self, input, target):
        assert len(input.size()) == 3 or input.size(1) == 1, "this loss is just designed for 2-cls classification."
        input = torch.sigmoid(input.squeeze())

        if self.smooth is not None:
            ls = LabelSmooth(2, self.smooth)
            target = ls(target)

        N = target.size(0) # batch size
        smooth = 1.0
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss
 
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
 
class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, reduction='mean', labelsmooth=None):
        # ignore index can be int or list
        super(MulticlassDiceLoss, self).__init__()
        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        C = input.shape[1]
        max = torch.max(input, dim=1).values.unsqueeze(dim=1)
        input = F.softmax(input - max, dim=1)

        if len(target.size()) <= 3:
            target = to_one_hot(target, C)
        
        if self.smooth is not None:
            ls = LabelSmooth(C, self.smooth)
            target = ls(target)

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
 
        dice = DiceLoss()
        totalLoss = 0

        if self.weight is None:
            self.weight = torch.ones(C).to(target.device) #uniform weights for all classes
        else:
            self.weight = torch.tensor(self.weight).to(target.device)
 
        for i in range(C):
            if self.ignore_index is None or i not in self.ignore_index:
                diceLoss = dice(input[:,i], target[:,i])
                if self.weight is not None:
                    diceLoss *= self.weight[i]
                totalLoss += diceLoss
        
        if self.reduction == 'none':
            return totalLoss
        elif self.reduction == 'mean':
            return totalLoss.mean()
        elif self.reduction == 'sum':
            return totalLoss.sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")

# class AugmentedCELoss(nn.Module):
class LabelSmooth(nn.Module):
    def __init__(self, n_classes=10, eps=0.1):
        super(LabelSmooth, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, labels):
        # labels.shape: [b,]
        if len(labels.size()) <= 3 or self.n_classes > 2:
            one_hot_key = to_one_hot(labels, self.n_classes)
        else:
            one_hot_key = labels
        mask = ~(one_hot_key > 0)
        smooth_labels = torch.masked_fill(one_hot_key, mask, self.eps / (self.n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - self.eps).to(labels.device)
        return smooth_labels

class CELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, reduction='none', labelsmooth=None):
        super(CELoss, self).__init__()
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        # labels.shape: [b,]
        C = input.shape[1]
        logpt = F.log_softmax(input, dim=1)
        
        if len(target.size()) <= 3:
            one_hot_key = to_one_hot(target, logpt.shape[1])
        else:
            one_hot_key = target
        
        if self.smooth is not None:
            ls = LabelSmooth(C, self.smooth)
            one_hot_key = ls(one_hot_key)
        
        if self.weight is None:
            weight = torch.ones(logpt.shape[1]).to(one_hot_key.device) #uniform weights for all classes
        else:
            weight = torch.tensor(self.weight).to(one_hot_key.device)
        
        for i in range(len(logpt.shape)):
            if i != 1:
                weight = torch.unsqueeze(weight, dim=i)
        
        s_weight = weight * one_hot_key
        for i in range(one_hot_key.shape[1]):
            if self.ignore_index is not None and i in self.ignore_index:
                one_hot_key[:,i] = - one_hot_key[:,i]
                s_weight[:,i] = 0
        s_weight = s_weight.sum(1)

        loss = -1 * weight * logpt * one_hot_key
        loss = loss.sum(1)
        if self.reduction == 'none':
            return torch.where(loss > 0, loss, torch.zeros_like(loss).to(loss.device))
        elif self.reduction == 'mean':
            if s_weight.sum() == 0:
                return loss[torch.where(loss > 0)].sum()
            else:
                return loss[torch.where(loss > 0)].sum() / s_weight[torch.where(loss > 0)].sum()
        elif self.reduction == 'sum':
            return loss[torch.where(loss > 0)].sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")

class BCELoss(nn.Module):
    # Just for Foreground and Background Segmentation (1 foreground class and 'background' class)
    def __init__(self, reduction='none', labelsmooth=None):
        super(BCELoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        # labels.shape: [b,]
        assert len(input.size()) == 3 or input.size(1) == 1, "this loss is just designed for 2-cls classification."
        B = target.size(0)
        logpt = F.logsigmoid(input.squeeze())
        logpt_1 = F.logsigmoid(-input.squeeze())
        one_hot_key = target
        
        if self.smooth is not None:
            ls = LabelSmooth(2, self.smooth)
            one_hot_key = ls(one_hot_key)

        loss = -1 * (logpt * one_hot_key + logpt_1 * (1 - one_hot_key))
        if self.reduction == 'none':
            return loss.view(B, -1).mean(1)
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
    
class OhemCELoss(nn.Module):
    def __init__(self, thresh, batch_num_min, weight=None, ignore_index=None, reduction='none', labelsmooth=None):
        super(OhemCELoss, self).__init__()

        # transfer thresh to -log(p) type
        # thresh indicates the minimum probability of being predicted as target
        assert thresh >= 0 and thresh <= 1, "thresh should be a float number in [0, 1]"
        if abs(thresh) < 1e-7:
            self.thresh = torch.tensor(0, dtype=torch.float)
        else:
            self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))

        # batch_num_min indicates the minimum number of training samples in a batch for back propagation
        self.batch_num_min = batch_num_min
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        C = input.shape[1]
        S = 1
        # sample number per batch per channel (H*W or H*W*D)
        for i in range(2, len(input.shape)):
            S *= input.shape[i]
        self.thresh.to(input.device)

        logpt = F.log_softmax(input, dim=1)

        if self.smooth is not None:
            ls = LabelSmooth(C, self.smooth)
            target = ls(target)
        
        if len(target.size()) <= 3:
            one_hot_key = to_one_hot(target, input.shape[1])
        else:
            one_hot_key = target
        
        if self.weight is None:
            weight = torch.ones(input.shape[1]).to(one_hot_key.device) #uniform weights for all classes
        else:
            weight = torch.tensor(self.weight).to(one_hot_key.device)
        
        for i in range(len(logpt.shape)):
            if i != 1:
                weight = torch.unsqueeze(weight, dim=i)
        
        s_weight = weight * one_hot_key
        for i in range(one_hot_key.shape[1]):
            if self.ignore_index is not None and i in self.ignore_index:
                one_hot_key[:,i] = - one_hot_key[:,i]
                s_weight[:,i] = 0
        s_weight = s_weight.sum(1).view(-1)

        loss = -1 * weight * logpt * one_hot_key
        loss = loss.sum(1).view(-1)
        loss, indices = torch.sort(loss, descending=True) # Find the top N (batch_num_min) samples with the largest loss
        s_weight = s_weight[indices] # rearrange the weights

        # Consider at least n_min pixels with the largest loss. 
        # If the loss of the smallest one of the first batch_num_min losses is still greater than the set threshold, 
        # then take all the elements that are greater than the threshold to calculate the loss.
        # Otherwise, calculate the first batch_num_min losses.
        if loss[self.batch_num_min] > (self.thresh * s_weight[indices[self.batch_num_min] // S]):
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.batch_num_min]

        if self.reduction == 'none':
            return torch.where(loss > 0, loss, torch.zeros_like(loss).to(loss.device))
        elif self.reduction == 'mean':
            if s_weight.sum() == 0:
                return loss[torch.where(loss > 0)].sum()
            else:
                return loss[torch.where(loss > 0)].sum() / s_weight[torch.where(loss > 0)].sum()
        elif self.reduction == 'sum':
            return loss[torch.where(loss > 0)].sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, weight=None, ignore_index=None, reduction='none', labelsmooth=None, norm=True, annealing_scale=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smooth = labelsmooth
        self.annealing_scale = annealing_scale
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction

        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")

        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        
        if self.annealing_scale is not None:
            assert isinstance(self.annealing_scale, (int, float))
            if self.annealing_scale < 0 or self.annealing_scale > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        
        self.norm = norm

    def forward(self, input, target):
        C = input.shape[1]

        max = torch.max(input, dim=1).values.unsqueeze(dim=1)
        logit = F.softmax(input - max, dim=1)

        logpt = F.log_softmax(input, dim=1)
        
        if len(target.size()) <= 3:
            one_hot_key = to_one_hot(target, C)
        else:
            one_hot_key = target
        
        if self.smooth is not None:
            ls = LabelSmooth(C, self.smooth)
            one_hot_key = ls(one_hot_key)

        if self.weight is None:
            weight = torch.ones(input.shape[1]).to(one_hot_key.device) #uniform weights for all classes
        else:
            weight = torch.tensor(self.weight).to(one_hot_key.device)
        
        for i in range(len(logpt.shape)):
            if i != 1:
                weight = torch.unsqueeze(weight, dim=i)
        
        s_weight = weight * one_hot_key
        for i in range(one_hot_key.shape[1]):
            if self.ignore_index is not None and i in self.ignore_index:
                one_hot_key[:,i] = - one_hot_key[:,i]
                s_weight[:,i] = 0
        s_weight = s_weight.sum(1)

        gamma = self.gamma

        loss = -1 * weight * torch.pow((1 - logit), gamma) * logpt * one_hot_key
        loss_ori = -1 * weight * logpt * one_hot_key
        loss_s = torch.where(loss > 0, loss, torch.zeros_like(loss).to(loss.device)).sum()
        loss_ori_s = torch.where(loss_ori > 0, loss_ori, torch.zeros_like(loss_ori).to(loss_ori.device)).sum()
        norm_const = loss_ori_s / loss_s

        loss = loss.sum(1)
        if self.norm:
            if self.annealing_scale is None:
                loss *= norm_const
            else:
                loss = loss + self.annealing_scale * (loss_ori - loss)

        if self.reduction == 'none':
            return torch.where(loss > 0, loss, torch.zeros_like(loss).to(loss.device))
        elif self.reduction == 'mean':
            if s_weight.sum() == 0:
                return loss[torch.where(loss > 0)].sum()
            else:
                return loss[torch.where(loss > 0)].sum() / s_weight[torch.where(loss > 0)].sum()
        elif self.reduction == 'sum':
            return loss[torch.where(loss > 0)].sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")

class ComposedLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=None, reduction='mean', labelsmooth=None, rate=1):
        super(ComposedLoss, self).__init__()
        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        self.rate = rate
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        
        self.celoss = CELoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, labelsmooth=self.smooth)
        self.diceloss = MulticlassDiceLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, labelsmooth=self.smooth)

    def forward(self, input, target):
        loss1 = self.celoss(input, target)
        loss2 = self.diceloss(input, target)
        return loss1 + self.rate * loss2

class ComposedFocalLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=None, reduction='mean', labelsmooth=None, rate=1):
        super(ComposedFocalLoss, self).__init__()
        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        self.rate = rate
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        
        self.focalloss = FocalLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, labelsmooth=self.smooth)
        self.diceloss = MulticlassDiceLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, labelsmooth=self.smooth)

    def forward(self, input, target):
        loss1 = self.focalloss(input, target)
        loss2 = self.diceloss(input, target)
        return loss1 + self.rate * loss2

class BinaryComposedLoss(nn.Module):

    def __init__(self, reduction='mean', labelsmooth=None, rate=1):
        super(BinaryComposedLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        self.smooth = labelsmooth
        self.rate = rate
        if self.smooth is not None:
            assert isinstance(self.smooth, (int, float))
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        
        self.bceloss = BCELoss(reduction=self.reduction, labelsmooth=self.smooth)
        self.bdcloss = BinaryDiceLoss(reduction=self.reduction, labelsmooth=self.smooth)

    def forward(self, input, target):
        loss1 = self.bceloss(input, target)
        loss2 = self.bdcloss(input, target)
        return loss1 + self.rate * loss2

if __name__ == '__main__':
    
    input = torch.autograd.Variable(torch.rand(3, 16, 256, 256))

    x = torch.zeros((3, 256, 256)).long()
    # x = torch.randint_like(x, low=0, high=16)
    # x[0, 1, 0] = 1
    # x[0, 1, 2] = 1
    # x[0, 2, 1] = 2
    # x[0, 2, 2] = 1
    # x[0, 2, 4] = 2
    # x[0, 3, 3] = 3
    # x[0, 4, 1] = 3
    # x[1, 1, 0] = 3
    # x[1, 1, 2] = 1
    # x[1, 2, 1] = 2
    # x[1, 2, 2] = 2
    # x[1, 2, 4] = 2
    # x[1, 3, 3] = 1
    # x[1, 4, 1] = 3
    # x[2, 1, 0] = 1
    # x[2, 1, 2] = 2
    # x[2, 2, 1] = 2
    # x[2, 2, 2] = 1
    # x[2, 2, 4] = 2
    # x[2, 3, 3] = 1
    # x[2, 4, 1] = 3

    # print(input)
    # print(x)
    # print(to_one_hot(x, 4))
    # mcd = MulticlassDiceLoss()
    # mcd1 = mcd(input, x)
    # print(mcd1)
    # mcd2 = MulticlassDiceLoss()
    # m = mcd2(input, x)
    # mcd3 = mcd2(input, x)
    # print(mcd3)

    # ls = LabelSmooth(4, 0.1)
    # weight = [0.1, 0.2, 0.3, 0.7]
    ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    cel = CELoss(reduction='mean', ignore_index=0, labelsmooth=None)
    # oce = OhemCELoss(thresh=0, batch_num_min=1, reduction='mean', ignore_index=None, labelsmooth=None)
    # fl = FocalLoss(reduction='mean', gamma=2, ignore_index=None, labelsmooth=None, norm=True)
    print(ce(input, x))
    print(cel(input, x))
    # print(oce(input, x))
    # print(fl(input, x))

    # bce = BCELoss(reduction='none')
    # bcel = nn.BCEWithLogitsLoss(reduction='mean')
    # bdc = BinaryDiceLoss(reduction='none')
    # bc = BinaryComposedLoss(reduction='mean')
    # print(bce(input, x))
    # print(bcel(input.squeeze(), x.float()))
    # print(bdc(input, x))
    # print(bc(input, x))