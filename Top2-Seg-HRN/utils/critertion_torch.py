from typing import IO
from matplotlib.pyplot import axis
import torch
from torch._C import dtype
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


# torch-based metric
class ConfusionMatrixBasedMetric:
    def __init__(self, input, target, classes_name_list=None):
        assert len(input.shape) == 4 or len(input.shape) == 5, "the input predicted tensor should be a 4-D tensor or 5-D tensor"
        assert len(target.shape) == 3, "the input label tensor should be a 3-D tensor"
        self.input = input
        self.class_num = input.shape[1]
        self.batch_num = input.shape[0]
        self.target = target.view(self.batch_num, -1)
        if classes_name_list is not None:
            assert isinstance(classes_name_list, list)
            assert self.class_num == len(classes_name_list)
            self.class_name_list = classes_name_list
        else:
            self.class_name_list = [i for i in range(self.class_num)]

        # build confusion matrix
        pred_index = torch.argmax(input, dim=1).view(self.batch_num, -1)
        target_array = self.target.cpu().numpy()
        pred_array = pred_index.cpu().numpy()
        self.confusion_matrix = torch.zeros((self.batch_num, self.class_num, self.class_num)).to(input.device)
        try:
            from sklearn.metrics import confusion_matrix
            for i in range(self.batch_num):
                x = confusion_matrix(target_array[i], pred_array[i])
                self.confusion_matrix[i] = torch.from_numpy(x).to(input.device)
        except ModuleNotFoundError:
            # this way is much slower than sklearn-implemented way
            for i in range(self.batch_num):
                for j in range(pred_index.shape[1]):
                    self.confusion_matrix[i, self.target[i, j], pred_index[i, j]] += 1
        self.total_confusion_matrix = self.confusion_matrix.sum(0)

        self.class_frequency = torch.zeros(self.class_num).to(input.device)
        for i in range(self.class_num):
            freq = len(torch.where(self.target == i)[0])
            self.class_frequency[i] = freq / float(self.target.numel())
    
    @staticmethod
    def to_one_hot(tensor, n_classes):
        device = tensor.device
        if len(tensor.size()) >= 4:
            tensor = torch.squeeze(tensor)
        tensor = tensor.long()
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w).to(device), 1).to(device)
        return one_hot
    
    def plot_confusion_matrix(self, batch_idx=None, savename=None, title='Confusion Matrix'):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8), dpi=100)
        np.set_printoptions(precision=2)

        if batch_idx is None:
            cm = self.total_confusion_matrix.numpy()
        else:
            cm = self.confusion_matrix[batch_idx].numpy()
        
        ind_array = np.arange(self.class_num)
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=15, va='center', ha='center')
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(self.class_num))
        plt.xticks(xlocations, self.class_name_list, rotation=90)
        plt.yticks(xlocations, self.class_name_list)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')
    
        # offset the tick
        tick_marks = np.array(range(self.class_num)) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
    
        # show confusion matrix
        if savename is not None:
            plt.savefig(savename, format='png')
        plt.show()
    
    def calculate_TPTNFPFN(self):
        TP = torch.zeros((self.batch_num, self.class_num)).to(self.confusion_matrix.device)
        TN = torch.zeros((self.batch_num, self.class_num)).to(self.confusion_matrix.device)
        FP = torch.zeros((self.batch_num, self.class_num)).to(self.confusion_matrix.device)
        FN = torch.zeros((self.batch_num, self.class_num)).to(self.confusion_matrix.device)
        for i in range(self.batch_num):
            cm = self.confusion_matrix[i]
            TP[i] = torch.diag(cm)
            FP[i] = cm.sum(axis=0) - torch.diag(cm)
            FN[i] = cm.sum(axis=1) - torch.diag(cm)
            TN[i] = cm.sum() - (FP[i] + FN[i] + TP[i])
        res_dict = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
        return res_dict
    
    def calculate_Metric(self, metric='IoU', class_idx=None, reduce=True, freq_weighted=False, beta=1):
        # 'reduce' means average over a whole batch
        res_dict = self.calculate_TPTNFPFN()
        assert metric in ['IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F-score', 'Kappa', 'DiceCoeff'], "Metric should be selected from {'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F-score', 'Kappa', 'DiceCoeff'}"
        if metric == 'IoU':
            metric_result = res_dict['TP'] / (res_dict['TP'] + res_dict['FP'] + res_dict['FN'])
        elif metric == 'Accuracy':
            metric_result = res_dict['TP'] / (res_dict['TP'] + res_dict['FP'] + res_dict['FN'] + res_dict['TN'])
        elif metric == 'Precision':
            metric_result = res_dict['TP'] / (res_dict['TP'] + res_dict['FP'])
        elif metric == 'Recall' or metric == 'Sensitivity':
            metric_result = res_dict['TP'] / (res_dict['TP'] + res_dict['FN'])
        elif metric == 'Specificity':
            metric_result = res_dict['TN'] / (res_dict['TN'] + res_dict['FP'])
        elif metric == 'F-score':
            precision = res_dict['TP'] / (res_dict['TP'] + res_dict['FP'])
            recall = res_dict['TP'] / (res_dict['TP'] + res_dict['FN'])
            metric_result = (1 + beta * beta) / (beta * beta / recall + 1 / precision)
        elif metric == 'DiceCoeff':
            precision = res_dict['TP'] / (res_dict['TP'] + res_dict['FP'])
            recall = res_dict['TP'] / (res_dict['TP'] + res_dict['FN'])
            metric_result = 2 / (1 / recall + 1 / precision)
        elif metric == 'Kappa':
            actual = self.confusion_matrix.sum(2)
            pred = self.confusion_matrix.sum(1)
            all = self.confusion_matrix.sum(1).sum(1)
            diag = []
            for i in range(self.confusion_matrix.shape[0]):
                diag.append(torch.diag(self.confusion_matrix[i]).sum())
            diag = torch.tensor(diag)
            p0 = diag / all
            pe = (actual * pred).sum(1) / (all * all)
            metric_result = (p0 - pe) / (1 - pe)

            actual_total = self.total_confusion_matrix.sum(1)
            pred_total = self.total_confusion_matrix.sum(0)
            all_total = self.total_confusion_matrix.sum(0).sum(0)
            diag_total = torch.diag(self.total_confusion_matrix).sum()
            p0_total = diag_total / all_total
            pe_total = (actual_total * pred_total).sum() / (all_total * all_total)
            metric_result_total = (p0_total - pe_total) / (1 - pe_total)
        else:
            raise ValueError("Metric should be selected from {'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F-score', 'Kappa', 'DiceCoeff'}")

        if class_idx is None: # all classes
            if freq_weighted: # FW-metric_result
                if metric != 'Kappa':
                    metric_result = metric_result * self.class_frequency
                    if reduce:
                        result = metric_result.sum(1).mean()
                    else:
                        result = metric_result.sum(1)
                else:
                    raise ValueError("Kappa Metric is not compatible with FW-metric method")
            else: #M-metric_result
                if metric != 'Kappa':
                    if reduce:
                        result = metric_result.mean()
                    else:
                        result = metric_result.mean(1)
                else:
                    if reduce:
                        result = metric_result_total
                    else:
                        result = metric_result
        else: # specific class
            if metric != 'Kappa':
                assert isinstance(class_idx, int) and class_idx >= 0 and class_idx < len(self.class_frequency), "class_idx should be int and within the range of len(self.class_frequency)"
                if reduce:
                    result = metric_result[:,class_idx].mean()
                else:
                    result = metric_result[:,class_idx]
            else:
                raise ValueError("There is no Kappa Metric for specific classes")
        return result