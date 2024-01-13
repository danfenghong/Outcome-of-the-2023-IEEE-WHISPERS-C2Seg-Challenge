from typing import IO
import torch
from torch._C import dtype
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class ConfusionMatrixBasedMetric:
    def __init__(self, num_classes, classes_name_list=None):
        self.num_classes = num_classes
        if classes_name_list is not None:
            assert isinstance(classes_name_list, list)
            assert self.num_classes == len(classes_name_list)
            self.class_name_list = classes_name_list
        else:
            self.class_name_list = [i for i in range(self.num_classes)]
        self.confusion_matrix = np.zeros((1, self.num_classes, self.num_classes))
        self.TP = np.zeros((1, self.num_classes))
        self.FN = np.zeros((1, self.num_classes))
        self.FP = np.zeros((1, self.num_classes))
        self.TN = np.zeros((1, self.num_classes))
        self.batch_num = 0
        self.sample_num = 0

        self.class_sample_num = np.zeros(self.num_classes)
        # self.class_frequency = torch.zeros(self.num_classes).to(input.device)

    
    def add_batch(self, target, input):
        assert len(input.shape) == 4 or len(input.shape) == 5, "the input predicted tensor should be a 4-D tensor or 5-D tensor"
        assert len(target.shape) == 3, "the input label tensor should be a 3-D tensor"
        sample_num_per_batch = input.shape[0]
        assert input.shape[1] == self.num_classes

        if self.num_classes > 1:
            pred_index = torch.argmax(input, dim=1)
        else:
            pred_index = torch.sigmoid(input).view(target.size())
            pred_index = torch.where(pred_index >= 0.5, torch.ones_like(pred_index), torch.zeros_like(pred_index))

        target_array = target.cpu().numpy()
        pred_array = pred_index.cpu().numpy()

        x = np.expand_dims(self._generate_matrix(target_array[0], pred_array[0]), axis=0)
        TP = np.expand_dims(np.diag(x[0]), axis=0)
        FP = np.expand_dims((x[0].sum(axis=0) - np.diag(x[0])), axis=0)
        FN = np.expand_dims((x[0].sum(axis=1) - np.diag(x[0])), axis=0)
        TN = np.expand_dims((x[0].sum() - (FP[0] + FN[0] + TP[0])), axis=0)
        for i in range(sample_num_per_batch - 1):
            temp_cm = np.expand_dims(self._generate_matrix(target_array[i + 1], pred_array[i + 1]), axis=0)
            temp_TP = np.expand_dims(np.diag(temp_cm[0]), axis=0)
            temp_FP = np.expand_dims((temp_cm[0].sum(axis=0) - np.diag(temp_cm[0])), axis=0)
            temp_FN = np.expand_dims((temp_cm[0].sum(axis=1) - np.diag(temp_cm[0])), axis=0)
            temp_TN = np.expand_dims((temp_cm[0].sum() - (temp_FP[0] + temp_FN[0] + temp_TP[0])), axis=0)
            x = np.concatenate((x, temp_cm), axis=0)
            TP = np.concatenate((TP, temp_TP), axis=0)
            FP = np.concatenate((FP, temp_FP), axis=0)
            FN = np.concatenate((FN, temp_FN), axis=0)
            TN = np.concatenate((TN, temp_TN), axis=0)
            del temp_cm, temp_TP, temp_FP, temp_FN, temp_TN
        if self.batch_num == 0:
            self.confusion_matrix = x
            self.TP = TP
            self.FP = FP
            self.FN = FN
            self.TN = TN
        else:
            self.confusion_matrix = np.concatenate((self.confusion_matrix, x), axis=0)
            self.TP = np.concatenate((self.TP, TP), axis=0)
            self.FP = np.concatenate((self.FP, FP), axis=0)
            self.FN = np.concatenate((self.FN, FN), axis=0)
            self.TN = np.concatenate((self.TN, TN), axis=0)
        del x, TP, FP, FN, TN
        self.batch_num += 1
        self.sample_num += sample_num_per_batch
        for i in range(self.num_classes):
            freq = len(np.where(target_array == i)[0])
            self.class_sample_num[i] += freq

    def _generate_matrix(self, gt_image, pre_image):
        if self.num_classes > 1:
            mask = (gt_image >= 0) & (gt_image < self.num_classes)
            label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
            count = np.bincount(label, minlength=self.num_classes**2)
            confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        else:
            mask = pre_image
            label = gt_image
            confusion_matrix = np.zeros((2, 2))
            confusion_matrix[0, 0] = (mask * label).sum() # TP
            confusion_matrix[1, 0] = (mask * (1 - label)).sum() # FP
            confusion_matrix[0, 1] = ((1 - mask) * label).sum() # FN
            confusion_matrix[1, 1] = ((1 - mask) * (1 - label)).sum() # TN
        return confusion_matrix

    def plot_confusion_matrix(self, sample_idx=None, savename=None, title='Confusion Matrix'):
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import axis
        plt.figure(figsize=(12, 8), dpi=100)
        np.set_printoptions(precision=2)

        if sample_idx is None:
            cm = self.confusion_matrix.sum(axis=0)
        else:
            cm = self.confusion_matrix[sample_idx]
        
        ind_array = np.arange(self.num_classes)
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=15, va='center', ha='center')
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(self.num_classes))
        plt.xticks(xlocations, self.class_name_list, rotation=90)
        plt.yticks(xlocations, self.class_name_list)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')
    
        # offset the tick
        tick_marks = np.array(range(self.num_classes)) + 0.5
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
    
    def calculate_Metric(self, metric='IoU', class_idx=None, reduce=True, freq_weighted=False, beta=1, binary=False):
        # 'reduce' means average over a whole batch
        self.class_frequency = self.class_sample_num / self.class_sample_num.sum()
        assert metric in ['IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F-score', 'Kappa', 'DiceCoeff'], "Metric should be selected from {'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F-score', 'Kappa', 'DiceCoeff'}"
        if metric == 'IoU':
            metric_result = self.TP / (self.TP + self.FP + self.FN + 1e-8)
        elif metric == 'Accuracy':
            metric_result = self.TP / (self.TP + self.FP + self.FN + self.TN)
        elif metric == 'Precision':
            metric_result = self.TP / (self.TP + self.FP + 1e-8)
        elif metric == 'Recall' or metric == 'Sensitivity':
            metric_result = self.TP / (self.TP + self.FN + 1e-8)
        elif metric == 'Specificity':
            metric_result = self.TN / (self.TN + self.FP)
        elif metric == 'F-score':
            precision = (self.TP / (self.TP + self.FP + 1e-8)) + 1e-8
            recall = (self.TP / (self.TP + self.FN + 1e-8)) + 1e-8
            metric_result = (1 + beta * beta) / (beta * beta / recall + 1 / precision)
        elif metric == 'DiceCoeff':
            precision = self.TP / (self.TP + self.FP) + 1e-8
            recall = self.TP / (self.TP + self.FN) + 1e-8
            metric_result = 2 / (1 / recall + 1 / precision)
        elif metric == 'Kappa':
            actual = self.confusion_matrix.sum(2)
            pred = self.confusion_matrix.sum(1)
            all = self.confusion_matrix.sum(1).sum(1)
            diag = []
            for i in range(self.confusion_matrix.shape[0]):
                diag.append(np.diag(self.confusion_matrix[i]).sum())
            diag = np.array(diag)
            p0 = diag / all
            pe = (actual * pred).sum(1) / (all * all)
            metric_result = (p0 - pe) / (1 - pe)

            self.total_confusion_matrix = self.confusion_matrix.sum(axis=0)

            actual_total = self.total_confusion_matrix.sum(1)
            pred_total = self.total_confusion_matrix.sum(0)
            all_total = self.total_confusion_matrix.sum(0).sum(0)
            diag_total = np.diag(self.total_confusion_matrix).sum()
            p0_total = diag_total / all_total
            pe_total = (actual_total * pred_total).sum() / (all_total * all_total)
            metric_result_total = (p0_total - pe_total) / (1 - pe_total)
        else:
            raise ValueError("Metric should be selected from {'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F-score', 'Kappa', 'DiceCoeff'}")

        if class_idx is None: # all classes
            if binary:
                if metric != 'Kappa':
                    metric_result = metric_result[:, 0]
                    if reduce:
                        result = metric_result.mean()
                    else:
                        result = metric_result
                else:
                    if reduce:
                        result = metric_result_total
                    else:
                        result = metric_result
            else:
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

if __name__ == '__main__':
    e = ConfusionMatrixBasedMetric(1)
    
    input = torch.autograd.Variable(torch.rand(3, 1, 256, 256))

    x = torch.zeros((3, 256, 256)).long()
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

    x[0, 1, 0] = 1
    x[0, 1, 2] = 1
    x[0, 2, 1] = 1
    x[0, 2, 2] = 1
    x[0, 2, 4] = 1
    x[0, 4, 1] = 1
    x[1, 1, 0] = 1
    x[1, 1, 2] = 1
    x[1, 2, 2] = 1
    x[1, 2, 4] = 1
    x[1, 3, 3] = 1
    x[1, 4, 1] = 1
    x[2, 1, 0] = 1
    x[2, 1, 2] = 1
    x[2, 2, 2] = 1
    x[2, 2, 4] = 1
    x[2, 4, 1] = 1

    e.add_batch(x, input)
    e.add_batch(x, input)
    # print(e.confusion_matrix)
    # print(e.TP)
    # print(e.FP)
    # print(e.FN)
    # print(e.TN)
    # recall = e.calculate_Metric(metric='Kappa', reduce=True, binary=True)
    recall = e.calculate_Metric(metric='Precision', reduce=False, binary=True)
    # print(e.confusion_matrix)
    print(recall)
    # e.add_batch(x, input)
    # e.plot_confusion_matrix(sample_idx=None)