import numpy as np
import torch

# ground truth mask is shape (B, H, W)
# pred mask is shape (B, H, W)

def global_accuracy_score(groundtruth_mask, pred_mask):
    total_correct = groundtruth_mask == pred_mask
    total_correct = total_correct.sum().item()
    total_pixels = groundtruth_mask.numel()
    return total_correct/total_pixels

def class_average_accuracy_score(groundtruth_mask, pred_mask, num_classes):
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for cls in range(num_classes):
        cls_mask = groundtruth_mask == cls
        class_total[cls] = cls_mask.sum().item()
        class_correct[cls] = ((pred_mask == cls) & cls_mask).sum().item()
    
    valid = class_total != 0
    total_accuracy = (class_correct[valid]/class_total[valid]).mean().item()
    return total_accuracy

def mean_IOU(groundtruth_mask, pred_mask, num_classes):
    class_IOU = torch.zeros(num_classes)

    for cls in range(num_classes):
        cls_mask = groundtruth_mask == cls
        intersection = ((pred_mask == cls) & cls_mask).sum().item()
        union = cls_mask.sum().items() + (pred_mask == cls).sum().items() - intersection
        class_IOU[cls] = (intersection/union) if union > 0 else 0

    total_accuracy = class_IOU.mean().item()
    return total_accuracy

def freq_weighted_IOU(groundtruth_mask, pred_mask, num_classes):
    class_IOU = torch.zeros(num_classes)
    total_pixels = groundtruth_mask.numel()

    for cls in range(num_classes):
        cls_mask = groundtruth_mask == cls
        pred_cls_mask = pred_mask == cls
        intersection = (pred_cls_mask & cls_mask).sum().item()
        union = cls_mask.sum().item() + pred_cls_mask.sum().item() - intersection
        IOU = (intersection/union) if union > 0 else 0
        cls_freq = cls_mask.sum().item()
        class_IOU[cls] = cls_freq * IOU / total_pixels
    
    return class_IOU.sum().item()

#micro
def precision_score(groundtruth_mask, pred_mask, num_classes):
    TPs = torch.zeros(num_classes)
    TP_FPs = torch.zeros(num_classes)

    for cls in range(num_classes):
        cls_mask = groundtruth_mask == cls
        pred_cls_mask = pred_mask == cls
        TP = (cls_mask & pred_cls_mask).sum().item()
        TP_FP = pred_cls_mask.sum().item()
        TPs[cls] = TP
        TP_FPs[cls] = TP_FP

    precision = TPs.sum().item()/TP_FPs.sum().item() if TP_FPs.sum().item() > 0 else 0
    return precision

#micro
def recall_score(groundtruth_mask, pred_mask, num_classes):
    TPs = torch.zeros(num_classes)
    TP_FNs = torch.zeros(num_classes)

    for cls in range(num_classes):
        cls_mask = groundtruth_mask == cls
        pred_cls_mask = pred_mask == cls
        TP = (cls_mask & pred_cls_mask).sum().item()
        TP_FN = cls_mask.sum().item()
        TPs[cls] = TP
        TP_FNs[cls] = TP_FN

    recall = TPs.sum().item()/TP_FNs.sum().item() if TP_FNs.sum().item() > 0 else 0
    return recall

#micro
def f1_score(groundtruth_mask, pred_mask, num_classes):
    precision = precision_score(groundtruth_mask, pred_mask, num_classes)
    recall = recall_score(groundtruth_mask, pred_mask, num_classes)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1
