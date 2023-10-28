import numpy as np
import torch

# ground truth mask is shape (B, H, W)
# pred mask is shape (B, H, W)

def global_accuracy_score(groundtruth_masks, pred_masks):
    total_correct = 0
    total_pixels = 0
    for groundtruth_mask, pred_mask in zip(groundtruth_masks, pred_masks):
        correct = groundtruth_mask == pred_mask
        total_correct += correct.sum().item()
        total_pixels += groundtruth_mask.numel()
    return total_correct/total_pixels

def class_average_accuracy_score(groundtruth_masks, pred_masks, num_classes):
    total_accuracy = torch.zeros(groundtruth_masks.shape[0])
    
    for idx, (groundtruth_mask, pred_mask) in enumerate(zip(groundtruth_masks, pred_masks)):
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)

        for cls in range(num_classes):
            cls_mask = groundtruth_mask == cls
            class_total[cls] = cls_mask.sum().item()
            class_correct[cls] = ((pred_mask == cls) & cls_mask).sum().item()
        
        valid = class_total != 0
        if valid.any():
            total_accuracy[idx] = (class_correct[valid] / class_total[valid]).mean().item()
        else:
            total_accuracy[idx] = 0.0
    
    return total_accuracy.mean().item()
    

def mean_IOU(groundtruth_masks, pred_masks, num_classes):
    total_IOU = torch.zeros(groundtruth_masks.shape[0])
    for idx, (groundtruth_mask, pred_mask) in enumerate(zip(groundtruth_masks, pred_masks)):
        class_IOU = torch.zeros(num_classes)

        for cls in range(num_classes):
            cls_mask = groundtruth_mask == cls
            intersection = ((pred_mask == cls) & cls_mask).sum().item()
            union = cls_mask.sum().item() + (pred_mask == cls).sum().item() - intersection
            class_IOU[cls] = (intersection/union) if union > 0 else 0

        total_IOU[idx] = class_IOU.mean().item()

    return total_IOU.mean().item()

def freq_weighted_IOU(groundtruth_masks, pred_masks, num_classes):
    total_IOU = torch.zeros(groundtruth_masks.shape[0])
    for idx, (groundtruth_mask, pred_mask) in enumerate(zip(groundtruth_masks, pred_masks)):
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
        
        total_IOU[idx] = class_IOU.sum().item()
    
    return total_IOU.mean().item()

#micro
def precision_score(groundtruth_masks, pred_masks, num_classes):
    total_precision = torch.zeros(groundtruth_masks.shape[0])
    for idx, (groundtruth_mask, pred_mask) in enumerate(zip(groundtruth_masks, pred_masks)):
        TPs = torch.zeros(num_classes)
        TP_FPs = torch.zeros(num_classes)

        for cls in range(num_classes):
            cls_mask = groundtruth_mask == cls
            pred_cls_mask = pred_mask == cls
            TP = (cls_mask & pred_cls_mask).sum().item()
            TP_FP = pred_cls_mask.sum().item()
            TPs[cls] = TP
            TP_FPs[cls] = TP_FP

        total_precision[idx] = TPs.sum().item()/TP_FPs.sum().item() if TP_FPs.sum().item() > 0 else 0

    return total_precision.mean().item()

#micro
def recall_score(groundtruth_masks, pred_masks, num_classes):
    total_recall = torch.zeros(groundtruth_masks.shape[0])
    for idx, (groundtruth_mask, pred_mask) in enumerate(zip(groundtruth_masks, pred_masks)):
        TPs = torch.zeros(num_classes)
        TP_FNs = torch.zeros(num_classes)

        for cls in range(num_classes):
            cls_mask = groundtruth_mask == cls
            pred_cls_mask = pred_mask == cls
            TP = (cls_mask & pred_cls_mask).sum().item()
            TP_FN = cls_mask.sum().item()
            TPs[cls] = TP
            TP_FNs[cls] = TP_FN

        total_recall[idx] = TPs.sum().item()/TP_FNs.sum().item() if TP_FNs.sum().item() > 0 else 0
    
    return total_recall.mean().item() 

#micro
def f1_score(groundtruth_mask, pred_mask, num_classes):
    precision = precision_score(groundtruth_mask, pred_mask, num_classes)
    recall = recall_score(groundtruth_mask, pred_mask, num_classes)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1
