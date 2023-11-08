import torch
import torch.nn as nn

def center_crop_tensor(target_dimension, input_tensor):
    delta_h = (input_tensor.shape[2] - target_dimension[2])//2
    delta_w = (input_tensor.shape[3] - target_dimension[3])//2
    end_h = target_dimension[2] + delta_h
    end_w = target_dimension[3] + delta_w
    return input_tensor[
        :,
        :,
        delta_h:end_h,
        delta_w:end_w
    ]