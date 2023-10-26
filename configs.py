import os
import sys

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

import matplotlib.pyplot as plt

import args
import vars as v
import models
from cli import *

def _get_voc_segmentation_dataloaders():
    _backup_print = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    norm_params = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.Resize(args.image_dimension),
        transforms.CenterCrop(args.image_dimension),
        transforms.ToTensor(),
        transforms.Normalize(*norm_params)
    ])
    transform_target = transforms.Compose([
        transforms.Resize(args.image_dimension),
        transforms.CenterCrop(args.image_dimension),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: torch.squeeze(x, 0))
    ])
    dataset = datasets.VOCSegmentation(
        root=args.data_path, year='2012', download=True, transform=transform_train, target_transform=transform_target
    )
    trainset, validset = random_split(dataset, lengths=[.9, .1])
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    validloader = DataLoader(
        validset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    sys.stdout = _backup_print

    v.trainloader = trainloader
    v.validloader = validloader
'''
    test_mask = trainset[0][1].numpy().squeeze()  # Remove the channel dimension
    plt.imshow(test_mask, cmap='tab20b')  # Use a colormap suitable for labels
    plt.colorbar()  # Optional: Add a colorbar to help interpret the values
    plt.show()
    test_im = trainset[0][0].numpy().transpose(1,2 , 0)
    plt.imshow(test_im)
    plt.show()
    uniques = np.unique(test_mask)
    print(uniques)
    print(trainset[0][1].numpy().shape)
'''
def segnet_voc_segmentation():
    v.model = models.segnetVGG()
    _get_voc_segmentation_dataloaders()
