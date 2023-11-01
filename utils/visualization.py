import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import numpy as np

import vars as v

voc_mask_colors = torch.tensor([
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 128),
    (128, 128, 128)
], dtype=torch.uint8)

def visualize(image, pred_mask, groundtruth_mask):
    image = image.numpy().transpose([1, 2, 0])
    pred_transformed = voc_mask_transform(pred_mask).numpy()
    groundtruth_transformed = voc_mask_transform(groundtruth_mask).numpy()

    fig, ax = plt.subplots(1, 4, figsize=(16, 8))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    ax[1].imshow(pred_transformed)
    ax[1].set_title('Predicted Mask')
    
    ax[2].imshow(groundtruth_transformed)
    ax[2].set_title('Ground Truth Mask')

    patches = [mpatches.Patch(color=voc_mask_colors[i].numpy()/255, label=v.mask_dict[i]) for i in range(len(voc_mask_colors))]
    ax[3].legend(handles=patches, bbox_to_anchor=(0.5, 1), loc='upper center')
    ax[3].axis('off')
    ax[3].set_title('Color Map Key')

    os.makedirs('../comparisons', exist_ok=True)
    plt.savefig(f'../comparisons/comparison_{v.current_epoch}.png')
    plt.close(fig)

def voc_mask_transform(mask):
    transformed_mask = torch.zeros((*mask.shape, 3), dtype=torch.uint8)
    for idx, color in enumerate(voc_mask_colors):
        transformed_mask[idx == mask] = color
    
    return transformed_mask
