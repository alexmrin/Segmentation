import matplotlib.pyplot as plt
import torch

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

def visualize(pred_mask, groundtruth_mask):
    pred_transformed = voc_mask_transform(pred_mask)
    groundtruth_transformed = voc_mask_transform(groundtruth_mask)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(pred_transformed, cmap='hsv')
    ax[0].set_title('Predicted Mask')
    
    ax[1].imshow(groundtruth_transformed, cmap='hsv')
    ax[1].set_title('Ground Truth Mask')
    plt.show()

def voc_mask_transform(mask):
    transformed_mask = torch.zeros((*mask.shape, 3), dtype=torch.uint8)
    for idx, color in enumerate(voc_mask_colors):
        transformed_mask[idx == mask] = color
    
    return transformed_mask

