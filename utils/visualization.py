import matplotlib.pyplot as plt

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
    transformed_mask = mask * 12.75
    return transformed_mask.astype(int)