import numpy as np

CITYSCAPES_COLORMAP = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)  # shape: (19, 3)

def decode_segmap(pred_mask, colormap=CITYSCAPES_COLORMAP):
    """
    pred_mask: (H, W), int64
    return: (H, W, 3), uint8
    """
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colormap):
        rgb[pred_mask == class_idx] = color
    return rgb