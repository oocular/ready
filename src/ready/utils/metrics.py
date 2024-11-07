import numpy as np 
import torch
import torch.nn.functional as F

"""
See pixel_accuracy, mIoU :
https://github.com/tanishqgautam/Drone-Image-Semantic-Segmentation/blob/main/semantic-segmentation-pytorch.ipynb

"""

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=1):
    """
        Mean Intersection over Union over defined number of classes.
        Args:
            pred_mask: predicted mask
            mask: ground truth mask
            smooth: smoothing value
            n_classes: number of classes
    """
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)