import numpy as np 
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score

"""
See pixel_accuracy, mIoU :
https://github.com/tanishqgautam/Drone-Image-Semantic-Segmentation/blob/main/semantic-segmentation-pytorch.ipynb

"""

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=1):
    """
        Mean Intersection over Union (miou and jaccard score is actually the same) over defined number of classes.
        Equation: IoU = (|X & Y|)/ (|X or Y|)
        Args:
            pred_mask: predicted mask
            mask: ground truth mask
            smooth: smoothing value
            n_classes: number of classes
    """
    # with torch.no_grad():
    #     pred_mask = F.softmax(pred_mask, dim=1)
    #     pred_mask = torch.argmax(pred_mask, dim=1)
    #     pred_mask = pred_mask.contiguous().view(-1)
    #     mask = mask.contiguous().view(-1)

    iou_per_class = []
    for clas in range(0, n_classes): #loop per pixel class
        true_class = pred_mask == clas
        true_label = mask == clas

        if np.sum(true_label) == 0: #no exist label in this loop
            iou_per_class.append(np.nan)
        else:
            intersect = np.logical_and(true_class, true_label).sum()
            union = np.logical_or(true_class, true_label).sum()

            iou = (intersect + smooth) / (union +smooth)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class)

def dice(pred_mask, mask, smooth=1e-10, n_classes=1):

    """
        Calculate Dice Coefficient over defined number of classes.
        Equation: Dice = (2*|X & Y|)/ (|X| + |Y|)
        Args:
            pred_mask: predicted mask
            mask: ground truth mask
            smooth: smoothing value
            n_classes: number of classes
    """

    # with torch.no_grad():
    #     pred_mask = F.softmax(pred_mask, dim=1)
    #     pred_mask = torch.argmax(pred_mask, dim=1)
    #     pred_mask = pred_mask.contiguous().view(-1)
    #     mask = mask.contiguous().view(-1)

    dice_per_class = []
    for clas in range(0, n_classes): #loop per pixel class
        true_class = pred_mask == clas
        true_label = mask == clas

        if np.sum(true_label) == 0: #no exist label in this loop
            dice_per_class.append(np.nan)
        else:
            intersect = np.logical_and(true_class, true_label).sum()
            total = np.sum(true_class) + np.sum(true_label)

            dice = 2 * (intersect + smooth) / (total +smooth)
            dice_per_class.append(dice)
    return np.nanmean(dice_per_class)
    

def evaluate(pred_mask, mask, smooth=1e-10, n_classes=1):

    """
        Evaluate model performance using pixel accuracy, f1, recall, precision, fbeta, mIoU, Dice Coefficient.
        Args:
            pred_mask: predicted mask
            mask: ground truth mask
            smooth: smoothing value
            n_classes: number of classes

        Output:
            accuracy: pixel accuracy
            f1: f1 score
            recall: recall score
            precision: precision score
            fbeta: fbeta score
            miou: mIoU score
            dice: Dice Coefficient
    """

    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        pred_mask = pred_mask.cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()

        accuracy = accuracy_score(mask, pred_mask)
        f1 = f1_score(mask, pred_mask, average='weighted')
        recall = recall_score(mask, pred_mask, average='weighted')
        precision = precision_score(mask, pred_mask, average='weighted')
        fbeta = fbeta_score(mask, pred_mask, beta=1, average='weighted')
        miou = mIoU(pred_mask, mask, smooth, n_classes)
        dice_score = dice(pred_mask, mask, smooth, n_classes)

        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'fbeta': fbeta,
            'miou': miou,
            'dice': dice_score
        }

        return metrics
    