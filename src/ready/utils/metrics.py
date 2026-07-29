import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score,
                             precision_score, recall_score)

"""
See pixel_accuracy, mIoU :
https://github.com/tanishqgautam/Drone-Image-Semantic-Segmentation/blob/main/semantic-segmentation-pytorch.ipynb

"""

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=1):
    """
        Mean Intersection over Union (IoU) over defined number of classes.
        IoU and jaccard score is actually the same! For reference, please see Table 3 below:
        Maier-Hein, Lena, Annika Reinke, Patrick Godau, Minu D. Tizabi, Florian Buettner, Evangelia Christodoulou, Ben Glocker, et al.
        ‘Metrics Reloaded: Recommendations for Image Analysis Validation’. Nature Methods 21, no. 2 (February 2024): 195–212. https://doi.org/10.1038/s41592-023-02151-z.


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


def evaluate(pred_mask, mask, smooth=1e-10, n_classes=1, **kwargs):

    """
        Evaluate model performance using pixel accuracy, f1, recall, precision, fbeta, mIoU, Dice Coefficient.

        Args:
            pred_mask: predicted mask
            mask: ground truth mask
            smooth: smoothing value, default is 1e-10.
            n_classes: number of classes, default is 1.
            average: averaging method for f1, recall, precision, fbeta. Default is 'weighted'.
                    can be any of 'binary', 'micro', 'macro', 'weighted', 'samples'.

        Returns:
            dict: A dictionary containing the following metrics:
                - **accuracy** (float):
                    Pixel-wise accuracy, measuring the percentage of correctly classified pixels.
                    Range: 0 (worst) to 1 (best).
                    Example: If 900 out of 1000 pixels are correct, accuracy = 0.9.
                - **f1** (float):
                    The F1 score, the harmonic mean of precision and recall.
                    Range: 0 (worst) to 1 (best).
                    Example: If precision = 0.8 and recall = 0.7, F1 = 2 * (0.8 * 0.7) / (0.8 + 0.7) = 0.747.
                - **recall** (float):
                    The proportion of true positives identified among all actual positives.
                    Range: 0 (worst) to 1 (best).
                    Example: If 70 out of 100 positive pixels are correctly identified, recall = 0.7.
                - **precision** (float):
                    The proportion of true positives among all predicted positives.
                    Range: 0 (worst) to 1 (best).
                    Example: If 80 out of 100 predicted positive pixels are correct, precision = 0.8.
                - **fbeta** (float):
                    The F-beta score, a weighted harmonic mean of precision and recall, where beta specifies the weight.
                    Range: 0 (worst) to 1 (best).
                    Example: With beta = 1, it equals the F1 score.
                - **miou** (float):
                    The mean Intersection over Union (IoU) across all classes, measuring overlap between predicted and true regions.
                    It is also called the Jaccard score.
                    Range: 0 (worst) to 1 (best).
                    Example: IoU = intersection / union. If overlap = 30 and union = 70, IoU = 0.429.
                - **dice** (float):
                    The Dice coefficient, measuring the proportion of overlap between two sets, normalised by their size.
                    Range: 0 (no overlap) to 1 (perfect overlap).
                    Example: Dice = 2 * (intersection) / (size_predicted + size_ground_truth).
                    If intersection = 30, size_predicted = 50, and size_ground_truth = 70, Dice = 2 * 30 / (50 + 70) = 0.5.

            Examples:
                >>> pred_mask = torch.tensor([[0, 1], [1, 1]])
                >>> mask = torch.tensor([[0, 1], [0, 1]])
                >>> metrics = evaluate(pred_mask, mask)
                >>> print("Pixel Accuracy:", metrics["accuracy"])
                Pixel Accuracy: 0.75
                >>> print("F1 Score:", metrics["f1"])
                F1 Score: 0.8571
                >>> print("Recall:", metrics["recall"])
                Recall: 1.0
                >>> print("Precision:", metrics["precision"])
                Precision: 0.75
                >>> print("F-Beta Score:", metrics["fbeta"])
                F-Beta Score: 0.8571
                >>> print("mIoU:", metrics["miou"])
                mIoU: 0.6667
                >>> print("Dice Coefficient:", metrics["dice"])
                Dice Coefficient: 0.8
    """

    average = kwargs.get('average', 'weighted')

    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        pred_mask = pred_mask.cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()

        accuracy = accuracy_score(mask, pred_mask)
        f1 = f1_score(mask, pred_mask, average=average)
        recall = recall_score(mask, pred_mask, average=average)
        precision = precision_score(mask, pred_mask, average=average)
        fbeta = fbeta_score(mask, pred_mask, beta=1, average=average)
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
