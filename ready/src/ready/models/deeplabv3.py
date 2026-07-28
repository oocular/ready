import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class DeepLabV3(nn.Module):
    """
    DeepLabV3 with MobileNetV3 large backbone
    """
    def __init__(self, nch_out=4):
        super(DeepLabV3, self).__init__()

        self.model = deeplabv3_mobilenet_v3_large(
            weights=None,       #no pretrained head so model can learn rather than memorise patterns
            weights_backbone="DEFAULT", # load pretrained weights for the backbone so model has visual knowledge before training begins
            num_classes=nch_out
        )
        #freeze the backbone parameters so that they are not updated during training
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """forward"""
        return self.model(x)['out']