import torch
import torch.nn as nn

from torchvision.models.segmentation import lraspp_mobilenet_v3_large
class LRASPP(nn.Module):
    """
    LRASPP with MobileNetV3 large backbone
    """
    def __init__(self, nch_out=4):
        super(LRASPP, self).__init__()

        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone="DEFAULT",
            num_classes=nch_out
        )

        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """forward"""
        return self.model(x)['out']