import torch.nn as nn

from .backbones.efficientnet import build_efficientnet
from .classifier import ClassifierHead


class Model(nn.Module):
    def __init__(self, backbone_name, num_classes=5):
        super().__init__()
        
        if backbone_name == 'efficientnet':
            self.backbone, in_features = build_efficientnet('efficientnet_b0')
        else:
            raise ValueError("Unsupported model")
        self.classifier = ClassifierHead(in_features, num_classes)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x