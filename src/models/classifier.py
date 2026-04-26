import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)