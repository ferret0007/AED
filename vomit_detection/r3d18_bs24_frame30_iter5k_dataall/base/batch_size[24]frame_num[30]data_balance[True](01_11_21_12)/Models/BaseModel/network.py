import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class network(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.model =  torchvision.models.video.r3d_18(pretrained=True)
        self.head = nn.Linear(512, 2)

    def model_forward(self, x):
        
        x = self.model.stem(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.flatten(1)
        return x

    def forward(self, x, output):
        x = self.model_forward(x)
        x = self.head(x)
        return x
