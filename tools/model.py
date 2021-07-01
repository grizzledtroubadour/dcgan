import os
import toml
from torchvision import models
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet


# ResNet18 
def load_resnet18(category_num=False, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    
    # change the output --> category_num
    if category_num:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, category_num)
    
    return model

# EfficientNet b4 model
def load_efficientnet_b4(category_num=False, pretrained=True):
    if pretrained == True:
        model = EfficientNet.from_pretrained('efficientnet-b4') # pretrained model
    else:
        model = EfficientNet.from_name('efficientnet-b4') # original model
    
    # change the output --> category_num
    if category_num:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, category_num, bias=True)
        
    return model