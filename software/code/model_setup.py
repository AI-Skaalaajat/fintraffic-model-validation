import torch.nn as nn
from torchvision import models, transforms
from typing import List, Tuple

def setup_resnet(layers: int,
                 pretrained: bool,
                 class_names: List[str],
                 device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if layers == 18:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights) if pretrained else models.resnet18()
    if layers == 34:
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights) if pretrained else models.resnet34()
    if layers == 50:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights) if pretrained else models.resnet50()
    if layers == 101:
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights) if pretrained else models.resnet101()
    if layers == 152:
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights) if pretrained else models.resnet152()
    
    model.to(device)
    preprocess = weights.transforms()
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(class_names) - 1).to(device)

    return model, preprocess