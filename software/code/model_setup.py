from functools import partial
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Tuple
from torchinfo import summary

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
        _freeze_parameters(model)

    model.fc = nn.Linear(model.fc.in_features, len(class_names) - 1).to(device)

    return model, preprocess

def setup_swin_transformer(version: int,
                           size: str,
                           pretrained: bool,
                           class_names: List[str],
                           device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if version == 1:
        if size == "tiny":
            weights = models.Swin_T_Weights.DEFAULT
            model = models.swin_t(weights) if pretrained else models.swin_t()
        elif size == "small":
            weights = models.Swin_S_Weights.DEFAULT
            model = models.swin_s(weights) if pretrained else models.swin_s()
        elif size == "base":
            weights = models.Swin_B_Weights.DEFAULT
            model = models.swin_b(weights) if pretrained else models.swin_b()

    elif version == 2:
        if size == "tiny":
            weights = models.Swin_V2_T_Weights.DEFAULT
            model = models.swin_v2_t(weights) if pretrained else models.swin_v2_t()
        elif size == "small":
            weights = models.Swin_V2_S_Weights.DEFAULT
            model = models.swin_v2_s(weights) if pretrained else models.swin_v2_s()
        elif size == "base":
            weights = models.Swin_V2_B_Weights.DEFAULT
            model = models.swin_v2_b(weights) if pretrained else models.swin_v2_b()

    model.to(device)
    preprocess = weights.transforms()

    if pretrained:
        _freeze_parameters(model)

    features = model.head.in_features
    model.head = nn.Linear(features, len(class_names) - 1).to(device)

    return model, preprocess

def setup_convnext(size: str,
                   pretrained: bool,
                   class_names: List[str],
                   device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if size == "tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights) if pretrained else models.convnext_tiny()
    elif size == "small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        model = models.convnext_small(weights) if pretrained else models.convnext_small()
    elif size == "base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = models.convnext_base(weights) if pretrained else models.convnext_base()
    elif size == "large":
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        model = models.convnext_large(weights) if pretrained else models.convnext_large()

    model.to(device)
    preprocess = weights.transforms()

    if pretrained:
        _freeze_parameters(model)

    lastconv_output_channels = 1024
    norm_layer = partial(models.convnext.LayerNorm2d, eps=1e-6)

    model.classifier = nn.Sequential(
        norm_layer(lastconv_output_channels),
        nn.Flatten(1),
        nn.Linear(lastconv_output_channels, len(class_names) - 1)
    ).to(device)

    return model, preprocess

def _freeze_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False