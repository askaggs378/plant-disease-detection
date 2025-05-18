import torch.nn as nn
from timm import create_model

def build_vit(num_classes):
    model = create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

