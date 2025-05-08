import torch.nn as nn
import torchvision.models as models

def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    in_feats = model.fc.in_features
    model.fc  = nn.Linear(in_feats, num_classes)
    return model
