# model_classification.py
import torch.nn as nn
import torchvision.models as models

def build_resnet(num_classes=2):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid()  # for multilabel; replace with Softmax if single-label
    )
    return model
