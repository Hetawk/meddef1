# alexnet_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import logging

class AlexNetModel(nn.Module):
    def __init__(self, input_channels=3, pretrained=False):
        logging.info(f"Initializing AlexNetModel with {input_channels} input channels.")
        super(AlexNetModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

        if pretrained:
            self.load_pretrained_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def load_pretrained_weights(self):
        # Load pretrained weights from torchvision models
        pretrained_model = models.alexnet(pretrained=True)

        # Transfer weights from pretrained model to self
        self.features.load_state_dict(pretrained_model.features.state_dict())
        self.classifier.load_state_dict(pretrained_model.classifier.state_dict())
