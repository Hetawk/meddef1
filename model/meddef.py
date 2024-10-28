import torch
import torch.nn as nn
from .resnet_model import *  # Import your ResNet50 model
from .densenet_model import *  # Import your DenseNet121 model


class MedDef(nn.Module):
    def __init__(self, num_classes, input_channels=3, pretrained=False):
        super(MedDef, self).__init__()

        # Initialize ResNet50
        self.resnet = ResNet50(pretrained=pretrained, input_channels=input_channels, num_classes=num_classes)

        # Initialize DenseNet121
        self.densenet = DenseNet121(pretrained=pretrained, input_channels=input_channels, num_classes=num_classes)

        # Remove the final fully connected layers from both models because we will use their feature outputs
        self.resnet.fc = nn.Identity()
        self.densenet.fc = nn.Identity()

        # Define a new fully connected layer that combines both ResNet50 and DenseNet121 feature outputs
        resnet_out_features = 2048  # Output size of ResNet50's features
        densenet_out_features = 1024  # Output size of DenseNet121's features
        combined_features = resnet_out_features + densenet_out_features

        # Final fully connected layer
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        # Pass input through ResNet50
        resnet_features = self.resnet(x)

        # Pass input through DenseNet121
        densenet_features = self.densenet(x)

        # Concatenate the features from ResNet50 and DenseNet121
        combined_features = torch.cat((resnet_features, densenet_features), dim=1)

        # Pass through the final fully connected layer
        output = self.fc(combined_features)
        return output
