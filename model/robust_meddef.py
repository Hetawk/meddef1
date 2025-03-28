import torch
import torch.nn as nn
import torch.nn.functional as F
from model.meddef.meddef1 import get_meddef1

class RobustnessWrapper(nn.Module):
    """Wrapper to enhance model robustness through architectural improvements"""
    
    def __init__(self, base_model, input_channels=3, num_classes=2):
        super().__init__()
        self.base_model = base_model
        
        # Extract feature dimensions from base model
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            features = self.base_model.features(dummy_input)
            self.feature_size = features.view(features.size(0), -1).size(1)
        
        # Add robustness-enhancing components
        self.noise_layer = GaussianNoise(0.1)
        self.feature_denoiser = FeatureDenoiser(self.feature_size)
        self.gradient_regularizer = GradientRegularizer()
        
        # Optional ensemble head
        self.ensemble_head = nn.Linear(num_classes * 2, num_classes)
        
    def forward(self, x):
        # Apply input noise during training
        if self.training:
            x = self.noise_layer(x)
        
        # Get base model features and logits
        features = self.base_model.features(x)
        
        # Apply feature denoising
        features_flat = features.view(features.size(0), -1)
        denoised_features = self.feature_denoiser(features_flat)
        denoised_features = denoised_features.view_as(features)
        
        # Get predictions from both regular and denoised features
        regular_logits = self.base_model.classifier(features.view(features.size(0), -1))
        denoised_logits = self.base_model.classifier(denoised_features.view(features.size(0), -1))
        
        # Combine predictions (during training we return both for loss calculation)
        if self.training:
            return regular_logits, denoised_logits
        else:
            # During inference, average the predictions
            return (regular_logits + denoised_logits) / 2.0

# Helper components
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class FeatureDenoiser(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, feature_size // 2)
        self.fc2 = nn.Linear(feature_size // 2, feature_size)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out + identity

class GradientRegularizer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, target):
        # Used in loss calculation, not in forward pass
        return x

def get_robust_meddef(depth, input_channels=3, num_classes=2, robust_method=None):
    """Create a robustness-enhanced MedDef model"""
    # Get base model
    base_model = get_meddef1(depth, input_channels, num_classes, robust_method)
    
    # Wrap with robustness enhancements
    model = RobustnessWrapper(base_model, input_channels, num_classes)
    
    return model
