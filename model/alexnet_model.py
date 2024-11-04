# # alexnet_model.py
#
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import AlexNet_Weights
#
#
# class AlexNetModel(nn.Module):
#     def __init__(self, input_channels=3, pretrained=False):
#         super(AlexNetModel, self).__init__()
#
#         # Load the PyTorch AlexNet model with optional pretrained weights
#         self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT if pretrained else None)
#
#         # Modify the first convolutional layer to accommodate different input channels if necessary
#         if input_channels != 3:
#             self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
#
#         # If pretrained, we need to reinitialize the first layer to handle different input channels
#         if pretrained and input_channels != 3:
#             self._initialize_first_layer()
#
#         # Add AdaptiveAvgPool2d to match PyTorch AlexNet
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#
#     def forward(self, x):
#         x = self.model.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.model.classifier(x)
#         return x
#
#     def _initialize_first_layer(self):
#         # Reinitialize the weights of the first conv layer for different input channels
#         original_conv1 = models.alexnet(weights=AlexNet_Weights.DEFAULT).features[0]
#         new_conv1 = nn.Conv2d(self.model.features[0].in_channels, self.model.features[0].out_channels,
#                               kernel_size=self.model.features[0].kernel_size,
#                               stride=self.model.features[0].stride,
#                               padding=self.model.features[0].padding)
#
#         with torch.no_grad():
#             for i in range(self.model.features[0].out_channels):
#                 new_conv1.weight.data[i] = original_conv1.weight.data[i].mean(dim=0, keepdim=True)
#
#         self.model.features[0] = new_conv1



# alexnet_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import logging

class AlexNetModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=None, pretrained=False):
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
            nn.Linear(4096, num_classes),
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




# from functools import partial
# from typing import Any, Optional
# import torch
# import torch.nn as nn
# import logging
#
# from torchvision.transforms import ImageClassification
# from torchvision.utils import _log_api_usage_once
# from torchvision.models._api import register_model, Weights, WeightsEnum
# from torchvision.models._meta import _IMAGENET_CATEGORIES
# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
#
# __all__ = ["AlexNet", "AlexNet_Weights", "alexnet"]
#
# class AlexNet(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.5, input_channels: int = 3) -> None:
#         logging.info(f"Initializing AlexNetModel with {input_channels} input channels.")
#         super().__init__()
#         _log_api_usage_once(self)
#         self.features = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.BatchNorm2d(192),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.BatchNorm2d(384),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
# class AlexNet_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             "num_params": 61100840,
#             "min_size": (63, 63),
#             "categories": _IMAGENET_CATEGORIES,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 56.522,
#                     "acc@5": 79.066,
#                 }
#             },
#             "_ops": 0.714,
#             "_file_size": 233.087,
#             "_docs": """
#                 These weights reproduce closely the results of the paper using a simplified training recipe.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
# @register_model()
# @handle_legacy_interface(weights=("pretrained", AlexNet_Weights.IMAGENET1K_V1))
# def alexnet(*, weights: Optional[AlexNet_Weights] = None, progress: bool = True, input_channels: int = 3, **kwargs: Any) -> AlexNet:
#     """AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`__.
#
#     .. note::
#         AlexNet was originally introduced in the `ImageNet Classification with
#         Deep Convolutional Neural Networks
#         <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`__
#         paper. Our implementation is based instead on the "One weird trick"
#         paper above.
#
#     Args:
#         weights (:class:`~torchvision.models.AlexNet_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.AlexNet_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         input_channels (int, optional): Number of input channels. Default is 3.
#         **kwargs: parameters passed to the ``torchvision.models.squeezenet.AlexNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.AlexNet_Weights
#         :members:
#     """
#
#     weights = AlexNet_Weights.verify(weights)
#
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#
#     model = AlexNet(input_channels=input_channels, **kwargs)
#
#     if weights is not None:
#         state_dict = weights.get_state_dict(progress=progress, check_hash=True)
#         if input_channels != 3:
#             # Modify the state dict to adjust for the different number of input channels
#             original_conv1_weight = state_dict['features.0.weight']
#             new_conv1_weight = torch.zeros((64, input_channels, 11, 11))
#             for i in range(64):
#                 new_conv1_weight[i] = original_conv1_weight[i].mean(dim=1, keepdim=True).expand_as(new_conv1_weight[i])
#             state_dict['features.0.weight'] = new_conv1_weight
#         model.load_state_dict(state_dict)
#
#     return model
