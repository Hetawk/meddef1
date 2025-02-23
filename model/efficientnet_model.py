<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.swish = Swish()

    def forward(self, x):
        batch, channel, _, _ = x.size()
        se = self.global_avg_pool(x)
        se = self.fc1(se)
        se = self.swish(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        return x * se

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, reduction=4, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.expand_ratio = expand_ratio
        self.stride = stride

        mid_channels = in_channels * expand_ratio

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(mid_channels) if expand_ratio != 1 else None

        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_channels)

        self.se = SEBlock(mid_channels, reduction=reduction)
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.swish = Swish()

    def forward(self, x):
        identity = x
        if self.expand_conv:
            x = self.swish(self.expand_bn(self.expand_conv(x)))
        x = self.swish(self.dw_bn(self.dwconv(x)))
        x = self.se(x)
        x = self.project_bn(self.project_conv(x))

        if self.stride == 1 and identity.size() == x.size():
            if self.drop_connect_rate:
                x = self.drop_connect(x)
            x += identity
        return x

    def drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(x.size(0), 1, 1, 1, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate=0.2, num_classes=None, input_channels=3):
        super(EfficientNet, self).__init__()

        def round_filters(filters, width_coefficient):
            multiplier = width_coefficient
            divisor = 8
            min_depth = filters
            new_filters = max(min_depth, int(filters * multiplier + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        def round_repeats(repeats, depth_coefficient):
            return int(depth_coefficient * repeats)

        base_channels = 32
        base_layers = [
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        self.out_channels = round_filters(base_channels, width_coefficient)
        self.stem_conv = nn.Conv2d(input_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.out_channels)
        self.swish = Swish()

        self.blocks = nn.ModuleList()
        in_channels = self.out_channels
        for t, c, n, k, s in base_layers:
            out_channels = round_filters(c, width_coefficient)
            repeats = round_repeats(n, depth_coefficient)
            for i in range(repeats):
                stride = s if i == 0 else 1
                self.blocks.append(MBConvBlock(in_channels, out_channels, t, stride, k))
                in_channels = out_channels

        self.out_channels = round_filters(1280, width_coefficient)
        self.head_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(self.out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        logging.info(f"Input shape: {x.shape}")
        x = self.swish(self.stem_bn(self.stem_conv(x)))
        logging.info(f"Shape after stem layers: {x.shape}")
        for i, block in enumerate(self.blocks):
            x = block(x)
            logging.info(f"Shape after block {i + 1}: {x.shape}")
        x = self.swish(self.head_bn(self.head_conv(x)))
        logging.info(f"Shape after head layers: {x.shape}")
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def EfficientNetB0(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB1(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB2(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB3(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB4(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB5(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB6(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.5, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB7(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5, num_classes=num_classes, input_channels=input_channels)
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.swish = Swish()

    def forward(self, x):
        batch, channel, _, _ = x.size()
        se = self.global_avg_pool(x)
        se = self.fc1(se)
        se = self.swish(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        return x * se

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, reduction=4, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.expand_ratio = expand_ratio
        self.stride = stride

        mid_channels = in_channels * expand_ratio

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(mid_channels) if expand_ratio != 1 else None

        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_channels)

        self.se = SEBlock(mid_channels, reduction=reduction)
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.swish = Swish()

    def forward(self, x):
        identity = x
        if self.expand_conv:
            x = self.swish(self.expand_bn(self.expand_conv(x)))
        x = self.swish(self.dw_bn(self.dwconv(x)))
        x = self.se(x)
        x = self.project_bn(self.project_conv(x))

        if self.stride == 1 and identity.size() == x.size():
            if self.drop_connect_rate:
                x = self.drop_connect(x)
            x += identity
        return x

    def drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(x.size(0), 1, 1, 1, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate=0.2, num_classes=None, input_channels=3):
        super(EfficientNet, self).__init__()

        def round_filters(filters, width_coefficient):
            multiplier = width_coefficient
            divisor = 8
            min_depth = filters
            new_filters = max(min_depth, int(filters * multiplier + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        def round_repeats(repeats, depth_coefficient):
            return int(depth_coefficient * repeats)

        base_channels = 32
        base_layers = [
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        self.out_channels = round_filters(base_channels, width_coefficient)
        self.stem_conv = nn.Conv2d(input_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.out_channels)
        self.swish = Swish()

        self.blocks = nn.ModuleList()
        in_channels = self.out_channels
        for t, c, n, k, s in base_layers:
            out_channels = round_filters(c, width_coefficient)
            repeats = round_repeats(n, depth_coefficient)
            for i in range(repeats):
                stride = s if i == 0 else 1
                self.blocks.append(MBConvBlock(in_channels, out_channels, t, stride, k))
                in_channels = out_channels

        self.out_channels = round_filters(1280, width_coefficient)
        self.head_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(self.out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        logging.info(f"Input shape: {x.shape}")
        x = self.swish(self.stem_bn(self.stem_conv(x)))
        logging.info(f"Shape after stem layers: {x.shape}")
        for i, block in enumerate(self.blocks):
            x = block(x)
            logging.info(f"Shape after block {i + 1}: {x.shape}")
        x = self.swish(self.head_bn(self.head_conv(x)))
        logging.info(f"Shape after head layers: {x.shape}")
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def EfficientNetB0(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB1(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB2(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB3(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB4(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB5(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB6(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.5, num_classes=num_classes, input_channels=input_channels)

def EfficientNetB7(input_channels=3, num_classes=None):
    return EfficientNet(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5, num_classes=num_classes, input_channels=input_channels)
>>>>>>> 16c5cfd9eac902321ee831908acfc69f3a52f936
