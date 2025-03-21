import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from model.attention.base_robust_method import BaseRobustMethod

# Pretrained model URLs (placeholder for future implementation)
model_urls = {
    'vit_tiny': None,
    'vit_small': None,
    'vit_base': None,
    'vit_large': None,
}


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding layer
    Splits image into patches and projects them to embedding dimension
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size if isinstance(
            img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(
            patch_size, tuple) else (patch_size, patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Create projection layer: combines convolution operation and flattening
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size,
                                    stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
        Returns:
            Patch embeddings of shape [batch_size, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # Project patches and flatten: [B, embed_dim, grid_h, grid_w] -> [B, embed_dim, num_patches]
        x = self.projection(x)
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention with advanced features
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # Scaling factor for dot product

        # Combined query, key, value projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, C] where B is batch size, N is sequence length, 
               C is embedding dimension
        Returns:
            Output tensor of shape [B, N, C]
        """
        B, N, C = x.shape

        # Project and reshape qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        # Unpack query, key, value
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, num_heads, N, head_dim]

        # Attention mechanism: (q @ k.transpose) * scale
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation used in Transformer blocks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # Drop path (stochastic depth)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    """

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with different dims
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation

    Configurable for different model sizes (tiny, small, base, large)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 representation_size=None,
                 pretrained=False,
                 robust_method: Optional[BaseRobustMethod] = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.pretrained = pretrained
        self.robust_method = robust_method

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Representation layer (optional pre-logits)
        if representation_size:
            self.has_logits = True
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.init_weights()

        logging.info(
            f"Initialized VisionTransformer with {depth} blocks, {num_heads} heads, {embed_dim} embedding dim")

        # Initialize with pretrained weights if needed
        if pretrained:
            logging.info("Pretrained weights loading is not implemented yet.")

    def init_weights(self):
        # Initialize patch_embed like a linear layer
        w = self.patch_embed.projection.weight
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Init head
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate position embeddings for variable input sizes"""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        # Class token position embedding stays the same
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        if npatch == N:
            return self.pos_embed

        # Interpolate patch position embeddings
        dim = x.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.permute(0, 3, 1, 2),  # [B, C, H, W]
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(
            0, 2, 3, 1).reshape(1, h * w, dim)

        # Combine with class token position embedding
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x):
        """Extract features before classification head"""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        if x.size(1) != self.pos_embed.size(1):
            # Need to interpolate position embeddings for different image sizes
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Process through transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Extract class token for classification
        x = x[:, 0]
        x = self.pre_logits(x)

        return x

    def forward_without_fc(self, x):
        """Extract features before classification head"""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        if x.size(1) != self.pos_embed.size(1):
            # Need to interpolate position embeddings for different image sizes
            h = w = int(math.sqrt(x.size(1) - 1))
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Process through transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Return both the class token and the sequence
        return x

    def forward(self, x):
        x = self.forward_without_fc(x)

        if self.robust_method:
            # Apply robust method if available
            # Note: we need to reshape for compatibility with robust methods
            B = x.shape[0]
            cls_token = x[:, 0]  # Extract class token

            # Reshape class token to 4D for robust method
            cls_token = cls_token.view(B, -1, 1, 1)
            cls_token, _ = self.robust_method(cls_token, cls_token, cls_token)

            # Convert back to expected shape
            cls_token = cls_token.view(B, -1)
            return cls_token  # Return the processed class token
        else:
            # Extract class token for classification
            x = x[:, 0]
            x = self.pre_logits(x)
            x = self.head(x)
            return x


# Decorator to check if num_classes is provided
def check_num_classes(func):
    def wrapper(*args, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes must be specified")
        return func(*args, **kwargs)
    return wrapper


@check_num_classes
def get_transformer(depth: str, pretrained: bool = False, input_channels: int = 3,
                    num_classes: int = None, robust_method: Optional[BaseRobustMethod] = None) -> VisionTransformer:
    """
    Get a Vision Transformer model with specified depth

    Args:
        depth: str - Model depth/size:
            - 'tiny': ViT-Tiny
            - 'small': ViT-Small 
            - 'base': ViT-Base
            - 'large': ViT-Large
        pretrained: bool - Whether to load pretrained weights
        input_channels: int - Number of input channels
        num_classes: int - Number of classes for classification
        robust_method: Optional[BaseRobustMethod] - Robust method to apply

    Returns:
        VisionTransformer model with specified configuration
    """
    depth_to_config = {
        'tiny': {'name': 'vit_tiny', 'patch_size': 16, 'embed_dim': 192, 'depth': 6, 'num_heads': 3, 'mlp_ratio': 4},
        'small': {'name': 'vit_small', 'patch_size': 16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'mlp_ratio': 4},
        'base': {'name': 'vit_base', 'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
        'large': {'name': 'vit_large', 'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4},
    }

    if depth not in depth_to_config:
        raise ValueError(
            f"Unsupported Vision Transformer depth: {depth}. Use one of: {list(depth_to_config.keys())}")

    config = depth_to_config[depth]
    logging.info(
        f"Creating {config['name']} model with {config['depth']} transformer blocks")

    model = VisionTransformer(
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=True,
        in_channels=input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        robust_method=robust_method
    )

    return model


# Update factory functions to use string-based depth
def vit_tiny(pretrained=False, **kwargs):
    return get_transformer('tiny', pretrained=pretrained, **kwargs)


def vit_small(pretrained=False, **kwargs):
    return get_transformer('small', pretrained=pretrained, **kwargs)


def vit_base(pretrained=False, **kwargs):
    return get_transformer('base', pretrained=pretrained, **kwargs)


def vit_large(pretrained=False, **kwargs):
    return get_transformer('large', pretrained=pretrained, **kwargs)
