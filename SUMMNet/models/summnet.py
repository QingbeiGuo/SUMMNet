# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, padding=3, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Mlp(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, linear_ratio=4., attn_drop=0., proj_drop=0., sr_ratios=1, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratios = sr_ratios
        if sr_ratios > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratios, stride=sr_ratios)
            self.norm = nn.LayerNorm(dim)

        if num_heads > 1:
            hidden_features = int(num_heads * linear_ratio)
            self.fc1 = nn.Linear(num_heads, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratios > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.num_heads > 1:
            attn = attn.permute(0, 2, 3, 1) # (B, h, N, N) -> (B, N, N, h)
            attn = self.fc1(attn)
            attn = self.act(attn)
            attn = self.fc2(attn)
            attn = attn.permute(0, 3, 1, 2) # (B, N, N, h) -> (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LFM(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()

        x = x + self.drop_path(self.dwconv(x))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp(self.norm(x)))

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class GFM(nn.Module):
    def __init__(self, dim, num_heads=8, linear_ratio=4., mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, sr_ratios=1):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pos_drop = nn.Dropout(p=drop)

        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, linear_ratio=linear_ratio, attn_drop=attn_drop, proj_drop=drop, sr_ratios=sr_ratios)

        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()

        # position embedded
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class Stage(nn.Module):
    def __init__(self, istage=0, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], dp_rates=0.0, 
                 layer_scale_init_value=1e-6, 
                 num_heads=[1, 2, 4, 8], linear_ratio=[1, 32, 16, 8], sr_ratios=[2, 2, 1, 1], mlp_ratio=4., drop=0., attn_drop=0., act_layer=nn.GELU
                 ):
        super().__init__()

        self.istage = istage
        layers = []
        cur = 0
        for i in range(4):
            if i == istage:
                s = int(depths[i] / 3)
                if i<2:
                    for j in range(depths[i]-s):
                       layers.append(LFM(dim=dims[i], mlp_ratio=mlp_ratio, drop=drop, drop_path=dp_rates[cur+j]))
                    for j in range(s):
                        layers.append(GFM(dim=dims[i], num_heads=num_heads[i], linear_ratio=linear_ratio[i], 
                                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=dp_rates[cur+depths[i]- s + j], sr_ratios=sr_ratios[i]))
                elif i>=2:
                    for j in range(s):
                       layers.append(LFM(dim=dims[i], mlp_ratio=mlp_ratio, drop=drop, drop_path=dp_rates[cur+depths[i]- s + j]))
                    for j in range(depths[i]-s):
                        layers.append(GFM(dim=dims[i], num_heads=num_heads[i], linear_ratio=linear_ratio[i], 
                                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=dp_rates[cur+j], sr_ratios=sr_ratios[i]))
            cur += depths[i]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i, block in enumerate(self.layers):
            x = block(x)

        return x 

class SUMMNet(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2),
                    nn.BatchNorm2d(dims[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stage = Stage(istage=i, depths=depths, dims=dims, dp_rates=dp_rates)
            self.stages.append(stage)
        self.norm = nn.BatchNorm2d(dims[-1])

        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)

        return x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def SUMMNet_tiny(pretrained=False,in_22k=False, **kwargs):
    model = SUMMNet(depths=[3, 3, 9, 3], dims=[64, 128, 320, 512], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def SUMMNet_small(pretrained=False,in_22k=False, **kwargs):
    model = SUMMNet(depths=[3, 6, 9, 6], dims=[64, 128, 320, 512], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def SUMMNet_base(pretrained=False, in_22k=False, **kwargs):
    model = SUMMNet(depths=[3, 6, 15, 6], dims=[64, 128, 320, 512], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def SUMMNet_large(pretrained=False, in_22k=False, **kwargs):
    model = SUMMNet(depths=[3, 6, 24, 6], dims=[96, 192, 320, 512], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def SUMMNet_huge(pretrained=False, in_22k=False, **kwargs):
    model = SUMMNet(depths=[6, 6, 30, 6], dims=[96, 192, 320, 512], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
