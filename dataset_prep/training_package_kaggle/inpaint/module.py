import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


_VGG = None


def _vgg():
    global _VGG
    if _VGG is None:
        _VGG = tv.models.vgg16(True)
    return _VGG


def _perceptual_loss(x_features, y_features):
    return sum(F.l1_loss(x, y) for x, y in zip(x_features, y_features))


def _style_loss(x_gram_matrices, y_gram_matrices, coefs):
    return sum(
        1 / c * F.l1_loss(x, y)
        for x, y, c in zip(x_gram_matrices, y_gram_matrices, coefs)
    )


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :]))
        w_tv = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1]))
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.shape[1] * t.shape[2] * t.shape[3]


class _PretrainedFeaturesGenerator(nn.Module):
    def __init__(self, module, layers, preprocessor=None, reshape=True):
        super().__init__()
        self._module = module
        for param in self._module:
            param.requires_grad = False
        self._layers = set(layers)
        self._preprocessor = preprocessor or (lambda x: x)
        self._reshape = reshape

    def forward(self, x):
        self._module.eval()
        x = self._preprocessor(x)
        layers = copy.deepcopy(self._layers)
        output = []
        for layer, module in self._module._modules.items():
            if not layers:
                break
            x = module(x)
            if layer in layers:
                output.append(x.view(*x.shape[:2], -1) if self._reshape else x)
                layers.remove(layer)
        return output


class _Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std, dtype=torch.float32).view(-1, 1, 1), requires_grad=False)

    def forward(self, img):
        return (img - self.mean) / self.std


def _calculate_gram_matrices(features):
    return [x.matmul(x.transpose(-2, -1)) for x in features]


class InpaintLoss(nn.Module):
    def __init__(self, valid_coef=1, hole_coef=6, perceptual_coef=0.05, style_coef=120, tv_coef=0.1, style_features_generator=None):
        super().__init__()
        self._valid_coef = valid_coef
        self._hole_coef = hole_coef
        self._perceptual_coef = perceptual_coef
        self._style_coef = style_coef
        self._tv_coef = tv_coef
        self._tv_loss = TVLoss()
        self._style_features_generator = (
            style_features_generator or
            _PretrainedFeaturesGenerator(
                _vgg().features, ('4', '9', '16'),
                _Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )
        )

    def forward(self, out, mask, gt):
        l1_valid_loss = F.l1_loss(mask * out, mask * gt)
        reversed_mask = 1 - mask
        l1_hole_loss = F.l1_loss(reversed_mask * out, reversed_mask * gt)
        out_features = self._style_features_generator(out)
        gt_features = self._style_features_generator(gt)
        out_perceptual_loss = _perceptual_loss(out_features, gt_features)
        comp = mask * gt + (1 - mask) * out
        comp_features = self._style_features_generator(comp)
        comp_perceptual_loss = _perceptual_loss(comp_features, gt_features)
        out_gram_matrices = _calculate_gram_matrices(out_features)
        gt_gram_matrices = _calculate_gram_matrices(gt_features)
        coefs = [x.shape[-2] * x.shape[-1] for x in out_features]
        out_style_loss = _style_loss(out_gram_matrices, gt_gram_matrices, coefs)
        comp_gram_matrices = _calculate_gram_matrices(comp_features)
        comp_style_loss = _style_loss(comp_gram_matrices, gt_gram_matrices, coefs)
        tv_loss = self._tv_loss(comp)
        loss = (
            self._valid_coef * l1_valid_loss +
            self._hole_coef * l1_hole_loss +
            self._perceptual_coef * (out_perceptual_loss + comp_perceptual_loss) +
            self._style_coef * (out_style_loss + comp_style_loss) +
            self._tv_coef * tv_loss
        )
        return loss.view((1,))


class _PartialConv2d(nn.Module):
    """Partial convolution layer (https://arxiv.org/abs/1804.07723)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_renorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_renorm = use_renorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.sum_conv = nn.Conv2d(in_channels, 1, kernel_size, stride=stride, padding=padding, bias=False)
        self.sum_conv.weight.data.fill_(1)
        self.sum_conv.weight.requires_grad_(False)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, mask):
        # Force float32 for mask counting/division — prevents AMP NaN
        x = x.float()
        mask = mask.float()
        assert x.shape == mask.shape, 'x and mask shapes must be equal'
        x_masked = x * mask
        x_after_conv = self.conv(x_masked)
        mask_sum = self.sum_conv(mask)

        if self.use_renorm:
            n_elements = self.in_channels * self.kernel_size * self.kernel_size
            mask_sum_safe = mask_sum.clamp(min=1e-8)
            renorm_factor = n_elements / mask_sum_safe
            renorm_factor = renorm_factor * (mask_sum > 0).float()
            x_after_conv_normed = x_after_conv * renorm_factor
        else:
            x_after_conv_normed = x_after_conv

        x_after_conv_normed += self.conv_bias.view(1, -1, 1, 1)

        updated_mask_single = (mask_sum > 0).type(torch.float32)
        updated_mask = torch.cat([updated_mask_single] * self.out_channels, dim=1)
        updated_mask = updated_mask.to(mask.device)
        return x_after_conv_normed, updated_mask


class _InpaintDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding='same', bn=True, use_renorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self.padding = padding
        self._use_bn = bn
        self.pconv = _PartialConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, use_renorm=use_renorm)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self._use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x, mask


class _InpaintUpBlock(nn.Module):
    def __init__(self, in_channels, in_channels_bridge, out_channels, kernel_size, padding='same', bn=True, upsample_mode='nearest', use_renorm=True, use_activation=True):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self._use_bn = bn
        self._use_activation = use_activation

        if upsample_mode == 'nearest':
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        elif upsample_mode == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise NotImplementedError('{} is not valid upsample_mode'.format(upsample_mode))

        self.pconv = _PartialConv2d(in_channels + in_channels_bridge, out_channels, kernel_size, padding=padding, use_renorm=use_renorm)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, mask, x_bridge, mask_bridge):
        x, mask = self.upsample(x), self.upsample(mask)
        x = torch.cat([x, x_bridge], dim=1).contiguous()
        mask = torch.cat([mask, mask_bridge], dim=1).contiguous()
        x, mask = self.pconv(x, mask)
        if self._use_bn:
            x = self.bn(x)
        if self._use_activation:
            x = self.leaky_relu(x)
        return x, mask


class SelfAttention(nn.Module):
    """Self-Attention layer (SA-GAN style, https://arxiv.org/abs/1805.08318).

    Allows the network to attend to distant spatial locations,
    enabling it to reason about global structure (e.g., matching
    both eyes, preserving facial symmetry across masked regions).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable residual weight

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x N x C'
        k = self.key(x).view(B, -1, H * W)                      # B x C' x N
        attn = torch.bmm(q, k)                                   # B x N x N
        attn = F.softmax(attn, dim=-1)
        v = self.value(x).view(B, -1, H * W)                    # B x C x N
        out = torch.bmm(v, attn.permute(0, 2, 1))               # B x C x N
        out = out.view(B, C, H, W)
        return self.gamma * out + x  # residual connection


class InpaintNet(nn.Module):
    """Image Inpainting Network with Self-Attention.

    Combines partial convolutions (https://arxiv.org/abs/1804.07723)
    with self-attention layers (https://arxiv.org/abs/1805.08318)
    for improved global coherence in face reconstruction.
    """

    def __init__(self, in_channels=3, out_channels=3, upsample_mode='nearest', bn=False, use_renorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down_blocks = nn.ModuleList([
            _InpaintDownBlock(in_channels, 64, 7, stride=2, padding='same', bn=False, use_renorm=use_renorm),
            _InpaintDownBlock(64, 128, 5, stride=2, padding='same', bn=False, use_renorm=use_renorm),
            _InpaintDownBlock(128, 256, 5, stride=2, padding='same', bn=bn, use_renorm=use_renorm),
            _InpaintDownBlock(256, 512, 3, stride=2, padding='same', bn=bn, use_renorm=use_renorm),
            _InpaintDownBlock(512, 512, 3, stride=2, padding='same', bn=bn, use_renorm=use_renorm),
            _InpaintDownBlock(512, 512, 3, stride=2, padding='same', bn=bn, use_renorm=use_renorm),
            _InpaintDownBlock(512, 512, 3, stride=2, padding='same', bn=bn, use_renorm=use_renorm),
        ])
        self.depth = len(self.down_blocks)

        # Self-attention at the bottleneck (smallest spatial resolution)
        self.bottleneck_attention = SelfAttention(512)

        self.up_blocks = nn.ModuleList([
            _InpaintUpBlock(512, 512, 512, 3, padding='same', bn=bn, upsample_mode=upsample_mode, use_renorm=use_renorm),
            _InpaintUpBlock(512, 512, 512, 3, padding='same', bn=bn, upsample_mode=upsample_mode, use_renorm=use_renorm),
            _InpaintUpBlock(512, 512, 512, 3, padding='same', bn=bn, upsample_mode=upsample_mode, use_renorm=use_renorm),
            _InpaintUpBlock(512, 256, 256, 3, padding='same', bn=bn, upsample_mode=upsample_mode, use_renorm=use_renorm),
            _InpaintUpBlock(256, 128, 128, 3, padding='same', bn=bn, upsample_mode=upsample_mode, use_renorm=use_renorm),
            _InpaintUpBlock(128, 64, 64, 3, padding='same', bn=bn, upsample_mode=upsample_mode, use_renorm=use_renorm),
            _InpaintUpBlock(64, 3, 3, 3, padding='same', bn=False, upsample_mode=upsample_mode, use_renorm=use_renorm, use_activation=False),
        ])

    def forward(self, x, mask):
        x_bridges, mask_bridges = [], []
        for i in range(self.depth):
            x_bridges.append(x)
            mask_bridges.append(mask)
            x, mask = self.down_blocks[i](x, mask)

        # Apply self-attention at the bottleneck
        x = self.bottleneck_attention(x)

        for i in range(self.depth):
            x, mask = self.up_blocks[i](x, mask, x_bridges[-i - 1], mask_bridges[-i - 1])
        return x, mask
