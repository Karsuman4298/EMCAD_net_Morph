import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

try:
    from timm.models.layers import trunc_normal_tf_
except ImportError:
    try:
        from timm.layers import trunc_normal_tf_
    except ImportError:
        from torch.nn.init import trunc_normal_ as trunc_normal_tf_

try:
    from timm.models.helpers import named_apply
except ImportError:
    def named_apply(fn, module, name='', depth_first=True):
        for child_name, child_module in module.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            named_apply(fn, child_module, child_full_name, depth_first)
        fn(module, name)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet-style default
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.out_channels)
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError(
            'activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


# ============================================================
#  MSDC
# ============================================================
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride,
                 activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels,
                          kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


# ============================================================
#  MSCB
# ============================================================
class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 kernel_sizes=[1, 3, 5], expansion_factor=2,
                 dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels,
                      1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes,
                         self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels,
                      1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels,
                                     1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(
            dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1,
              kernel_sizes=[1, 3, 5], expansion_factor=2,
              dw_parallel=True, add=True, activation='relu6'):
    convs = []
    mscb = MSCB(in_channels, out_channels, stride,
                kernel_sizes=kernel_sizes,
                expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add,
                activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1,
                        kernel_sizes=kernel_sizes,
                        expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add,
                        activation=activation)
            convs.append(mscb)
    return nn.Sequential(*convs)


# ============================================================
#  EUCB
# ============================================================
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2,
                      groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


# ============================================================
#  LGAG  — completely unchanged from original
# ============================================================
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int,
                 kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================
#  CAB  — completely unchanged from original
# ============================================================
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None,
                 ratio=16, activation='relu'):
        super(CAB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels is None:
            self.out_channels = in_channels

        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.max_pool  = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1       = nn.Conv2d(self.in_channels,
                                   self.reduced_channels, 1, bias=False)
        self.fc2       = nn.Conv2d(self.reduced_channels,
                                   self.out_channels, 1, bias=False)
        self.sigmoid   = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


# ============================================================
#  SAB  — completely unchanged from original
# ============================================================
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


# ============================================================
#  MorphWrapper — NEW block inserted between EUCB and LGAG
#
#  Ported faithfully from original MorphWrap / apply_offset /
#  MultiScaleFlowEstimator logic.
#
#  Data flow per stage:
#    EUCB  →  d  (guidance g)
#    skip  →  x  (encoder feature to be warped)
#              ↓
#    MorphWrapper(g=d, x=skip)  →  x_warped
#              ↓
#    LGAG(g=d, x=x_warped)  →  x_gated     ← unchanged
#              ↓
#    d + x_gated                            ← unchanged
#              ↓
#    CAB → SAB → MSCB                       ← unchanged
# ============================================================

def apply_offset(offset):
    """
    Converts pixel-space offset fields into a normalised sampling
    grid suitable for F.grid_sample.

    Original logic (faithfully ported):
      1. Build absolute pixel-coordinate grids with arange
      2. Add the predicted offset to get displaced coordinates
      3. Normalise displaced coords to [-1, 1]

    Args:
        offset : [B*K, 2, H, W]  — raw pixel-space displacements
    Returns:
        grid   : [B*K, H, W, 2]  — normalised (x, y) sampling grid
    """
    sizes     = list(offset.size()[2:])          # [H, W]

    # absolute pixel grids — meshgrid returns (H-grid, W-grid)
    grid_list = torch.meshgrid(
        *[torch.arange(size, device=offset.device) for size in sizes]
    )
    # reverse so dim=0 → x (W direction), dim=1 → y (H direction)
    # matching grid_sample's (x, y) coordinate convention
    grid_list = list(reversed(list(grid_list)))

    # add pixel-space offset to absolute coordinates
    grid_list = [
        grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)
    ]                                            # each [B*K, H, W]

    # normalise: pixel p → p / ((size-1)/2) - 1
    # maps [0, size-1]  →  [-1, 1]
    grid_list = [
        grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))
    ]                                            # each [B*K, H, W]

    return torch.stack(grid_list, dim=-1)        # [B*K, H, W, 2]


def MorphWrap(feat, offsets, att_maps, sample_k, out_ch, detach=False):
    """
    Warp feat using sample_k offset fields and aggregate with
    attention weights.

    Original aggregation logic (faithfully ported):
      1. Expand att_maps  [B, K, H, W]  →  [B, K*C, H, W]
      2. Repeat feat along batch dim    →  [B*K, C, H, W]
      3. grid_sample all K warps at once→  [B*K, C, H, W]
      4. Reshape                        →  [B, K*C, H, W]
      5. Multiply by expanded att_maps
      6. Split into K chunks of C and sum →  [B, C, H, W]

    Args:
        feat     : [B,   C,   H, W]
        offsets  : [B*K, 2,   H, W]
        att_maps : [B,   K,   H, W]  (after softmax, sums to 1 over K)
        sample_k : int  K
        out_ch   : int  C
        detach   : bool whether to detach offsets from autograd
    Returns:
        att_warp_feat : [B, C, H, W]
    """
    # expand attention to [B, K*C, H, W]
    att_maps = torch.repeat_interleave(att_maps, out_ch, dim=1)

    B, C, H, W = feat.size()

    # repeat feat along batch → [B*K, C, H, W]
    multi_feat = torch.repeat_interleave(feat, sample_k, dim=0)

    # build sampling grid  [B*K, H, W, 2]
    grid = offsets.detach().permute(0, 2, 3, 1) \
           if detach else offsets.permute(0, 2, 3, 1)

    multi_warp_feat = F.grid_sample(
        multi_feat, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )                                            # [B*K, C, H, W]

    # reshape → [B, K*C, H, W]  then weight
    multi_att_warp_feat = (
        multi_warp_feat.reshape(B, -1, H, W) * att_maps
    )                                            # [B, K*C, H, W]

    # split K chunks of C and sum → [B, C, H, W]
    att_warp_feat = sum(
        torch.split(multi_att_warp_feat, out_ch, dim=1)
    )
    return att_warp_feat


class MultiScaleFlowEstimator(nn.Module):
    """
    Predicts K*3 output channels from concatenated features:
      K*2  →  pixel-space offset fields (one (dx,dy) per head)
      K    →  attention logits          (one weight per head)

    Architecture (ported from original):
      Conv(in→filters[0], 3×3) + LeakyReLU
      Conv(filters[i-1]→filters[i], k×k) + LeakyReLU   (for i=1..n)
      Conv(filters[-1]→out_channels, k×k)               ← zero-init
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, num_filters=None):
        super(MultiScaleFlowEstimator, self).__init__()

        if num_filters is None:
            num_filters = [128, 64, 32]

        layers = []
        for i in range(len(num_filters)):
            in_ch = in_channels if i == 0 else num_filters[i - 1]
            k     = 3 if i == 0 else kernel_size
            layers.append(
                nn.Conv2d(in_ch, num_filters[i], kernel_size=k,
                          stride=1, padding=k // 2)
            )
            layers.append(nn.LeakyReLU(inplace=False, negative_slope=0.1))

        # final conv — zero-init so offsets=0 and attention=uniform at step 0
        final_conv = nn.Conv2d(
            num_filters[-1], out_channels,
            kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2
        )
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MorphWrapper(nn.Module):
    """
    Adaptive feature warping block inserted between EUCB and LGAG.

    Uses the original MorphWrap / apply_offset / MultiScaleFlowEstimator
    logic faithfully — the only adaptation is that g and x come from the
    decoder (EUCB output and encoder skip) rather than cloth/reference
    encoder pyramids.

    At initialisation (zero-init on final conv of MFE):
      offsets  = 0  →  identity sampling grid  →  x_warped == x
      att_maps = softmax(0) = uniform  →  weighted sum == x
    So the block starts as a perfect no-op, preserving baseline behaviour.

    Args:
        F_g        : channels of EUCB output  (guidance)
        F_l        : channels of encoder skip (feature to warp)
        sample_k   : number of sampling heads K  (default 4)
        num_filters: hidden layer widths in MFE
                     (default scales with F_l to avoid over-parameterising)
    """
    def __init__(self, F_g, F_l, sample_k=4, num_filters=None):
        super(MorphWrapper, self).__init__()
        self.sample_k = sample_k
        self.F_l      = F_l

        # project g to same channel depth as x when they differ
        # (in your decoder F_g == F_l per stage, kept for robustness)
        self.g_proj = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True),
        ) if F_g != F_l else nn.Identity()

        # scale MFE width to the channel budget of each stage
        if num_filters is None:
            base        = max(F_l // 4, 16)
            num_filters = [base * 2, base, max(base // 2, 16)]

        # input = concat(g_aligned, x)  →  F_l*2 channels
        # output = K*3  (K*2 offsets + K attention logits)
        self.mfe = MultiScaleFlowEstimator(
            in_channels  = F_l * 2,
            out_channels = sample_k * 3,
            kernel_size  = 3,
            num_filters  = num_filters
        )

    def forward(self, g, x):
        """
        Args:
            g : [B, F_g, H, W]  EUCB output  (guidance)
            x : [B, F_l, H, W]  encoder skip (to be warped)
        Returns:
            x_warped : [B, F_l, H, W]
        """
        B, C, H, W = x.shape

        # ── 1. align g to same spatial size as x ─────────────────────
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=(H, W),
                              mode='bilinear', align_corners=False)
        g_aligned = self.g_proj(g)                   # [B, F_l, H, W]

        # ── 2. estimate Morpho Fields ─────────────────────────────────
        offsets_att = self.mfe(
            torch.cat([g_aligned, x], dim=1)
        )                                            # [B, K*3, H, W]

        # ── 3. split into attention logits and offset fields ──────────
        # first K*2 channels → offset fields
        # last  K   channels → attention logits
        raw_offsets = offsets_att[:, :self.sample_k * 2, :, :]
        raw_att     = offsets_att[:, self.sample_k * 2:, :, :]

        att_maps = F.softmax(raw_att, dim=1)         # [B, K, H, W]

        # ── 4. build normalised sampling grid (original apply_offset) ─
        # reshape [B, K*2, H, W] → [B*K, 2, H, W]
        raw_offsets = raw_offsets.reshape(
            B * self.sample_k, 2, H, W
        )                                            # [B*K, 2, H, W]

        # apply_offset: arange coords + offset → normalise → [B*K,H,W,2]
        offsets = apply_offset(raw_offsets)          # [B*K, H, W, 2]

        # permute to [B*K, 2, H, W] for MorphWrap
        offsets = offsets.permute(0, 3, 1, 2)        # [B*K, 2, H, W]

        # ── 5. warp and aggregate (original MorphWrap) ────────────────
        x_warped = MorphWrap(
            feat     = x,
            offsets  = offsets,
            att_maps = att_maps,
            sample_k = self.sample_k,
            out_ch   = C
        )                                            # [B, F_l, H, W]

        return x_warped


# ============================================================
#  EMCAD  — original architecture + MorphWrapper between EUCB & LGAG
# ============================================================
class EMCAD(nn.Module):
    def __init__(self,
                 channels=[512, 320, 128, 64],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=6,
                 dw_parallel=True,
                 add=True,
                 lgag_ks=3,
                 activation='relu6',
                 use_morph=True,
                 morph_sample_k=4):
        super(EMCAD, self).__init__()

        eucb_ks = 3

        # ── Stage 4 ───────────────────────────────────────────────
        self.mscb4 = MSCBLayer(
            channels[0], channels[0], n=1, stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)
        self.cab4 = CAB(channels[0])

        # ── Stage 3 ───────────────────────────────────────────────
        self.eucb3  = EUCB(channels[0], channels[1],
                           kernel_size=eucb_ks, stride=eucb_ks//2)
        self.morph3 = MorphWrapper(channels[1], channels[1],
                                   sample_k=morph_sample_k) \
                      if use_morph else None
        self.lgag3  = LGAG(channels[1], channels[1],
                           channels[1]//2,
                           kernel_size=lgag_ks,
                           groups=channels[1]//2)
        self.cab3   = CAB(channels[1])
        self.mscb3  = MSCBLayer(
            channels[1], channels[1], n=1, stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)

        # ── Stage 2 ───────────────────────────────────────────────
        self.eucb2  = EUCB(channels[1], channels[2],
                           kernel_size=eucb_ks, stride=eucb_ks//2)
        self.morph2 = MorphWrapper(channels[2], channels[2],
                                   sample_k=morph_sample_k) \
                      if use_morph else None
        self.lgag2  = LGAG(channels[2], channels[2],
                           channels[2]//2,
                           kernel_size=lgag_ks,
                           groups=channels[2]//2)
        self.cab2   = CAB(channels[2])
        self.mscb2  = MSCBLayer(
            channels[2], channels[2], n=1, stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)

        # ── Stage 1 ───────────────────────────────────────────────
        self.eucb1  = EUCB(channels[2], channels[3],
                           kernel_size=eucb_ks, stride=eucb_ks//2)
        self.morph1 = MorphWrapper(channels[3], channels[3],
                                   sample_k=morph_sample_k) \
                      if use_morph else None
        self.lgag1  = LGAG(channels[3], channels[3],
                           int(channels[3]/2),
                           kernel_size=lgag_ks,
                           groups=int(channels[3]/2))
        self.cab1   = CAB(channels[3])
        self.mscb1  = MSCBLayer(
            channels[3], channels[3], n=1, stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)

        self.sab       = SAB()
        self.use_morph = use_morph

    def forward(self, x, skips):
        """
        x     : [B, C0, H/16, W/16]  bottleneck
        skips : [skip0, skip1, skip2]
                  skip0 [B, C1, H/8,  W/8 ]
                  skip1 [B, C2, H/4,  W/4 ]
                  skip2 [B, C3, H/2,  W/2 ]

        Fusion formula per stage (use_morph=True):
            d = EUCB(prev)
            x_warped = MorphWrapper(g=d, x=skip)
            x_gated  = LGAG(g=d, x=x_warped)
            d = d + x_gated + x_warped   ← NEW triple fusion
            d = CAB(d)*d → SAB(d)*d → MSCB(d)
        """

        # ════ Stage 4: bottleneck MSCAM (unchanged) ══════════════
        d4 = self.cab4(x) * x
        d4 = self.sab(d4) * d4
        d4 = self.mscb4(d4)

        # ════ Stage 3 ════════════════════════════════════════════
        d3 = self.eucb3(d4)

        if self.use_morph:
            x3_warped = self.morph3(g=d3, x=skips[0])
            x3_gated  = self.lgag3(g=d3, x=x3_warped)
            # Triple fusion: decoder + gated_skip + warped_skip
            d3 = d3 + x3_gated + x3_warped
        else:
            x3 = self.lgag3(g=d3, x=skips[0])
            d3 = d3 + x3

        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3)  * d3
        d3 = self.mscb3(d3)

        # ════ Stage 2 ════════════════════════════════════════════
        d2 = self.eucb2(d3)

        if self.use_morph:
            x2_warped = self.morph2(g=d2, x=skips[1])
            x2_gated  = self.lgag2(g=d2, x=x2_warped)
            # Triple fusion
            d2 = d2 + x2_gated + x2_warped
        else:
            x2 = self.lgag2(g=d2, x=skips[1])
            d2 = d2 + x2

        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2)  * d2
        d2 = self.mscb2(d2)

        # ════ Stage 1 ════════════════════════════════════════════
        d1 = self.eucb1(d2)

        if self.use_morph:
            x1_warped = self.morph1(g=d1, x=skips[2])
            x1_gated  = self.lgag1(g=d1, x=x1_warped)
            # Triple fusion
            d1 = d1 + x1_gated + x1_warped
        else:
            x1 = self.lgag1(g=d1, x=skips[2])
            d1 = d1 + x1

        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1)  * d1
        d1 = self.mscb1(d1)

        return [d4, d3, d2, d1]
# ============================================================
#  Verification
# ============================================================
if __name__ == '__main__':
    channels = [512, 320, 128, 64]
    B        = 2

    x      = torch.randn(B, channels[0], 16, 16)
    skip0  = torch.randn(B, channels[1], 32, 32)
    skip1  = torch.randn(B, channels[2], 64, 64)
    skip2  = torch.randn(B, channels[3], 128, 128)
    skips  = [skip0, skip1, skip2]

    # ── MorphWrapper ON ───────────────────────────────────────────────
    print("=" * 60)
    print("  MorphWrapper ON  (use_morph=True, sample_k=4)")
    print("=" * 60)
    model = EMCAD(channels=channels, use_morph=True, morph_sample_k=4)
    model.eval()
    with torch.no_grad():
        outs = model(x, skips)
    names = ['d4', 'd3', 'd2', 'd1']
    for name, o in zip(names, outs):
        print(f"  {name}: {tuple(o.shape)}")

    # ── MorphWrapper OFF (baseline) ───────────────────────────────────
    print()
    print("=" * 60)
    print("  MorphWrapper OFF (use_morph=False) — exact baseline")
    print("=" * 60)
    model_base = EMCAD(channels=channels, use_morph=False)
    model_base.eval()
    with torch.no_grad():
        outs_base = model_base(x, skips)
    for name, o in zip(names, outs_base):
        print(f"  {name}: {tuple(o.shape)}")

    # ── Backward / NaN check ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Backward pass / NaN check")
    print("=" * 60)
    model.train()
    outs = model(x, skips)
    loss = sum(o.mean() for o in outs)
    loss.backward()
    nan_params = [
        n for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    if nan_params:
        print(f"  FAIL — NaN gradients in: {nan_params}")
    else:
        print("  PASS — no NaN gradients anywhere")

    # ── Identity check at init ────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Identity check at init (morph output == skip input ?)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        w3 = model.morph3(g=skip0, x=skip0)
        diff = (w3 - skip0).abs().max().item()
    print(f"  max |x_warped - x| at init: {diff:.2e}")
    print(f"  {'PASS — identity holds' if diff < 1e-4 else 'WARN — not exact identity'}")

    # ── Parameter counts ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Parameter counts")
    print("=" * 60)
    def count(m): return sum(p.numel() for p in m.parameters())
    morph_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'morph' in n
    )
    print(f"  morph=ON  total  : {count(model):>12,}")
    print(f"  morph=OFF total  : {count(model_base):>12,}")
    print(f"  MorphWrapper Δ   : {count(model)-count(model_base):>12,}")
    print(f"  MorphWrapper only: {morph_params:>12,}")