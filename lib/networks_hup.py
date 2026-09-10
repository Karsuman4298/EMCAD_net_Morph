# networks_hup.py — EMCADNet extended for HUPAnno + MorphWrapper
#
# Changes vs previous version:
#   1. Imports EMCAD from decoder.py (your updated file) instead of lib.decoders
#   2. Passes use_morph and morph_sample_k to EMCAD
#   3. use_morph / morph_sample_k wired through EMCADNet constructor
#   4. Everything else unchanged
# lib/networks_hup.py
# lib/networks_hup.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import (pvt_v2_b0, pvt_v2_b1, pvt_v2_b2,
                        pvt_v2_b3, pvt_v2_b4, pvt_v2_b5)
from lib.resnet import (resnet18, resnet34, resnet50,
                         resnet101, resnet152)

# ── FIXED: decoder code is inside lib/decoders.py ────────────────────────────
from lib.decoders import EMCAD



class EMCADNet(nn.Module):

    def __init__(self,
                 num_classes=1,
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dw_parallel=True,
                 add=True,
                 lgag_ks=3,
                 activation='relu6',
                 encoder='pvt_v2_b2',
                 pretrain=True,
                 pretrained_dir='./pretrained_pth/pvt/',
                 use_morph=True,        # NEW: toggle MorphWrapper
                 morph_sample_k=4):     # NEW: sampling heads K
        super().__init__()

        # grayscale → RGB adapter
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # ── Backbone ──────────────────────────────────────────────────────────
        encoder_cfg = {
            'pvt_v2_b0': (pvt_v2_b0, 'pvt_v2_b0.pth', [256, 160,  64,  32]),
            'pvt_v2_b1': (pvt_v2_b1, 'pvt_v2_b1.pth', [512, 320, 128,  64]),
            'pvt_v2_b2': (pvt_v2_b2, 'pvt_v2_b2.pth', [512, 320, 128,  64]),
            'pvt_v2_b3': (pvt_v2_b3, 'pvt_v2_b3.pth', [512, 320, 128,  64]),
            'pvt_v2_b4': (pvt_v2_b4, 'pvt_v2_b4.pth', [512, 320, 128,  64]),
            'pvt_v2_b5': (pvt_v2_b5, 'pvt_v2_b5.pth', [512, 320, 128,  64]),
            'resnet18' : (lambda: resnet18(pretrained=pretrain),  None,
                          [512, 256, 128,  64]),
            'resnet34' : (lambda: resnet34(pretrained=pretrain),  None,
                          [512, 256, 128,  64]),
            'resnet50' : (lambda: resnet50(pretrained=pretrain),  None,
                          [2048, 1024, 512, 256]),
            'resnet101': (lambda: resnet101(pretrained=pretrain), None,
                          [2048, 1024, 512, 256]),
            'resnet152': (lambda: resnet152(pretrained=pretrain), None,
                          [2048, 1024, 512, 256]),
        }
        if encoder not in encoder_cfg:
            print(f'Encoder "{encoder}" not found — defaulting to pvt_v2_b2.')
            encoder = 'pvt_v2_b2'

        build_fn, pth_name, channels = encoder_cfg[encoder]
        self.backbone = build_fn()

        if pretrain and pth_name is not None:
            ckpt_path  = pretrained_dir + pth_name
            save_model = torch.load(ckpt_path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items()
                          if k in model_dict}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        print(f'Backbone {encoder}: '
              f'{sum(p.numel() for p in self.backbone.parameters()):,} params')

        # ── EMCAD Decoder (FIX: use_morph and morph_sample_k now passed) ──────
        self.decoder = EMCAD(
            channels         = channels,
            kernel_sizes     = kernel_sizes,
            expansion_factor = expansion_factor,
            dw_parallel      = dw_parallel,
            add              = add,
            lgag_ks          = lgag_ks,
            activation       = activation,
            use_morph        = use_morph,        # ← NEW
            morph_sample_k   = morph_sample_k,   # ← NEW
        )
        print(f'EMCAD decoder (use_morph={use_morph}, K={morph_sample_k}): '
              f'{sum(p.numel() for p in self.decoder.parameters()):,} params')

        # ── Segmentation output heads (unchanged) ─────────────────────────────
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        # ── 4-class CCG head (unchanged) ──────────────────────────────────────
        #   0 = certain BG   (Omega_O / Omega_RB)
        #   1 = global uncertain  (non-LRP Omega_Delta)
        #   2 = LRP uncertain     (inside patch, between tight rings)
        #   3 = certain FG   (Omega_I / Omega_RF)
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[3] // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3] // 2, 4, kernel_size=1)
        )

        # ── Spatially-varying confidence head (unchanged) ─────────────────────
        self.conf_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[3] // 4,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3] // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # ── CCL embedding head (unchanged) ────────────────────────────────────
        embed_dim = 128
        self.embed_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[3] // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3] // 2, embed_dim, kernel_size=1)
        )
        self.embed_dim  = embed_dim
        self.queue_size = 1024
        self.register_buffer(
            'neg_queue',
            F.normalize(torch.randn(embed_dim, self.queue_size), dim=0)
        )
        self.register_buffer(
            'queue_ptr',
            torch.zeros(1, dtype=torch.long)
        )

    def forward(self, x, mode='test'):
        if x.size(1) == 1:
            x = self.conv(x)

        # backbone: x1=stage1(large), x2=stage2, x3=stage3, x4=stage4(small)
        x1, x2, x3, x4 = self.backbone(x)

        # decoder: bottleneck=x4, skips=[x3,x2,x1] from coarse to fine
        # EMCAD.forward(x, skips):
        #   skips[0] → used at stage 3  (channels[1], H/8)
        #   skips[1] → used at stage 2  (channels[2], H/4)
        #   skips[2] → used at stage 1  (channels[3], H/2)
        # x3 is channels[1] @ H/8, x2 is channels[2] @ H/4, x1 is channels[3] @ H/2
        dec_outs = self.decoder(x4, [x3, x2, x1])
        # dec_outs = [d4, d3, d2, d1]
        #   d4: [B, channels[0], H/16, W/16]
        #   d3: [B, channels[1], H/8,  W/8 ]
        #   d2: [B, channels[2], H/4,  W/4 ]
        #   d1: [B, channels[3], H/2,  W/2 ]

        p4 = F.interpolate(self.out_head4(dec_outs[0]),
                           scale_factor=32, mode='bilinear',
                           align_corners=False)
        p3 = F.interpolate(self.out_head3(dec_outs[1]),
                           scale_factor=16, mode='bilinear',
                           align_corners=False)
        p2 = F.interpolate(self.out_head2(dec_outs[2]),
                           scale_factor=8,  mode='bilinear',
                           align_corners=False)
        p1 = F.interpolate(self.out_head1(dec_outs[3]),
                           scale_factor=4,  mode='bilinear',
                           align_corners=False)

        if mode == 'test':
            return [p4, p3, p2, p1]

        H, W = x.shape[2], x.shape[3]
        d1   = dec_outs[3]   # finest decoder feature [B, channels[3], H/2, W/2]

        # 4-class CCG logits → upsample to input resolution
        cls_logits = F.interpolate(
            self.cls_head(d1), size=(H, W),
            mode='bilinear', align_corners=False
        )   # [B, 4, H, W]

        # confidence map → upsample to input resolution
        conf_map = F.interpolate(
            self.conf_head(d1), size=(H, W),
            mode='bilinear', align_corners=False
        )   # [B, 1, H, W]

        # L2-normalised CCL embeddings → upsample to input resolution
        embeddings = F.normalize(
            F.interpolate(
                self.embed_head(d1), size=(H, W),
                mode='bilinear', align_corners=False
            ), dim=1
        )   # [B, 128, H, W]

        return [p4, p3, p2, p1], cls_logits, embeddings, conf_map

    @torch.no_grad()
    def update_queue(self, new_vecs):
        """Enqueue new negative vectors into the memory queue (FIFO)."""
        n   = new_vecs.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + n
        if end <= self.queue_size:
            self.neg_queue[:, ptr:end] = new_vecs.T
        else:
            first = self.queue_size - ptr
            self.neg_queue[:, ptr:]       = new_vecs[:first].T
            self.neg_queue[:, :n - first] = new_vecs[first:].T
        self.queue_ptr[0] = end % self.queue_size


# ── Quick verification ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Testing EMCADNet with MorphWrapper ON")
    print("=" * 60)
    model = EMCADNet(
        use_morph=True, morph_sample_k=4
    ).cuda()
    x = torch.randn(2, 3, 352, 352).cuda()

    # test mode
    preds = model(x, mode='test')
    print('Test  preds:', [tuple(p.shape) for p in preds])

    # train mode
    preds, cls, emb, conf = model(x, mode='train')
    print('Train preds:', [tuple(p.shape) for p in preds])
    print('cls_logits :', tuple(cls.shape))
    print('embeddings :', tuple(emb.shape))
    print('conf_map   :', tuple(conf.shape))

    print("\n" + "=" * 60)
    print("  Testing EMCADNet with MorphWrapper OFF (baseline)")
    print("=" * 60)
    model_base = EMCADNet(
        use_morph=False
    ).cuda()
    preds_base = model_base(x, mode='test')
    print('Test  preds:', [tuple(p.shape) for p in preds_base])

    print("\n" + "=" * 60)
    print("  Parameter counts")
    print("=" * 60)
    def count(m): return sum(p.numel() for p in m.parameters())
    print(f'  morph=ON  : {count(model):>12,}')
    print(f'  morph=OFF : {count(model_base):>12,}')
    print(f'  Δ (morph) : {count(model)-count(model_base):>12,}')