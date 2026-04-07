import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.ops import resize

from .decode_seg import ATMSingleHeadSeg


class _ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CATAggregator(nn.Module):
    """Cost-volume aggregation module."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        text_guidance_dim: int = 512,
        text_guidance_proj_dim: int = 128,
        appearance_guidance_dim: int = 512,
        appearance_guidance_proj_dim: int = 128,
        decoder_dims=(64, 32),
        decoder_guidance_dims=(0, 0),
        decoder_guidance_proj_dims=(0, 0),
        prompt_channel: int = 1,
        pad_len: int = 0,
        drop: float = 0.1,
        use_temp_scale: bool = True,
        init_logit_scale: Optional[float] = None,
        use_cost_std: bool = True,
        use_class_norm: bool = False,
        use_class_linear: bool = False,
        class_hidden_dim: int = None,
        class_nheads: int = 4,
        class_layers: int = 1,
        class_pool: tuple = (6, 6),
        class_attention: str = 'linear',
        class_pad_len: int = 0,
        class_alpha: float = 0.5,
        class_drop: float = 0.1,
        class_blocks_type: str = 'linear',        # 'linear' / 'decoupled'
        class_num_prototypes: int = 32,
        num_novel_prototypes: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.prompt_channel = prompt_channel
        self.num_layers = num_layers
        self.use_temp_scale = use_temp_scale
        self.use_cost_std = use_cost_std
        self.use_class_norm = use_class_norm
        self.class_blocks_type = (class_blocks_type or 'linear').lower()
        self.class_num_prototypes = int(class_num_prototypes)
        self.num_novel_prototypes = int(num_novel_prototypes)
        self.prototype_diversity_weight = 0.0

        if self.use_temp_scale:
            default_scale = init_logit_scale if init_logit_scale is not None else math.log(1.0 / 0.07)
            self.logit_scale = nn.Parameter(torch.tensor(default_scale, dtype=torch.float32))
        else:
            self.register_buffer('logit_scale', torch.tensor(0.0, dtype=torch.float32), persistent=False)

        # Prompt-wise correlation embedding.
        self.corr_embed = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guid_proj = nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, padding=1)
        self.text_proj = nn.Linear(text_guidance_dim, text_guidance_proj_dim)

        self.spatial_gate = nn.Conv2d(appearance_guidance_proj_dim, hidden_dim, kernel_size=1)
        self.text_gate = nn.Linear(text_guidance_proj_dim, 1)

        layers = []
        in_dim = hidden_dim
        for _ in range(num_layers):
            layers.append(_ConvBlock2d(in_dim, in_dim))
        self.layers2d = nn.Sequential(*layers)

        self.use_class_linear = use_class_linear
        self.class_alpha = float(class_alpha)
        if self.use_class_linear:
            ch = class_hidden_dim or hidden_dim
            self.class_pool = class_pool if isinstance(class_pool, tuple) else (class_pool, class_pool)
            n_layers = max(1, class_layers)
            blocks = []
            for _ in range(n_layers):
                if self.class_blocks_type == 'decoupled':
                    blk = _ClassDecoupledAggBlock(
                        hidden_dim=ch,
                        guidance_dim=text_guidance_proj_dim,
                        nheads=class_nheads,
                        num_prototypes=self.class_num_prototypes,
                        num_novel_prototypes=self.num_novel_prototypes,
                        drop=class_drop,
                    )
                else:
                    blk = _ClassLinearAggBlock(
                        hidden_dim=ch,
                        guidance_dim=text_guidance_proj_dim,
                        nheads=class_nheads,
                        attention_type=class_attention,
                        drop=class_drop,
                    )
                blocks.append(blk)
            self.class_blocks = nn.ModuleList(blocks)

        dec = []
        for d in decoder_dims:
            dec.append(_ConvBlock2d(in_dim, d))
            in_dim = d
        self.decoder = nn.Sequential(*dec) if len(dec) > 0 else nn.Identity()
        self.head = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)

    def get_diversity_loss(self) -> torch.Tensor:
        """Aggregate prototype diversity loss."""
        if not getattr(self, "use_class_linear", False):
            return torch.tensor(0.0, device=self.corr_embed.weight.device)
        if not hasattr(self, "class_blocks"):
            return torch.tensor(0.0, device=self.corr_embed.weight.device)

        losses = []
        for blk in self.class_blocks:
            if isinstance(blk, _ClassDecoupledAggBlock):
                losses.append(blk.get_diversity_loss())
        if not losses:
            return torch.tensor(0.0, device=self.corr_embed.weight.device)
        return torch.stack(losses, dim=0).mean()

    def forward(
        self,
        img_feats: torch.Tensor,
        text_feats: torch.Tensor,
        appearance_guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = img_feats.shape

        if text_feats.dim() == 4:
            text_feats = text_feats.squeeze(2)
        img = F.normalize(img_feats, dim=1)
        text = F.normalize(text_feats, dim=-1)
        text = text.to(img.dtype)

        T = text.shape[1]

        text4 = text.unsqueeze(2)
        corr = torch.einsum('bchw,btpc->bpthw', img, text4)

        if self.use_temp_scale:
            corr = corr * self.logit_scale.exp().to(corr.dtype)

        if self.use_cost_std:
            mean_hw = corr.mean(dim=(3, 4), keepdim=True)
            var_hw = corr.var(dim=(3, 4), keepdim=True, unbiased=False)
            std_hw = torch.sqrt(var_hw + 1e-6)
            corr = (corr - mean_hw) / std_hw

        if self.use_class_norm:
            mean_t = corr.mean(dim=2, keepdim=True)
            var_t = corr.var(dim=2, keepdim=True, unbiased=False)
            std_t = torch.sqrt(var_t + 1e-6)
            corr = (corr - mean_t) / std_t

        corr_bt = corr.permute(0, 2, 1, 3, 4).reshape(B * T, self.prompt_channel, H, W)
        corr_embed = F.gelu(self.corr_embed(corr_bt))

        if appearance_guidance is None:
            appearance_guidance = img_feats
        g = self.guid_proj(appearance_guidance)
        s_map = self.spatial_gate(g)
        s_map_bt = s_map.unsqueeze(1).repeat(1, T, 1, 1, 1).reshape(B * T, self.hidden_dim, H, W)
        corr_embed = corr_embed * torch.sigmoid(s_map_bt) + corr_embed

        proj_dtype = self.text_proj.weight.dtype
        t_proj = self.text_proj(text.to(proj_dtype))
        w_class = torch.sigmoid(self.text_gate(t_proj)).reshape(B * T, 1, 1, 1)
        corr_embed = corr_embed * (1.0 + w_class)

        corr_embed = self.layers2d(corr_embed)

        if self.use_class_linear:
            x = corr_embed.reshape(B, T, self.hidden_dim, H, W)
            t_proj = self.text_proj(text)  # B x T x Tp
            Bh, Bw = self.class_pool
            x_flat = x.reshape(B * T, self.hidden_dim, H, W)
            x_pool = F.adaptive_avg_pool2d(x_flat, (Bh, Bw))
            x_pool = x_pool.reshape(B, T, self.hidden_dim, Bh, Bw)
            for blk in self.class_blocks:
                x_pool = blk(x_pool, t_proj)
            x_up = F.interpolate(
                x_pool.reshape(B * T, self.hidden_dim, Bh, Bw),
                size=(H, W), mode='bilinear', align_corners=False)
            x_up = x_up.reshape(B, T, self.hidden_dim, H, W)
            x = x + self.class_alpha * x_up
            corr_embed = x.reshape(B * T, self.hidden_dim, H, W)
        corr_embed = self.decoder(corr_embed)
        logits_bt = self.head(corr_embed)

        logits = logits_bt.reshape(B, T, 1, H, W).squeeze(2)
        return logits


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class _LinearAttention(nn.Module):
    """Multi-head linear attention."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q = _elu_feature_map(Q)
        K = _elu_feature_map(K)
        v_length = V.size(1)
        V = V / v_length
        KV = torch.einsum('nshd,nshv->nhdv', K, V)
        Z = 1.0 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        out = torch.einsum('nlhd,nhdv,nlh->nlhv', Q, KV, Z) * v_length
        return out.contiguous()


class _ClassLinearAggBlock(nn.Module):
    """Class-wise aggregation block."""
    def __init__(self, hidden_dim: int, guidance_dim: int, nheads: int = 4,
                 attention_type: str = 'linear', drop: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.guidance_dim = guidance_dim
        self.nheads = nheads
        self.attn_type = attention_type
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if attention_type == 'linear':
            self.attn = _LinearAttention()
        else:
            self.attn = None
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_bhw = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        g = guidance.unsqueeze(1).unsqueeze(1).expand(B, H, W, T, guidance.size(-1))
        g = g.reshape(B * H * W, T, guidance.size(-1))
        xq = torch.cat([x_bhw, g], dim=-1)
        xk = torch.cat([x_bhw, g], dim=-1)
        q = self.q(xq)
        k = self.k(xk)
        v = self.v(x_bhw)
        Hd = C // self.nheads
        Hh = self.nheads
        if C % self.nheads != 0 or Hd == 0:
            Hh = 1
            Hd = C
        q = q.view(B * H * W, T, Hh, Hd)
        k = k.view(B * H * W, T, Hh, Hd)
        v = v.view(B * H * W, T, Hh, Hd)
        if self.attn is not None:
            out = self.attn(q, k, v)
        else:
            scale = Hd ** -0.5
            A = torch.einsum('nlhd,nshd->nlsh', q * scale, k)
            A = torch.softmax(A, dim=2)
            out = torch.einsum('nlsh,nshd->nlhd', A, v)
        out = out.reshape(B * H * W, T, C)
        out = out + self.ln1(x_bhw)
        out = out + self.ffn(self.ln2(out))
        out = out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return out


class _ClassDecoupledAggBlock(nn.Module):
    """Decoupled aggregation that routes classes through shared prototypes."""
    def __init__(
        self,
        hidden_dim: int,
        guidance_dim: int,
        nheads: int = 4,
        num_prototypes: int = 32,
        num_novel_prototypes: int = 0,
        drop: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.guidance_dim = guidance_dim
        self.nheads = nheads
        self.num_base_prototypes = int(num_prototypes)
        self.num_novel_prototypes = int(num_novel_prototypes)

        self.base_prototypes = nn.Parameter(torch.empty(1, self.num_base_prototypes, hidden_dim))
        nn.init.normal_(self.base_prototypes, std=0.02)
        if self.num_novel_prototypes > 0:
            self.novel_prototypes = nn.Parameter(torch.empty(1, self.num_novel_prototypes, hidden_dim))
            nn.init.normal_(self.novel_prototypes, std=0.02)
        else:
            self.register_parameter('novel_prototypes', None)
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k_proto = nn.Linear(hidden_dim, hidden_dim)
        self.v_proto = nn.Linear(hidden_dim, hidden_dim)
        self.attn = _LinearAttention()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(drop),
        )

    @property
    def prototypes(self) -> torch.Tensor:
        if self.novel_prototypes is None:
            return self.base_prototypes
        return torch.cat([self.base_prototypes, self.novel_prototypes], dim=1)

    def init_novel_prototypes(self, novel_text_proj: torch.Tensor, std: float = 0.01) -> None:
        if self.novel_prototypes is None:
            return
        if novel_text_proj.dim() != 2 or novel_text_proj.size(1) != self.hidden_dim:
            raise ValueError(
                f"novel_text_proj must be (N,{self.hidden_dim}); got {tuple(novel_text_proj.shape)}"
            )
        num_classes = novel_text_proj.size(0)
        if num_classes == 0:
            return

        total = self.novel_prototypes.size(1)
        reps = total // num_classes
        rem = total % num_classes
        chunks = []
        for i in range(num_classes):
            k = reps + (1 if i < rem else 0)
            if k > 0:
                chunks.append(novel_text_proj[i : i + 1].expand(k, -1))
        init = torch.cat(chunks, dim=0)[:total]
        with torch.no_grad():
            self.novel_prototypes.data[0].copy_(init)
            if std and std > 0:
                self.novel_prototypes.data.add_(torch.randn_like(self.novel_prototypes.data) * float(std))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        old_key = prefix + 'prototypes'
        new_key = prefix + 'base_prototypes'
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def get_diversity_loss(self) -> torch.Tensor:
        """Encourage prototype diversity."""
        protos = F.normalize(self.prototypes.squeeze(0), dim=-1)
        cosine_sim = torch.matmul(protos, protos.t())
        n = cosine_sim.size(0)
        if n <= 1:
            return cosine_sim.new_tensor(0.0)
        diag_mask = torch.eye(n, device=cosine_sim.device, dtype=torch.bool)
        cosine_sim = cosine_sim.masked_fill(diag_mask, 0.0)
        diversity_loss = torch.abs(cosine_sim).mean()
        return diversity_loss

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        N = B * H * W
        x_bhw = x.permute(0, 3, 4, 1, 2).reshape(N, T, C)
        g = guidance.unsqueeze(1).unsqueeze(1).expand(B, H, W, T, guidance.size(-1))
        g = g.reshape(N, T, guidance.size(-1))
        q_in = torch.cat([x_bhw, g], dim=-1)
        q = self.q(q_in)
        prototypes = self.prototypes.expand(N, -1, -1)
        k = self.k_proto(prototypes)
        v = self.v_proto(prototypes)
        Hh = self.nheads
        Hd = C // Hh if Hh > 0 else C
        if C % Hh != 0 or Hd == 0:
            Hh = 1
            Hd = C
        q = q.view(N, T, Hh, Hd)
        P = k.size(1)
        k = k.view(N, P, Hh, Hd)
        v = v.view(N, P, Hh, Hd)
        out = self.attn(q, k, v)
        out = out.reshape(N, T, C)
        out = out + self.ln1(x_bhw)
        out = out + self.ffn(self.ln2(out))
        out = out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return out


@HEADS.register_module()
class GalaFusionHead(BaseDecodeHead):
    """Fusion head for ATM logits and cost-volume aggregation."""

    def __init__(
        self,
        img_size,
        in_channels,
        seen_idx,
        all_idx,
        channels=512,
        num_classes=20,
        num_layers=3,
        num_heads=8,
        use_proj=True,
        use_stages=1,
        embed_dims=512,
        crop_train=False,
        fusion_alpha: float = 0.6,
        inference_weight: float = 0.01,
        use_dynamic_fusion: bool = True,
        use_atm_branch: bool = True,
        novel_class: Optional[list] = None,
        text_embedding_path: Optional[str] = None,
        novel_init: bool = True,
        novel_init_std: float = 0.01,
        cat_agg: Optional[Dict] = None,
        proto_div_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.fusion_alpha = fusion_alpha
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_atm_branch = use_atm_branch
        self.novel_class = list(novel_class) if novel_class is not None else []
        self.text_embedding_path = text_embedding_path
        self.novel_init = bool(novel_init)
        self.novel_init_std = float(novel_init_std)
        self._novel_initialized = False
        self.proto_div_weight = float(proto_div_weight)
        self.inference_weight = float(inference_weight)

        self.atm = ATMSingleHeadSeg(
            img_size=img_size,
            in_channels=in_channels,
            seen_idx=seen_idx,
            all_idx=all_idx,
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            use_stages=use_stages,
            use_proj=use_proj,
            crop_train=crop_train,
            inference_weight=self.inference_weight,
            channels=channels,
            num_classes=num_classes,
        )

        cat_agg = cat_agg or {}
        self.agg = CATAggregator(**cat_agg)

        if self.use_dynamic_fusion and self.use_atm_branch:
            self.fuse_gate = nn.Conv2d(2, 1, kernel_size=1)
            nn.init.zeros_(self.fuse_gate.weight)
            alpha = float(self.fusion_alpha)
            alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
            bias_val = math.log(alpha) - math.log(1.0 - alpha)
            self.fuse_gate.bias.data.fill_(bias_val)
        else:
            self.fuse_gate = None

        if hasattr(self, 'conv_seg'):
            delattr(self, 'conv_seg')

    def init_weights(self):
        if hasattr(self.atm, 'init_weights'):
            self.atm.init_weights()

    def _maybe_init_novel_prototypes(self) -> None:
        if self._novel_initialized:
            return
        self._novel_initialized = True

        if not self.novel_init:
            return
        if not self.text_embedding_path or not self.novel_class:
            return
        if not hasattr(self.agg, "class_blocks"):
            return

        decoupled_blocks = []
        for blk in self.agg.class_blocks:
            if isinstance(blk, _ClassDecoupledAggBlock) and blk.novel_prototypes is not None:
                decoupled_blocks.append(blk)
        if not decoupled_blocks:
            return

        import numpy as np

        device = next(self.parameters()).device
        proj_dtype = self.agg.text_proj.weight.dtype

        text_emb = np.load(self.text_embedding_path)
        text_emb = torch.from_numpy(text_emb).to(device=device, dtype=proj_dtype)
        novel_ids = torch.tensor(self.novel_class, device=device, dtype=torch.long)
        novel_text = text_emb.index_select(0, novel_ids)
        with torch.no_grad():
            novel_text_proj = self.agg.text_proj(novel_text)

        for blk in decoupled_blocks:
            if novel_text_proj.size(1) != blk.hidden_dim:
                raise ValueError(
                    f"text_proj dim {novel_text_proj.size(1)} != prototype hidden_dim {blk.hidden_dim}"
                )
            blk.init_novel_prototypes(novel_text_proj.to(dtype=blk.base_prototypes.dtype), std=self.novel_init_std)

    def forward_train(self, inputs_both, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        out = self.forward(inputs_both, self_training)
        gt = gt_semantic_seg.clone()
        if self_training:
            pseudo_masks = out["pred_masks"].detach().sigmoid()
            pseudo_masks[:, self.seen_idx, :, :] = -1
            pseudo_seg = pseudo_masks.argmax(dim=1).unsqueeze(1)
            gt[gt == -1] = pseudo_seg[gt == -1]
        gt[gt == -1] = 255
        losses = self.losses(out, gt)

        if self.proto_div_weight > 0.0:
            div_loss = self.agg.get_diversity_loss()
            losses["loss_proto_div"] = self.proto_div_weight * div_loss
        return losses

    def forward_test(self, inputs_both, img_metas, test_cfg, self_training):
        return self.forward(inputs_both, self_training)

    def _extract_lateral(self, inputs_both):
        """Extract fused lateral features."""
        inputs = inputs_both[0][0]
        x = []
        for stage_ in inputs[: self.atm.use_stages]:
            x.append(self.atm.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()

        laterals = []
        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.atm.input_proj, self.atm.proj_norm)):
            lateral = norm_(proj_(x_))
            if idx == 0:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    l_ = self.atm.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.atm.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        lateral = laterals[-1]
        img_feats = self.atm.d3_to_d4(lateral)
        return lateral, img_feats

    def _atm_logits(self, inputs_both):
        """Compute ATM logits."""
        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]

        x = []
        for stage_ in inputs[: self.atm.use_stages]:
            x.append(self.atm.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()

        laterals = []
        attns = []
        maps_size = []
        qs = []

        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.atm.input_proj, self.atm.proj_norm)):
            lateral = norm_(proj_(x_))
            if idx == 0:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    l_ = self.atm.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.atm.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        lateral = laterals[-1]

        q = self.atm.q_proj(self.atm.get_qs(text_token, cls_token))
        q = q.transpose(0, 1)

        for idx, decoder_ in enumerate(self.atm.decoder):
            q_, attn_ = decoder_(q, lateral.transpose(0, 1))
            for q_i, attn in zip(q_, attn_):
                attn = attn.transpose(-1, -2)
                attn = self.atm.d3_to_d4(attn)
                maps_size.append(attn.size()[-2:])
                qs.append(q_i.transpose(0, 1))
                attns.append(attn)

        outputs_seg_masks = []
        size = maps_size[-1]
        for i_attn, attn in enumerate(attns):
            outputs_seg_masks.append(
                F.interpolate(attn, size=size, mode='bilinear', align_corners=False)
            )

        pred = F.interpolate(
            outputs_seg_masks[-1],
            size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False,
        )
        return pred

    def forward(self, inputs_both, self_training: Optional[bool] = None):
        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]
        self._maybe_init_novel_prototypes()

        atm_logits = None
        if self.use_atm_branch:
            atm_logits = self._atm_logits(inputs_both)

        lateral, img_feats = self._extract_lateral(inputs_both)

        B = img_feats.size(0)
        T = text_token.size(0)
        text_feats = text_token.unsqueeze(0).expand(B, -1, -1)

        cat_logits = self.agg(
            img_feats=img_feats,
            text_feats=text_feats,
            appearance_guidance=img_feats,
        )

        cat_logits = resize(
            input=cat_logits,
            size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False)

        if self.use_dynamic_fusion and self.fuse_gate is not None and atm_logits is not None:
            logits_stack = torch.stack([cat_logits, atm_logits], dim=2)
            BT = B * T
            H, W = cat_logits.shape[-2:]
            gate_in = logits_stack.reshape(BT, 2, H, W)
            gate = torch.sigmoid(self.fuse_gate(gate_in))
            gate = gate.reshape(B, T, H, W)
            fused_logits = gate * cat_logits + (1.0 - gate) * atm_logits
        elif atm_logits is not None:
            fused_logits = self.fusion_alpha * cat_logits + (1.0 - self.fusion_alpha) * atm_logits
        else:
            fused_logits = cat_logits

        if self.training:
            return {"pred_masks": fused_logits}
        else:
            weight = 0.0 if self_training else self.inference_weight
            pred = self.semantic_inference(fused_logits, self.seen_idx, weight)
            return pred

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:, seen_idx] = mask_pred[:, seen_idx] - weight
        return mask_pred

    def d3_to_d4(self, t: torch.Tensor) -> torch.Tensor:
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t: torch.Tensor) -> torch.Tensor:
        return t.flatten(-2).transpose(-1, -2)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        if isinstance(seg_logit, dict):
            seg_label = seg_label.squeeze(1)
            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index=self.ignore_index
            )
            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss
