"""MTIL policy: faithful port of the official MTIL paper architecture.

Mirrors `MTIL/train/mamba_policy.py`:
  * DINOv2 ViT-L/14 backbone, frozen, layer-(-4) **spatial features** (CLS dropped)
  * SpatialAdapter: 3 convs + flatten + linear -> embed_dim, then LayerNorm/ReLU/Dropout
  * CrossCameraAttention (multihead, 16 heads, when n_cams > 1)
  * CrossModalAttention with deep state projection (14 -> 128 -> 512 -> d_model)
    where camera features are the query and projected state is key/value
  * 4x Block (Mamba-2 + 4x MLP w/ GELU, LayerNorm pre-residual)
  * Action head: linear -> chunk_size x action_dim
"""

from __future__ import annotations

import contextlib
import logging
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_mtil import MTILConfig

logger = logging.getLogger(__name__)


# ImageNet normalization (DINOv2 was trained with these stats).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINOV2_PATCH_SIZE = 14


class SpatialAdapter(nn.Module):
    """Compress (B, C_in, H_p, W_p) DINOv2 spatial features to (B, embed_dim).

    Mirrors the official `spatial_adapter` exactly — a 3-conv stack with
    stride-2 in the last conv, flatten, linear, LayerNorm, ReLU, Dropout(0.10).
    """

    def __init__(self, in_dim: int, embed_dim: int, patch_grid: int, dropout: float = 0.10):
        super().__init__()
        # After Conv stride-2 with kernel 3, padding 1: ceil(patch_grid / 2).
        post = (patch_grid + 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
        )
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(128 * post * post, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class CrossCameraAttention(nn.Module):
    """Self-attention on the concatenated camera vector. Mirrors official.

    The official treats `cat([cam1_feat, cam2_feat, ...], dim=1)` as a single
    token of size `embed_dim * n_cams` and runs a 16-head self-attention on it
    with residual + LayerNorm. With one token, self-attention reduces to a
    learnable linear transform followed by the residual/norm.
    """

    def __init__(self, d_model: int, num_heads: int = 16):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, d_model) — concatenated cam features as a single token.
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)


class CrossModalAttention(nn.Module):
    """Camera tokens (Q) attend to deeply-projected state (KV).

    Official: state goes through Linear(state_dim->128) -> GELU ->
    Linear(128->512) -> Dropout(0.2) -> Linear(512->d_model). Then a single
    multi-head attention with cam features as query, projected state as K/V,
    residual-add into camera features and LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        num_heads: int = 8,
        dropout: float = 0.20,
    ):
        super().__init__()
        if state_dim > 0:
            self.state_proj: nn.Sequential | None = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.GELU(),
                nn.Linear(128, 512),
                nn.Dropout(dropout),
                nn.Linear(512, d_model),
            )
            self.register_parameter("learned_kv", None)
        else:
            self.state_proj = None
            # Fallback when no state feature exists: a single learnable token.
            self.learned_kv = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.learned_kv, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cam_q: Tensor, state: Tensor | None) -> Tensor:
        # cam_q: (B, 1, d_model). state: (B, state_dim) or None.
        if self.state_proj is not None:
            assert state is not None
            kv = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)
        else:
            kv = self.learned_kv.expand(cam_q.shape[0], -1, -1)
        attn_out, _ = self.attn(cam_q, kv, kv, need_weights=False)
        return self.norm(cam_q + attn_out)


class MTILBlock(nn.Module):
    """One block of the official policy stack: Mamba2 + 4x MLP, pre-norm.

    Layout follows `mamba_ssm.modules.block.Block`: residual = LayerNorm
    (residual then Mamba2), residual = LayerNorm (residual then MLP).
    """

    def __init__(
        self,
        d_model: int,
        mamba_cfg: dict,
        layer_idx: int,
    ):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.mixer = Mamba2(
            d_model=d_model,
            d_state=mamba_cfg["d_state"],
            d_conv=mamba_cfg["d_conv"],
            expand=mamba_cfg["expand"],
            headdim=mamba_cfg["headdim"],
            layer_idx=layer_idx,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm residual stack — equivalent to the official Block path.
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MTILPolicy(PreTrainedPolicy):
    config_class = MTILConfig
    name = "mtil"

    def __init__(self, config: MTILConfig, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # --- DINOv2 image encoder (frozen, layer-(-4) spatial features) ---
        from transformers import Dinov2Model

        self.dinov2 = Dinov2Model.from_pretrained(config.dinov2_variant)
        self._dinov2_hidden = self.dinov2.config.hidden_size
        if config.freeze_dinov2:
            for p in self.dinov2.parameters():
                p.requires_grad_(False)
            self.dinov2.eval()

        # Forward hook on layer -4 (paper's `self.dino.blocks[-4]`). We capture
        # only this layer's output instead of using `output_hidden_states=True`,
        # which would retain all 25 hidden states in memory simultaneously
        # (~13 GB at B*T=1024 with DINOv2-Large, the source of an OOM).
        self._dinov2_layer_features: Tensor | None = None

        def _capture_layer_minus_4(_module, _input, output):
            # Dinov2Layer returns a tuple (hidden_states, ...). Take only the
            # hidden states; storing the whole tuple keeps refs we don't need.
            self._dinov2_layer_features = output[0] if isinstance(output, tuple) else output

        self.dinov2.encoder.layer[-4].register_forward_hook(_capture_layer_minus_4)

        self.register_buffer(
            "_imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # Patch grid implied by image_size (square images, multiple of 14).
        S = config.dinov2_image_size
        if S % _DINOV2_PATCH_SIZE != 0:
            raise ValueError(
                f"dinov2_image_size ({S}) must be a multiple of {_DINOV2_PATCH_SIZE}."
            )
        self._patch_grid = S // _DINOV2_PATCH_SIZE

        # --- Per-camera spatial adapter (one shared adapter, applied per cam) ---
        self.image_keys = sorted(config.image_features.keys())
        n_cams = len(self.image_keys)
        if n_cams == 0:
            raise ValueError("MTILPolicy requires at least one image feature.")
        self._n_cams = n_cams
        # Each camera's spatial features are compressed to (B, embed_dim);
        # `embed_dim == d_model` matches the official paper's setup.
        self.spatial_adapter = SpatialAdapter(
            in_dim=self._dinov2_hidden,
            embed_dim=config.d_model,
            patch_grid=self._patch_grid,
            dropout=config.spatial_dropout,
        )
        # Concatenated camera vector dimensionality.
        self._concat_dim = config.d_model * n_cams

        # --- Cross-camera attention (only if multiple cameras) ---
        if n_cams > 1:
            self.cross_cam: nn.Module = CrossCameraAttention(
                d_model=self._concat_dim, num_heads=config.cross_cam_heads
            )
        else:
            self.cross_cam = nn.Identity()
        # Project concatenated camera vector down to d_model (paper: in_proj).
        self.in_proj = nn.Linear(self._concat_dim, config.d_model)

        # --- Cross-modal attention: cam Q, deeply-projected state KV ---
        self._state_dim = (
            config.robot_state_feature.shape[0] if config.robot_state_feature is not None else 0
        )
        self.cross_modal = CrossModalAttention(
            d_model=config.d_model,
            state_dim=self._state_dim,
            num_heads=config.cross_modal_heads,
            dropout=config.cross_modal_dropout,
        )

        # --- Mamba-2 stack: 4 blocks of (Mamba2 + MLP), pre-norm residuals ---
        mamba_cfg = {
            "d_state": config.mamba_d_state,
            "d_conv": config.mamba_d_conv,
            "expand": config.mamba_expand,
            "headdim": config.mamba_headdim,
        }
        self.blocks = nn.ModuleList(
            [
                MTILBlock(d_model=config.d_model, mamba_cfg=mamba_cfg, layer_idx=i)
                for i in range(config.n_mamba_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        # --- Action head: (d_model) -> (chunk_size, action_dim) ---
        self._action_dim = config.action_feature.shape[0]
        self.action_head = nn.Linear(config.d_model, config.chunk_size * self._action_dim)

        # --- Inference state ---
        self._inference_state: list | None = None
        self._action_queue: deque[Tensor] = deque()
        self._temporal_ensembler: ACTTemporalEnsembler | None = None
        self.reset()

    # ---- PreTrainedPolicy abstract methods ----

    def reset(self) -> None:
        self._inference_state = None
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        if self.config.temporal_ensemble_coeff is not None:
            self._temporal_ensembler = ACTTemporalEnsembler(
                self.config.temporal_ensemble_coeff, self.config.chunk_size
            )

    def get_optim_params(self) -> list[dict]:
        encoder_params = [p for n, p in self.named_parameters() if n.startswith("dinov2.")]
        body_params = [p for n, p in self.named_parameters() if not n.startswith("dinov2.")]
        groups = [{"params": [p for p in body_params if p.requires_grad]}]
        if any(p.requires_grad for p in encoder_params):
            groups.append(
                {
                    "params": [p for p in encoder_params if p.requires_grad],
                    "lr": self.config.optimizer_lr_dinov2,
                    "weight_decay": 0.0,
                }
            )
        return groups

    # ---- Encoder ----

    def _normalize_for_dinov2(self, images: Tensor) -> Tensor:
        S = self.config.dinov2_image_size
        if images.shape[-2:] != (S, S):
            images = F.interpolate(images, size=(S, S), mode="bilinear", align_corners=False)
        return (images - self._imagenet_mean) / self._imagenet_std

    def _extract_spatial(self, image: Tensor) -> Tensor:
        """Run DINOv2 and return layer-(-4) spatial features as (B, D, H_p, W_p).

        Uses a forward hook on `dinov2.encoder.layer[-4]` so only that layer's
        output is retained in memory (vs. `output_hidden_states=True`, which
        keeps all 25). DINOv2 is run in micro-batches of `dinov2_micro_batch`
        when set, to bound the per-call activation peak.
        """
        normed = self._normalize_for_dinov2(image)
        ctx = torch.no_grad() if self.config.freeze_dinov2 else contextlib.nullcontext()
        micro = self.config.dinov2_micro_batch
        N_total = normed.shape[0]
        chunks: list[Tensor] = []
        with ctx:
            if micro is None or micro >= N_total:
                self._dinov2_layer_features = None
                _ = self.dinov2(pixel_values=normed)
                chunks.append(self._dinov2_layer_features)
            else:
                for start in range(0, N_total, micro):
                    self._dinov2_layer_features = None
                    _ = self.dinov2(pixel_values=normed[start : start + micro])
                    chunks.append(self._dinov2_layer_features)
        feats = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        feats = feats[:, 1:, :]  # drop CLS
        B, N, D = feats.shape
        H = W = self._patch_grid
        if N != H * W:
            raise RuntimeError(
                f"DINOv2 returned {N} patch tokens but expected {H * W} for "
                f"image_size={self.config.dinov2_image_size}."
            )
        return feats.permute(0, 2, 1).contiguous().view(B, D, H, W)

    def _encode_step(self, images_per_cam: list[Tensor], state: Tensor | None) -> Tensor:
        """Encode (N, ...) frames per camera into (N, d_model)."""
        feats_per_cam: list[Tensor] = []
        for img in images_per_cam:
            spatial = self._extract_spatial(img)  # (N, D, H, W)
            feats_per_cam.append(self.spatial_adapter(spatial))  # (N, embed_dim)
        # Concatenate along feature dim: (N, embed_dim * n_cams). Matches the
        # official's `torch.cat(feats_all, dim=1)`.
        cam_cat = torch.cat(feats_per_cam, dim=1)
        # Cross-camera self-attention treats the concatenated vector as a
        # single token (paper: `.unsqueeze(1)` -> attention -> `.squeeze(1)`).
        if self._n_cams > 1:
            cam_cat = self.cross_cam(cam_cat.unsqueeze(1)).squeeze(1)
        # Project down to d_model and unsqueeze for cross-modal Q.
        cam_q = self.in_proj(cam_cat).unsqueeze(1)  # (N, 1, d_model)
        fused = self.cross_modal(cam_q, state).squeeze(1)  # (N, d_model)
        return fused

    def _encode_sequence(self, batch: dict[str, Tensor], B: int, T: int) -> Tensor:
        """Encode a (B, T, ...) batch into (B, T, d_model)."""
        if self._state_dim > 0:
            state = batch[OBS_STATE]
            if state.dim() == 3:
                state = state.reshape(B * T, self._state_dim)
            # Train-time per-feature Gaussian noise on the (already MEAN_STD-
            # normalized) state. Mirrors the official paper's per-joint noise.
            std = self.config.state_noise_std
            if self.training and std > 0.0:
                state = state + torch.randn_like(state) * std
        else:
            state = None

        images_flat: list[Tensor] = []
        for k in self.image_keys:
            img = batch[k]
            if img.dim() == 5:
                img = img.reshape(B * T, *img.shape[2:])
            images_flat.append(img)
        x = self._encode_step(images_flat, state)  # (B*T, d_model)
        return x.view(B, T, -1)

    # ---- Training ----

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        actions_gt = batch[ACTION]
        if actions_gt.dim() != 4:
            raise ValueError(
                f"MTILPolicy.forward expects action shape (B, T, chunk, dim); got {tuple(actions_gt.shape)}. "
                "Did you forget to use mtil_collate_fn / MTILEpisodeBatchSampler?"
            )
        B, T = actions_gt.shape[:2]

        x = self._encode_sequence(batch, B, T)  # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)

        a_hat = self.action_head(x).view(B, T, self.config.chunk_size, self._action_dim)

        action_pad = batch.get("action_is_pad")
        if action_pad is None:
            action_pad = torch.zeros(B, T, self.config.chunk_size, dtype=torch.bool, device=a_hat.device)
        frame_pad = batch.get("frame_is_pad")
        if frame_pad is None:
            frame_pad = torch.zeros(B, T, dtype=torch.bool, device=a_hat.device)
        valid = (~action_pad) & (~frame_pad.unsqueeze(-1))

        sq_err = (a_hat - actions_gt) ** 2
        denom = valid.sum().clamp_min(1) * a_hat.shape[-1]
        loss = (sq_err * valid.unsqueeze(-1)).sum() / denom

        return loss, {"mse_loss": loss.detach()}

    # ---- Inference ----

    def _allocate_inference_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list:
        cache = []
        for blk in self.blocks:
            conv_state, ssm_state = blk.mixer.allocate_inference_cache(
                batch_size, max_seqlen=1, dtype=dtype
            )
            cache.append([conv_state.to(device), ssm_state.to(device)])
        return cache

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Single-step inference. Advances the recurrent state by one frame."""
        self.eval()

        sample_action = batch.get(ACTION)
        if sample_action is not None and sample_action.dim() == 2:
            B = sample_action.shape[0]
        else:
            first_img = batch[self.image_keys[0]]
            B = first_img.shape[0]

        promoted: dict[str, Tensor] = {}
        for k in self.image_keys:
            v = batch[k]
            promoted[k] = v.unsqueeze(1) if v.dim() == 4 else v
        if self._state_dim > 0:
            v = batch[OBS_STATE]
            promoted[OBS_STATE] = v.unsqueeze(1) if v.dim() == 2 else v

        x = self._encode_sequence(promoted, B, 1)  # (B, 1, d_model)
        device = x.device
        dtype = x.dtype

        if self._inference_state is None:
            self._inference_state = self._allocate_inference_cache(B, device, dtype)

        for i, blk in enumerate(self.blocks):
            normed = blk.norm1(x)
            conv_state, ssm_state = self._inference_state[i]
            out, conv_state_new, ssm_state_new = blk.mixer.step(normed, conv_state, ssm_state)
            self._inference_state[i] = [conv_state_new, ssm_state_new]
            x = x + out
            x = x + blk.mlp(blk.norm2(x))
        x = self.final_norm(x).squeeze(1)

        chunk = self.action_head(x).view(B, self.config.chunk_size, self._action_dim)
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            chunk = self.predict_action_chunk(batch)
            return self._temporal_ensembler.update(chunk)

        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
