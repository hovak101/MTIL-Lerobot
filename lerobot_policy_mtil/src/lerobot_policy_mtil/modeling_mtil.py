"""MTIL policy: Mamba-2 backbone over a DINOv2 + state encoder."""

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
from .sampler_mtil import DINO_FEATURES_KEY

logger = logging.getLogger(__name__)


# ImageNet normalization (DINOv2 was trained with these stats).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CrossModalAttentionBlock(nn.Module):
    """State-as-query cross-attention over per-camera DINOv2 [CLS] tokens.

    Implements the "Cross-modal Attention" stage shown in Fig. 1 of the MTIL
    paper. The state token attends to the per-camera image tokens, allowing
    input-dependent fusion (vs the fixed concat → linear baseline).

    Variable-N camera support: ``cam_proj`` is a single linear layer applied
    to every camera's [CLS] vector — works for any number of cameras with no
    architecture change.

    If the dataset has no state feature, the query is a learnable token.
    """

    def __init__(
        self,
        dinov2_hidden: int,
        state_dim: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cam_proj = nn.Linear(dinov2_hidden, d_model)
        if state_dim > 0:
            self.state_proj: nn.Linear | None = nn.Linear(state_dim, d_model)
            self.register_parameter("query_token", None)
        else:
            self.state_proj = None
            self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.query_token, std=0.02)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, cam_features: Tensor, state: Tensor | None) -> Tensor:
        """Fuse multi-cam features and state into one ``(N, d_model)`` token.

        Args:
            cam_features: ``(N, C, dinov2_hidden)`` per-camera [CLS] tokens.
            state: ``(N, state_dim)`` or ``None`` if no state feature.
        """
        kv = self.cam_proj(cam_features)  # (N, C, d_model)
        if self.state_proj is not None:
            assert state is not None
            q = self.state_proj(state).unsqueeze(1)  # (N, 1, d_model)
        else:
            q = self.query_token.expand(cam_features.shape[0], -1, -1)  # (N, 1, d_model)

        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), kv, need_weights=False)
        x = q + self.attn_dropout(attn_out)  # residual, (N, 1, d_model)
        x = x + self.ffn_dropout(self.ffn(self.norm_ffn(x)))
        return x.squeeze(1)  # (N, d_model)


class MTILPolicy(PreTrainedPolicy):
    config_class = MTILConfig
    name = "mtil"

    def __init__(self, config: MTILConfig, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # --- DINOv2 image encoder (lazy import: heavy + may not be installed in all envs) ---
        from transformers import Dinov2Model

        self.dinov2 = Dinov2Model.from_pretrained(config.dinov2_variant)
        self._dinov2_hidden = self.dinov2.config.hidden_size
        if config.freeze_dinov2:
            for p in self.dinov2.parameters():
                p.requires_grad_(False)
            self.dinov2.eval()

        # ImageNet normalization buffers (broadcast over (..., C, H, W)).
        self.register_buffer(
            "_imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # --- Per-step cross-modal fusion: cams + state -> d_model token ---
        self.image_keys = sorted(config.image_features.keys())
        n_cams = len(self.image_keys)
        if n_cams == 0:
            raise ValueError("MTILPolicy requires at least one image feature.")
        self._state_dim = (
            config.robot_state_feature.shape[0] if config.robot_state_feature is not None else 0
        )
        self.cross_modal = CrossModalAttentionBlock(
            dinov2_hidden=self._dinov2_hidden,
            state_dim=self._state_dim,
            d_model=config.d_model,
            n_heads=config.cross_attn_heads,
            dropout=config.dropout,
        )
        # Single dropout module reused after each Mamba-2 residual addition.
        # nn.Dropout is a no-op in .eval() mode, so this is automatically off
        # during inference and validation.
        self.mamba_dropout = nn.Dropout(config.dropout)

        # --- Mamba-2 stack (lazy import — requires CUDA + Triton at runtime) ---
        from mamba_ssm.modules.mamba2 import Mamba2

        self.mamba_layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=config.d_model,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    headdim=config.mamba_headdim,
                    layer_idx=i,
                )
                for i in range(config.n_mamba_layers)
            ]
        )
        # Pre-norm style residual; final norm before the action head.
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.n_mamba_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        # --- Action head: per-step (d_model) -> (chunk_size, action_dim) ---
        self._action_dim = config.action_feature.shape[0]
        self.action_head = nn.Linear(config.d_model, config.chunk_size * self._action_dim)

        # --- Inference state (NOT registered buffers — must not get serialized) ---
        self._inference_state: list | None = None  # [(conv_state, ssm_state) per layer]
        self._action_queue: deque[Tensor] = deque()
        self._temporal_ensembler: ACTTemporalEnsembler | None = None
        self.reset()

    # ----------------------------------------------------------------------
    # PreTrainedPolicy abstract methods
    # ----------------------------------------------------------------------

    def reset(self) -> None:
        """Clear hidden state, action queue, and ensembler. Call between episodes."""
        self._inference_state = None
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        if self.config.temporal_ensemble_coeff is not None:
            self._temporal_ensembler = ACTTemporalEnsembler(
                self.config.temporal_ensemble_coeff, self.config.chunk_size
            )

    def get_optim_params(self) -> list[dict]:
        # Keep DINOv2 in its own group so LR can be controlled (or zero'd when frozen).
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

    # ----------------------------------------------------------------------
    # Encoder
    # ----------------------------------------------------------------------

    def _normalize_for_dinov2(self, images: Tensor) -> Tensor:
        """Resize to dinov2_image_size and apply ImageNet normalization.

        Input: ``(N, 3, H, W)`` floats in ``[0, 1]``. Output: ``(N, 3, S, S)``.
        """
        S = self.config.dinov2_image_size
        if images.shape[-2:] != (S, S):
            images = F.interpolate(images, size=(S, S), mode="bilinear", align_corners=False)
        return (images - self._imagenet_mean) / self._imagenet_std

    def _encode_step(self, images_per_cam: list[Tensor], state: Tensor | None) -> Tensor:
        """Encode a single time step into ``(N, d_model)``.

        Args:
            images_per_cam: list of ``(N, 3, H, W)`` tensors, one per camera.
            state: ``(N, state_dim)`` or ``None`` if no state feature.
        """
        cls_per_cam: list[Tensor] = []
        ctx = torch.no_grad() if self.config.freeze_dinov2 else contextlib.nullcontext()
        with ctx:
            for img in images_per_cam:
                normed = self._normalize_for_dinov2(img)
                out = self.dinov2(pixel_values=normed)
                cls_per_cam.append(out.last_hidden_state[:, 0])  # (N, dinov2_hidden)
        cam_features = torch.stack(cls_per_cam, dim=1)  # (N, C, dinov2_hidden)
        return self.cross_modal(cam_features, state)

    def _encode_step_from_features(self, features: Tensor, state: Tensor | None) -> Tensor:
        """Cached-feature path: features already encode all cameras' [CLS] tokens.

        Args:
            features: ``(N, n_cams, dinov2_hidden)`` precomputed [CLS] tokens.
            state: ``(N, state_dim)`` or ``None``.
        """
        # Up-cast cached fp16 features to the model's working dtype.
        features = features.to(self.cross_modal.cam_proj.weight.dtype)
        # Train-time augmentation on cached features: cheap stand-in for image
        # transforms (which would invalidate the cache). Off in eval mode so
        # validation and inference see the deterministic features.
        std = self.config.dino_feature_noise_std
        if self.training and std > 0.0:
            features = features + torch.randn_like(features) * std
        return self.cross_modal(features, state)

    def _encode_sequence(self, batch: dict[str, Tensor], B: int, T: int) -> Tensor:
        """Encode a (B, T, ...) batch into (B, T, d_model)."""
        if self._state_dim > 0:
            state = batch[OBS_STATE]
            if state.dim() == 3:
                state = state.reshape(B * T, self._state_dim)
        else:
            state = None

        # Fast path: cached DINOv2 features in the batch.
        if DINO_FEATURES_KEY in batch:
            feats = batch[DINO_FEATURES_KEY]
            # (B, T, n_cams, hidden) → (B*T, n_cams, hidden); inference may
            # promote (B, n_cams, hidden) → (B, 1, n_cams, hidden) externally.
            if feats.dim() == 4:
                feats = feats.reshape(B * T, *feats.shape[2:])
            x = self._encode_step_from_features(feats, state)
            return x.view(B, T, -1)

        # Slow path: live image encode through DINOv2.
        images_flat = []
        for k in self.image_keys:
            img = batch[k]  # (B, T, C, H, W) or (B*T, C, H, W) at inference (T=1 promoted)
            if img.dim() == 5:
                img = img.reshape(B * T, *img.shape[2:])
            images_flat.append(img)
        x = self._encode_step(images_flat, state)  # (B*T, d_model)
        return x.view(B, T, -1)

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Compute training loss over a ``(B, T, ...)`` batch."""
        actions_gt = batch[ACTION]
        if actions_gt.dim() != 4:
            raise ValueError(
                f"MTILPolicy.forward expects action shape (B, T, chunk, dim); got {tuple(actions_gt.shape)}. "
                "Did you forget to use mtil_collate_fn / MTILEpisodeBatchSampler?"
            )
        B, T = actions_gt.shape[:2]

        x = self._encode_sequence(batch, B, T)  # (B, T, d_model)

        # Stateless per sub-sequence in v1 (see module docstring).
        for layer, norm in zip(self.mamba_layers, self.layer_norms, strict=True):
            x = x + self.mamba_dropout(layer(norm(x)))
        x = self.final_norm(x)

        a_hat = self.action_head(x).view(B, T, self.config.chunk_size, self._action_dim)

        action_pad = batch.get("action_is_pad")
        if action_pad is None:
            action_pad = torch.zeros(B, T, self.config.chunk_size, dtype=torch.bool, device=a_hat.device)
        frame_pad = batch.get("frame_is_pad")
        if frame_pad is None:
            frame_pad = torch.zeros(B, T, dtype=torch.bool, device=a_hat.device)
        valid = (~action_pad) & (~frame_pad.unsqueeze(-1))  # (B, T, chunk)

        sq_err = (a_hat - actions_gt) ** 2  # (B, T, chunk, dim)
        denom = valid.sum().clamp_min(1) * a_hat.shape[-1]
        loss = (sq_err * valid.unsqueeze(-1)).sum() / denom

        # Return loss tensor (no .item()) — caller converts at log boundary so
        # we don't force a CUDA sync every training step.
        return loss, {"mse_loss": loss.detach()}

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    def _allocate_inference_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list:
        cache = []
        for layer in self.mamba_layers:
            conv_state, ssm_state = layer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype)
            cache.append([conv_state.to(device), ssm_state.to(device)])
        return cache

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Single-step inference. Advances the recurrent state by one frame.

        Returns ``(B, chunk_size, action_dim)``.
        """
        self.eval()

        # Promote single-frame batch to T=1: (B, ...) -> (B, 1, ...)
        sample_action = batch.get(ACTION)
        if sample_action is not None and sample_action.dim() == 2:
            B = sample_action.shape[0]
        else:
            # Infer B from first image feature
            first_img = batch[self.image_keys[0]]
            B = first_img.shape[0]

        # Build a T=1 batch dict.
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

        # Allocate or reuse inference cache.
        if self._inference_state is None:
            self._inference_state = self._allocate_inference_cache(B, device, dtype)

        # Run one step through every Mamba layer.
        for i, (layer, norm) in enumerate(zip(self.mamba_layers, self.layer_norms, strict=True)):
            normed = norm(x)
            conv_state, ssm_state = self._inference_state[i]
            # Mamba2.step expects (B, 1, D) hidden states. Returns (B, 1, D), (conv, ssm).
            out, conv_state_new, ssm_state_new = layer.step(normed, conv_state, ssm_state)
            self._inference_state[i] = [conv_state_new, ssm_state_new]
            x = x + out
        x = self.final_norm(x).squeeze(1)  # (B, d_model)

        chunk = self.action_head(x).view(B, self.config.chunk_size, self._action_dim)
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action per env step. Manages chunk queue / temporal aggregation."""
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            chunk = self.predict_action_chunk(batch)
            return self._temporal_ensembler.update(chunk)

        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
