"""MTIL policy configuration — defaults match the official paper."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("mtil")
@dataclass
class MTILConfig(PreTrainedConfig):
    """Configuration for the Mamba Temporal Imitation Learning (MTIL) policy.

    Defaults reproduce the architecture and optimizer used in the official
    paper implementation (https://github.com/yulinzhouZYL/MTIL,
    `MambaConfig` + `LitMambaModel` in train.py). Only deviations from the
    paper are those required to live inside lerobot's policy/processor/
    sampler framework.
    """

    # --- Action chunking (paper: future_steps = 16) ---
    n_obs_steps: int = 1
    chunk_size: int = 16
    # Replans every 16 env steps by default — i.e. consume the entire chunk
    # before re-running the policy. Override on the CLI to replan more often.
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # DINOv2 has its own ImageNet norm
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- Encoder (paper: DINOv2 ViT-L/14, frozen, layer -4 spatial features) ---
    dinov2_variant: str = "facebook/dinov2-large"
    freeze_dinov2: bool = True
    # Square multiple of 14. Paper uses 640x480 native; 224 keeps DINOv2-Large
    # cost tractable on consumer GPUs (16 GB and below). Bump to 308/392 if
    # you have headroom and want closer parity with the paper's resolution.
    dinov2_image_size: int = 224
    # Per-call micro-batch size for the live DINOv2 forward. Each step encodes
    # `B*T*n_cams` images; running them all in one DINOv2 call peaks at several
    # GB of intermediate activations. None = single call (fastest, highest
    # peak); a small int = chunked calls (slower per step, lower peak). 64 is
    # a good default for 16 GB GPUs at d_model=2048.
    dinov2_micro_batch: int | None = 64

    # --- Cross-camera self-attention (paper: 16 heads on d_model) ---
    cross_cam_heads: int = 16

    # --- Cross-modal attention: cam Q, deeply-projected state KV ---
    cross_modal_heads: int = 8
    # Dropout inside the deep state projection (14 -> 128 -> 512 -> d_model);
    # paper hardcodes 0.20 between the 512 linear and the d_model linear.
    cross_modal_dropout: float = 0.20

    # --- Spatial adapter dropout (paper: 0.10 after the LayerNorm + ReLU) ---
    spatial_dropout: float = 0.10

    # --- Mamba-2 backbone (paper: d_model=2048, d_state=512, headdim=128, 4 layers) ---
    d_model: int = 2048
    n_mamba_layers: int = 4
    mamba_d_state: int = 512
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 128

    # --- Inference temporal aggregation (off by default) ---
    temporal_ensemble_coeff: float | None = None

    # --- Training memory management ---
    # Number of episodes laid out in parallel along the batch dimension.
    # Paper trains with batch_size=1; lerobot's full-episode parallel scan
    # makes batch_episodes>1 cheap when GPU memory allows.
    batch_episodes: int = 2
    # None = full-episode forward (paper-faithful: Mamba parallel scan over
    # the entire episode). Setting an int truncates with cold-start hidden
    # state per sub-sequence — escape hatch only.
    max_seq_len: int | None = None

    # --- Optimizer (paper: AdamW lr=2e-4, wd=5e-4) ---
    optimizer_lr: float = 2e-4
    optimizer_weight_decay: float = 5e-4
    optimizer_lr_dinov2: float = 0.0  # encoder is frozen

    # --- LR schedule (paper: CosineAnnealingLR, T_max=200 epochs, eta_min=0.5e-6) ---
    # We translate to step counts here. `lr_decay_steps` auto-scales when
    # --steps is shorter (see CosineDecayWithWarmupSchedulerConfig).
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 100_000
    lr_min: float = 5e-7

    # --- Regularization ---
    # Per-feature Gaussian noise added to the (already MEAN_STD-normalized)
    # state during training only. Paper applies 0.02 x per-joint-std on raw
    # radians; in normalized units that is 0.02 flat.
    state_noise_std: float = 0.02

    # --- Validation ---
    val_split: float = 0.1
    val_freq: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must not exceed chunk_size ({self.chunk_size})."
            )
        if self.d_model % self.mamba_headdim != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by mamba_headdim ({self.mamba_headdim})."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError(
                "temporal_ensemble_coeff requires n_action_steps=1 (one inference per env step)."
            )
        if self.batch_episodes < 1:
            raise ValueError("batch_episodes must be >= 1.")
        if self.max_seq_len is not None and self.max_seq_len < 1:
            raise ValueError("max_seq_len must be None or >= 1.")
        if self.n_obs_steps != 1:
            raise ValueError("MTIL consumes one frame at a time (n_obs_steps must be 1).")
        if self.d_model % self.cross_modal_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by cross_modal_heads "
                f"({self.cross_modal_heads})."
            )
        if self.d_model % self.cross_cam_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by cross_cam_heads "
                f"({self.cross_cam_heads})."
            )
        if not 0.0 <= self.cross_modal_dropout < 1.0:
            raise ValueError(
                f"cross_modal_dropout must be in [0.0, 1.0); got {self.cross_modal_dropout}."
            )
        if not 0.0 <= self.spatial_dropout < 1.0:
            raise ValueError(
                f"spatial_dropout must be in [0.0, 1.0); got {self.spatial_dropout}."
            )
        if self.state_noise_std < 0.0:
            raise ValueError(
                f"state_noise_std must be >= 0.0; got {self.state_noise_std}."
            )
        if self.dinov2_micro_batch is not None and self.dinov2_micro_batch < 1:
            raise ValueError(
                f"dinov2_micro_batch must be None or >= 1; got {self.dinov2_micro_batch}."
            )
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0.0, 1.0); got {self.val_split}.")
        if self.val_freq < 1:
            raise ValueError(f"val_freq must be >= 1; got {self.val_freq}.")

        try:
            import mamba_ssm  # noqa: F401
            import causal_conv1d  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MTIL requires the `mamba-ssm` and `causal-conv1d` packages "
                "(CUDA + Triton wheels). Install with:\n"
                "    pip install mamba-ssm causal-conv1d\n"
                f"Original import error: {e}"
            ) from e

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("MTIL requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("MTIL requires an 'action' output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.lr_warmup_steps,
            num_decay_steps=self.lr_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.lr_min,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
