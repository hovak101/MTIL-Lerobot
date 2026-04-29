"""MTIL policy configuration."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("mtil")
@dataclass
class MTILConfig(PreTrainedConfig):
    """Configuration for the Mamba Temporal Imitation Learning (MTIL) policy.

    MTIL encodes the entire trajectory history into a Mamba-2 recurrent hidden state
    and predicts an action chunk conditioned on `(o_t, h_t)`. Training requires
    sequential, in-order, full-episode trajectories — supplied by `train_mtil.py`'s
    custom sampler. The standard `lerobot-train` script's IID sampling is unsuitable.
    """

    # --- Action chunking ---
    n_obs_steps: int = 1
    chunk_size: int = 50
    # Default replans every 20 env steps (chunk_size still 50, so the head
    # always supervises a 50-step horizon).
    n_action_steps: int = 20

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # DINOv2 has its own normalization
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- Encoder ---
    dinov2_variant: str = "facebook/dinov2-base"
    freeze_dinov2: bool = True
    dinov2_image_size: int = 224

    # --- Cross-modal fusion (paper Fig. 1) ---
    # State token cross-attends to per-camera DINOv2 [CLS] tokens, then FFN.
    # Output is a single d_model token per timestep, fed into the Mamba stack.
    cross_attn_heads: int = 8

    # --- Mamba-2 backbone ---
    d_model: int = 512
    # Tuned for full-episode parallel scan on a single 24 GB GPU. Per-layer
    # scan activation memory scales linearly with depth, so 4 layers fit
    # comfortably while 6 was borderline on long-episode datasets.
    n_mamba_layers: int = 4
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64

    # --- Inference temporal aggregation ---
    # If set, n_action_steps must equal 1 (matches ACT contract).
    temporal_ensemble_coeff: float | None = None

    # --- Training memory management ---
    # Number of episodes laid out in parallel along the batch dimension.
    # Default 2 leaves head-room for unusually long episodes on a 24 GB GPU;
    # raise on bigger hardware.
    batch_episodes: int = 2
    # None (default, RECOMMENDED) = full-episode forward: each batch covers
    # entire episodes (padded to the longest in the group of `batch_episodes`).
    # Mamba-2's parallel scan carries hidden state through the whole episode in
    # a single forward — this is the paper's algorithm.
    #
    # Setting `max_seq_len` is a MEMORY ESCAPE HATCH WITH DEGRADED FIDELITY:
    # Mamba-2 has no public initial-state forward API, so each truncated
    # sub-sequence cold-starts from zero hidden state during training. The
    # paper's full-history claim is no longer honoured. Only use this if a
    # specific episode genuinely will not fit in memory at full length.
    max_seq_len: int | None = None

    # --- Optimizer ---
    optimizer_lr: float = 1e-4
    # Bumped from AdamW's standard 1e-4 to 1e-2 because real-world IL datasets
    # at the user's scale (50–100 episodes) overfit aggressively. Lower this if
    # you have substantially more demonstrations.
    optimizer_weight_decay: float = 1e-2
    # Kept separate so the encoder can be fine-tuned at a smaller LR if unfrozen.
    optimizer_lr_dinov2: float = 0.0

    # --- Regularization ---
    # Single dropout knob driving three sites: after cross-attn out-proj,
    # after the cross-modal FFN, and after each Mamba-2 residual addition.
    # Set to 0.0 for strict paper parity.
    dropout: float = 0.1
    # Gaussian noise stdev added to cached DINOv2 features at training time
    # only. Cheap "augmentation" that doesn't invalidate the cache. 0.0 disables;
    # try 0.02 for SO-101-scale data.
    dino_feature_noise_std: float = 0.0

    # --- Validation ---
    # Fraction of *episodes* (not frames) held out for val-loss tracking.
    # Episode-level — frame-level would leak in-episode history into val.
    val_split: float = 0.1
    # Run a val pass every `val_freq` optimizer steps; the best-val checkpoint
    # is saved separately under `<output_dir>/checkpoints/best/`.
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
        if self.d_model % self.cross_attn_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by cross_attn_heads "
                f"({self.cross_attn_heads})."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0); got {self.dropout}.")
        if self.dino_feature_noise_std < 0.0:
            raise ValueError(
                f"dino_feature_noise_std must be >= 0.0; got {self.dino_feature_noise_std}."
            )
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0.0, 1.0); got {self.val_split}.")
        if self.val_freq < 1:
            raise ValueError(f"val_freq must be >= 1; got {self.val_freq}.")

        # Fail fast if mamba_ssm / causal_conv1d aren't installed. These are
        # CUDA + Triton wheels; the model module imports them lazily, so
        # without this probe the user sees the failure only after the dataset
        # has already loaded.
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

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        # Single frame from the dataset's POV; the sampler stitches frames into
        # (B, T, ...) batches at the DataLoader layer.
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
