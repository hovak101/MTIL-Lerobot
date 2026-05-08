"""Training entry point for the MTIL BYOP plugin.

Why a separate script: ``lerobot.scripts.lerobot_train`` builds the DataLoader
inline with hardcoded ``shuffle=True`` and no hook for a custom sampler /
collate. MTIL needs sequential, in-order, per-episode batches so the Mamba
hidden state evolves correctly. We reuse LeRobot's helpers (config parser,
``make_policy``, ``make_pre_post_processors``, optimizer factory, checkpointing)
and replace only the data pipeline + training loop.

Reference for the modified block: ``lerobot/scripts/lerobot_train.py:367-391``.

Run: ``lerobot-policy-mtil-train --policy.type=mtil --dataset.repo_id=...``

Note: do NOT add ``from __future__ import annotations`` to this module — LeRobot's
``parser.wrap()`` resolves the type hint of ``cfg`` at runtime via
``inspect.getfullargspec``, which would return a string instead of the class.
"""

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, format_big_number, init_logging, inside_slurm

from .configuration_mtil import MTILConfig
from .sampler_mtil import (
    MTILEpisodeBatchSampler,
    MTILEpisodicDataset,
    mtil_collate_fn,
)

logger = logging.getLogger(__name__)


_SAMPLER_STATE_FILENAME = "mtil_sampler_state.json"
_UNSET = object()  # sentinel for "argument not supplied" vs "explicit None"


def _build_mtil_dataloader(
    dataset,
    mtil_cfg: MTILConfig,
    num_workers: int,
    pin_memory: bool,
    feature_cache: torch.Tensor | None = None,
    metadata_cache: dict[str, torch.Tensor] | None = None,
    image_cache: dict[str, torch.Tensor] | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool = True,
    episode_starts: list[int] | None = None,
    episode_ends: list[int] | None = None,
    sampler_seed: int = 0,
    max_seq_len=_UNSET,
):
    image_keys = sorted(mtil_cfg.image_features.keys())
    wrapped = MTILEpisodicDataset(
        dataset,
        feature_cache=feature_cache,
        camera_keys=image_keys,
        metadata_cache=metadata_cache,
        image_cache=image_cache,
    )
    if episode_starts is None or episode_ends is None:
        episode_starts = list(dataset.meta.episodes["dataset_from_index"])
        episode_ends = list(dataset.meta.episodes["dataset_to_index"])
    sampler = MTILEpisodeBatchSampler(
        dataset=wrapped,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        batch_episodes=mtil_cfg.batch_episodes,
        max_seq_len=mtil_cfg.max_seq_len if max_seq_len is _UNSET else max_seq_len,
        seed=sampler_seed,
    )

    def _collate(samples):
        return mtil_collate_fn(samples, batch_episodes=mtil_cfg.batch_episodes)

    dl_kwargs: dict = dict(
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    # PyTorch only honours prefetch_factor when num_workers > 0; passing it in
    # the num_workers=0 case raises ValueError.
    if num_workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    loader = torch.utils.data.DataLoader(wrapped, **dl_kwargs)
    # Keep the sampler reachable from the loader for resume bookkeeping; the
    # underlying object is the same, this is just a convenience handle.
    loader.mtil_sampler = sampler  # type: ignore[attr-defined]
    return loader


def _load_sampler_state(checkpoint_path: Path | str | None) -> dict:
    """Return the saved sampler state dict (epoch + best_val_loss) for the given checkpoint."""
    default = {"epoch": 0, "best_val_loss": float("inf")}
    if checkpoint_path is None:
        return default
    p = Path(checkpoint_path) / _SAMPLER_STATE_FILENAME
    if not p.exists():
        return default
    try:
        loaded = json.loads(p.read_text())
        return {
            "epoch": int(loaded.get("epoch", 0)),
            "best_val_loss": float(loaded.get("best_val_loss", float("inf"))),
        }
    except (ValueError, OSError, json.JSONDecodeError) as e:
        logger.warning(
            "MTIL sampler state at %s unreadable (%s); restarting from epoch 0.", p, e
        )
        return default


def _save_sampler_state(
    checkpoint_dir: Path | str, epoch: int, best_val_loss: float = float("inf")
) -> None:
    p = Path(checkpoint_dir) / _SAMPLER_STATE_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"epoch": int(epoch), "best_val_loss": float(best_val_loss)}))


def _split_episodes_train_val(
    episode_starts: list[int],
    episode_ends: list[int],
    val_split: float,
    seed: int,
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    """Deterministically partition episodes into train/val by episode index.

    Returns ``((train_starts, train_ends), (val_starts, val_ends))``. Episode
    order is shuffled with the given seed before slicing so different seeds
    produce different splits.
    """
    import random

    n_eps = len(episode_starts)
    n_val = max(1, int(round(n_eps * val_split))) if val_split > 0 else 0
    n_val = min(n_val, n_eps - 1)  # always leave at least one train episode
    rng = random.Random(seed)
    order = list(range(n_eps))
    rng.shuffle(order)
    val_idx = sorted(order[:n_val])
    train_idx = sorted(order[n_val:])
    train = (
        [episode_starts[i] for i in train_idx],
        [episode_ends[i] for i in train_idx],
    )
    val = (
        [episode_starts[i] for i in val_idx],
        [episode_ends[i] for i in val_idx],
    )
    return train, val


@torch.no_grad()
def _run_validation(policy, val_loader, preprocessor, accelerator) -> float:
    """One pass over ``val_loader``, returns mean MSE loss (no_grad, eval mode)."""
    if val_loader is None:
        return float("nan")
    was_training = policy.training
    policy.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        # Match the train loop's per-batch image float-cast (cameras may not be
        # present at all in the cache path, but be safe for the live path).
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                batch[k] = v.to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        with accelerator.autocast():
            loss, _ = policy.forward(batch)
        total_loss += loss.detach().float().item()
        n_batches += 1
    if was_training:
        policy.train()
    return total_loss / max(n_batches, 1)


class _FrameImageDataset(torch.utils.data.Dataset):
    """Returns only image-key tensors for a given frame index.

    Used by ``_precompute_dinov2_cache`` to iterate the dataset without paying
    for action-chunk lookups or other per-frame fields we don't need.
    """

    def __init__(self, base_dataset, image_keys: list[str]):
        self._base = base_dataset
        self._keys = image_keys

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        return {k: item[k] for k in self._keys}


def _frames_collate(samples):
    out = {}
    for k in samples[0]:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out


@torch.no_grad()
def _precompute_dinov2_cache(
    policy,
    base_dataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    """Run frozen DINOv2 over every frame, return ``(N, n_cams, hidden)`` fp16 CPU tensor.

    Cache is indexed by ``real_idx`` (the same key passed to ``base[real_idx]``).
    Output dtype is fp16 to keep RAM small; the policy upcasts on consumption.
    """
    image_keys = sorted(policy.config.image_features.keys())
    n_cams = len(image_keys)
    num_frames = len(base_dataset)
    hidden = policy._dinov2_hidden

    cache = torch.empty(num_frames, n_cams, hidden, dtype=torch.float16)

    loader = torch.utils.data.DataLoader(
        _FrameImageDataset(base_dataset, image_keys),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_frames_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    was_training = policy.dinov2.training
    policy.dinov2.eval()
    idx = 0
    pbar = tqdm(total=num_frames, desc="Precomputing DINOv2", unit="f")
    for batch in loader:
        N = batch[image_keys[0]].shape[0]
        cam_feats = []
        for k in image_keys:
            img = batch[k].to(device, non_blocking=True)
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0
            normed = policy._normalize_for_dinov2(img)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = policy.dinov2(pixel_values=normed)
            cls = out.last_hidden_state[:, 0]  # (N, hidden)
            cam_feats.append(cls)
        stacked = torch.stack(cam_feats, dim=1)  # (N, n_cams, hidden)
        cache[idx : idx + N] = stacked.to(torch.float16).cpu()
        idx += N
        pbar.update(N)
    pbar.close()
    if was_training:
        policy.dinov2.train()
    return cache


@torch.no_grad()
def _precompute_image_cache(
    base_dataset,
    image_keys: list[str],
    image_size: int,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, torch.Tensor]:
    """Decode every frame once, resize to (S, S), store as uint8 in CPU RAM.

    Returns a dict keyed by camera name with tensors of shape
    ``(num_frames, 3, S, S)`` dtype ``uint8``. Total RAM ~= num_frames *
    n_cams * 3 * S^2 bytes.

    The bottleneck before this cache was libsvtav1 video decode + per-frame
    parquet lookups (see ``_precompute_metadata_cache`` for the parquet half).
    Workers parallelize video decode; the result is read-only and shared
    across DataLoader workers via copy-on-write.
    """
    num_frames = len(base_dataset)
    cache = {
        k: torch.empty(num_frames, 3, image_size, image_size, dtype=torch.uint8)
        for k in image_keys
    }

    loader = torch.utils.data.DataLoader(
        _FrameImageDataset(base_dataset, image_keys),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_frames_collate,
        pin_memory=False,
        persistent_workers=False,
    )

    idx = 0
    pbar = tqdm(total=num_frames, desc="Precomputing images", unit="f")
    for batch in loader:
        first = batch[image_keys[0]]
        N = first.shape[0]
        for k in image_keys:
            img = batch[k]  # float32 in [0,1] (lerobot default) or uint8
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0
            if img.shape[-2:] != (image_size, image_size):
                img = torch.nn.functional.interpolate(
                    img, size=(image_size, image_size), mode="bilinear", align_corners=False
                )
            img_u8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
            cache[k][idx : idx + N] = img_u8
        idx += N
        pbar.update(N)
    pbar.close()
    return cache


@torch.no_grad()
def _precompute_metadata_cache(
    base_dataset,
    *,
    image_keys: list[str],
    feature_cache: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Cache every frame's non-image fields into a dict of stacked tensors.

    With ``action_delta_indices=range(chunk_size)``, LeRobot's dataset reader
    issues ~``chunk_size + 1`` parquet row lookups per frame (the frame itself
    plus one row per future action). At ``B*T`` >> 1 these dominate per-step
    time. Precomputing once at startup turns ``__getitem__`` into pure tensor
    indexing.

    Memory cost is tiny — for a 24K-frame dataset with chunk_size=50 and a
    14-dim action it's ~70 MB. Workers inherit the cache via copy-on-write
    shared memory (fork mode), so per-worker RAM stays flat.
    """
    # Build a single MTILEpisodicDataset wrapper just to reuse its
    # `_get_item_skip_videos` helper. The per-frame call avoids video decode.
    wrapped = MTILEpisodicDataset(
        base_dataset, feature_cache=feature_cache, camera_keys=image_keys
    )
    image_set = set(image_keys)
    num_frames = len(base_dataset)
    accum: dict[str, list[torch.Tensor]] = {}
    pbar = tqdm(total=num_frames, desc="Precomputing metadata", unit="f")
    for real_idx in range(num_frames):
        item = wrapped._get_item_skip_videos(real_idx)
        for k, v in item.items():
            if k in image_set:
                continue  # images come from feature_cache
            if isinstance(v, torch.Tensor):
                accum.setdefault(k, []).append(v)
            # Non-tensor fields (strings) are dropped — collate drops them too.
        pbar.update(1)
    pbar.close()
    return {k: torch.stack(vs, dim=0) for k, vs in accum.items()}


def _mtil_update_step(policy, batch, optimizer, lr_scheduler, accelerator, grad_clip_norm):
    start_time = time.perf_counter()
    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss, grad_norm, output_dict, time.perf_counter() - start_time


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    if not isinstance(cfg.policy, MTILConfig):
        raise ValueError(
            f"train_mtil.py is only valid for --policy.type=mtil; got {type(cfg.policy).__name__}."
        )

    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    cfg.validate()

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # bf16 on CUDA: makes accelerator.autocast() actually cast. Mamba-2 + the
        # action head + DINOv2 (during precompute) all run faster; bf16 has the
        # numerical range to skip loss scaling.
        mixed_precision = "bf16" if cfg.policy.device == "cuda" else "no"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=cfg.policy.device == "cpu",
            mixed_precision=mixed_precision,
        )
    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    if is_main:
        logging.info(pformat(cfg.to_dict()))

    wandb_logger = (
        WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project and is_main) else None
    )
    if wandb_logger is None and is_main:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main:
        dataset = make_dataset(cfg)

    if is_main:
        logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    accelerator.wait_for_everyone()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    # Cache decoded images (uint8 at dinov2_image_size) and per-frame metadata
    # in CPU RAM. DINOv2-Large still runs live on GPU every step — only the
    # video decode + parquet lookups are eliminated. Workers see the caches
    # via copy-on-write (Linux fork mode), so per-worker RAM stays flat.
    feature_cache: torch.Tensor | None = None  # legacy [CLS] cache, unused
    image_cache: dict[str, torch.Tensor] | None = None
    metadata_cache: dict[str, torch.Tensor] | None = None
    if cfg.policy.freeze_dinov2 and not cfg.dataset.image_transforms.enable:
        image_keys = sorted(cfg.policy.image_features.keys())
        if is_main:
            logging.info("Precomputing image cache (decode every frame once into uint8 RAM tensor)")
        image_cache = _precompute_image_cache(
            dataset,
            image_keys=image_keys,
            image_size=cfg.policy.dinov2_image_size,
            batch_size=64,
            num_workers=max(cfg.num_workers, 2),
        )
        if is_main:
            total_gb = sum(t.numel() * t.element_size() for t in image_cache.values()) / 1e9
            shapes = {k: tuple(v.shape) for k, v in image_cache.items()}
            logging.info(f"Image cache ready: {shapes} uint8 ({total_gb:.2f} GB)")

        if is_main:
            logging.info("Precomputing per-frame metadata cache (state, action chunks, ...)")
        metadata_cache = _precompute_metadata_cache(
            dataset, image_keys=image_keys, feature_cache=None
        )
        if is_main:
            total_mb = sum(t.numel() * t.element_size() for t in metadata_cache.values()) / 1e6
            shapes = {k: tuple(v.shape) for k, v in metadata_cache.items()}
            logging.info(f"Metadata cache ready: {shapes} ({total_mb:.1f} MB)")
    elif is_main:
        logging.warning(
            colored(
                "Image and metadata caches disabled "
                f"(freeze_dinov2={cfg.policy.freeze_dinov2}, "
                f"image_transforms.enable={cfg.dataset.image_transforms.enable}). "
                "Each step will run video decode + parquet lookups for every frame "
                "— expect heavy CPU/disk load and low GPU utilization.",
                "yellow",
                attrs=["bold"],
            )
        )

    if is_main:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    sampler_resume_epoch = 0
    best_val_loss = float("inf")
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        resume_state = _load_sampler_state(cfg.checkpoint_path)
        sampler_resume_epoch = resume_state["epoch"]
        best_val_loss = resume_state["best_val_loss"]
        if is_main:
            if sampler_resume_epoch > 0:
                logging.info(f"Restored MTIL sampler epoch {sampler_resume_epoch}.")
            if best_val_loss < float("inf"):
                logging.info(f"Restored best_val_loss = {best_val_loss:.4f}.")

    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    # Estimate the longest episode for memory-pressure logging.
    ep_starts = list(dataset.meta.episodes["dataset_from_index"])
    ep_ends = list(dataset.meta.episodes["dataset_to_index"])
    longest_ep = max((e - s for s, e in zip(ep_starts, ep_ends)), default=0)
    if is_main:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable=} ({format_big_number(num_learnable)})")
        logging.info(f"{num_total=} ({format_big_number(num_total)})")
        logging.info(
            f"MTIL batch shape: B={cfg.policy.batch_episodes}, "
            f"max_seq_len={cfg.policy.max_seq_len}, chunk_size={cfg.policy.chunk_size}, "
            f"longest_episode={longest_ep} frames"
        )
        # Rough activation-memory estimate per Mamba-2 layer at peak sub-seq:
        # B * T * d_inner * d_state * 2 bytes (bf16). Used only as a UX hint.
        d_inner = cfg.policy.mamba_expand * cfg.policy.d_model
        peak_T = cfg.policy.max_seq_len if cfg.policy.max_seq_len is not None else longest_ep
        per_layer_bytes = (
            cfg.policy.batch_episodes * peak_T * d_inner * cfg.policy.mamba_d_state * 2
        )
        per_layer_mb = per_layer_bytes / (1024 * 1024)
        total_mb = per_layer_mb * cfg.policy.n_mamba_layers
        logging.info(
            f"Approx Mamba scan activation memory: {per_layer_mb:.0f} MB/layer × "
            f"{cfg.policy.n_mamba_layers} layers ≈ {total_mb:.0f} MB"
        )
        if longest_ep > 2000 and cfg.policy.max_seq_len is None:
            logging.warning(
                colored(
                    f"Longest episode is {longest_ep} frames — full-episode parallel "
                    "scan may exhaust GPU memory. If you see CUDA OOM, lower "
                    "batch_episodes, n_mamba_layers, or set max_seq_len.",
                    "yellow",
                    attrs=["bold"],
                )
            )
        if cfg.policy.max_seq_len is not None:
            logging.warning(
                colored(
                    f"max_seq_len={cfg.policy.max_seq_len} is set. This is a memory "
                    "escape hatch with DEGRADED FIDELITY: each truncated sub-sequence "
                    "cold-starts from zero Mamba state, breaking the paper's full-history "
                    "claim. Prefer max_seq_len=None when the longest episode fits.",
                    "yellow",
                    attrs=["bold"],
                )
            )

    # Episode-level train/val split. Episode-level (not frame-level) avoids
    # leaking in-episode history into validation. Same `cfg.seed` ⇒ same split.
    (train_starts, train_ends), (val_starts, val_ends) = _split_episodes_train_val(
        episode_starts=ep_starts,
        episode_ends=ep_ends,
        val_split=cfg.policy.val_split,
        seed=cfg.seed if cfg.seed is not None else 0,
    )
    if is_main:
        logging.info(
            f"Episode split: train={len(train_starts)}, val={len(val_starts)} "
            f"(val_split={cfg.policy.val_split}, seed={cfg.seed})"
        )
        if dataset.num_episodes < 75:
            logging.warning(
                colored(
                    f"Dataset has only {dataset.num_episodes} episodes — at this scale "
                    "MTIL will overfit train loss quickly. Watch val_loss (logged every "
                    f"val_freq={cfg.policy.val_freq} steps), not train loss. The MTIL "
                    "paper used 100 demos per real-world task; consider recording more.",
                    "yellow",
                    attrs=["bold"],
                )
            )
        if 0 < len(val_starts) < 5:
            logging.warning(
                colored(
                    f"Only {len(val_starts)} validation episode(s); val_loss will be "
                    "high-variance. Lower val_split or collect more data.",
                    "yellow",
                    attrs=["bold"],
                )
            )

    dataloader = _build_mtil_dataloader(
        dataset,
        cfg.policy,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        feature_cache=feature_cache,
        metadata_cache=metadata_cache,
        image_cache=image_cache,
        prefetch_factor=getattr(cfg, "prefetch_factor", None),
        persistent_workers=getattr(cfg, "persistent_workers", True),
        episode_starts=train_starts,
        episode_ends=train_ends,
        sampler_seed=0,
    )
    # Apply restored sampler epoch (zero on a fresh run). Keep a reference to
    # the unwrapped sampler for saving its state at checkpoint time.
    mtil_sampler = dataloader.mtil_sampler  # type: ignore[attr-defined]
    mtil_sampler._epoch = sampler_resume_epoch

    # Val loader: shares the underlying caches but runs over held-out episodes.
    # num_workers=0 because val passes are short and worker churn outweighs
    # parallelism. max_seq_len=None forces full-episode val for paper fidelity.
    val_loader = None
    if len(val_starts) > 0:
        val_loader = _build_mtil_dataloader(
            dataset,
            cfg.policy,
            num_workers=0,
            pin_memory=device.type == "cuda",
            feature_cache=feature_cache,
            metadata_cache=metadata_cache,
            image_cache=image_cache,
            prefetch_factor=None,
            persistent_workers=False,
            episode_starts=val_starts,
            episode_ends=val_ends,
            sampler_seed=1,
            max_seq_len=None,
        )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main:
        progbar = tqdm(
            total=cfg.steps - step, desc="MTIL training", unit="step", disable=inside_slurm()
        )
    else:
        progbar = nullcontext()

    # Buffer loss/grad_norm tensors on-device between log steps to avoid one CUDA
    # sync per training step. We sync the whole buffer in one shot at log time.
    defer_metrics = cfg.log_freq > 0 and is_main
    pending_loss: list[torch.Tensor] = []
    pending_grad: list[torch.Tensor] = []

    with progbar:
        for _ in range(step, cfg.steps):
            t0 = time.perf_counter()
            batch = next(dl_iter)
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - t0

            loss, grad_norm, output_dict, update_s = _mtil_update_step(
                policy, batch, optimizer, lr_scheduler, accelerator, cfg.optimizer.grad_clip_norm
            )
            if defer_metrics:
                pending_loss.append(loss.detach())
                pending_grad.append(grad_norm.detach())
            elif is_main:
                # Eager path (log_freq <= 0): keep legacy behaviour.
                train_tracker.loss = loss.item()
                train_tracker.grad_norm = grad_norm.item()
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.update_s = update_s

            step += 1
            if is_main:
                progbar.update(1)
            train_tracker.step()

            if defer_metrics and step % cfg.log_freq == 0:
                # Single sync for the whole window of pending values.
                losses = torch.stack(pending_loss).float().cpu().tolist()
                grads = torch.stack(pending_grad).float().cpu().tolist()
                pending_loss.clear()
                pending_grad.clear()
                for lv, gv in zip(losses, grads):
                    train_tracker.loss = lv
                    train_tracker.grad_norm = gv
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log = train_tracker.to_dict()
                    if output_dict:
                        wandb_log.update(
                            {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in output_dict.items()}
                        )
                    wandb_logger.log_dict(wandb_log, step)
                train_tracker.reset_averages()

            # Periodic validation pass + best-checkpoint save.
            if (
                val_loader is not None
                and step % cfg.policy.val_freq == 0
                and is_main
            ):
                val_loss = _run_validation(
                    accelerator.unwrap_model(policy),
                    val_loader,
                    preprocessor,
                    accelerator,
                )
                logging.info(f"step:{step}  val_loss:{val_loss:.4f}  best:{best_val_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"val_loss": val_loss}, step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if cfg.save_checkpoint:
                        best_dir = Path(cfg.output_dir) / "checkpoints" / "best"
                        save_checkpoint(
                            checkpoint_dir=best_dir,
                            step=step,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                        _save_sampler_state(best_dir, mtil_sampler._epoch, best_val_loss)
                        logging.info(
                            f"New best val_loss={val_loss:.4f}; saved to {best_dir}"
                        )
                    # Auto-push the new best to HF Hub so a remote rollout PC
                    # always pulls the strongest snapshot. Gated on the same
                    # flag stock lerobot_train.py uses; non-fatal on failure.
                    if cfg.policy.push_to_hub:
                        try:
                            accelerator.unwrap_model(policy).push_model_to_hub(cfg)
                            preprocessor.push_to_hub(cfg.policy.repo_id)
                            postprocessor.push_to_hub(cfg.policy.repo_id)
                            logging.info(f"Pushed new best to {cfg.policy.repo_id}.")
                        except Exception as e:
                            logging.warning(f"Best-checkpoint HF push failed: {e}")

            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                if is_main:
                    logging.info(f"Checkpoint at step {step}")
                    ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(
                        checkpoint_dir=ckpt_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    # Persist the sampler's epoch and best_val_loss alongside
                    # the checkpoint so resumes continue the deterministic
                    # episode order and don't lose the best-so-far record.
                    _save_sampler_state(ckpt_dir, mtil_sampler._epoch, best_val_loss)
                    update_last_checkpoint(ckpt_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(ckpt_dir)
                accelerator.wait_for_everyone()

    if is_main:
        logging.info("End of MTIL training")
        # Final HF Hub push, mirrors lerobot_train.py:561-568. Pushes the
        # current (final-step) policy and processors to cfg.policy.repo_id.
        # Gated on the existing push_to_hub flag; non-fatal on failure so a
        # successful training run is never lost to a network blip.
        if cfg.policy.push_to_hub:
            try:
                unwrapped = accelerator.unwrap_model(policy)
                unwrapped.push_model_to_hub(cfg)
                preprocessor.push_to_hub(cfg.policy.repo_id)
                postprocessor.push_to_hub(cfg.policy.repo_id)
                logging.info(f"Pushed final policy to {cfg.policy.repo_id}.")
            except Exception as e:
                logging.warning(f"End-of-training HF push failed: {e}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
