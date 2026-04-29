"""Episode-aware time-major batching for MTIL."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

# Per-frame keys we attach in MTILEpisodicDataset to flow metadata to the collate.
_PAD_KEY = "_frame_is_pad"
_FIRST_KEY = "_is_first_subseq"

# Key used to flow precomputed DINOv2 [CLS] features from the dataset wrapper
# through the collate fn into the policy. Shape per-frame: (n_cams, hidden_dim).
# Must start with "observation." or `batch_to_transition` will drop it during
# the preprocessor's round-trip through `EnvTransition`.
DINO_FEATURES_KEY = "observation.dino_cls_features"


class MTILEpisodicDataset(Dataset):
    """Wraps a LeRobotDataset; lookups go through a per-epoch ``plan``.

    ``plan[i] = (real_idx, is_pad, is_first_subseq)`` — the sampler decides per
    logical index whether the underlying frame is padding (replicated) and
    whether this slot lies at the start of an episode's first sub-sequence.

    Optional ``feature_cache`` carries precomputed DINOv2 [CLS] features keyed
    by the absolute frame index. When provided, ``__getitem__`` skips video
    decoding entirely and emits cached features under ``DINO_FEATURES_KEY``.
    """

    def __init__(
        self,
        base_dataset,
        feature_cache: torch.Tensor | None = None,
        camera_keys: list[str] | None = None,
        metadata_cache: dict[str, torch.Tensor] | None = None,
    ):
        self.base = base_dataset
        self._plan: list[tuple[int, bool, bool]] = []
        self._feature_cache = feature_cache
        # Keys we need to drop / replace with cached features in the cache path.
        self._camera_keys = camera_keys or []
        # Stacked-tensor cache of every per-frame non-image field (state,
        # chunked action, action_is_pad, episode_index, etc.). When set, the
        # __getitem__ fast path becomes pure tensor indexing — no parquet I/O.
        self._metadata_cache = metadata_cache
        # On the first cached lookup, we compare the (video-skipped) key set
        # against a live full-decode call so a future LeRobot DatasetReader API
        # change fails loudly instead of silently dropping fields.
        self._cache_keys_validated: bool = False

    def set_plan(self, plan: list[tuple[int, bool, bool]]) -> None:
        self._plan = plan

    def __len__(self) -> int:
        return max(len(self._plan), len(self.base))

    def __getitem__(self, key) -> dict[str, Any]:
        # Worker-safe path: the sampler yields plan tuples directly as keys, so
        # `__getitem__` does not depend on shared `_plan` state. Falls back to
        # the legacy int-index lookup if a plain int comes in.
        if isinstance(key, tuple):
            real_idx, is_pad, is_first = key
        else:
            real_idx, is_pad, is_first = self._plan[key]
        if self._metadata_cache is not None:
            # Fast path: O(1) slicing into pre-stacked tensors. No parquet I/O,
            # no video decode. Used when both DINOv2 features AND metadata
            # (state, action chunks, action_is_pad, ...) are precomputed.
            item = {k: v[real_idx] for k, v in self._metadata_cache.items()}
            if self._feature_cache is not None:
                item[DINO_FEATURES_KEY] = self._feature_cache[real_idx]
        elif self._feature_cache is not None:
            item = self._get_item_skip_videos(real_idx)
            item[DINO_FEATURES_KEY] = self._feature_cache[real_idx]
        else:
            item = self.base[real_idx]
        item[_PAD_KEY] = torch.tensor(is_pad, dtype=torch.bool)
        item[_FIRST_KEY] = torch.tensor(is_first, dtype=torch.bool)
        return item

    def _get_item_skip_videos(self, real_idx: int) -> dict[str, Any]:
        """Lookup that omits video decode (cache path replaces raw images).

        Mirrors the tensor portion of ``DatasetReader.get_item`` (see
        ``lerobot/datasets/dataset_reader.py``) but skips ``_query_videos`` —
        the cache supplies pre-extracted [CLS] tokens, so decoding video frames
        is wasted work.

        Drops the ``task``/``subtask`` string enrichment that the upstream
        method adds: ``mtil_collate_fn`` discards string-valued keys anyway, so
        producing them is dead work.

        On the first call, compares the produced key set with a live full
        ``base[real_idx]`` (minus camera tensors) and logs a loud warning if
        they diverge — early signal that a LeRobot upgrade may have shifted
        the reader's contract.
        """
        reader = self.base._ensure_reader()
        if reader.hf_dataset is None:
            reader.load_and_activate()

        item = reader.hf_dataset[real_idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        if reader.delta_indices is not None:
            query_indices, padding = reader._get_query_indices(abs_idx, ep_idx)
            query_result = reader._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for k, v in query_result.items():
                item[k] = v

        if not self._cache_keys_validated:
            self._cache_keys_validated = True
            self._validate_cache_keys(item, real_idx)
        return item

    def _validate_cache_keys(self, fast_item: dict[str, Any], real_idx: int) -> None:
        """One-shot guard: warn if the fast path's key set diverges from live."""
        try:
            live = self.base[real_idx]
        except Exception as e:  # pragma: no cover — best-effort diagnostic
            import logging

            logging.getLogger(__name__).warning(
                "MTIL cache-path key validation skipped (live lookup raised %s).", e
            )
            return
        # Camera image keys are intentionally absent in the fast path; ignore
        # them. Also ignore string-valued keys that the collate drops anyway.
        ignore = set(self._camera_keys) | {"task", "subtask"}
        fast_keys = set(fast_item) - ignore
        live_keys = set(live) - ignore
        missing = live_keys - fast_keys
        if missing:
            import logging

            logging.getLogger(__name__).warning(
                "MTIL cache path is missing keys produced by the live LeRobot "
                "DatasetReader: %s. This usually means LeRobot's reader API has "
                "changed since this plugin was written; verify the cache path "
                "still produces what MTILPolicy.forward expects.",
                sorted(missing),
            )


class MTILEpisodeBatchSampler(Sampler[list[int]]):
    """Yields B*T logical indices per batch in time-major layout: ``samples[t*B + b]``.

    A new plan is built each ``__iter__`` (each epoch). ``B`` is fixed to
    ``batch_episodes``; if fewer episodes remain in the final group, missing
    slots are filled with padded data so every batch has the same shape.
    """

    def __init__(
        self,
        dataset: MTILEpisodicDataset,
        episode_starts: list[int],
        episode_ends: list[int],
        batch_episodes: int,
        max_seq_len: int | None,
        seed: int = 0,
    ):
        if batch_episodes < 1:
            raise ValueError("batch_episodes must be >= 1")
        if len(episode_starts) != len(episode_ends):
            raise ValueError("episode_starts and episode_ends length mismatch")
        self.dataset = dataset
        self.episode_starts = list(episode_starts)
        self.episode_ends = list(episode_ends)
        self.B = batch_episodes
        self.T_max = max_seq_len
        self.seed = seed
        self._epoch = 0
        # Cache __len__ — computed deterministically from episode lengths.
        self._cached_len: int | None = None

    def _split_episode(self, ep_idx: int) -> list[list[int]]:
        """Return the list of sub-sequence index-lists for one episode."""
        s, e = self.episode_starts[ep_idx], self.episode_ends[ep_idx]
        frames = list(range(s, e))
        if not frames:
            return []
        if self.T_max is None:
            return [frames]
        return [frames[i : i + self.T_max] for i in range(0, len(frames), self.T_max)]

    def _build_plan(self, rng: random.Random) -> tuple[list[tuple[int, bool, bool]], list[list[int]]]:
        """Returns (plan, batch_logical_lists). Side-effect-free."""
        ep_order = list(range(len(self.episode_starts)))
        rng.shuffle(ep_order)

        plan: list[tuple[int, bool, bool]] = []
        batches: list[list[int]] = []

        # Iterate over groups of B episodes.
        for g0 in range(0, len(ep_order), self.B):
            group = ep_order[g0 : g0 + self.B]
            ep_segments = [self._split_episode(ep) for ep in group]
            # Pad group up to B with empty episodes (entirely padded slots).
            while len(ep_segments) < self.B:
                ep_segments.append([])

            n_rounds = max((len(segs) for segs in ep_segments), default=0)
            for sub_idx in range(n_rounds):
                # Per-slot sub-sequence frame list (or [] if exhausted).
                sub_per_b: list[list[int]] = [
                    segs[sub_idx] if sub_idx < len(segs) else [] for segs in ep_segments
                ]
                T = max((len(s) for s in sub_per_b), default=0)
                if T == 0:
                    continue

                # Pick a fallback real index for entirely-empty slots — must be a
                # valid frame in the dataset; use the first frame of the first
                # non-empty episode in this group.
                fallback = next((s[0] for s in sub_per_b if s), None)
                if fallback is None:
                    continue

                base = len(plan)
                for t in range(T):
                    for b in range(self.B):
                        sub = sub_per_b[b]
                        if not sub:
                            real_idx, is_pad = fallback, True
                            is_first = False
                        elif t < len(sub):
                            real_idx, is_pad = sub[t], False
                            is_first = (sub_idx == 0) and (t == 0)
                        else:
                            real_idx, is_pad = sub[-1], True
                            is_first = False
                        plan.append((real_idx, is_pad, is_first))
                batches.append(list(range(base, base + T * self.B)))

        return plan, batches

    def __iter__(self) -> Iterator[list[tuple[int, bool, bool]]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        plan, batches = self._build_plan(rng)
        # Keep `set_plan` for the legacy int-index path; with workers >= 1 the
        # workers won't see this anyway, which is fine because we now yield
        # plan tuples directly.
        self.dataset.set_plan(plan)
        for batch in batches:
            yield [plan[i] for i in batch]

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        # Same logic as _build_plan but only counting batches.
        n_groups = (len(self.episode_starts) + self.B - 1) // self.B
        total = 0
        for g0 in range(0, len(self.episode_starts), self.B):
            group = list(range(g0, min(g0 + self.B, len(self.episode_starts))))
            n_rounds_per_ep = []
            for ep in group:
                length = self.episode_ends[ep] - self.episode_starts[ep]
                if length == 0:
                    n_rounds_per_ep.append(0)
                elif self.T_max is None:
                    n_rounds_per_ep.append(1)
                else:
                    n_rounds_per_ep.append((length + self.T_max - 1) // self.T_max)
            total += max(n_rounds_per_ep, default=0)
        # Tiny correctness check
        assert n_groups >= 0
        self._cached_len = total
        return total


def mtil_collate_fn(
    samples: list[dict[str, Any]], *, batch_episodes: int
) -> dict[str, Any]:
    """Collate a flat list of B*T per-frame samples into ``(B, T, ...)`` tensors.

    Layout in ``samples`` is time-major: ``samples[t * B + b]`` is slot ``b`` at
    time ``t``. Tensors are reshaped to ``(B, T, ...)`` for direct consumption by
    ``MTILPolicy.forward``.

    Emits two extra keys derived from per-frame markers attached by the dataset
    wrapper:
      - ``frame_is_pad``: ``(B, T)`` bool — True at sampler-padded positions.
      - ``is_first_subseq``: ``(B,)`` bool — True iff slot ``b``'s sub-sequence
        is the first of its episode (signal to reset the carried Mamba state).
    """
    B = batch_episodes
    BT = len(samples)
    if BT % B != 0:
        raise ValueError(f"len(samples)={BT} not divisible by batch_episodes={B}")
    T = BT // B

    # Drop string-valued per-frame metadata that doesn't tensor-collate cleanly.
    drop_keys = {"task", "subtask"}

    # Reorder time-major (t*B + b) → (b, t) by reshaping.
    out: dict[str, Any] = {}
    keys = [k for k in samples[0] if k not in drop_keys]
    for key in keys:
        first = samples[0][key]
        if not isinstance(first, torch.Tensor):
            # Skip non-tensor fields (e.g., scalars without batch broadcast). The
            # policy doesn't need them.
            continue
        # Stack as (BT, ...) in original time-major order, then reshape.
        stacked = torch.stack([s[key] for s in samples], dim=0)  # (BT, ...)
        rest = stacked.shape[1:]
        # (T, B, *rest) → (B, T, *rest)
        out[key] = stacked.view(T, B, *rest).transpose(0, 1).contiguous()

    # Derived fields from per-frame markers.
    pad_bt = out.pop(_PAD_KEY)  # (B, T) bool
    first_bt = out.pop(_FIRST_KEY)  # (B, T) bool
    out["frame_is_pad"] = pad_bt
    out["is_first_subseq"] = first_bt[:, 0].contiguous()  # only t=0 matters per episode
    return out
