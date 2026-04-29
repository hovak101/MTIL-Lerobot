"""MTIL: Mamba Temporal Imitation Learning — LeRobot BYOP plugin."""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Install lerobot before using lerobot_policy_mtil."
    ) from e

from .configuration_mtil import MTILConfig
from .modeling_mtil import MTILPolicy
from .processor_mtil import make_mtil_pre_post_processors

__all__ = [
    "MTILConfig",
    "MTILPolicy",
    "make_mtil_pre_post_processors",
]
