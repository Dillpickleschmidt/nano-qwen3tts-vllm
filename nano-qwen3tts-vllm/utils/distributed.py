"""Distributed utilities that handle single-GPU (no dist) gracefully."""

import torch.distributed as dist


def get_world_size() -> int:
    """Get world size, returning 1 if dist is not initialized (single-GPU)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Get rank, returning 0 if dist is not initialized (single-GPU)."""
    return dist.get_rank() if dist.is_initialized() else 0
