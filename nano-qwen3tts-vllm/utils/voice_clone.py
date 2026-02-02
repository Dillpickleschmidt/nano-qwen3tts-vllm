"""Voice clone helpers - load and use pre-created voice prompts."""

from pathlib import Path
from typing import Any

import torch


def load_voice_prompt(prompt_path: Path, device: str = "cuda") -> dict[str, Any]:
    """Load voice prompt from .pt file to dict format for generation.py.

    Args:
        prompt_path: Path to the .pt file containing the voice prompt
        device: Target device for tensors

    Returns:
        Dictionary with keys: ref_code, ref_spk_embedding, x_vector_only_mode,
        icl_mode, ref_text - format expected by prepare_inputs()
    """
    prompt_data = torch.load(prompt_path, weights_only=False)
    items = prompt_data["items"]

    def get_field(item, key, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def to_device(val):
        if val is None:
            return None
        if hasattr(val, "to"):
            return val.to(device)
        return val

    return dict(
        ref_code=[to_device(get_field(item, "ref_code")) for item in items],
        ref_spk_embedding=[to_device(get_field(item, "ref_spk_embedding")) for item in items],
        x_vector_only_mode=[get_field(item, "x_vector_only_mode", False) for item in items],
        icl_mode=[get_field(item, "icl_mode", True) for item in items],
        ref_text=[get_field(item, "ref_text") for item in items],
    )


def prepare_speaker_embeds(
    voice_clone_prompt: dict,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Move speaker embeddings to correct device/dtype.

    Args:
        voice_clone_prompt: Dict from load_voice_prompt()
        device: Target device
        dtype: Target dtype

    Returns:
        List of speaker embedding tensors
    """
    return [
        emb.to(device).to(dtype)
        for emb in voice_clone_prompt["ref_spk_embedding"]
    ]
