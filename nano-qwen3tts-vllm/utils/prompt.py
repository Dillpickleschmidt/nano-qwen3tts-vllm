"""Prompt construction for custom-voice TTS.

Kept in sync with Qwen3-TTS qwen_tts/inference/qwen3_tts_model.py generate_custom_voice():
- _build_assistant_text, _build_instruct_text match exactly
- _tokenize_texts uses same processor(text=..., return_tensors="pt", padding=True) and unsqueeze(0) if dim==1
- prepare_custom_voice_prompt returns (input_ids, instruct_ids, speakers, languages) with same semantics
"""
import torch
from transformers.processing_utils import ProcessorMixin
from typing import List, Optional, Union, Any


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]

def _build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

def _build_instruct_text(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"

def _tokenize_texts(texts: List[str], processor: ProcessorMixin, device: torch.device) -> List[torch.Tensor]:
    input_ids = []
    for text in texts:
        input = processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"].to(device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        input_ids.append(input_id)
    return input_ids

def prepare_custom_voice_prompt(
    text: Union[str, List[str]], 
    speaker: Union[str, List[str]],
    language: Union[str, List[str]],
    instruct: Optional[Union[str, List[str]]] = None,
    non_streaming_mode: bool = True,
    processor: ProcessorMixin = None,
    model_size: str = "0b6",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    texts = _ensure_list(text)
    languages = _ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
    speakers = _ensure_list(speaker)
    if model_size in "0b6": # for 0b6 model, instruct is not supported
        instruct = None
    instructs = _ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

    if len(languages) == 1 and len(texts) > 1:
        languages = languages * len(texts)
    if len(speakers) == 1 and len(texts) > 1:
        speakers = speakers * len(texts)
    if len(instructs) == 1 and len(texts) > 1:
        instructs = instructs * len(texts)

    if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
        raise ValueError(
            f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )
        
    input_ids = _tokenize_texts([_build_assistant_text(t) for t in texts], processor, device)
    

    instruct_ids: List[Optional[torch.Tensor]] = []
    for ins in instructs:
        if ins is None or ins == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(_tokenize_texts([_build_instruct_text(ins)], processor, device)[0])
    
    return input_ids, instruct_ids, speakers, languages
    
    