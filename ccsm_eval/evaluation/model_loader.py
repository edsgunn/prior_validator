"""Model loading utilities for CCSM Phase 1 evaluation.

Supports HuggingFace transformers (standard) and vLLM (fast batched inference).
Models are loaded once and reused across all evaluations for a given run.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LoadedModel:
    """Container for a loaded model and tokenizer."""

    def __init__(
        self,
        model_id: str,
        model,
        tokenizer,
        backend: str,
        device: str,
    ):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend   # "transformers" or "vllm"
        self.device = device


def load_model(
    model_name: str,
    hf_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    use_vllm: bool = False,
    gpu_memory_utilization: float = 0.85,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> LoadedModel:
    """Load a pretrained model for inference.

    Args:
        model_name:  Short identifier used in result filenames.
        hf_path:     HuggingFace model ID or local path.
        device:      "cuda", "cpu", or "mps".
        dtype:       "bfloat16", "float16", or "float32".
        use_vllm:    If True, use vLLM for faster batched inference.
        gpu_memory_utilization: vLLM memory fraction (ignored for transformers).
        max_model_len: Override context length (vLLM).

    Returns:
        LoadedModel wrapping the model + tokenizer.
    """
    import torch

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    if use_vllm:
        return _load_vllm(
            model_name, hf_path, gpu_memory_utilization, max_model_len,
            tensor_parallel_size, pipeline_parallel_size,
        )
    else:
        return _load_transformers(model_name, hf_path, device, torch_dtype)


def _load_transformers(
    model_name: str, hf_path: str, device: str, torch_dtype
) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info(f"Loading {hf_path} with transformers on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if device not in ("cpu", "mps") else device
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Loaded {hf_path}")
    return LoadedModel(
        model_id=model_name,
        model=model,
        tokenizer=tokenizer,
        backend="transformers",
        device=device,
    )


def _load_vllm(
    model_name: str,
    hf_path: str,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> LoadedModel:
    from vllm import LLM
    from transformers import AutoTokenizer

    tp = tensor_parallel_size
    pp = pipeline_parallel_size
    logger.info(f"Loading {hf_path} with vLLM (tp={tp}, pp={pp}) ...")
    kwargs = dict(
        model=hf_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if tp * pp > 1:
        kwargs["distributed_executor_backend"] = "ray"

    llm = LLM(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loaded {hf_path} with vLLM")
    return LoadedModel(
        model_id=model_name,
        model=llm,
        tokenizer=tokenizer,
        backend="vllm",
        device="cuda",
    )
