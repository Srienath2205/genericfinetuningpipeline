"""
Generic model loader.
Supports:
- Kaggle Models via `kagglehub`
- Hugging Face model ids
Handles 4-bit/8-bit/none quantization for Transformers models with PEFT.
"""

from typing import Tuple, Optional
import os

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = object  # type: ignore[misc]
    torch = None  # type: ignore


def _maybe_quant_config(quantization: str) -> Optional["BitsAndBytesConfig"]:
    try:
        if quantization == "4bit":
            # Use proper torch dtype (not a string)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, "bfloat16", None),
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        return None
    return None


def load_model_and_tokenizer(
    model_source: str,
    model_name: str,
    quantization: str = "4bit",
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    for_training: bool = False,
) -> Tuple["AutoTokenizer", "AutoModelForCausalLM", str]:
    """
    Returns (tokenizer, model, model_path)
    """
    if model_source == "kaggle":
        try:
            import kagglehub  # installed in Colab first cell
        except Exception as e:
            raise RuntimeError(
                "kagglehub is not available in this environment. "
                "Run in Colab after installing deps in the first cell."
            ) from e
        model_path = kagglehub.model_download(model_name)
    elif model_source == "hf":
        model_path = model_name
    else:
        raise ValueError(f"Unknown model_source: {model_source}")

    quant_cfg = _maybe_quant_config(quantization)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if quant_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=quant_cfg,
        )
    else:
        dtype = getattr(torch, torch_dtype, None) if torch is not None else None
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
        )

    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Optional: prep for k-bit training (QLoRA/LoRA with quantized base)
    if for_training and quant_cfg is not None:
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        except Exception:
            # Non-fatal: training will still run, but QLoRA may be less stable
            pass

    return tok, model, model_path
