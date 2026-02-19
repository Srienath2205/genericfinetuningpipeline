"""
Generic model loader.
Supports:
- Kaggle Models via `kagglehub` (with CLI fallback in Colab)
- Hugging Face model ids
- Local filesystem paths
Handles 4-bit/8-bit/none quantization for Transformers models with PEFT.
"""

from typing import Tuple, Optional
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Optional imports (handled gracefully if unavailable)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = object  # type: ignore[misc]
    torch = None  # type: ignore


# ----------------------------
# Helpers
# ----------------------------
def _preferred_compute_dtype():
    """Prefer bfloat16 if present, otherwise fallback to float16."""
    if torch is None:
        return None
    return getattr(torch, "bfloat16", None) or getattr(torch, "float16", None)


def _to_torch_dtype(name: str):
    """Map a string like 'bfloat16' to a torch dtype, if torch is available."""
    if torch is None or not isinstance(name, str):
        return None
    return getattr(torch, name, None)


def _maybe_quant_config(quantization: str) -> Optional["BitsAndBytesConfig"]:
    try:
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_preferred_compute_dtype(),
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        return None
    return None


def _parse_kaggle_model_handle(handle: str):
    """
    Parse a Kaggle Models handle like:
      'google/gemma-3/pyTorch/gemma-3-1b-it'
    Returns (owner, model, framework, variant)
    """
    parts = handle.strip("/").split("/")
    if len(parts) < 4:
        raise ValueError(
            f"Kaggle model handle must look like 'owner/name/framework/variant', got: {handle}"
        )
    owner, name, framework, variant = parts[0], parts[1], parts[2], "/".join(parts[3:])
    framework = framework.lower()  # CLI expects lowercase (e.g., 'pytorch')
    return owner, name, framework, variant


def _ensure_kaggle_cli():
    """
    Ensure Kaggle CLI is installed & available. Tries to pip install if missing.
    """
    from shutil import which
    if which("kaggle") is not None:
        return True
    # Try to install
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "kaggle"])
        return which("kaggle") is not None
    except Exception:
        return False


def _download_with_kaggle_cli(handle: str, out_dir: str) -> str:
    """
    Download a Kaggle Model with the Kaggle CLI and unzip it locally.
    Returns the directory path containing model files.
    """
    ok = _ensure_kaggle_cli()
    if not ok:
        raise RuntimeError(
            "Kaggle CLI is not available and auto-install failed. "
            "Try: pip install kaggle  (and verify credentials)."
        )

    owner, name, framework, variant = _parse_kaggle_model_handle(handle)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CLI command
    cmd = [
        "kaggle", "models", "download",
        "-m", f"{owner}/{name}",
        "--framework", framework,
        "--variant", variant,
        "-p", str(out),
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Kaggle CLI download failed for {handle}. "
            f"Make sure:\n"
            f"  • KAGGLE_USERNAME/KAGGLE_KEY are set (or ~/.kaggle/kaggle.json exists)\n"
            f"  • You accepted the license on the exact variant page\n"
            f"  • The framework/variant are correct\n"
            f"Original error: {e}"
        ) from e

    # Unzip any zip files into out_dir
    zips = list(out.glob("*.zip"))
    for zf in zips:
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(out)
        finally:
            # Keep or remove ZIPs—prefer removing to avoid confusion
            try:
                zf.unlink()
            except Exception:
                pass

    return str(out.resolve())


# ----------------------------
# Main loader
# ----------------------------
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
    model_source: 'kaggle' | 'hf' | 'local'
    model_name:
      - kaggle: 'owner/name/framework/variant' (e.g., 'google/gemma-3/pyTorch/gemma-3-1b-it')
      - hf:     Hugging Face model id (e.g., 'mistralai/Mistral-7B-Instruct-v0.3')
      - local:  Local directory path with model files
    """

    # Resolve model_path
    if model_source == "kaggle":
        force_cli = os.environ.get("FORCE_KAGGLE_CLI", "0") == "1"
        if not force_cli:
            try:
                import kagglehub  # installed in Colab first cell
                # Try kagglehub first (fast path)
                model_path = kagglehub.model_download(model_name)
            except Exception as e:
                msg = str(e)
                # Known Colab 403 path: fall back to CLI
                if "403" in msg or "ColabHTTPError" in msg:
                    force_cli = True
                else:
                    # Other errors bubble up (auth, not found, etc.)
                    # But give a more actionable message
                    raise RuntimeError(
                        f"kagglehub.model_download failed for '{model_name}'. "
                        f"If you are in Colab and see 403/permission issues, set FORCE_KAGGLE_CLI=1 "
                        f"or ensure license acceptance and correct credentials.\n\nOriginal error:\n{e}"
                    ) from e

        if force_cli:
            # Use CLI fallback
            safe_out = model_name.replace("/", "_")
            out_dir = os.path.join("models", safe_out)
            model_path = _download_with_kaggle_cli(model_name, out_dir)

    elif model_source == "hf":
        # Hugging Face ID; transformers can pull directly
        model_path = model_name

    elif model_source == "local":
        # Local directory path with model files
        if not Path(model_name).exists():
            raise FileNotFoundError(f"Local model path not found: {model_name}")
        model_path = model_name

    else:
        raise ValueError(f"Unknown model_source: {model_source}. Use 'kaggle' | 'hf' | 'local'.")

    # Quantization config (if any)
    quant_cfg = _maybe_quant_config(quantization)

    # Tokenizer
    # Prefer fast tokenizer; fall back to slow if fast fails
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Model
    if quant_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=quant_cfg,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
    else:
        dtype = _to_torch_dtype(torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )

    # Runtime tweaks
    try:
        model.config.use_cache = False  # helps training/gradient checkpointing
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
