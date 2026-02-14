import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
from safetensors import safe_open


# --- Result structure for a single LoRA analysis --- #

@dataclass
class LoraAnalysis:
    file_path: str
    model_family: str              # e.g. "Flux"
    base_model_code: Optional[str] # e.g. "FLX"

    lora_type: str                 # e.g. "Flux (UNet blocks only)"
    rank: Optional[int]            # placeholder, we can try to detect later
    block_layout: Optional[str]    # e.g. flux_transformer_38, flux_unet_57

    block_weights: List[float]        # normalised 0–1
    raw_block_strengths: List[float]  # unnormalised values for reference

    notes: Optional[str] = None    # free-form info


# --- CORE UTILITIES --- #

def _normalise_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def _load_safetensors_as_torch(path: str) -> Dict[str, torch.Tensor]:
    """
    Load all tensors from a .safetensors file as PyTorch tensors.
    Using Torch avoids NumPy's issues with bfloat16.
    """
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


# --- PATTERNS FOR DIFFERENT FLUX STYLES --- #
#
# 1) "Flux transformer" LoRA
#    transformer.single_transformer_blocks.<idx>....
#
# 2) "Flux UNet double_blocks" LoRA
#    lora_unet_double_blocks_<idx>_...
#
# 3) "Flux TE-only" LoRA
#    lora_te1_text_model_encoder_layers_<idx>_...
#    lora_te2_text_model_encoder_layers_<idx>_...
#

_flux_transformer_pattern = re.compile(
    r"transformer\.single_transformer_blocks\.(\d+)\.", re.IGNORECASE
)

_flux_double_pattern = re.compile(
    r"lora_unet_double_blocks_(\d+)_", re.IGNORECASE
)

_flux_single_pattern = re.compile(
    r"lora_unet_single_blocks_(\d+)_", re.IGNORECASE
)

_flux_te_layer_pattern = re.compile(
    r"lora_te[12]_text_model_encoder_layers_(\d+)_", re.IGNORECASE
)


def _accumulate_block_strengths(
    blocks: Dict[int, List[torch.Tensor]]
) -> Tuple[List[int], List[float], List[float]]:
    """
    Given a mapping: block_index -> list of tensors,
    compute raw and normalised strengths.
    Returns:
        indices (sorted),
        raw_strengths (per index in indices),
        norm_strengths (per index in indices)
    """
    if not blocks:
        return [], [], []

    indices = sorted(blocks.keys())
    raw_strengths: List[float] = []

    for idx in indices:
        strength = 0.0
        for t in blocks[idx]:
            strength += float(t.norm().item())
        raw_strengths.append(strength)

    max_val = max(raw_strengths) if raw_strengths else 0.0
    if max_val > 0:
        norm_strengths = [round(v / max_val, 6) for v in raw_strengths]
    else:
        norm_strengths = [0.0 for _ in raw_strengths]

    return indices, raw_strengths, norm_strengths


def _compute_flux_unet_57_strengths(
    double_blocks: Dict[int, List[torch.Tensor]],
    single_blocks: Dict[int, List[torch.Tensor]],
) -> Tuple[List[float], List[float]]:
    """
    Compute ordered [DOUBLE_0..18] + [SINGLE_0..37] raw and normalised strengths.
    Missing indices are represented as 0.0.
    """
    raw_strengths: List[float] = []

    for idx in range(19):
        tensors = double_blocks.get(idx, [])
        raw_strengths.append(sum(float(t.norm().item()) for t in tensors))

    for idx in range(38):
        tensors = single_blocks.get(idx, [])
        raw_strengths.append(sum(float(t.norm().item()) for t in tensors))

    max_val = max(raw_strengths) if raw_strengths else 0.0
    if max_val > 0:
        norm_strengths = [round(v / max_val, 6) for v in raw_strengths]
    else:
        norm_strengths = [0.0 for _ in raw_strengths]

    return raw_strengths, norm_strengths


# --- FLUX BLOCK ANALYSIS --- #

def _analyse_flux_blocks(path: str, base_model_code: Optional[str]) -> LoraAnalysis:
    """
    Inspect a Flux LoRA (.safetensors).

    We currently support three main patterns:

    1) Flux transformer-style blocks:
       transformer.single_transformer_blocks.<idx>....

    2) Flux UNet "double_blocks" LoRA:
       lora_unet_double_blocks_<idx>_...

    3) Flux text-encoder-only LoRA:
       lora_te1_text_model_encoder_layers_<idx>_...
       lora_te2_text_model_encoder_layers_<idx>_...
    """
    file_path = _normalise_path(path)
    tensors = _load_safetensors_as_torch(file_path)

    transformer_blocks: Dict[int, List[torch.Tensor]] = {}
    double_blocks: Dict[int, List[torch.Tensor]] = {}
    single_blocks: Dict[int, List[torch.Tensor]] = {}
    te_blocks: Dict[int, List[torch.Tensor]] = {}

    # --- Scan all tensors once and bucket them --- #
    for name, arr in tensors.items():
        # 1) transformer.single_transformer_blocks.<idx>.*
        m = _flux_transformer_pattern.search(name)
        if m:
            idx = int(m.group(1))
            transformer_blocks.setdefault(idx, []).append(arr)
            continue

        # 2) lora_unet_double_blocks_<idx>_...
        m = _flux_double_pattern.search(name)
        if m:
            idx = int(m.group(1))
            double_blocks.setdefault(idx, []).append(arr)
            continue

        # 2b) lora_unet_single_blocks_<idx>_...
        m = _flux_single_pattern.search(name)
        if m:
            idx = int(m.group(1))
            single_blocks.setdefault(idx, []).append(arr)
            continue

        # 3) lora_te[1 or 2]_text_model_encoder_layers_<idx>_...
        m = _flux_te_layer_pattern.search(name)
        if m:
            idx = int(m.group(1))
            te_blocks.setdefault(idx, []).append(arr)
            continue

    # --- Case 1: Transformer-style Flux LoRA --- #
    if transformer_blocks:
        indices, raw_strengths, norm_strengths = _accumulate_block_strengths(transformer_blocks)

        notes = (
            f"Flux transformer blocks detected at indices: {indices}. "
            "Block weights are normalised so the strongest block = 1.0."
        )

        return LoraAnalysis(
            file_path=file_path,
            model_family="Flux",
            base_model_code=base_model_code,
            lora_type="Flux (single_transformer_blocks)",
            rank=None,
            block_layout="flux_transformer_38",
            block_weights=norm_strengths,
            raw_block_strengths=raw_strengths,
            notes=notes,
        )

    # --- Case 2: UNet double+single blocks Flux LoRA --- #
    if double_blocks and single_blocks:
        raw_strengths, norm_strengths = _compute_flux_unet_57_strengths(
            double_blocks=double_blocks,
            single_blocks=single_blocks,
        )

        notes_parts = [
            "Flux UNet double+single blocks detected. "
            "Computed ordered layout: DOUBLE_0..18 + SINGLE_0..37 (57 total). "
            "Block weights are normalised so the strongest block = 1.0."
        ]
        if te_blocks:
            notes_parts.append(
                f"Additional TE layers present (indices: {sorted(te_blocks.keys())}), "
                "but only UNet block tensors are used for block-strength computation."
            )

        return LoraAnalysis(
            file_path=file_path,
            model_family="Flux",
            base_model_code=base_model_code,
            lora_type="Flux (UNet double+single blocks)",
            rank=None,
            block_layout="flux_unet_57",
            block_weights=norm_strengths,
            raw_block_strengths=raw_strengths,
            notes=" ".join(notes_parts),
        )

    # --- Case 2: UNet double_blocks Flux LoRA --- #
    if double_blocks:
        indices, raw_strengths, norm_strengths = _accumulate_block_strengths(double_blocks)

        notes_parts = [
            f"Flux UNet double_blocks detected at indices: {indices}. "
            "Block weights are normalised so the strongest block = 1.0."
        ]
        if te_blocks:
            notes_parts.append(
                f"Additional TE layers present (indices: {sorted(te_blocks.keys())}), "
                "but only UNet double_blocks are used for block-strength computation in this engine version."
            )

        notes = " ".join(notes_parts)

        return LoraAnalysis(
            file_path=file_path,
            model_family="Flux",
            base_model_code=base_model_code,
            lora_type="Flux (UNet double_blocks)",
            rank=None,
            block_layout="flux_unet_double",
            block_weights=norm_strengths,
            raw_block_strengths=raw_strengths,
            notes=notes,
        )

    # --- Case 3: TE-only Flux LoRA --- #
    if te_blocks:
        indices, raw_strengths, norm_strengths = _accumulate_block_strengths(te_blocks)

        notes = (
            "Flux text-encoder-only LoRA detected. "
            f"TE layer indices: {indices}. "
            "Block weights represent per-layer strengths, normalised so the strongest layer = 1.0."
        )

        return LoraAnalysis(
            file_path=file_path,
            model_family="Flux",
            base_model_code=base_model_code,
            lora_type="Flux (text-encoder only)",
            rank=None,
            block_layout="flux_te_layers",
            block_weights=norm_strengths,
            raw_block_strengths=raw_strengths,
            notes=notes,
        )

    # --- Fallback: unknown Flux format --- #
    raise ValueError(
        "No recognised Flux-style structures found.\n"
        "- Expected one of:\n"
        "  * transformer.single_transformer_blocks.<idx>.* (Flux transformer), or\n"
        "  * lora_unet_double_blocks_<idx>_* (Flux UNet double_blocks), or\n"
        "  * lora_te[1/2]_text_model_encoder_layers_<idx>_* (Flux TE-only).\n"
        "This file may use an unexpected format."
    )


# --- PUBLIC ENTRY POINT --- #

def inspect_lora(path: str, base_model_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect a LoRA .safetensors file and return a dictionary of analysis results.

    For now, only Flux / Flux Krea (FLX / FLK) are supported for block analysis.
    Other base_model_code values will raise NotImplementedError.

    Returns a plain dict so it’s easy to JSON-serialise or store in a DB.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    code_upper = (base_model_code or "").upper()

    if code_upper in ("FLX", "FLK", ""):
        analysis = _analyse_flux_blocks(path, base_model_code=code_upper or None)
    elif code_upper in ("W21", "W22"):
        analysis = LoraAnalysis(
            file_path=_normalise_path(path),
            model_family="WAN",
            base_model_code=code_upper,
            lora_type="WAN (unimplemented)",
            rank=None,
            block_layout=None,
            block_weights=[],
            raw_block_strengths=[],
            notes=(
                "WAN block extraction is not implemented yet. "
                "This placeholder preserves metadata safely for indexing/API responses."
            ),
        )
    else:
        raise NotImplementedError(
            f"Block analysis for base_model_code='{base_model_code}' is not implemented yet. "
            "Currently supported: FLX, FLK (Flux)."
        )

    return asdict(analysis)


# --- SIMPLE CLI TEST HARNESS --- #

def _cli_main():
    print("=== Delta Inspector Engine (Flux v0.5, Torch backend) ===")
    print("This is the backend engine used by the LoRA Master tool.")
    print("Currently supports:")
    print("- Flux transformer-style LoRA (transformer.single_transformer_blocks.<idx>.*)")
    print("- Flux UNet double_blocks LoRA (lora_unet_double_blocks_<idx>_*)")
    print("- Flux TE-only LoRA (lora_te[1/2]_text_model_encoder_layers_<idx>_*)")
    print()
    print("Enter a path to a .safetensors file to inspect.")
    print("You can paste a full path, or drag+drop a file into this window.")
    print()

    file_path = input("LoRA .safetensors path: ").strip().strip('"')

    if not file_path:
        print("No path provided, aborting.")
        return

    file_path = _normalise_path(file_path)

    if not os.path.isfile(file_path):
        print(f"ERROR: '{file_path}' is not a valid file.")
        return

    try:
        result = inspect_lora(file_path, base_model_code="FLX")
    except Exception as e:
        print("\nERROR while inspecting LoRA:")
        print(str(e))
        return

    print("\n=== Analysis Result ===")
    print(f"File: {result['file_path']}")
    print(f"Model family   : {result['model_family']}")
    print(f"Base model code: {result['base_model_code']}")
    print(f"LoRA type      : {result['lora_type']}")
    print(f"Rank           : {result['rank']}")
    print(f"Notes          : {result['notes']}")
    print()

    bw = result["block_weights"]
    raw = result["raw_block_strengths"]

    print(f"Detected {len(bw)} block(s) with weights.")
    if bw:
        print("Normalised block weights (0–1, max block = 1):")
        print(bw)
        print()
        print("Raw block strengths (for reference):")
        print(raw)
    else:
        print("No per-block weights computed for this LoRA style (yet).")
    print()
    print("This data will later be stored in the database and used for:")
    print("- Pattern generation")
    print("- LoRA comparison")
    print("- Conflict-free stacking suggestions")


if __name__ == "__main__":
    _cli_main()
