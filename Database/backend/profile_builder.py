import os
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Import the analysis engine (your existing file)
from delta_inspector_engine import inspect_lora  # :contentReference[oaicite:1]{index=1}


# -------------------------------------------------------------
# CONFIG: where profile JSON files are stored
# -------------------------------------------------------------
PROFILES_ROOT = r"E:\LoRA Project\Database\backend\profiles"

# Buckets you’ve already created on disk
PROFILE_BUCKETS = {
    "flux": "flux",
    "flux_krea": "flux_krea",
    "illustrious": "illustrious",
    "pony": "pony",
    "sd": "sd",
    "sdxl": "sdxl",
    "wan21": "wan21",
    "wan22": "wan22",
    "other": "other",  # catch-all for anything unknown
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalise_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


# -------------------------------------------------------------
# Model / bucket detection
# -------------------------------------------------------------
def _detect_bucket(lora_path: str, analysis: Dict[str, Any]) -> Tuple[str, str]:
    """
    Decide which model family + bucket this LoRA belongs to.

    Returns:
        (model_family_label, bucket_name)

    model_family_label: human-facing label ("Flux", "Pony", "WAN 2.2", etc.)
    bucket_name: one of PROFILE_BUCKETS values, e.g. "flux", "sdxl", "wan22".
    """
    name = os.path.basename(lora_path).lower()
    engine_family = (analysis.get("model_family") or "").strip()  # e.g. "Flux"

    # Pony
    if "pony" in name:
        return "Pony", PROFILE_BUCKETS.get("pony", "pony")

    # WAN 2.2
    if "wan2.2" in name or "wan22" in name or "wan-2.2" in name:
        return "WAN 2.2", PROFILE_BUCKETS.get("wan22", "wan22")

    # WAN 2.1
    if "wan2.1" in name or "wan21" in name or "wan-2.1" in name:
        return "WAN 2.1", PROFILE_BUCKETS.get("wan21", "wan21")

    # SDXL
    if "sdxl" in name or "xl" in name:
        return "SDXL", PROFILE_BUCKETS.get("sdxl", "sdxl")

    # SD 1.5 / classic SD
    if "sd15" in name or "sd1.5" in name or "sd-v1" in name:
        return "SD 1.x", PROFILE_BUCKETS.get("sd", "sd")

    # Flux Krea
    if "krea" in name:
        return "Flux (Krea)", PROFILE_BUCKETS.get("flux_krea", "flux_krea")

    # Illustrious
    if "illustrious" in name or "illu" in name:
        return "Illustrious", PROFILE_BUCKETS.get("illustrious", "illustrious")

    # Default for anything the engine flagged as Flux
    if engine_family.lower() == "flux":
        return "Flux", PROFILE_BUCKETS.get("flux", "flux")

    # Fallback
    return engine_family or "Unknown", PROFILE_BUCKETS.get("other", "other")


# -------------------------------------------------------------
# Profile builder
# -------------------------------------------------------------
def build_profile_for_lora(
    lora_path: str,
    base_model_code: Optional[str] = "FLX",
) -> Dict[str, Any]:
    """
    High-level entry point:

    1) Analyse a LoRA using delta_inspector_engine.inspect_lora()
    2) Build a unified profile dict
    3) Save JSON into the appropriate profiles bucket
    4) Return the profile dict

    Currently expects Flux/Flux-Krea (base_model_code FLX/FLK).
    Other base model codes can be added later.
    """
    lora_path = _normalise_path(lora_path)

    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"No such file: {lora_path}")

    # --- 1) Run the engine analysis --- #
    # This returns a dict matching the LoraAnalysis dataclass in the engine.
    analysis: Dict[str, Any] = inspect_lora(lora_path, base_model_code=base_model_code)

    # --- 2) File info --- #
    file_name = os.path.basename(lora_path)
    stem, ext = os.path.splitext(file_name)

    try:
        size_bytes = os.path.getsize(lora_path)
    except OSError:
        size_bytes = 0
    size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0.0

    # --- 3) Model family / bucket assignment --- #
    model_family_label, bucket = _detect_bucket(lora_path, analysis)

    # --- 4) Block stats --- #
    block_weights = analysis.get("block_weights") or []
    raw_block_strengths = analysis.get("raw_block_strengths") or []

    # Length should match, but we’re not going to be precious about it
    block_count = max(len(block_weights), len(raw_block_strengths))

    max_norm = max(block_weights) if block_weights else 0.0
    mean_norm = (
        float(sum(block_weights) / len(block_weights))
        if block_weights
        else 0.0
    )

    # --- 5) Profile dict --- #
    profile: Dict[str, Any] = {
        "schema_version": "0.1",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",

        "file": {
            "path": lora_path,
            "name": file_name,
            "stem": stem,
            "ext": ext,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
        },

        "model": {
            # What the engine thinks the family is (e.g. "Flux")
            "engine_family": analysis.get("model_family"),
            # Human-facing family label we inferred (e.g. "Flux (Krea)")
            "family_label": model_family_label,
            # Which bucket under PROFILES_ROOT we stored it in
            "bucket": bucket,
            # Base model code (e.g. FLX / FLK)
            "base_model_code": analysis.get("base_model_code"),
            # Engine-reported LoRA type string
            "lora_type": analysis.get("lora_type"),
        },

        "blocks": {
            "normalised_weights": block_weights,
            "raw_strengths": raw_block_strengths,
            "count": block_count,
            "stats": {
                "max_norm": max_norm,
                "mean_norm": mean_norm,
            },
        },

        "engine": {
            # Raw analysis dict (so the web app has full access if needed)
            "analysis": analysis,
        },

        "classification": {
            # To be enriched later: role/usage type etc.
            "role": None,              # e.g. "character", "style", "clothing", "pose"
            "tags": [],                # we’ll fill from LoRA Manager / CivitAI later
            "manual_override": False,  # set to True if you hand-edit this later
            "notes": analysis.get("notes"),
        },
    }

    # --- 6) Write JSON to the appropriate bucket --- #
    target_dir = os.path.join(PROFILES_ROOT, bucket)
    _ensure_dir(target_dir)

    profile_path = os.path.join(target_dir, f"{stem}_profile.json")

    try:
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        print(f"[PROFILE] Saved: {profile_path}")
    except Exception as e:
        print(f"[PROFILE] ERROR writing profile JSON: {e}")

    return profile


# -------------------------------------------------------------
# Simple CLI harness for testing
# -------------------------------------------------------------
def _cli_main() -> None:
    print("=== LoRA Profile Builder v0.1 (profiles in backend\\profiles) ===")
    print("This uses delta_inspector_engine.inspect_lora() as the analysis backend.")
    print()
    print("Enter a path to a .safetensors file to build a profile for.")
    print("You can paste a full path, or drag+drop the file into this window.")
    print()

    lora_path = input("LoRA .safetensors path: ").strip().strip('"')
    if not lora_path:
        print("No path provided, aborting.")
        return

    # For now, default to FLX (Flux) – we’ll add Pony/SDXL/WAN later
    base_model_code = input("Base model code [default FLX]: ").strip().upper() or "FLX"

    try:
        profile = build_profile_for_lora(lora_path, base_model_code=base_model_code)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return
    except NotImplementedError as e:
        print("\nERROR: The analysis engine does not yet support this base model code.")
        print(str(e))
        return
    except Exception as e:
        print("\nERROR while building profile:")
        print(str(e))
        return

    print("\n=== Profile Summary ===")
    print(f"File         : {profile['file']['path']}")
    print(f"Bucket       : {profile['model']['bucket']}")
    print(f"Family (raw) : {profile['model']['engine_family']}")
    print(f"Family (label): {profile['model']['family_label']}")
    print(f"LoRA type    : {profile['model']['lora_type']}")
    print(f"Blocks       : {profile['blocks']['count']}")
    print(f"Max norm     : {profile['blocks']['stats']['max_norm']}")
    print(f"Mean norm    : {profile['blocks']['stats']['mean_norm']}")
    print()
    print("Profile JSON written. This is what the web UI will consume for:")
    print("- Overview / Dashboard")
    print("- Patterns")
    print("- Compare")
    print("- Delta Lab")


if __name__ == "__main__":
    _cli_main()
