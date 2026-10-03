"""Capture observable numeric behavior from an explicitly supplied local loader.

Run with Python: capture_inspire_contract.py <lora_block_weight.py> <output.json>
Reads source without importing Comfy, Torch or executing module startup. Only
the named pure/parser/weight-assignment methods are executed against fake patch
objects. Upstream source is GPL-3.0; it is not copied into this repository.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace


def capture(source_path):
    source = Path(source_path).read_bytes()
    tree = ast.parse(source.decode("utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LoraLoaderBlockWeight")
    methods = {"validate", "convert_vector_value", "norm_value", "block_spec_parser", "load_lbw"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"is_numeric_string", "parse_unet_num"}]
    body.append(cls)
    selected = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    context = {
        "re": re, "np": SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
        "comfy": SimpleNamespace(lora=SimpleNamespace(
            model_lora_keys_unet=lambda _: {}, model_lora_keys_clip=lambda _, keys: keys,
            load_lora=lambda lora, _: lora,
        )),
        "load_preset_dict": lambda: {},
    }
    exec(compile(selected, str(source_path), "exec"), context)
    loader = context["LoraLoaderBlockWeight"]
    model = SimpleNamespace(model=None)
    clip = SimpleNamespace(cond_stage_model=None)
    all_keys = [f"diffusion_model.double_blocks.{i}.img_attn.proj.weight" for i in range(19)]
    all_keys += [f"diffusion_model.single_blocks.{i}.linear1.weight" for i in range(38)]
    base = "diffusion_model.img_in.weight"
    cases = [
        ("full_flux", all_keys + [base], [0.25] + [i / 100 for i in range(1, 58)]),
        ("sparse_architecture_vector_misapplies", [all_keys[3], all_keys[17], all_keys[20], base], [0.25] + [i / 100 for i in range(1, 58)]),
        ("sparse_consumed_vector", [all_keys[3], all_keys[17], all_keys[20], base], [0.25, 0.04, 0.18, 0.21] + [0] * 8),
        ("equal_index_boundary", [all_keys[3], all_keys[22], base], [0.25, 0.4, 0.8] + [0] * 9),
        ("repeated_patch_in_same_block", [all_keys[3], "diffusion_model.double_blocks.3.img_mlp.0.weight", base], [0.25, 0.4] + [0] * 10),
        ("short_vector_reuses_last", all_keys + [base], [0.25] + [i / 100 for i in range(1, 12)]),
        ("unknown_transformer_uses_base", ["diffusion_model.transformer_blocks.0.weight", base], [0.25] + [0.8] * 11),
        ("scientific_notation_rejected", [all_keys[0]], [1.0, "1e-05"] + [0] * 10),
    ]
    results = []
    for name, keys, values in cases:
        csv = ",".join(map(str, values))
        try:
            weighted, muted, populated = loader.load_lbw(model, clip, dict.fromkeys(keys, "dummy_patch"), False, 0, 1.0, 1.0, csv)
            weights = {key: value[1] for key, value in weighted.items()}
            weights.update(dict.fromkeys(muted, 0.0))
            result = {"applied_weights": weights, "populated_vector": populated}
        except ValueError:
            result = {"error": "invalid_block_vector"}
        results.append({"name": name, "patch_keys": keys, "input_csv": csv, **result})
    return {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_project": "https://github.com/ltdrdata/ComfyUI-Inspire-Pack",
        "source_license": "GPL-3.0 (source inspected locally, not vendored)",
        "captured_methods": sorted(methods),
        "scope": "Numeric non-inverse block assignment with synthetic resolved patches; not tensor application or image quality.",
        "cases": results,
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(capture(sys.argv[1]), indent=2) + "\n", encoding="utf-8")
