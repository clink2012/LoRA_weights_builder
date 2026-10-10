"""Capture H3 behavior from supplied pinned upstream source, without Comfy startup.

Run with the optional CPU environment: capture_h3_contract.py <source.py> <out.json>
Only H3 methods and their pure dependencies are selected with AST. File loading,
Comfy target resolution, saving and rendering are replaced by synthetic sentinels.
No upstream source is copied into the fixture or application.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace


def capture(source_path):
    import torch

    source = Path(source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != "0a0cac0e3d3de4be0128eddbd1f2803045e2f1542d3219232e4a0c1a6468ed45":
        raise ValueError("Supply the exact characterized upstream H3 source; changed source needs a new contract.")
    tree = ast.parse(source.decode("utf-8"))
    names = {"_get_architecture_blocks", "_parse_block_weights_string",
             "_extract_block_id_minimax_h3", "_coerce_scalar_strength", "_scale_minimax_h3_tensor"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "MiniMaxH3SelectiveLoRALoader")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {"INPUT_TYPES", "load_lora"}]
    seen = {}
    state = {}

    def record_load(model, clip, tensors, model_strength, clip_strength):
        seen.update(tensors=tensors, model_strength=model_strength, clip_strength=clip_strength)
        return model, clip

    context = {
        "re": re, "torch": torch, "Dict": dict, "List": list, "Optional": __import__("typing").Optional,
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda _: True)),
        "folder_paths": SimpleNamespace(get_full_path=lambda *_: "synthetic.safetensors",
                                         get_filename_list=lambda _: ["synthetic.safetensors"]),
        "comfy": SimpleNamespace(sd=SimpleNamespace(load_lora_for_models=record_load)),
        "load_file": lambda _: state,
        "MINIMAX_H3_PRESETS": {"All Blocks": set(range(50)), "All Off": set()},
        "_save_minimax_h3_filtered_lora": lambda *_: (_ for _ in ()).throw(AssertionError("Saving must not occur")),
    }
    selected = ast.fix_missing_locations(ast.Module(body=[*functions, cls], type_ignores=[]))
    exec(compile(selected, str(source_path), "exec"), context)
    loader = context["MiniMaxH3SelectiveLoRALoader"]()
    a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    b = torch.tensor([[2., 1.], [-1., 3.]], dtype=torch.float64)
    native = "diffusion_model.blocks."
    specs = [
        ("full_order", [(f"{native}{i}.attn.qkv_proj", "native") for i in range(50)],
         [i / 100 for i in range(50)] + [0.37], 0.8),
        ("sparse_signed_and_other", [(f"{native}3.attn.qkv_proj", "native"),
          (f"{native}49.mlp.fc1", "native"), ("diffusion_model.token_refiner.blocks.0.attn.qkv_proj", "native"),
          ("diffusion_model.img_in", "native")],
         [(-0.625 if i == 3 else 0.1256789 if i == 49 else 1.) for i in range(50)] + [-0.25], -0.8),
        ("kohya_signed", [("lora_unet_blocks_7_attn_qkv_proj", "kohya")],
         [(-0.5 if i == 7 else 1.) for i in range(50)] + [1.], 0.9),
        ("zero_removes_main_and_other", [(f"{native}3.attn.qkv_proj", "native"),
          ("diffusion_model.token_refiner.blocks.0.attn.qkv_proj", "native")],
         [(0. if i == 3 else 1.) for i in range(50)] + [0.], 1.),
        ("short_50_defaults_other", [("diffusion_model.img_in", "native")], [0.4] * 50, 1.),
        ("malformed_falls_back_to_preset", [(f"{native}3.attn.qkv_proj", "native")], "bad,input", 1.),
        ("out_of_range_main_is_dropped", [(f"{native}50.attn.qkv_proj", "native")], [1.] * 51, 1.),
        ("bare_blocks_are_other", [("blocks.3.attn.qkv_proj", "native")],
         [(0. if i == 3 else 1.) for i in range(50)] + [0.5], 1.),
    ]
    cases = []
    for name, modules, weights, outer in specs:
        state = {}
        pairs = []
        for module, fmt in modules:
            down, up = (".lora_A.weight", ".lora_B.weight") if fmt == "native" else (".lora_down.weight", ".lora_up.weight")
            state[module + down], state[module + up] = a.clone(), b.clone()
            state[module + ".alpha"] = torch.tensor(3., dtype=torch.float64)
            pairs.append((module, down, up))
        seen.clear()
        csv = weights if isinstance(weights, str) else ",".join(map(str, weights))
        result = loader.load_lora(object(), object(), "synthetic.safetensors", outer, "All Blocks",
                                  block_weights_string=csv, save_refined_lora=False)
        tensors = seen.get("tensors", {})
        records = []
        for module, down, up in pairs:
            retained = module + up in tensors
            update = ((tensors[module + up] @ tensors[module + down]) *
                      tensors[module + ".alpha"] / 2 * outer).tolist() if retained else [[0., 0.], [0., 0.]]
            records.append({"module": module, "retained": retained,
                "down_unchanged": torch.equal(tensors[module + down], a) if retained else None,
                "alpha_unchanged": float(tensors[module + ".alpha"]) == 3. if retained else None,
                "effective_update": update})
        cases.append({"name": name, "input_csv": csv, "outer_strength": outer,
                      "applied_model_strength": seen.get("model_strength"),
                      "applied_clip_strength": seen.get("clip_strength"),
                      "returned_csv": result["result"][3], "modules": records})
    inputs = loader.INPUT_TYPES()
    scale = context["_scale_minimax_h3_tensor"]
    return {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_revision": "47d5962a651e61a39afcf06c7cf26614454c5fe7",
        "source_project": "https://github.com/shootthesound/comfyUI-Realtime-Lora",
        "scope": "Synthetic CPU factors before stubbed Comfy loading; not checkpoint mapping or rendering.",
        "required_inputs": sorted(inputs["required"]),
        "unscaled_suffixes": {suffix: torch.equal(scale("example" + suffix, a, -0.5), a)
                              for suffix in (".alpha", ".dora_scale", ".reshape_weight", ".unknown_factor")},
        "cases": cases,
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(capture(sys.argv[1]), indent=2) + "\n", encoding="utf-8")
