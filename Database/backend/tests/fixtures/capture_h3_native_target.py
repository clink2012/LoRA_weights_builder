"""Symbolically capture pinned local native H3 constructor shapes and aliases.

No model files, tensors, torch imports, inference or upstream module startup.
The result is a conditional source target, never a detected owner checkpoint.
Run with a read-only Comfy root and an output JSON path.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from capture_flux1_target import Module, Shape, Linear, Norm, Sequential, ModelBaseTypes, extract

PINNED = {
    "comfy/ldm/minimax/model.py": "99bb765eaa8c4fcc5d279d7aa63f54cc92eeffdc9ecb4a9629996dfb214aceb8",
    "comfy/lora.py": "fce6903ca8150611b3f6477c3772bfef0662a2b01747a579ff56b415a44c29bc",
}


class BufferedModule(Module):
    def register_buffer(self, name, value):
        setattr(self, name, value)


def capture(root):
    root = Path(root)
    for relative, expected in PINNED.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source changed: {relative}; review before recapturing.")
    nn = SimpleNamespace(Module=BufferedModule, ModuleList=lambda items: Sequential(*items))
    context = dict(nn=nn, torch=SimpleNamespace(float32="float32", empty=lambda *shape, **_: Shape(*shape)), ComfyAttention=Module)
    model_path = root / "comfy/ldm/minimax/model.py"
    names = ("TimeEmbedder", "Attention", "MLP", "AdalnProj", "RefinerBlock", "TokenRefiner", "DiTBlock", "FinalLayer", "MiniMaxH3Model")
    exec(compile(extract(model_path, names), str(model_path), "exec"), context)
    instance = context["MiniMaxH3Model"](operations=SimpleNamespace(Linear=Linear, RMSNorm=Norm))
    shapes = instance.state_dict("diffusion_model.")
    class NativeH3:
        def state_dict(self): return shapes
        model_config = SimpleNamespace(unet_config={})
    types = ModelBaseTypes()
    types.MiniMaxH3 = NativeH3
    context["comfy"] = SimpleNamespace(utils=SimpleNamespace(unet_to_diffusers=lambda _: {}), model_base=types)
    mapping = root / "comfy/lora.py"
    exec(compile(extract(mapping, function_names=("model_lora_keys_unet",)), str(mapping), "exec"), context)
    aliases = context["model_lora_keys_unet"](NativeH3(), {})
    shapes2d = {key: list(value.shape) for key, value in shapes.items() if key.endswith(".weight") and len(value.shape) == 2}
    return {
        "contract_id": "h3-native-constructor-defaults-v1",
        "label": "Native H3 constructor defaults (conditional source target)",
        "source_sha256": PINNED,
        "scope": "Pinned constructor shapes and actual native generic/bare/Kohya aliases; not a loaded checkpoint, task identity, patch application or export.",
        "main_count": len(vars(instance.blocks)), "refiner_count": len(vars(instance.token_refiner.blocks)),
        "parameters": {"hidden_size": instance.hidden_size, "attention_heads": instance.blocks.__dict__["0"].attn.heads,
                       "attention_head_dim": instance.blocks.__dict__["0"].attn.head_dim,
                       "time_embed_dim": instance.time_embedder.proj_out.weight.shape[0],
                       "adaln_curves": False, "gate_compress": False},
        "target_shapes": shapes2d,
        "aliases": {alias: target for alias, target in aliases.items() if target in shapes2d},
        "non_matrix_targets": {key: list(value.shape) for key, value in shapes.items() if key not in shapes2d},
        "checkpoint_verified": False, "export_verified": False,
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(capture(sys.argv[1]), indent=2) + "\n", encoding="utf-8")
