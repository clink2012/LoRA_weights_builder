"""Record native FLUX target shapes/aliases by symbolically running local Comfy constructors.

No torch import, tensor allocation, model file or startup code. Source is read-only.
The supplied parameters define a conditional standard FLUX.1 dev target, not a
detected user checkpoint. GPL source is executed locally and is not vendored.
"""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


class Module:
    def __init__(self, *args, **kwargs):
        pass

    def state_dict(self, prefix=""):
        result = {}
        for name, value in vars(self).items():
            if isinstance(value, Shape):
                result[prefix + name] = value
            elif isinstance(value, Module):
                result.update(value.state_dict(prefix + name + "."))
        return result


class Shape:
    def __init__(self, *shape):
        self.shape = shape


class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True, **kwargs):
        self.weight = Shape(out_dim, in_dim)
        if bias:
            self.bias = Shape(out_dim)


class Norm(Module):
    def __init__(self, dim, elementwise_affine=True, **kwargs):
        if elementwise_affine:
            self.weight = Shape(dim)


class Sequential(Module):
    def __init__(self, *items):
        for index, item in enumerate(items):
            setattr(self, str(index), item)


class ModelBaseTypes:
    def __getattr__(self, _):
        return type("UnrelatedModel", (), {})


def extract(path, class_names=(), function_names=()):
    body = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            if node.name != "FluxParams":
                node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"]
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            body.append(node)
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def capture(root):
    root = Path(root)
    layers = root / "comfy/ldm/flux/layers.py"
    model = root / "comfy/ldm/flux/model.py"
    mapping = root / "comfy/lora.py"
    adapter = root / "comfy/weight_adapter/lora.py"
    nn = SimpleNamespace(Module=Module, Sequential=Sequential, ModuleList=lambda items: Sequential(*items), SiLU=Module, GELU=Module, Identity=Module)
    context = dict(nn=nn, torch=SimpleNamespace(nn=nn), dataclass=dataclass, ComfyAttention=Module, EmbedND=Module)
    exec(compile(extract(layers, ("MLPEmbedder", "QKNorm", "SelfAttention", "Modulation", "DoubleStreamBlock", "SingleStreamBlock", "LastLayer"), ("build_mlp",)), str(layers), "exec"), context)
    exec(compile(extract(model, ("FluxParams", "Flux")), str(model), "exec"), context)
    params = dict(in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=True, txt_ids_dims=[])
    instance = context["Flux"](operations=SimpleNamespace(Linear=Linear, LayerNorm=Norm, RMSNorm=Norm), **params)
    shapes = instance.state_dict("diffusion_model.")
    # This captures native generic aliases only; unrelated diffusers/model-family
    # conversion branches are deliberately not invoked by the synthetic target.
    context["comfy"] = SimpleNamespace(utils=SimpleNamespace(unet_to_diffusers=lambda _: {}), model_base=ModelBaseTypes())
    exec(compile(extract(mapping, function_names=("model_lora_keys_unet",)), str(mapping), "exec"), context)
    proxy = SimpleNamespace(state_dict=lambda: shapes, model_config=SimpleNamespace(unet_config=params))
    aliases = context["model_lora_keys_unet"](proxy, {})
    shapes2d = {key: list(value.shape) for key, value in shapes.items() if key.endswith(".weight") and len(value.shape) == 2}
    adapter_class = next(node for node in ast.parse(adapter.read_text(encoding="utf-8")).body if isinstance(node, ast.ClassDef) and node.name == "LoRAAdapter")
    adapter_class.body = [node for node in adapter_class.body if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "load"}]
    adapter_tree = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), adapter_class], type_ignores=[]))
    adapter_context = {"WeightAdapterBase": Module}
    exec(compile(adapter_tree, str(adapter), "exec"), adapter_context)
    adapter_cases = []
    for prefix in ("lora_unet_double_blocks_0_img_attn_proj", "diffusion_model.single_blocks.37.linear2"):
        target = aliases[prefix]
        out_dim, in_dim = shapes2d[target]
        up, down = prefix + ".lora_up.weight", prefix + ".lora_down.weight"
        for complete in (True, False):
            tensors = {up: Shape(out_dim, 4)}
            if complete:
                tensors[down] = Shape(4, in_dim)
            try:
                observed = adapter_context["LoRAAdapter"].load(prefix, tensors, 4.0, None, set())
                result = {"loaded_keys": sorted(observed.loaded_keys), "up_shape": list(observed.weights[0].shape), "down_shape": list(observed.weights[1].shape)}
            except KeyError:
                result = {"error": "missing_pair_key"}
            adapter_cases.append({"prefix": prefix, "target": target, "complete": complete, **result})
    return dict(
        contract_id="flux1-dev-native-v1", label="FLUX.1 dev (standard 19 double / 38 single)",
        parameters=params, source_sha256={str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in (layers, model, mapping, adapter)},
        scope="Symbolic target constructor shapes and native generic aliases; conditional target contract, not a loaded checkpoint or runtime patch verification.",
        target_shapes=shapes2d,
        aliases={alias: target for alias, target in aliases.items() if target in shapes2d},
        native_adapter_observations=adapter_cases,
    )


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(capture(sys.argv[1]), indent=2) + "\n", encoding="utf-8")
