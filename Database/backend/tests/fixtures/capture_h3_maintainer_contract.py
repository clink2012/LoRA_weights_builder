"""Capture only pinned maintainer H3 parser/execute against synthetic patches.

Run: capture_h3_maintainer_contract.py <pinned_source.py> <out.json>
No Comfy imports, files, model tensors or generation. Upstream not vendored.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace


def capture(source_path):
    source = Path(source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != "3d57625d2fb907bbbb8902d293287df7db5d721b4032a544eaa1a971a82f29d8":
        raise ValueError("Supply the exact pinned H3 maintainer source.")
    tree = ast.parse(source.decode("utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"parse_block_overrides", "patch_group"}]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "FL_MiniMaxH3LoraBlockLoader")
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute"]
    selected.append(cls)
    loaded = {}
    calls = []

    class NativeH3:
        blocks = [None] * 50
        token_refiner = SimpleNamespace(blocks=[None] * 2)

    class Model:
        def __init__(self, native=True, reject=False):
            self.native, self.reject = native, reject
            self.model = object()
            self.applied = {"prior_adapter": 0.7}
            self.attachments = {}

        def get_model_object(self, _):
            return NativeH3() if self.native else object()

        def clone(self):
            result = Model(self.native, self.reject)
            result.applied = dict(self.applied)
            return result

        def set_attachments(self, name, value):
            self.attachments[name] = value

        def add_patches(self, patches, strength):
            if self.reject:
                return []
            for key, patch in patches.items():
                assert patch is loaded[key]  # Exact patch and alpha object retained.
                self.applied[str(key)] = strength
            return list(patches)

    def load_file(*_, **kwargs):
        calls.append(kwargs)
        return loaded, {"example": "synthetic"}

    context = {
        "math": math, "re": re, "MiniMaxH3Model": NativeH3,
        "folder_paths": SimpleNamespace(get_full_path_or_raise=lambda *_: "synthetic"),
        "io": SimpleNamespace(NodeOutput=lambda *args: args),
        "comfy": SimpleNamespace(utils=SimpleNamespace(load_torch_file=load_file),
            lora_convert=SimpleNamespace(convert_lora=lambda weights: weights),
            lora=SimpleNamespace(model_lora_keys_unet=lambda *_: {}, load_lora=lambda weights, _: weights)),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(source_path), "exec"), context)
    loader = context[cls.name]
    keys = ["diffusion_model.blocks.3.attn.qkv_proj.weight", "diffusion_model.blocks.49.mlp.fc1.weight",
            ("diffusion_model.token_refiner.blocks.1.attn.qkv_proj.weight", (0, 0, 1)), "diffusion_model.img_in.weight"]
    specs = [
        ("neutral", 1., 1., 1., 1., "", True, False, keys),
        ("signed_sparse_separate_refiner", -0.8, 0.6, 0.25, -0.3,
         "blocks.3=-0.625\nblocks.49=0.1256789\nrefiner.1=0.4", True, False, keys),
        ("later_overrides_replace_group", 0.9, 0.6, 1., 1.,
         "blocks.0-9=0.2 # first\nblocks.3=-0.5", True, False, keys),
        ("zero_group_drops_patch", 1., 0., 0., 0., "", True, False, keys),
        ("overall_zero_bypasses_loading", 0., 1., 1., 1., "", True, False, keys),
        ("invalid_override_rejected", 1., 1., 1., 1., "blocks.50=1", True, False, keys),
        ("invalid_number_rejected", 1., 1., 1., 1., "refiner.0=nan", True, False, keys),
        ("non_native_model_rejected", 1., 1., 1., 1., "", False, False, keys),
        ("unaccepted_nonzero_patch_rejected", 1., 1., 1., 1., "", True, True, keys),
        ("empty_resolved_patches_rejected", 1., 1., 1., 1., "", True, False, []),
    ]
    cases = []
    for name, outer, main, refiner, other, text, native, reject, patch_keys in specs:
        loaded = {key: object() for key in patch_keys}
        calls.clear()
        model = Model(native, reject)
        result = {"name": name, "outer": outer, "main": main, "refiner": refiner,
                  "other": other, "overrides": text}
        try:
            output, report = loader.execute(model, "synthetic", outer, main, refiner, other, text)
            result.update(applied=output.applied, report_summary=report.splitlines()[0],
                          metadata_attached=bool(output.attachments))
        except ValueError as error:
            result["error"] = str(error)
        assert model.applied == {"prior_adapter": 0.7} and model.attachments == {}
        result["source_model_unchanged"] = True
        result["file_load_calls"] = len(calls)
        cases.append(result)
    return {"source_sha256": hashlib.sha256(source).hexdigest(),
            "source_revision": "4e6379770e7f4fc48651b76d03284a7eee7df8a4",
            "scope": "Stubbed native H3 model and resolved patches; complete source-key mapping and render behavior unverified.",
            "main_count": 50, "refiner_count": 2, "cases": cases}


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(capture(sys.argv[1]), indent=2) + "\n", encoding="utf-8")
