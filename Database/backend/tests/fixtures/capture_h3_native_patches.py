"""Pinned native ordinary LoRA loading, registration and CPU calculation.

Execute selected methods only, without Comfy startup, owner models or inference.
Reduced-size synthetic matrices do not verify INT8/ConvRot or native cloning.
"""
import ast
import contextlib
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional
import uuid

from capture_flux1_target import extract

PINNED = {
    'comfy/weight_adapter/lora.py': '5e30c8b8a22be6459883cb5d758fa76f725cc4688d4048200145b885567c4cec',
    'comfy/lora.py': 'fce6903ca8150611b3f6477c3772bfef0662a2b01747a579ff56b415a44c29bc',
    'comfy/lora_convert.py': '6c199c3404828838b5745aafc1b0759566a45287f5c3e9cebd31f248cc179245',
    'comfy/model_patcher.py': 'fc43f038835aeded2e64a0490f9d37795fb9d3c9490d9210ba11f1201aec4fd8',
}


def capture(root):
    root = Path(root)
    for relative, digest in PINNED.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned native patch source changed; review before capture.')
    import torch
    adapter_path = root / 'comfy/weight_adapter/lora.py'
    cls = next(n for n in ast.parse(adapter_path.read_text(encoding='utf-8')).body
               if isinstance(n, ast.ClassDef) and n.name == 'LoRAAdapter')
    cls.body = [n for n in cls.body if isinstance(n, ast.Assign) or
                (isinstance(n, ast.FunctionDef) and n.name in {'__init__', 'load', 'calculate_weight'})]
    context = {'torch': torch, 'Optional': Optional, 'WeightAdapterBase': object, 'logging': logging, 'uuid': uuid}
    context['comfy'] = SimpleNamespace(model_management=SimpleNamespace(
        cast_to_device=lambda value, device, dtype: value.to(device=device, dtype=dtype)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(adapter_path), 'exec'), context)
    context['weight_adapter'] = SimpleNamespace(adapters=[context['LoRAAdapter']])
    for relative, names in [('comfy/lora.py', ['load_lora']), ('comfy/lora_convert.py', ['convert_lora'])]:
        path = root / relative
        exec(compile(extract(path, function_names=names), str(path), 'exec'), context)
    patcher_path = root / 'comfy/model_patcher.py'
    patcher_cls = next(n for n in ast.parse(patcher_path.read_text(encoding='utf-8')).body
                       if isinstance(n, ast.ClassDef) and n.name == 'ModelPatcher')
    method = next(n for n in patcher_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'add_patches')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(patcher_path), 'exec'), context)

    target = 'diffusion_model.blocks.0.attn.qkv_proj.weight'
    class Patcher:
        # State and ownership are synthetic; only add_patches is native source.
        add_patches = context['add_patches']
        model = SimpleNamespace(state_dict=lambda: {target: None})
        def __init__(self): self.patches = {}
        def use_ejected(self): return contextlib.nullcontext()

    up = torch.tensor([[1., -2.], [3., 4.]], dtype=torch.float64)
    down = torch.tensor([[2., 1., -1.], [-3., 2., 4.]], dtype=torch.float64)
    module = 'blocks.0.attn.qkv_proj'
    observations = []
    for format_name, up_suffix, down_suffix in [
            ('AB', '.lora_B.weight', '.lora_A.weight'),
            ('up_down', '.lora_up.weight', '.lora_down.weight'),
            ('AB_default', '.lora_B.default.weight', '.lora_A.default.weight')]:
        for alpha in (None, 4., -2.):
            weights = {module + up_suffix: up, module + down_suffix: down}
            if alpha is not None: weights[module + '.alpha'] = torch.tensor(alpha, dtype=torch.float64)
            patches = context['load_lora'](context['convert_lora'](weights), {module: target})
            patch = patches[target]
            assert patch.weights[0] is up and patch.weights[1] is down and patch.weights[2] == alpha
            for outer, block in [(1., 1.), (.8, .35), (-.8, .35), (.8, -.35), (.8, 0.)]:
                base = torch.ones((2, 3), dtype=torch.float64)
                strength = outer * block
                output = patch.calculate_weight(base.clone(), target, strength, 1., None, lambda x: x,
                                                intermediate_dtype=torch.float64)
                expected = base + strength * (1. if alpha is None else alpha / 2) * (up @ down)
                assert torch.equal(output, expected)
                observations.append(dict(format=format_name, alpha=alpha, outer=outer, block=block, output=output.tolist()))

    # Native registration appends independent adapters; it does not validate
    # factor dimensions or calculate the effective update at this stage.
    patcher = Patcher()
    accepted_first = patcher.add_patches(patches, .7)
    accepted_second = patcher.add_patches(patches, -.2)
    rejected_unknown = patcher.add_patches({'unknown': patch}, 1.)
    assert accepted_first == [target] and accepted_second == [target] and rejected_unknown == []
    assert [entry[0] for entry in patcher.patches[target]] == [.7, -.2]
    assert all(entry[1] is patch for entry in patcher.patches[target])

    errors = []
    class LogCapture(logging.Handler):
        def emit(self, record): errors.append(record.getMessage())
    handler = LogCapture()
    logging.getLogger().addHandler(handler)
    try:
        bad_base = torch.ones((3, 3), dtype=torch.float64)
        result = patch.calculate_weight(bad_base.clone(), target, 1., 1., None, lambda x: x,
                                        intermediate_dtype=torch.float64)
    finally:
        logging.getLogger().removeHandler(handler)
    assert torch.equal(result, bad_base) and errors
    return dict(source_sha256=PINNED, scope='Selected native methods on reduced-size synthetic CPU matrices; no owner checkpoint or quantized base.',
                up=up.tolist(), down=down.tolist(), rank=2, cases=observations,
                registration_strengths=[entry[0] for entry in patcher.patches[target]],
                patch_objects_preserved=True, unknown_target_rejected=True,
                dimension_mismatch_logs_error_and_returns_unchanged=True,
                quantized_base_verified=False, native_clone_verified=False, export_verified=False)


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(json.dumps(capture(sys.argv[1]), indent=2) + '\n', encoding='utf-8')
