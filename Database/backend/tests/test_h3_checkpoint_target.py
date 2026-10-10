import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux_header_coverage import CoverageError, MAX_HEADER_BYTES, read_header
from h3_checkpoint_target import describe_checkpoint, read_checkpoint_header, read_quantization_descriptors
from h3_native_coverage import CONTRACT, resolve_header

CURVE = json.loads((Path(__file__).resolve().parents[1] / 'contracts/h3-native-curve8-source-v1.json').read_text())


def write_file(path, header, payload=b'\0' * 8):
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
    return 8 + len(raw)


def tensor(dtype='I8', shape=None, offsets=None):
    return {'dtype': dtype, 'shape': [2, 4] if shape is None else shape,
            'data_offsets': [0, 8] if offsets is None else offsets}


def constructor_header():
    return {key.removeprefix('diffusion_model.'): {'dtype': 'F32', 'shape': shape}
            for key, shape in {**CURVE['target_shapes'], **CURVE['non_matrix_targets']}.items()}


def test_separate_checkpoint_reader_accepts_int8_without_relaxing_lora_reader(tmp_path):
    path = tmp_path / 'model.safetensors'
    length = write_file(path, {'blocks.0.weight': tensor(), '__metadata__': {'config': '{}'}})
    header, identity = read_checkpoint_header(path)
    assert header['blocks.0.weight']['dtype'] == 'I8'
    assert identity['header_bytes_read'] == length
    assert not identity['tensor_payload_read'] and not identity['tensor_contents_verified']
    with pytest.raises(CoverageError, match='floating-point'): read_header(path)


def test_only_small_json_payloads_are_read_and_not_reported_as_header_only(tmp_path):
    path = tmp_path / 'model.safetensors'
    descriptor = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    header = {'blocks.0.weight': tensor(), 'blocks.0.comfy_quant': tensor('U8', [len(descriptor)], [8, 8 + len(descriptor)])}
    write_file(path, header, b'\xff' * 8 + descriptor)
    values, evidence = read_quantization_descriptors(path)
    assert values['blocks.0']['format'] == 'int8_tensorwise'
    assert evidence['descriptor_payload_bytes_read'] == len(descriptor) and evidence['tensor_payload_read']
    assert not evidence['matrix_payload_read'] and not evidence['scale_payload_read']
    assert not evidence['quantization_verified'] and not evidence['rotation_verified']


@pytest.mark.parametrize('payload', [b'{"x":1,"x":2}', b'[]', b'\xff'])
def test_malformed_or_duplicate_quant_descriptors_cannot_be_used(tmp_path, payload):
    path = tmp_path / 'model.safetensors'
    write_file(path, {'blocks.0.comfy_quant': tensor('U8', [len(payload)], [0, len(payload)])}, payload)
    with pytest.raises(CoverageError): read_quantization_descriptors(path)


def test_descriptor_limits_apply_before_any_payload_read(tmp_path):
    path = tmp_path / 'model.safetensors'
    write_file(path, {'blocks.0.comfy_quant': tensor('U8', [1025], [0, 1025])}, b'\0' * 1025)
    with pytest.raises(CoverageError, match='bounded'): read_quantization_descriptors(path)
    header = {f'blocks.{i}.comfy_quant': tensor('U8', [2], [i*2, i*2+2]) for i in range(257)}
    write_file(path, header, b'{}' * 257)
    with pytest.raises(CoverageError, match='Too many'): read_quantization_descriptors(path)


@pytest.mark.parametrize('header', [
    [], {}, {'x': tensor(dtype='UNKNOWN')}, {'x': tensor(shape=[True, 4])},
    {'x': tensor(shape=[0, 4])}, {'x': tensor(shape=[2**32, 1])},
    {'x': tensor(offsets=[1, 9])}, {'x': tensor(offsets=[0, 7])},
    {'x': tensor(), 'y': tensor()}, {'x': tensor(), '__metadata__': {'config': 3}},
    {'x': tensor(offsets=[False, 8])}, {'x': tensor(dtype='F32')},
])
def test_storage_shapes_offsets_and_metadata_fail_closed(tmp_path, header):
    path = tmp_path / 'model.safetensors'
    write_file(path, header)
    with pytest.raises(CoverageError): read_checkpoint_header(path)


def test_duplicate_keys_and_length_budget_are_rejected(tmp_path):
    path = tmp_path / 'model.safetensors'
    raw = b'{"x":{},"x":{}}'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw)
    with pytest.raises(CoverageError, match='Duplicate'): read_checkpoint_header(path)
    path.write_bytes(struct.pack('<Q', MAX_HEADER_BYTES + 1))
    with pytest.raises(CoverageError, match='bounded'): read_checkpoint_header(path)
    path.write_bytes(b'bad')
    with pytest.raises(CoverageError, match='Missing'): read_checkpoint_header(path)


def test_reader_detects_replaced_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = tmp_path / 'model.safetensors'
    write_file(path, {'x': tensor()})
    original = Path.stat
    def replacement_stat(self, *args, **kwargs):
        stat = original(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_dev=stat.st_dev, st_ino=stat.st_ino + 1,
                                   st_size=stat.st_size, st_mtime_ns=stat.st_mtime_ns)
        return stat
    monkeypatch.setattr(Path, 'stat', replacement_stat)
    with pytest.raises(CoverageError, match='changed'): read_checkpoint_header(path)


def test_explicit_curve_target_differs_from_defaults_in_all_affected_targets():
    assert CURVE['constructor_overrides'] == {'adaln_curve_grid': 1025, 'time_embed_dim': 8}
    assert CURVE['main_count'] == 50 and CURVE['refiner_count'] == 2
    assert CURVE['parameters']['adaln_curves'] and CURVE['parameters']['time_embed_dim'] == 8
    assert len(CURVE['target_shapes']) == 264 and len(CURVE['non_matrix_targets']) == 268
    assert CURVE['target_shapes']['diffusion_model.blocks.49.adaln_proj.linear.weight'] == [96768, 8]
    assert CURVE['target_shapes']['diffusion_model.final_layer.adaln_proj.linear.weight'] == [10752, 8]
    assert CURVE['non_matrix_targets']['diffusion_model.adaln_t_table'] == [1025, 8]
    assert not any('time_embedder' in key for key in CURVE['target_shapes'])
    assert CURVE['target_shapes']['diffusion_model.blocks.0.attn.qkv_proj.weight'] == CONTRACT['target_shapes']['diffusion_model.blocks.0.attn.qkv_proj.weight']


def test_native_pair_coverage_uses_explicit_target_and_never_approves_export():
    pair = {'blocks.0.adaln_proj.linear.lora_A.weight': {'shape': [4, 8]},
            'blocks.0.adaln_proj.linear.lora_B.weight': {'shape': [96768, 4]}}
    assert not resolve_header(pair)['all_tensors_mapped']
    result = resolve_header(pair, contract=CURVE)
    assert result['all_tensors_mapped'] and result['contract_id'] == CURVE['contract_id']
    assert result['source_target_conditional'] and not result['checkpoint_verified'] and not result['export_verified']
    pair['blocks.0.adaln_proj.linear.lora_A.weight']['shape'] = [4, 2688]
    assert resolve_header(pair)['all_tensors_mapped']
    assert not resolve_header(pair, contract=CURVE)['all_tensors_mapped']


def test_every_constructor_state_shape_is_checked_with_quant_aux_separate():
    header = constructor_header()
    header['blocks.0.attn.qkv_proj.weight']['dtype'] = 'I8'
    header['blocks.0.attn.qkv_proj.weight_scale'] = {'shape': [21504, 1], 'dtype': 'F32'}
    header['blocks.0.attn.qkv_proj.comfy_quant'] = {'shape': [72], 'dtype': 'U8'}
    result = describe_checkpoint(header, CURVE)
    assert result['constructor_state_shapes_match'] and result['matched_state_count'] == 532
    assert result['quant_auxiliary_count'] == 2 and result['int8_matrix_count'] == 1
    for flag in ('task_identity_verified', 'quantization_verified', 'rotation_verified', 'patch_application_verified', 'measurements_available', 'export_verified'):
        assert result[flag] is False
    header['blocks.49.adaln_proj.linear.weight']['shape'] = [96768, 2688]
    del header['token_refiner.blocks.1.mlp.fc1.weight']
    header['unknown'] = {'shape': [1], 'dtype': 'F32'}
    result = describe_checkpoint(header, CURVE)
    assert not result['constructor_state_shapes_match']
    assert {i['code'] for i in result['issues']} == {'target_shape_mismatch', 'missing_target', 'unaccounted_checkpoint_tensor'}


def test_packed_quant_dimensions_and_metadata_overrides_need_separate_evidence():
    header = constructor_header()
    header['blocks.0.attn.qkv_proj.weight']['shape'] = [21504, 2688]
    header['__metadata__'] = {'config': '{"transformer":{"time_embed_dim":2688}}'}
    result = describe_checkpoint(header, CURVE)
    assert not result['constructor_state_shapes_match']
    assert {i['code'] for i in result['issues']} == {'target_shape_mismatch', 'metadata_config_requires_review'}


@pytest.mark.parametrize('parameters', [{'unknown': 2}, {'time_embed_dim': True}, {'num_layers': 257}, {'gate_compress': 1}])
def test_symbolic_capture_rejects_unbounded_parameters_before_constructor(tmp_path, monkeypatch, parameters):
    sys.path.insert(0, str(Path(__file__).parent / 'fixtures'))
    import capture_h3_native_target as capture_module
    # Empty source pin set isolates validation; only constructor definitions would
    # be extracted, and the invalid parameters fail before construction.
    monkeypatch.setattr(capture_module, 'PINNED', {})
    model = tmp_path / 'comfy/ldm/minimax/model.py'
    model.parent.mkdir(parents=True)
    model.write_text('class MiniMaxH3Model:\n def __init__(self, **kwargs):\n  raise RuntimeError("must not construct")\n')
    with pytest.raises(ValueError): capture_module.capture(tmp_path, parameters=parameters)
