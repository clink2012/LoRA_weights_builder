"""Identification never promotes header observations to verified model support."""
from pathlib import Path
import json
import struct
import subprocess
import sys

import pytest

from adapter_identity import classify_header, identify_file
from flux_header_coverage import CoverageError, MAX_HEADER_BYTES, read_header


def pair(module, form="ab"):
    suffixes = ("lora_A", "lora_B") if form == "ab" else ("lora_down", "lora_up")
    return {f"{module}.{suffix}.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
            for suffix in suffixes}


def write_header(tmp_path, header, payload=None):
    header = json.loads(json.dumps(header))
    cursor = 0
    if payload is None:
        for key, item in header.items():
            if key != "__metadata__":
                item["data_offsets"] = [cursor, cursor + 16]
                cursor += 16
        payload = bytes(cursor)
    raw = json.dumps(header).encode()
    path = tmp_path / 'sample.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
    return path, raw


def test_sparse_ltx_av_observation_has_no_total_depth_or_version_authority():
    header = {**pair('diffusion_model.transformer_blocks.47.audio_attn1.to_q'),
              **pair('diffusion_model.transformer_blocks.2.video_to_audio_attn.to_q'),
              **pair('diffusion_model.transformer_blocks.2.attn1.to_q'),
              **pair('diffusion_model.adaln_single.linear')}
    result = classify_header(header, 'LTXV2_5')
    assert result['family_candidates'] == ['ltx']
    assert result['observed_groups']['ltx.transformer'] == {'indices': [2, 47], 'observed_count': 2, 'total_depth': None}
    assert result['version_status'] == 'unproven'
    assert result['modality_tensor_counts'] == {'audio': 2, 'cross_stream': 2, 'other': 2, 'video': 2}
    assert result['export_verified'] is False and result['architecture_verified'] is False


def test_video_only_transformer_pattern_is_explicitly_ambiguous():
    result = classify_header(pair('transformer_blocks.1.attn1.to_q'))
    assert any('other architectures' in reason for reason in result['limitations'])
    assert result['version_status'] == 'unproven'


def test_h3_main_refiner_and_unclassified_modules_are_retained():
    header = {**pair('lora_unet_blocks_49_attn_qkv_proj', 'up_down'),
              **pair('diffusion_model.token_refiner.blocks.1.attn.qkv_proj'),
              **pair('diffusion_model.final_layer.video_out')}
    result = classify_header(header, 'MiniMax-H3')
    assert result['family_candidates'] == ['h3']
    assert result['observed_groups']['h3.main']['indices'] == [49]
    assert result['observed_groups']['h3.refiner']['indices'] == [1]
    assert result['tensor_count'] == 6
    assert result['mixed_pair_formats'] is True
    assert result['incomplete_pair_count'] == 0
    assert any('FL2VA versus Ref2VA' in reason for reason in result['limitations'])


def test_generic_mlp_pattern_does_not_establish_h3_identity():
    result = classify_header(pair('blocks.0.mlp.fc1'))
    assert result['family_candidates'] == ['h3']
    assert any('not unique to H3' in reason for reason in result['limitations'])
    assert result['architecture_verified'] is False


def test_flux_1_2_ambiguity_and_folder_mismatch_are_not_relabelled():
    result = classify_header({**pair('diffusion_model.double_blocks.0.img_attn.proj'),
                              '__metadata__': {'ss_base_model_version': 'flux2_klein_9b'}}, 'LTXV2')
    assert result['family_candidates'] == ['flux']
    assert result['metadata']['claims']['ss_base_model_version'] == 'flux2_klein_9b'
    assert result['metadata']['trust'] == 'self_declared'
    assert result['disagreements'] == ['Folder hint disagrees with observed family-like key topology.']
    assert result['version_status'] == 'unproven'
    assert any('FLUX.1 from FLUX.2' in reason for reason in result['limitations'])


def test_conflicting_metadata_and_mixed_topology_remain_visible():
    result = classify_header({**pair('transformer_blocks.0.audio_attn1.to_q'),
                              **pair('double_blocks.0.img_attn.proj'),
                              '__metadata__': {'ss_base_model_version': 'minimax_h3', 'modelspec.architecture': 'flux/lora'}})
    assert result['family_candidates'] == ['flux', 'ltx']
    assert len(result['disagreements']) == 2
    assert any('Multiple family-like' in reason for reason in result['limitations'])


def test_metadata_alone_cannot_establish_identity_and_unknown_formats_are_visible():
    result = classify_header({'other.lokr_w1': {'shape': [2, 2]},
                              '__metadata__': {'ss_base_model_version': 'ltx-2.5'}})
    assert result['family_candidates'] == []
    assert result['adapter_formats'] == {'unsupported': 1}
    assert result['unsupported_tensor_examples'] == ['other.lokr_w1']
    assert result['metadata']['claim_families'] == ['ltx']


def test_incomplete_and_mixed_pairs_are_not_silently_treated_as_supported():
    header = pair('blocks.0.attn.qkv_proj')
    header.update(pair('blocks.0.attn.qkv_proj', 'up_down'))
    header['blocks.1.attn.qkv_proj.alpha'] = {'shape': []}
    result = classify_header(header)
    assert result['mixed_module_format_count'] == 1
    assert result['incomplete_pair_count'] == 1
    assert result['export_verified'] is False


def test_include_metadata_is_backwards_compatible_and_reads_header_once(tmp_path, monkeypatch):
    path, raw = write_header(tmp_path, {**pair('transformer_blocks.0.attn1.to_q'), '__metadata__': {'model_version': '2.5.0'}})
    legacy_tensors, legacy_identity = read_header(path)
    assert '__metadata__' not in legacy_tensors
    calls, reads = [], []
    original = Path.open

    class Tracked:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            assert size >= 0
            assert self.stream.tell() + size <= 8 + len(raw)
            return self.stream.read(size)

    def tracked_open(actual, *args, **kwargs):
        calls.append(actual)
        return Tracked(original(actual, *args, **kwargs))

    monkeypatch.setattr(Path, 'open', tracked_open)
    result = identify_file(path)
    assert reads == [8, len(raw)] and calls == [path]
    assert result['metadata']['claims'] == {'model_version': '2.5.0'}
    assert result['file_identity'] == {**legacy_identity, 'basis': 'header_stat'}
    assert result['file_identity']['header_bytes_read'] == 8 + len(raw)


@pytest.mark.parametrize('metadata', [None, [], {'model_version': 2.5}, {'config': {'nested': True}}])
def test_metadata_validation_only_changes_explicit_metadata_reader_mode(tmp_path, metadata):
    path, _ = write_header(tmp_path, {**pair('blocks.0.attn.qkv_proj'), '__metadata__': metadata})
    assert read_header(path)[0]  # Preserve the historical FLUX reader's behaviour.
    with pytest.raises(CoverageError, match='metadata'):
        identify_file(path)


@pytest.mark.parametrize('problem', ['oversize', 'truncated', 'overlap', 'gap', 'shape', 'duplicate', 'nonfloat'])
def test_structurally_invalid_or_unsupported_headers_do_not_gain_identity(tmp_path, problem):
    header = pair('transformer_blocks.0.attn1.to_q')
    path, _ = write_header(tmp_path, header)
    if problem == 'oversize':
        path.write_bytes(struct.pack('<Q', MAX_HEADER_BYTES + 1))
    elif problem == 'truncated':
        path.write_bytes(struct.pack('<Q', 100) + b'{}')
    elif problem == 'duplicate':
        raw = b'{"x":{},"x":{}}'
        path.write_bytes(struct.pack('<Q', len(raw)) + raw)
    else:
        entries = list(header.values())
        entries[0]['data_offsets'], entries[1]['data_offsets'] = [0, 16], [16, 32]
        if problem == 'overlap': entries[1]['data_offsets'] = [0, 16]
        if problem == 'gap': entries[1]['data_offsets'] = [20, 36]
        if problem == 'shape': entries[0]['shape'] = [True, 2]
        if problem == 'nonfloat': entries[0]['dtype'] = 'I8'
        write_header(tmp_path, header, bytes(36 if problem == 'gap' else 32))
    with pytest.raises(CoverageError):
        identify_file(path)


def test_long_claim_is_bounded_and_labelled_truncated():
    result = classify_header({**pair('blocks.0.attn.qkv_proj'), '__metadata__': {'ss_network_module': 'x' * 5000}})
    assert len(result['metadata']['claims']['ss_network_module']) == 4096
    assert result['metadata']['claim_values_truncated'] is True


def test_unreasonably_large_block_index_is_unclassified_instead_of_crashing():
    result = classify_header(pair('transformer_blocks.' + '9' * 5000 + '.audio_attn1.to_q'))
    assert result['family_candidates'] == []
    assert result['unclassified_target_tensor_count'] == 2


@pytest.mark.parametrize('inside', [True, False])
def test_cli_accepts_one_in_root_header_and_rejects_outside_paths(tmp_path, inside):
    root = tmp_path / 'library'
    root.mkdir()
    path, _ = write_header(root if inside else tmp_path, pair('blocks.0.attn.qkv_proj'))
    script = Path(__file__).resolve().parents[3] / 'tools' / 'identify_lora.py'
    result = subprocess.run([sys.executable, '-B', str(script), str(path), '--root', str(root)],
                            capture_output=True, text=True, timeout=10)
    output = json.loads(result.stdout)
    assert result.returncode == (0 if inside else 1)
    assert output['status'] == ('observations_only' if inside else 'unavailable')
    assert output['tensor_payload_read'] is False and output['export_verified'] is False
