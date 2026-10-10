import json
from pathlib import Path
import sys

import pytest

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/h3_native_patches.json').read_text())


def test_registered_patch_counts_do_not_establish_calculation_or_quantized_support():
    assert FIXTURE['registration_strengths'] == [.7, -.2]
    assert FIXTURE['patch_objects_preserved'] and FIXTURE['unknown_target_rejected']
    assert FIXTURE['dimension_mismatch_logs_error_and_returns_unchanged']
    for flag in ('quantized_base_verified', 'native_clone_verified', 'export_verified'):
        assert FIXTURE[flag] is False


@pytest.mark.parametrize('format_name', ['AB', 'up_down', 'AB_default'])
@pytest.mark.parametrize('alpha', [None, 4., -2.])
def test_independent_arithmetic_matches_captured_native_loading_and_signed_scaling(format_name, alpha):
    up, down = FIXTURE['up'], FIXTURE['down']
    product = [[sum(up[i][r] * down[r][j] for r in range(2)) for j in range(3)] for i in range(2)]
    cases = [case for case in FIXTURE['cases'] if case['format'] == format_name and case['alpha'] == alpha]
    assert len(cases) == 5
    for case in cases:
        gain = case['outer'] * case['block'] * (1. if alpha is None else alpha / FIXTURE['rank'])
        for i in range(2):
            assert case['output'][i] == pytest.approx([1. + gain * v for v in product[i]], abs=1e-12)


def test_patch_source_drift_is_rejected_before_source_execution_or_tensor_import(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent / 'fixtures'))
    from capture_h3_native_patches import PINNED, capture
    assert FIXTURE['source_sha256'] == PINNED
    for relative in PINNED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('raise RuntimeError("must never execute")')
    with pytest.raises(ValueError, match='source changed'): capture(tmp_path)
