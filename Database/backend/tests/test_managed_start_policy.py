from copy import deepcopy
import pytest
from managed_start_policy import propose_managed_start
from test_role_start_policy import inputs


def test_character_coat_shoes_and_lighting_have_independent_role_starts():
    metrics, entries = inputs([[1]*4 for _ in range(4)], ['person', 'coat', 'shoes', 'lighting'])
    before = deepcopy((metrics, entries))
    result = propose_managed_start(metrics, entries)
    assert [rule['starting_factor'] for rule in result['role_rules']] == [.9, .65, .65, .6]
    assert [entry['values'][1] for entry in result['entries'][:3]] == [.9, .65, .65]
    assert .48 <= result['entries'][3]['values'][1] < .6
    assert all(entry['values'][0] == 1 for entry in result['entries'])
    assert all(entry['before_values'] == [1]*58 for entry in result['entries'])
    assert result['blocks'][1]['energy_before'] == 16
    assert result['blocks'][1]['energy_after'] < 16
    assert (metrics, entries) == before
    assert result['calibrated'] is False and result['image_quality_verified'] is False


def test_real_block_measurements_determine_additional_adjustments_without_norm_normalization():
    metrics, entries = inputs([[144, 0], [0, 9]], ['character', 'lighting'], weights=[.5, -1], strengths=[.8, 1])
    result = propose_managed_start(metrics, entries)
    assert result['entries'][0]['values'][1] == .45
    assert result['entries'][1]['values'][1] == -.6
    graph = result['contribution_graphs'][0]
    assert graph['original_norms'][1] == 12
    assert graph['before_norms'][1] == pytest.approx(4.8)
    assert graph['proposed_norms'][1] == pytest.approx(4.32)
    assert graph['guidance'][1]['proposed_multiplier'] == .45
    # Alter one slot's real overlap, leaving other slots independent.
    metrics, entries = inputs([[1, 1], [1, 1]], ['character', 'lighting'])
    metrics['pairs'][0]['block_inner_products'][2] = 0
    result = propose_managed_start(metrics, entries)
    assert result['entries'][1]['values'][1] < .6
    assert result['entries'][1]['values'][2] == .6


def test_cancellation_guard_reverts_role_cuts_that_increase_signed_energy():
    metrics, entries = inputs([[1, -1], [-1, 1]], ['character', 'clothing'])
    result = propose_managed_start(metrics, entries)
    assert result['changes'] == []
    assert all(entry['values'] == [1]*58 for entry in result['entries'])
    assert result['blocks'][1]['energy_before'] == result['blocks'][1]['energy_after'] == 0
    assert result['blocks'][1]['reverted']


def test_zero_absent_and_unknown_intent_do_not_acquire_guessed_content():
    metrics, entries = inputs([[1, 0], [0, 1]], ['character', 'unknown-purpose'])
    metrics['sources'][0]['block_squared_norms'][2] = 0
    entries[0]['values'][3] = 0
    result = propose_managed_start(metrics, entries)
    assert result['entries'][0]['values'][1:4] == [.9, 1, 0]
    assert result['entries'][1]['values'] == [1]*58
    assert result['contribution_graphs'][0]['guidance'][2]['state'] == 'inactive'


def test_saved_outer_strengths_participate_in_actual_signed_stack_energy():
    metrics, entries = inputs([[1, 1], [1, 1]], ['character', 'lighting'], strengths=[1, 10])
    result = propose_managed_start(metrics, entries)
    assert result['entries'][1]['values'][1] == .6  # Far below positive-pressure threshold.
    assert result['blocks'][1]['energy_before'] == 121
    assert result['blocks'][1]['energy_after'] == pytest.approx(6.9**2)
