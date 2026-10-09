from copy import deepcopy
import math

import pytest

from role_start_policy import POLICY_VERSION, propose_role_start, role_rule
from gentle_balance_policy import PolicyError
from test_gentle_balance_policy import fixture


def inputs(gram, roles, **kwargs):
    metrics, entries = fixture(gram, **kwargs)
    for entry, role in zip(entries, roles):
        entry['role'] = role
    return metrics, entries


def test_four_roles_keep_independent_contributions_and_reduce_only_lighting():
    metrics, entries = inputs([[1]*4 for _ in range(4)], ['person', 'coat', 'shoes', 'lighting'])
    original = deepcopy((metrics, entries))
    result = propose_role_start(metrics, entries)
    assert result['policy_version'] == POLICY_VERSION
    assert [e['priority'] for e in result['entries']] == [2, 2, 2, 0]
    assert [e['values'] for e in result['entries'][:3]] == [[1]*58]*3
    assert result['entries'][3]['values'] == [1] + [.8]*57
    assert result['blocks'][1]['energy_before'] == 16
    assert result['blocks'][1]['energy_after'] == pytest.approx(3.8**2)
    assert (metrics, entries) == original


def test_proposal_is_block_selective_and_does_not_normalize_values_into_weights():
    metrics, entries = inputs([[1, 1], [1, 1]], ['character', 'style'])
    metrics['pairs'][0]['block_inner_products'][2] = 0
    metrics['sources'][1]['block_squared_norms'][3] = 0
    metrics['pairs'][0]['block_inner_products'][3] = 0
    result = propose_role_start(metrics, entries)
    assert result['entries'][1]['values'][1:4] == [.8, 1, 1]
    graph = result['contribution_graphs'][1]
    assert graph['guidance'][1]['state'] == 'suggested'
    assert graph['guidance'][2]['state'] == 'review'
    assert graph['guidance'][3]['state'] == 'inactive'
    assert graph['guidance'][1]['trial_interval']['min'] == .8


def test_measured_graph_uses_actual_magnitudes_and_supporting_strengths():
    result = propose_role_start(*inputs([[144, 0], [0, 9]], ['character', 'clothing'], weights=[.5, 1], strengths=[.8, 1]))
    assert result['changes'] == []
    graph = result['contribution_graphs'][0]
    assert graph['original_norms'] == [12]*58
    assert graph['before_norms'] == pytest.approx([4.8]*58)
    assert graph['proposed_norms'] == graph['before_norms']
    assert graph['fixed_plot_peak'] == 12
    assert graph['guidance'][1]['state'] == 'protected'
    assert graph['guidance'][1]['before_multiplier'] == .5


def test_role_defaults_do_not_claim_to_repair_protected_identity_clothing_clash():
    result = propose_role_start(*inputs([[1, 1], [1, 1]], ['character', 'clothing']))
    assert result['changes'] == []
    assert all(e['values'] == [1]*58 for e in result['entries'])
    assert result['calibrated'] is False
    assert result['image_quality_verified'] is False


def test_owner_override_precedes_role_and_is_retained_in_explanation():
    result = propose_role_start(*inputs([[1, 1], [1, 1]], ['character', 'clothing']), {'1': 0})
    assert result['entries'][1]['values'] == [1] + [.8]*57
    assert result['role_rules'][1]['priority_basis'] == 'owner_override'
    assert result['role_rules'][1]['priority'] == 0


def test_unknown_intent_is_normal_and_does_not_guess_a_profile():
    assert role_rule('custom thing')['default_priority'] == 1
    result = propose_role_start(*inputs([[1, 1], [1, 1]], ['unknown', 'utility']))
    assert result['changes'] == []


def test_signed_multipliers_and_cancellation_keep_original_direction():
    result = propose_role_start(*inputs([[1, 1], [1, 1]], ['character', 'style'], weights=[-1, -1]))
    assert result['entries'][1]['values'][1] == -.8
    assert result['contribution_graphs'][1]['proposed_norms'][1] == .8
    result = propose_role_start(*inputs([[1, 1], [1, 1]], ['character', 'style'], weights=[1, -1]))
    assert result['changes'] == []


@pytest.mark.parametrize('override', [{'missing': 0}, {'1': True}, {'1': 3}, []])
def test_invalid_owner_priorities_are_rejected(override):
    with pytest.raises(PolicyError):
        propose_role_start(*inputs([[1, 1], [1, 1]], ['character', 'style']), override)


@pytest.mark.parametrize('role', ['', None, 3])
def test_missing_role_is_not_silently_invented(role):
    with pytest.raises(PolicyError):
        propose_role_start(*inputs([[1, 1], [1, 1]], ['character', role]))


def test_order_invariance_matches_by_identity_and_keeps_different_norms():
    metrics, entries = inputs([[4, 2], [2, 1]], ['character', 'style'], weights=[.5, 1])
    first = propose_role_start(metrics, entries)
    reversed_metrics, _ = inputs([[1, 2], [2, 4]], ['style', 'character'])
    second = propose_role_start(reversed_metrics, entries[::-1])
    assert {e['stable_id']: e['values'] for e in first['entries']} == {e['stable_id']: e['values'] for e in second['entries']}
    assert first['contribution_graphs'][0]['original_norms'][1] == 2
    assert first['contribution_graphs'][1]['original_norms'][1] == 1


def test_all_zero_graph_has_finite_scale_and_preserves_values():
    result = propose_role_start(*inputs([[0, 0], [0, 0]], ['character', 'style']))
    assert result['changes'] == []
    assert all(g['fixed_plot_peak'] == 0 for g in result['contribution_graphs'])
    assert all(math.isfinite(b['relative_original_norm']) for g in result['contribution_graphs'] for b in g['guidance'])
