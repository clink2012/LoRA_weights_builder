from copy import deepcopy
import pytest

from gentle_balance_policy import PolicyError, propose_gentle_balance
from inspire_export import ARCHITECTURE_LABELS


def fixture(gram, *, weights=None, priorities=None, strengths=None):
    count = len(gram)
    def slots(value):
        return [value] * len(ARCHITECTURE_LABELS)
    metrics = {"status": "complete", "engine_version": "effective_native_lora_v1",
               "slot_labels": list(ARCHITECTURE_LABELS), "outer_model_strength_applied": False,
               "block_weights_applied": False,
               "sources": [{"source_index": i, "block_squared_norms": slots(gram[i][i])} for i in range(count)],
               "pairs": [{"left_index": i, "right_index": j, "block_inner_products": slots(gram[i][j])}
                         for i in range(count) for j in range(i + 1, count)]}
    entries = [{"stable_id": str(i), "profile_version_id": "v" + str(i),
                "values": slots((weights or [1] * count)[i]),
                "strength_model": (strengths or [1] * count)[i],
                "priority": (priorities or [2] + [0] * (count - 1))[i]} for i in range(count)]
    return metrics, entries


def test_known_identical_updates_reduce_lower_priority_by_twenty_percent_only():
    args = fixture([[1, 1], [1, 1]])
    originals = deepcopy(args)
    result = propose_gentle_balance(*args)
    assert result["entries"][0]["values"] == [1] * 58
    assert result["entries"][1]["values"] == [1] + [0.8] * 57
    assert result["blocks"][1]["energy_before"] == 4
    assert result["blocks"][1]["energy_after"] == pytest.approx(3.24)
    assert len(result["changes"]) == 57
    assert args == originals


@pytest.mark.parametrize("gram,kwargs", [
    ([[1, -1], [-1, 1]], {}), ([[1, 0], [0, 1]], {}),
    ([[0, 0], [0, 0]], {}), ([[1, 1], [1, 1]], {"priorities": [1, 1]}),
    ([[1, 1], [1, 1]], {"strengths": [1, 0]}),
])
def test_cancellation_orthogonal_zero_and_equal_priority_do_not_trigger(gram, kwargs):
    assert propose_gentle_balance(*fixture(gram, **kwargs))["changes"] == []


def test_signed_values_keep_sign_and_bounds_are_ordered():
    result = propose_gentle_balance(*fixture([[1, 1], [1, 1]], weights=[-1, -1]))
    assert result["entries"][1]["values"][1] == -0.8
    assert result["changes"][0]["min"] == -1
    assert result["changes"][0]["max"] == -0.8


def test_signed_strength_affects_pressure():
    assert propose_gentle_balance(*fixture([[1, 1], [1, 1]], strengths=[1, -1]))["changes"] == []


def test_reducing_cancellation_in_three_way_stack_reverts_whole_block():
    result = propose_gentle_balance(*fixture([[1] * 3] * 3, weights=[1, 1, -1.95], priorities=[2, 0, 2]))
    assert result["changes"] == []
    assert result["blocks"][1]["reverted"] is True
    assert result["blocks"][1]["energy_after"] == pytest.approx(0.0025)


def test_zero_weight_stays_zero_and_default_base_is_preserved():
    args = fixture([[1, 1], [1, 1]])
    args[1][1]["values"][3] = 0
    result = propose_gentle_balance(*args)
    assert result["entries"][1]["values"][0] == 1
    assert result["entries"][1]["values"][3] == 0


def test_unchanged_large_integer_values_remain_exact_including_base():
    metrics, entries = fixture([[0, 0], [0, 0]], weights=[9007199254740993, 1])
    result = propose_gentle_balance(metrics, entries)
    assert result["changes"] == []
    assert result["entries"][0]["before_values"] == entries[0]["values"]
    assert result["entries"][0]["values"] == entries[0]["values"]
    assert type(result["entries"][0]["values"][0]) is int


def test_permuting_sources_does_not_change_proposed_values_by_identity():
    metrics, entries = fixture([[4, 2], [2, 1]], weights=[0.5, 1])
    first = propose_gentle_balance(metrics, entries)
    reordered, _ = fixture([[1, 2], [2, 4]])
    second = propose_gentle_balance(reordered, entries[::-1])
    assert {e["stable_id"]: e["values"] for e in first["entries"]} == {e["stable_id"]: e["values"] for e in second["entries"]}


@pytest.mark.parametrize("mutation", [
    lambda m, e: m.update(status="failed"),
    lambda m, e: m.update(block_weights_applied=True),
    lambda m, e: m["pairs"].clear(),
    lambda m, e: m["pairs"].append(deepcopy(m["pairs"][0])),
    lambda m, e: m["sources"][0]["block_squared_norms"].__setitem__(1, -1),
    lambda m, e: m["pairs"][0]["block_inner_products"].__setitem__(1, 2),
    lambda m, e: e[0].update(priority=True),
    lambda m, e: e[0].update(strength_model=float("nan")),
    lambda m, e: e[0].update(strength_model=1e308),
    lambda m, e: e[0].update(strength_model=10**308),
    lambda m, e: e[1].update(stable_id=e[0]["stable_id"]),
    lambda m, e: e[0]["values"].pop(),
])
def test_incomplete_invalid_or_overflow_inputs_fail_without_proposal(mutation):
    metrics, entries = fixture([[1, 1], [1, 1]])
    mutation(metrics, entries)
    with pytest.raises(PolicyError):
        propose_gentle_balance(metrics, entries)
