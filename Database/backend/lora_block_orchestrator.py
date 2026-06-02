from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lora_composer import weights_to_csv
from lora_energy_overlap import (
    LoRAEnergyInput,
    OVERLAP_THRESHOLD,
    canonicalize_role,
    compute_lora_energy_metrics,
    dot_overlap,
)

MAX_SOFTENING_PASSES = 20
OVERLAP_EPSILON = 1e-6


@dataclass(frozen=True)
class LoraBlockOrchestratorInput:
    stable_id: str
    filename: Optional[str]
    role: str
    base_model_code: Optional[str]
    block_layout: Optional[str]
    text_encoder_contributor: bool
    affect_text_encoder: bool
    strength_model: float
    strength_text_encoder: float
    block_weights: List[float]


@dataclass(frozen=True)
class LoraBlockOrchestratorOutput:
    stable_id: str
    filename: Optional[str]
    role: str
    base_model_code: Optional[str]
    block_layout: Optional[str]
    text_encoder_contributor: bool
    affect_text_encoder: bool
    strength_model: float
    strength_text_encoder: Optional[float]
    block_weights: List[float]
    block_weights_csv: str
    notes: List[str]


AdjustableGroupKey = Tuple[str, str, int]
WorstPair = Tuple[float, LoraBlockOrchestratorInput, LoraBlockOrchestratorInput]


def _same_adjustable_block_space(
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
) -> Tuple[bool, str]:
    left_role = canonicalize_role(left.role)
    right_role = canonicalize_role(right.role)
    if left_role != right_role:
        return False, left_role

    # `other` is intentionally excluded in the first behaviour pass. Unknown role
    # taxonomy is not strong enough evidence for block-vector collision handling.
    if left_role == "other":
        return False, left_role

    if (left.block_layout or "").lower() != (right.block_layout or "").lower():
        return False, left_role

    if len(left.block_weights) != len(right.block_weights):
        return False, left_role

    return True, left_role


def _adjustable_group_key(entry: LoraBlockOrchestratorInput) -> Optional[AdjustableGroupKey]:
    role = canonicalize_role(entry.role)
    if role == "other":
        return None
    return (role, (entry.block_layout or "").lower(), len(entry.block_weights))


def _overlap_for_vectors(
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
    left_weights: List[float],
    right_weights: List[float],
) -> float:
    left_metrics = compute_lora_energy_metrics(
        LoRAEnergyInput(
            stable_id=left.stable_id,
            role=left.role,
            block_weights=left_weights,
            raw_strength_factor=left.strength_model,
        )
    )
    right_metrics = compute_lora_energy_metrics(
        LoRAEnergyInput(
            stable_id=right.stable_id,
            role=right.role,
            block_weights=right_weights,
            raw_strength_factor=right.strength_model,
        )
    )
    return dot_overlap(
        left_metrics.normalized_energy_vector,
        right_metrics.normalized_energy_vector,
    )


def _total_energy(entry: LoraBlockOrchestratorInput, weights: List[float]) -> float:
    return sum(abs(float(value)) for value in weights) * abs(float(entry.strength_model))


def _choose_adjustment_target(
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
    left_weights: List[float],
    right_weights: List[float],
) -> LoraBlockOrchestratorInput:
    left_energy = _total_energy(left, left_weights)
    right_energy = _total_energy(right, right_weights)

    if left_energy < right_energy:
        return left
    if right_energy < left_energy:
        return right

    # Deterministic tie-break: stable_id lexical order keeps the earlier item
    # intact and adjusts the later one.
    return right if right.stable_id > left.stable_id else left


def _reduce_pair_overlap(
    *,
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
    adjusted_weights: Dict[str, List[float]],
    overlap_threshold: float,
) -> Tuple[Optional[str], List[int], float, float]:
    left_weights = adjusted_weights[left.stable_id]
    right_weights = adjusted_weights[right.stable_id]
    initial_overlap = _overlap_for_vectors(left, right, left_weights, right_weights)

    if initial_overlap <= overlap_threshold or initial_overlap <= 0.0:
        return None, [], initial_overlap, initial_overlap

    target = _choose_adjustment_target(left, right, left_weights, right_weights)
    target_weights = adjusted_weights[target.stable_id]

    # Reduce directionally important shared blocks first. This is deliberately
    # not uniform whole-vector scaling because cosine/L2 overlap is unchanged by
    # uniform scaling of one vector.
    contribution_by_index = sorted(
        range(len(target_weights)),
        key=lambda idx: (-(abs(left_weights[idx]) * abs(right_weights[idx])), idx),
    )

    changed_indices: List[int] = []
    current_overlap = initial_overlap

    for idx in contribution_by_index:
        original_value = target_weights[idx]
        if abs(original_value) <= 0.0:
            continue

        # First check whether removing this one block can get us below the
        # threshold. If not, keep it removed only when it still improves overlap
        # and continue to the next strongest shared block.
        target_weights[idx] = 0.0
        zero_overlap = _overlap_for_vectors(left, right, left_weights, right_weights)

        if zero_overlap > overlap_threshold:
            if zero_overlap < current_overlap:
                changed_indices.append(idx)
                current_overlap = zero_overlap
            else:
                target_weights[idx] = original_value
            continue

        # Binary-search the highest retained value for this block that keeps the
        # measured cosine overlap at or below the threshold.
        low = 0.0
        high = 1.0
        for _ in range(40):
            mid = (low + high) / 2.0
            target_weights[idx] = original_value * mid
            candidate_overlap = _overlap_for_vectors(left, right, left_weights, right_weights)
            if candidate_overlap <= overlap_threshold:
                low = mid
            else:
                high = mid

        target_weights[idx] = original_value * low
        current_overlap = _overlap_for_vectors(left, right, left_weights, right_weights)
        changed_indices.append(idx)
        break

    if not changed_indices:
        return None, [], initial_overlap, current_overlap

    return target.stable_id, changed_indices, initial_overlap, current_overlap


def _find_worst_violating_pair(
    group: List[LoraBlockOrchestratorInput],
    adjusted_weights: Dict[str, List[float]],
    *,
    overlap_threshold: float,
) -> Optional[WorstPair]:
    worst: Optional[WorstPair] = None

    for left_pos, left in enumerate(group):
        for right in group[left_pos + 1:]:
            overlap = _overlap_for_vectors(
                left,
                right,
                adjusted_weights[left.stable_id],
                adjusted_weights[right.stable_id],
            )
            if overlap <= overlap_threshold + OVERLAP_EPSILON:
                continue

            if worst is None:
                worst = (overlap, left, right)
                continue

            worst_overlap, worst_left, worst_right = worst
            candidate_pair = (left.stable_id, right.stable_id)
            worst_pair = (worst_left.stable_id, worst_right.stable_id)
            if overlap > worst_overlap + OVERLAP_EPSILON or (
                abs(overlap - worst_overlap) <= OVERLAP_EPSILON
                and candidate_pair < worst_pair
            ):
                worst = (overlap, left, right)

    return worst


def _format_softening_note(
    *,
    role: str,
    peer_id: str,
    initial_overlap: float,
    final_overlap: float,
    overlap_threshold: float,
    changed_indices: List[int],
    pass_index: int,
) -> str:
    if final_overlap <= overlap_threshold + OVERLAP_EPSILON:
        outcome = "reached threshold for this pass"
    else:
        outcome = "reduced overlap; threshold not reached"

    return (
        f"Same-role ({role}) block overlap softening {outcome} against {peer_id}: "
        f"overlap={initial_overlap:.4f}->{final_overlap:.4f}, "
        f"threshold={overlap_threshold:.4f}, "
        f"adjusted_blocks={changed_indices}, pass={pass_index}."
    )


def _soften_same_role_overlaps(
    inputs: List[LoraBlockOrchestratorInput],
    adjusted_weights: Dict[str, List[float]],
    notes_by_id: Dict[str, List[str]],
    *,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    max_passes: int = MAX_SOFTENING_PASSES,
) -> None:
    groups: Dict[AdjustableGroupKey, List[LoraBlockOrchestratorInput]] = {}
    for entry in sorted(inputs, key=lambda item: item.stable_id):
        key = _adjustable_group_key(entry)
        if key is None:
            continue
        groups.setdefault(key, []).append(entry)

    for (role, _layout, _block_count), group in sorted(groups.items(), key=lambda item: item[0]):
        if len(group) < 2:
            continue

        passes_used = 0
        for pass_index in range(1, max_passes + 1):
            worst = _find_worst_violating_pair(
                group,
                adjusted_weights,
                overlap_threshold=overlap_threshold,
            )
            if worst is None:
                break

            passes_used = pass_index
            _worst_overlap, left, right = worst
            target_id, changed_indices, initial_overlap, final_overlap = _reduce_pair_overlap(
                left=left,
                right=right,
                adjusted_weights=adjusted_weights,
                overlap_threshold=overlap_threshold,
            )

            if target_id is None:
                notes_by_id[left.stable_id].append(
                    f"Phase 8.5 same-role ({role}) block overlap softening stopped best-effort: "
                    f"pair {left.stable_id}/{right.stable_id} remains overlap={initial_overlap:.4f} "
                    f"above threshold={overlap_threshold:.4f}; no lower-overlap block adjustment was found."
                )
                notes_by_id[right.stable_id].append(
                    f"Phase 8.5 same-role ({role}) block overlap softening stopped best-effort: "
                    f"pair {left.stable_id}/{right.stable_id} remains overlap={initial_overlap:.4f} "
                    f"above threshold={overlap_threshold:.4f}; no lower-overlap block adjustment was found."
                )
                break

            peer_id = right.stable_id if target_id == left.stable_id else left.stable_id
            notes_by_id[target_id].append(
                _format_softening_note(
                    role=role,
                    peer_id=peer_id,
                    initial_overlap=initial_overlap,
                    final_overlap=final_overlap,
                    overlap_threshold=overlap_threshold,
                    changed_indices=changed_indices,
                    pass_index=pass_index,
                )
            )

            if final_overlap >= initial_overlap - OVERLAP_EPSILON:
                notes_by_id[target_id].append(
                    f"Phase 8.5 same-role ({role}) block overlap softening stopped best-effort: "
                    f"latest adjustment did not materially reduce overlap "
                    f"({initial_overlap:.4f}->{final_overlap:.4f})."
                )
                break

        remaining = _find_worst_violating_pair(
            group,
            adjusted_weights,
            overlap_threshold=overlap_threshold,
        )
        if remaining is not None:
            remaining_overlap, left, right = remaining
            note = (
                f"Phase 8.5 same-role ({role}) block overlap softening stopped best-effort "
                f"after {passes_used} pass(es): pair {left.stable_id}/{right.stable_id} "
                f"remains overlap={remaining_overlap:.4f} above threshold={overlap_threshold:.4f}."
            )
            notes_by_id[left.stable_id].append(note)
            notes_by_id[right.stable_id].append(note)


def orchestrate_lora_block_payloads(
    inputs: List[LoraBlockOrchestratorInput],
) -> List[LoraBlockOrchestratorOutput]:
    """Return one stack-aware payload per selected LoRA.

    This is the first behavioural Phase 8.5 pass. It keeps one output per LoRA
    and applies deterministic same-role, same-layout block-vector softening when
    measured overlap exceeds the fixed overlap threshold. The legacy averaged
    `combined` payload remains outside this module.
    """
    adjusted_weights: Dict[str, List[float]] = {
        entry.stable_id: [float(value) for value in entry.block_weights]
        for entry in inputs
    }
    notes_by_id: Dict[str, List[str]] = {
        entry.stable_id: []
        for entry in inputs
    }

    _soften_same_role_overlaps(inputs, adjusted_weights, notes_by_id)

    outputs: List[LoraBlockOrchestratorOutput] = []
    for entry in inputs:
        block_weights = adjusted_weights[entry.stable_id]
        affect_text_encoder = bool(entry.affect_text_encoder and entry.text_encoder_contributor)
        strength_text_encoder: Optional[float]
        if affect_text_encoder:
            strength_text_encoder = float(entry.strength_text_encoder)
        else:
            strength_text_encoder = 0.0

        notes = list(notes_by_id[entry.stable_id])
        if not notes:
            notes.append(
                "Phase 8.5: no same-role block-vector softening was applied; "
                "per-LoRA block weights are preserved."
            )

        outputs.append(
            LoraBlockOrchestratorOutput(
                stable_id=entry.stable_id,
                filename=entry.filename,
                role=entry.role,
                base_model_code=entry.base_model_code,
                block_layout=entry.block_layout,
                text_encoder_contributor=bool(entry.text_encoder_contributor),
                affect_text_encoder=affect_text_encoder,
                strength_model=float(entry.strength_model),
                strength_text_encoder=strength_text_encoder,
                block_weights=block_weights,
                block_weights_csv=weights_to_csv(block_weights),
                notes=notes,
            )
        )

    return outputs
