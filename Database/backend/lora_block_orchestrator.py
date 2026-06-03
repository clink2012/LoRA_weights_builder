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

MIN_SOFTENING_PASSES = 20
SOFTENING_PASSES_PER_PAIR_BLOCK = 4
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
GroupViolationScore = Tuple[float, int, float]
PairAdjustmentCandidate = Tuple[str, Tuple[int, ...], float, float, List[float]]
WorstPair = Tuple[float, LoraBlockOrchestratorInput, LoraBlockOrchestratorInput]


@dataclass(frozen=True)
class PairAdjustment:
    target_id: str
    peer_id: str
    changed_indices: Tuple[int, ...]
    initial_overlap: float
    final_overlap: float
    target_weights: List[float]
    score: Tuple[float, int, float, int, float, str, str, str, Tuple[int, ...]]


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


def _candidate_pair_overlap(
    *,
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
    adjusted_weights: Dict[str, List[float]],
    target_id: str,
    target_weights: List[float],
) -> float:
    left_weights = target_weights if target_id == left.stable_id else adjusted_weights[left.stable_id]
    right_weights = target_weights if target_id == right.stable_id else adjusted_weights[right.stable_id]
    return _overlap_for_vectors(left, right, left_weights, right_weights)


def _build_pair_adjustment_candidates(
    *,
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
    adjusted_weights: Dict[str, List[float]],
    target: LoraBlockOrchestratorInput,
    block_index: int,
    overlap_threshold: float,
) -> List[PairAdjustmentCandidate]:
    initial_overlap = _overlap_for_vectors(
        left,
        right,
        adjusted_weights[left.stable_id],
        adjusted_weights[right.stable_id],
    )
    if initial_overlap <= overlap_threshold or initial_overlap <= 0.0:
        return []

    original_weights = list(adjusted_weights[target.stable_id])
    original_value = original_weights[block_index]
    if abs(original_value) <= 0.0:
        return []

    candidates: List[PairAdjustmentCandidate] = []

    # Always score the fully zeroed block when it improves the selected pair.
    # A threshold-tight retained value may be locally nicer for this pair while
    # still worsening another group member, so group-level scoring needs both
    # options in its candidate set.
    zero_weights = list(original_weights)
    zero_weights[block_index] = 0.0
    zero_overlap = _candidate_pair_overlap(
        left=left,
        right=right,
        adjusted_weights=adjusted_weights,
        target_id=target.stable_id,
        target_weights=zero_weights,
    )
    if zero_overlap < initial_overlap - OVERLAP_EPSILON:
        candidates.append(
            (
                target.stable_id,
                (block_index,),
                initial_overlap,
                zero_overlap,
                zero_weights,
            )
        )

    if zero_overlap > overlap_threshold:
        return candidates

    # Binary-search the highest retained value for this block that keeps the
    # measured pair cosine overlap at or below the threshold.
    retained_weights = list(original_weights)
    low = 0.0
    high = 1.0
    for _ in range(40):
        mid = (low + high) / 2.0
        retained_weights[block_index] = original_value * mid
        candidate_overlap = _candidate_pair_overlap(
            left=left,
            right=right,
            adjusted_weights=adjusted_weights,
            target_id=target.stable_id,
            target_weights=retained_weights,
        )
        if candidate_overlap <= overlap_threshold:
            low = mid
        else:
            high = mid

    retained_weights[block_index] = original_value * low
    final_overlap = _candidate_pair_overlap(
        left=left,
        right=right,
        adjusted_weights=adjusted_weights,
        target_id=target.stable_id,
        target_weights=retained_weights,
    )
    if final_overlap < initial_overlap - OVERLAP_EPSILON and (
        not candidates
        or abs(retained_weights[block_index] - zero_weights[block_index]) > OVERLAP_EPSILON
    ):
        candidates.append(
            (
                target.stable_id,
                (block_index,),
                initial_overlap,
                final_overlap,
                retained_weights,
            )
        )

    return candidates


def _rank_pair_candidate(
    candidate: PairAdjustmentCandidate,
    target: LoraBlockOrchestratorInput,
    *,
    overlap_threshold: float,
) -> Tuple[int, float, float]:
    _target_id, _changed_indices, _initial_overlap, final_overlap, target_weights = candidate
    retained_energy = _total_energy(target, target_weights)
    if final_overlap <= overlap_threshold + OVERLAP_EPSILON:
        return 0, -retained_energy, final_overlap
    return 1, final_overlap, -retained_energy


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

    # Reduce directionally important shared blocks first. This is deliberately
    # not uniform whole-vector scaling because cosine/L2 overlap is unchanged by
    # uniform scaling of one vector.
    contribution_by_index = sorted(
        range(len(adjusted_weights[target.stable_id])),
        key=lambda idx: (-(abs(left_weights[idx]) * abs(right_weights[idx])), idx),
    )

    for idx in contribution_by_index:
        candidates = _build_pair_adjustment_candidates(
            left=left,
            right=right,
            adjusted_weights=adjusted_weights,
            target=target,
            block_index=idx,
            overlap_threshold=overlap_threshold,
        )
        if not candidates:
            continue

        target_id, changed_indices, _initial, final_overlap, target_weights = min(
            candidates,
            key=lambda item: _rank_pair_candidate(
                item,
                target,
                overlap_threshold=overlap_threshold,
            ),
        )
        adjusted_weights[target_id] = target_weights
        return target_id, list(changed_indices), initial_overlap, final_overlap

    return None, [], initial_overlap, initial_overlap


def _weights_for_group_score(
    entry: LoraBlockOrchestratorInput,
    adjusted_weights: Dict[str, List[float]],
    override: Optional[Tuple[str, List[float]]],
) -> List[float]:
    if override is not None and entry.stable_id == override[0]:
        return override[1]
    return adjusted_weights[entry.stable_id]


def _group_violation_score(
    group: List[LoraBlockOrchestratorInput],
    adjusted_weights: Dict[str, List[float]],
    *,
    overlap_threshold: float,
    override: Optional[Tuple[str, List[float]]] = None,
) -> GroupViolationScore:
    max_violation = 0.0
    violation_count = 0
    total_violation = 0.0

    for left_pos, left in enumerate(group):
        for right in group[left_pos + 1:]:
            overlap = _overlap_for_vectors(
                left,
                right,
                _weights_for_group_score(left, adjusted_weights, override),
                _weights_for_group_score(right, adjusted_weights, override),
            )
            violation = overlap - overlap_threshold
            if violation <= OVERLAP_EPSILON:
                continue
            max_violation = max(max_violation, violation)
            violation_count += 1
            total_violation += violation

    return max_violation, violation_count, total_violation


def _score_not_worse(
    candidate: GroupViolationScore,
    current: GroupViolationScore,
) -> bool:
    if candidate[0] < current[0] - OVERLAP_EPSILON:
        return True
    if candidate[0] > current[0] + OVERLAP_EPSILON:
        return False
    if candidate[1] != current[1]:
        return candidate[1] < current[1]
    return candidate[2] <= current[2] + OVERLAP_EPSILON


def _choose_best_group_adjustment(
    group: List[LoraBlockOrchestratorInput],
    adjusted_weights: Dict[str, List[float]],
    *,
    overlap_threshold: float,
) -> Optional[PairAdjustment]:
    current_score = _group_violation_score(
        group,
        adjusted_weights,
        overlap_threshold=overlap_threshold,
    )
    if current_score[1] == 0:
        return None

    best: Optional[PairAdjustment] = None

    for left_pos, left in enumerate(group):
        for right in group[left_pos + 1:]:
            left_weights = adjusted_weights[left.stable_id]
            right_weights = adjusted_weights[right.stable_id]
            initial_overlap = _overlap_for_vectors(left, right, left_weights, right_weights)
            if initial_overlap <= overlap_threshold + OVERLAP_EPSILON:
                continue

            # Evaluate both peers instead of only the lower-energy target. In
            # multi-LoRA groups, the lower-energy member can be involved in
            # several remaining violations; editing the other side may be the
            # only group-improving move. When group scores tie, keep the older
            # deterministic lower-energy/stable-id preference.
            preferred_target = _choose_adjustment_target(left, right, left_weights, right_weights)
            targets = sorted(
                (left, right),
                key=lambda entry: (0 if entry.stable_id == preferred_target.stable_id else 1, entry.stable_id),
            )
            for target in targets:
                target_preference = 0 if target.stable_id == preferred_target.stable_id else 1
                contribution_by_index = sorted(
                    range(len(adjusted_weights[target.stable_id])),
                    key=lambda idx: (-(abs(left_weights[idx]) * abs(right_weights[idx])), idx),
                )

                for idx in contribution_by_index:
                    candidates = _build_pair_adjustment_candidates(
                        left=left,
                        right=right,
                        adjusted_weights=adjusted_weights,
                        target=target,
                        block_index=idx,
                        overlap_threshold=overlap_threshold,
                    )
                    for candidate in candidates:
                        target_id, changed_indices, pair_initial, pair_final, target_weights = candidate
                        candidate_score = _group_violation_score(
                            group,
                            adjusted_weights,
                            overlap_threshold=overlap_threshold,
                            override=(target_id, target_weights),
                        )
                        if not _score_not_worse(candidate_score, current_score):
                            continue

                        peer_id = right.stable_id if target_id == left.stable_id else left.stable_id
                        retained_energy = _total_energy(target, target_weights)
                        ranking_score = (
                            candidate_score[0],
                            candidate_score[1],
                            candidate_score[2],
                            target_preference,
                            -retained_energy,
                            left.stable_id,
                            right.stable_id,
                            target_id,
                            changed_indices,
                        )

                        adjustment = PairAdjustment(
                            target_id=target_id,
                            peer_id=peer_id,
                            changed_indices=changed_indices,
                            initial_overlap=pair_initial,
                            final_overlap=pair_final,
                            target_weights=target_weights,
                            score=ranking_score,
                        )
                        if best is None or adjustment.score < best.score:
                            best = adjustment

    return best


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


def _max_softening_passes_for_group(
    group: List[LoraBlockOrchestratorInput],
    *,
    min_passes: int,
) -> int:
    if len(group) < 2:
        return 0

    pair_count = len(group) * (len(group) - 1) // 2
    block_count = max(1, len(group[0].block_weights))
    scaled_passes = pair_count * block_count * SOFTENING_PASSES_PER_PAIR_BLOCK
    return max(min_passes, scaled_passes)


def _format_softening_note(
    *,
    role: str,
    peer_id: str,
    initial_overlap: float,
    final_overlap: float,
    overlap_threshold: float,
    changed_indices: Tuple[int, ...],
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
        f"adjusted_blocks={list(changed_indices)}, pass={pass_index}."
    )


def _soften_same_role_overlaps(
    inputs: List[LoraBlockOrchestratorInput],
    adjusted_weights: Dict[str, List[float]],
    notes_by_id: Dict[str, List[str]],
    *,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    max_passes: int = MIN_SOFTENING_PASSES,
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
        pass_budget = _max_softening_passes_for_group(group, min_passes=max_passes)
        for pass_index in range(1, pass_budget + 1):
            if _group_violation_score(
                group,
                adjusted_weights,
                overlap_threshold=overlap_threshold,
            )[1] == 0:
                break

            adjustment = _choose_best_group_adjustment(
                group,
                adjusted_weights,
                overlap_threshold=overlap_threshold,
            )
            if adjustment is None:
                remaining = _find_worst_violating_pair(
                    group,
                    adjusted_weights,
                    overlap_threshold=overlap_threshold,
                )
                if remaining is not None:
                    remaining_overlap, left, right = remaining
                    note = (
                        f"Phase 8.5 same-role ({role}) block overlap softening stopped best-effort: "
                        f"pair {left.stable_id}/{right.stable_id} remains overlap={remaining_overlap:.4f} "
                        f"above threshold={overlap_threshold:.4f}; no group-improving block adjustment was found."
                    )
                    notes_by_id[left.stable_id].append(note)
                    notes_by_id[right.stable_id].append(note)
                break

            passes_used = pass_index
            adjusted_weights[adjustment.target_id] = list(adjustment.target_weights)
            notes_by_id[adjustment.target_id].append(
                _format_softening_note(
                    role=role,
                    peer_id=adjustment.peer_id,
                    initial_overlap=adjustment.initial_overlap,
                    final_overlap=adjustment.final_overlap,
                    overlap_threshold=overlap_threshold,
                    changed_indices=adjustment.changed_indices,
                    pass_index=pass_index,
                )
            )

        remaining = _find_worst_violating_pair(
            group,
            adjusted_weights,
            overlap_threshold=overlap_threshold,
        )
        if remaining is not None:
            remaining_overlap, left, right = remaining
            note = (
                f"Phase 8.5 same-role ({role}) block overlap softening stopped best-effort "
                f"after {passes_used} pass(es), budget={pass_budget}: pair {left.stable_id}/{right.stable_id} "
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
