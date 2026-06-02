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


def _soften_same_role_overlaps(
    inputs: List[LoraBlockOrchestratorInput],
    adjusted_weights: Dict[str, List[float]],
    notes_by_id: Dict[str, List[str]],
    *,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> None:
    by_id = {entry.stable_id: entry for entry in inputs}
    ordered_ids = sorted(by_id)

    for left_pos, left_id in enumerate(ordered_ids):
        left = by_id[left_id]
        for right_id in ordered_ids[left_pos + 1:]:
            right = by_id[right_id]
            should_adjust, role = _same_adjustable_block_space(left, right)
            if not should_adjust:
                continue

            left_weights = adjusted_weights[left.stable_id]
            right_weights = adjusted_weights[right.stable_id]
            overlap = _overlap_for_vectors(left, right, left_weights, right_weights)
            if overlap <= overlap_threshold or overlap <= 0.0:
                continue

            scale_factor = overlap_threshold / overlap
            left_changed = False
            right_changed = False

            for idx, (left_value, right_value) in enumerate(zip(left_weights, right_weights)):
                if abs(left_value) <= 0.0 or abs(right_value) <= 0.0:
                    continue

                left_abs = abs(left_value)
                right_abs = abs(right_value)
                if left_abs > right_abs:
                    right_weights[idx] = right_value * scale_factor
                    right_changed = True
                elif right_abs > left_abs:
                    left_weights[idx] = left_value * scale_factor
                    left_changed = True
                else:
                    # Deterministic tie-break: stable_id lexical order keeps the
                    # first item intact and softens the later one.
                    right_weights[idx] = right_value * scale_factor
                    right_changed = True

            detail = (
                f"Same-role ({role}) block overlap softening applied against "
                f"{{peer}}: overlap={overlap:.4f}, scale={scale_factor:.4f}."
            )
            if left_changed:
                notes_by_id[left.stable_id].append(detail.format(peer=right.stable_id))
            if right_changed:
                notes_by_id[right.stable_id].append(detail.format(peer=left.stable_id))


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
