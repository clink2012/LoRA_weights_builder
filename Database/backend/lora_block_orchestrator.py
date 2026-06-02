from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from lora_composer import weights_to_csv


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


def orchestrate_lora_block_payloads(
    inputs: List[LoraBlockOrchestratorInput],
) -> List[LoraBlockOrchestratorOutput]:
    """Return one stack-aware payload per selected LoRA.

    This is the Phase 8.5 skeleton only. It deliberately preserves current
    per-LoRA block vectors so adding the module does not change runtime behaviour.
    Future amendments will add role-aware, block-level adjustment here.
    """
    outputs: List[LoraBlockOrchestratorOutput] = []

    for entry in inputs:
        block_weights = [float(value) for value in entry.block_weights]
        affect_text_encoder = bool(entry.affect_text_encoder and entry.text_encoder_contributor)
        strength_text_encoder: Optional[float]
        if affect_text_encoder:
            strength_text_encoder = float(entry.strength_text_encoder)
        else:
            strength_text_encoder = 0.0

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
                notes=[
                    "Phase 8.5 skeleton: scanned per-LoRA block weights are preserved; "
                    "stack-aware block-vector adjustment is not implemented yet."
                ],
            )
        )

    return outputs
