from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_block_orchestrator import (  # noqa: E402
    LoraBlockOrchestratorInput,
    orchestrate_lora_block_payloads,
)


def _input(
    stable_id: str,
    *,
    role: str = "character",
    text_encoder_contributor: bool = False,
    affect_text_encoder: bool = True,
    strength_text_encoder: float = 0.0,
    weights: list[float] | None = None,
) -> LoraBlockOrchestratorInput:
    return LoraBlockOrchestratorInput(
        stable_id=stable_id,
        filename=f"{stable_id}.safetensors",
        role=role,
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        text_encoder_contributor=text_encoder_contributor,
        affect_text_encoder=affect_text_encoder,
        strength_model=1.25,
        strength_text_encoder=strength_text_encoder,
        block_weights=weights or [0.2, 0.4, 0.6],
    )


def test_orchestrator_returns_one_payload_per_lora_in_order() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001"),
        _input("FLX-BBB-002", role="style"),
    ])

    assert [output.stable_id for output in outputs] == ["FLX-AAA-001", "FLX-BBB-002"]
    assert [output.role for output in outputs] == ["character", "style"]


def test_orchestrator_preserves_scanned_block_vectors_for_skeleton() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001", weights=[0.12345, 0.5, 1.0]),
    ])

    assert outputs[0].block_weights == [0.12345, 0.5, 1.0]
    assert outputs[0].block_weights_csv == "0.1235,0.5000,1.0000"
    assert outputs[0].notes


def test_orchestrator_disables_text_encoder_when_no_text_encoder_tensors() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input(
            "FLX-AAA-001",
            text_encoder_contributor=False,
            affect_text_encoder=True,
            strength_text_encoder=2.0,
        ),
    ])

    assert outputs[0].text_encoder_contributor is False
    assert outputs[0].affect_text_encoder is False
    assert outputs[0].strength_text_encoder == 0.0


def test_orchestrator_preserves_text_encoder_strength_when_allowed() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input(
            "FLX-AAA-001",
            text_encoder_contributor=True,
            affect_text_encoder=True,
            strength_text_encoder=0.75,
        ),
    ])

    assert outputs[0].text_encoder_contributor is True
    assert outputs[0].affect_text_encoder is True
    assert outputs[0].strength_text_encoder == 0.75
