from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from lora_energy_overlap import ROLE_HIERARCHY, canonicalize_role


@dataclass(frozen=True)
class LoRARolePolicy:
    role: str
    priority: int
    default_model_strength: float
    default_clip_strength: float
    intent_label: str
    protects_identity: bool = False
    preserves_composition: bool = False
    treat_as_flavour: bool = False


ROLE_POLICIES: Dict[str, LoRARolePolicy] = {
    "character": LoRARolePolicy(
        role="character",
        priority=100,
        default_model_strength=0.90,
        default_clip_strength=0.70,
        intent_label="identity anchor",
        protects_identity=True,
    ),
    "style": LoRARolePolicy(
        role="style",
        priority=70,
        default_model_strength=0.60,
        default_clip_strength=0.40,
        intent_label="look / flavour",
        treat_as_flavour=True,
    ),
    "clothing": LoRARolePolicy(
        role="clothing",
        priority=75,
        default_model_strength=0.65,
        default_clip_strength=0.45,
        intent_label="outfit detail",
    ),
    "environment": LoRARolePolicy(
        role="environment",
        priority=50,
        default_model_strength=0.45,
        default_clip_strength=0.30,
        intent_label="scene context",
        preserves_composition=True,
    ),
    "utility": LoRARolePolicy(
        role="utility",
        priority=40,
        default_model_strength=0.35,
        default_clip_strength=0.00,
        intent_label="helper / detail / pose",
        preserves_composition=True,
    ),
    "other": LoRARolePolicy(
        role="other",
        priority=20,
        default_model_strength=0.30,
        default_clip_strength=0.00,
        intent_label="unknown intent",
    ),
}


def get_role_policy(role: str) -> LoRARolePolicy:
    """Return the deterministic advisory policy for a folder-derived role.

    This is an advisory layer only. It does not override scanned tensor facts,
    folder-derived role, explicit user settings, or the overlap engine.
    """

    canonical_role = canonicalize_role(role)
    return ROLE_POLICIES[canonical_role]


def list_role_policies() -> Tuple[LoRARolePolicy, ...]:
    """Return policies in canonical role hierarchy order."""

    return tuple(ROLE_POLICIES[role] for role in ROLE_HIERARCHY)
