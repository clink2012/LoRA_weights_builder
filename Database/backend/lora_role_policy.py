from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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


@dataclass(frozen=True)
class LoRARoleStrengthRecommendation:
    """Advisory role-aware strength recommendation.

    Phase 8.8b contract:
    - This is recommendation metadata only.
    - It does not mutate requested strengths, corrected strengths, block weights,
      overlap handling, clip enforcement, or composer maths.
    """

    role: str
    requested_model_strength: float
    overlap_corrected_model_strength: float
    role_default_model_strength: float
    recommended_model_strength: float
    requested_clip_strength: float
    role_default_clip_strength: float
    recommended_clip_strength: float
    clip_contributor: bool
    applied_to_math: bool = False
    basis: str = "role_policy_advisory"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "requested_model_strength": self.requested_model_strength,
            "overlap_corrected_model_strength": self.overlap_corrected_model_strength,
            "role_default_model_strength": self.role_default_model_strength,
            "recommended_model_strength": self.recommended_model_strength,
            "requested_clip_strength": self.requested_clip_strength,
            "role_default_clip_strength": self.role_default_clip_strength,
            "recommended_clip_strength": self.recommended_clip_strength,
            "clip_contributor": self.clip_contributor,
            "applied_to_math": self.applied_to_math,
            "basis": self.basis,
        }


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


def build_role_strength_recommendation(
    role: str,
    *,
    requested_model_strength: float,
    overlap_corrected_model_strength: float,
    requested_clip_strength: float = 0.0,
    clip_contributor: bool = False,
) -> LoRARoleStrengthRecommendation:
    """Build advisory role-aware strength targets for a LoRA.

    The recommendation deliberately uses role defaults as the target values while
    preserving both the raw requested model strength and the overlap-corrected
    model strength for later UI/explanation work.
    """

    policy = get_role_policy(role)
    recommended_clip_strength = policy.default_clip_strength if clip_contributor else 0.0

    return LoRARoleStrengthRecommendation(
        role=policy.role,
        requested_model_strength=float(requested_model_strength),
        overlap_corrected_model_strength=float(overlap_corrected_model_strength),
        role_default_model_strength=float(policy.default_model_strength),
        recommended_model_strength=float(policy.default_model_strength),
        requested_clip_strength=float(requested_clip_strength),
        role_default_clip_strength=float(policy.default_clip_strength),
        recommended_clip_strength=float(recommended_clip_strength),
        clip_contributor=bool(clip_contributor),
        applied_to_math=False,
        basis="role_policy_advisory",
    )


def build_role_recommendation_notes(role: str) -> Tuple[str, ...]:
    """Return advisory, non-mutating recommendation notes for a role.

    These notes explain how a role should be considered by later recommendation
    logic. They deliberately do not change strengths or block weights.
    """

    policy = get_role_policy(role)
    notes = [
        f"Phase 8.8: role policy marks this LoRA as {policy.intent_label}; recommendations are advisory only."
    ]

    if policy.protects_identity:
        notes.append(
            "Phase 8.8: identity anchor detected; preserve character influence unless explicit overlap handling requires softening."
        )

    if policy.treat_as_flavour:
        notes.append(
            "Phase 8.8: flavour layer detected; keep it supportive so it does not overpower identity or outfit anchors."
        )

    if policy.preserves_composition:
        notes.append(
            "Phase 8.8: composition-preserving helper detected; prefer lower supportive strength unless the user boosts it."
        )

    if policy.role == "clothing":
        notes.append(
            "Phase 8.8: outfit detail role detected; allow moderate influence while avoiding identity collision."
        )

    if policy.role == "other":
        notes.append(
            "Phase 8.8: unknown role detected; keep conservative defaults until the LoRA intent is clearer."
        )

    return tuple(notes)
