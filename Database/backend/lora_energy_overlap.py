from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

ROLE_HIERARCHY: Tuple[str, ...] = (
    "character",
    "style",
    "clothing",
    "environment",
    "utility",
    "other",
)

# Hard-coded, non-user-editable role target caps.
ROLE_BUDGETS: Dict[str, float] = {
    "character": 0.35,
    "style": 0.25,
    "clothing": 0.15,
    "environment": 0.10,
    "utility": 0.08,
    "other": 0.07,
}

# Deterministic canonicalization for folder-derived roles that are not part of the
# budgeted hierarchy.
# NOTE: derive_role_from_path can return "pose" for "04 - Action" style folders.
# We map it to "utility" to preserve its intent without inventing a new hierarchy tier.
ROLE_CANONICAL_MAP: Dict[str, str] = {
    "pose": "utility",
    "action": "utility",
}

OVERLAP_THRESHOLD = 0.85


@dataclass(frozen=True)
class LoRAEnergyInput:
    stable_id: str
    role: str
    block_weights: List[float]
    raw_strength_factor: float


@dataclass(frozen=True)
class LoRAEnergyMetrics:
    stable_id: str
    role: str
    raw_strength_factor: float
    energy_blocks: List[float]
    total_energy: float
    normalized_energy_vector: List[float]


def canonicalize_role(role: str) -> str:
    """Return a deterministic, budget-compatible role.

    Folder-derived roles are mandatory input at the API boundary, but this module
    defensively normalizes and canonicalizes to protect overlap math and keep
    role budgets deterministic.
    """
    raw = (role or "").strip().lower()
    if not raw:
        return "other"
    raw = ROLE_CANONICAL_MAP.get(raw, raw)
    return raw if raw in ROLE_BUDGETS else "other"


def compute_lora_energy_metrics(entry: LoRAEnergyInput) -> LoRAEnergyMetrics:
    energy_blocks = [
        abs(float(weight)) * abs(float(entry.raw_strength_factor))
        for weight in entry.block_weights
    ]
    total_energy = sum(energy_blocks)
    if total_energy == 0.0:
        normalized = [0.0 for _ in energy_blocks]
    else:
        normalized = [value / total_energy for value in energy_blocks]

    return LoRAEnergyMetrics(
        stable_id=entry.stable_id,
        role=canonicalize_role(entry.role),
        raw_strength_factor=float(entry.raw_strength_factor),
        energy_blocks=energy_blocks,
        total_energy=total_energy,
        normalized_energy_vector=normalized,
    )


def dot_overlap(left: List[float], right: List[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Normalized energy vectors must have equal length.")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def build_overlap_matrix(metrics: List[LoRAEnergyMetrics]) -> Dict[str, Dict[str, float]]:
    matrix: Dict[str, Dict[str, float]] = {}
    for i, left in enumerate(metrics):
        row: Dict[str, float] = {}
        for j, right in enumerate(metrics):
            if (
                j < i
                and right.stable_id in matrix
                and left.stable_id in matrix[right.stable_id]
            ):
                row[right.stable_id] = matrix[right.stable_id][left.stable_id]
            else:
                row[right.stable_id] = dot_overlap(
                    left.normalized_energy_vector, right.normalized_energy_vector
                )
        matrix[left.stable_id] = row
    return matrix


def allocate_strengths_with_role_budget_and_overlap(
    metrics: List[LoRAEnergyMetrics],
    *,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> Dict[str, float]:
    if not metrics:
        return {}

    total_requested_abs_strength = sum(abs(m.raw_strength_factor) for m in metrics)
    if total_requested_abs_strength == 0.0:
        return {m.stable_id: 0.0 for m in metrics}

    by_role: Dict[str, List[LoRAEnergyMetrics]] = {role: [] for role in ROLE_HIERARCHY}
    for m in metrics:
        by_role[m.role if m.role in by_role else "other"].append(m)

    base_allocations: Dict[str, float] = {}
    for role in ROLE_HIERARCHY:
        role_items = by_role[role]
        if not role_items:
            continue

        role_cap = ROLE_BUDGETS[role] * total_requested_abs_strength
        role_energy_total = sum(item.total_energy for item in role_items)
        if role_energy_total == 0.0:
            for item in role_items:
                base_allocations[item.stable_id] = 0.0
            continue

        role_demand = sum(abs(item.raw_strength_factor) for item in role_items)
        allocatable = min(role_cap, role_demand)

        for item in role_items:
            share = item.total_energy / role_energy_total
            base_allocations[item.stable_id] = allocatable * share

    # Overlap matrix must be built once deterministically.
    overlap = build_overlap_matrix(metrics)

    corrected_abs: Dict[str, float] = {}
    for role in ROLE_HIERARCHY:
        role_items = by_role[role]
        if not role_items:
            continue

        for item in role_items:
            max_overlap = 0.0
            for peer in role_items:
                if peer.stable_id == item.stable_id:
                    continue
                max_overlap = max(max_overlap, overlap[item.stable_id][peer.stable_id])

            factor = 1.0
            if max_overlap > overlap_threshold and max_overlap > 0.0:
                factor = overlap_threshold / max_overlap

            corrected_abs[item.stable_id] = base_allocations.get(item.stable_id, 0.0) * factor

    signed: Dict[str, float] = {}
    for item in metrics:
        signed[item.stable_id] = corrected_abs.get(item.stable_id, 0.0) * (
            -1.0 if item.raw_strength_factor < 0 else 1.0
        )

    return signed
