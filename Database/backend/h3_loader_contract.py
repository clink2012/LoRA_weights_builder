"""Offline candidate contract only; not an enabled H3 exporter or target resolver.

The older selective loader indexes 50 main blocks with non-main weights LAST.
The preferred maintainer candidate separates main, refiner and other controls.
Both differ from the pinned FLUX Inspire contract.
No architecture, checkpoint, format or visual compatibility is inferred here.
"""
from __future__ import annotations

from decimal import Decimal
import math
from typing import Sequence

LOADER_REVISION = "47d5962a651e61a39afcf06c7cf26614454c5fe7"
LOADER_SHA256 = "0a0cac0e3d3de4be0128eddbd1f2803045e2f1542d3219232e4a0c1a6468ed45"
SLOT_LABELS = tuple([f"MAIN {i}" for i in range(50)] + ["OTHER"])
MAINTAINER_REVISION = "4e6379770e7f4fc48651b76d03284a7eee7df8a4"
MAINTAINER_SHA256 = "3d57625d2fb907bbbb8902d293287df7db5d721b4032a544eaa1a971a82f29d8"


def candidate_vector(block_weights: Sequence[float], *, other_weight: float,
                     loader_source_sha256: str) -> dict:
    """Describe a strict research vector, without granting copy/export readiness.

    Require all slots, an explicit OTHER value and the characterized source hash.
    In particular, never rely on the loader's malformed-input fallback, implicit
    OTHER=1, sparse compression or rounded return string.
    """
    if loader_source_sha256 != LOADER_SHA256:
        raise ValueError("The H3 candidate loader source is not the characterized version.")
    raw = [*block_weights, other_weight]
    if len(raw) != len(SLOT_LABELS) or any(isinstance(value, (bool, str)) for value in raw):
        raise ValueError("H3 needs 50 numeric main-block values and explicit OTHER.")
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("H3 candidate values must be finite numbers.") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("H3 candidate values must be finite numbers.")
    return {
        "status": "characterized_candidate",
        "export_verified": False,
        "loader_revision": LOADER_REVISION,
        "loader_source_sha256": LOADER_SHA256,
        "slot_labels": list(SLOT_LABELS),
        "slot_values": values,
        "candidate_numeric_csv": ",".join(format(Decimal(str(value)), "f") for value in values),
        "reason": "Synthetic loader behavior only; target mapping and workflow acceptance remain unverified.",
    }


def maintainer_candidate(main_weights: Sequence[float], refiner_weights: Sequence[float], *,
                         main_block_count: int, refiner_block_count: int,
                         other_weight: float, model_strength: float,
                         loader_source_sha256: str) -> dict:
    """Research serialization for independently counted native H3 target stacks.

    Every main/refiner multiplier is an explicit text override. OTHER and outer
    strength are separate widgets. Counts must come from a later target contract;
    this function cannot prove them from an adapter's highest observed index.
    """
    if loader_source_sha256 != MAINTAINER_SHA256:
        raise ValueError("The H3 maintainer source is not the characterized version.")
    if any(type(count) is not int or not 0 <= count <= 1000
           for count in (main_block_count, refiner_block_count)) or main_block_count == 0:
        raise ValueError("Explicit bounded target block counts are required.")
    if len(main_weights) != main_block_count or len(refiner_weights) != refiner_block_count:
        raise ValueError("Include every target main and refiner block, even for sparse adapters.")
    raw = [*main_weights, *refiner_weights, other_weight, model_strength]
    if any(isinstance(value, (bool, str)) for value in raw):
        raise ValueError("H3 strengths must be finite numeric values between -10 and 10.")
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("H3 strengths must be finite numeric values between -10 and 10.") from error
    if any(not math.isfinite(value) or not -10 <= value <= 10 for value in values):
        raise ValueError("H3 strengths must be finite numeric values between -10 and 10.")
    labels = [f"MAIN {i}" for i in range(main_block_count)] + [f"REFINER {i}" for i in range(refiner_block_count)] + ["OTHER"]
    override_labels = [f"blocks.{i}" for i in range(main_block_count)] + [f"refiner.{i}" for i in range(refiner_block_count)]
    return {
        "status": "characterized_candidate", "export_verified": False,
        "loader_revision": MAINTAINER_REVISION, "loader_source_sha256": MAINTAINER_SHA256,
        "slot_labels": labels, "slot_values": values[:-1],
        "block_overrides": "\n".join(f"{label}={format(Decimal(str(value)), 'f')}"
                                    for label, value in zip(override_labels, values[:-2])),
        "blocks_strength": 1., "refiner_strength": 1.,
        "other_strength": values[-2], "strength_model": values[-1],
        "reason": "Synthetic native-loader behavior only; complete source-to-target coverage and workflow acceptance remain unverified.",
    }
