"""Uncalibrated, bounded parameter-space experiment; never an image score."""
from copy import deepcopy
import math

from inspire_export import ARCHITECTURE_LABELS

POLICY_VERSION = "gentle_positive_alignment_v1"
MAX_REDUCTION = 0.20
PRESSURE_THRESHOLD = 0.5


class PolicyError(ValueError):
    pass


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError("All measurement and profile values must be finite numbers")
    try:
        if math.isfinite(value):
            return value
    except OverflowError:
        pass
    raise PolicyError("All measurement and profile values must be finite numbers")


def _vector(value):
    if not isinstance(value, list) or len(value) != len(ARCHITECTURE_LABELS):
        raise PolicyError("Every vector must match the complete canonical slot order")
    return [_number(item) for item in value]


def _finite(value):
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise PolicyError("The proposed experiment exceeds finite arithmetic")
    return value


def _energy(coefficients, gram):
    terms = [coefficients[i] * coefficients[j] * gram[i][j]
             for i in range(len(coefficients)) for j in range(len(coefficients))]
    for term in terms:
        _finite(term)
    try:
        value, magnitude = math.fsum(terms), math.fsum(abs(term) for term in terms)
    except (OverflowError, ValueError) as exc:
        raise PolicyError("The proposed experiment exceeds finite arithmetic") from exc
    tolerance = 1e-10 * magnitude
    if value < -tolerance:
        raise PolicyError("The supplied measurements have inconsistent signed energy")
    return max(value, 0.0), tolerance


def propose_gentle_balance(metrics, entries):
    """Inputs are server-owned measured receipts and pinned saved profiles.

    The caller must separately prove current source/selection identity. This
    pure function cannot grant export authority or persist profile versions.
    Priorities are explicit owner choices: 0 flexible, 1 normal, 2 protect.
    """
    if (not isinstance(metrics, dict) or metrics.get("status") != "complete"
            or metrics.get("engine_version") != "effective_native_lora_v1"
            or metrics.get("slot_labels") != list(ARCHITECTURE_LABELS)
            or metrics.get("outer_model_strength_applied") is not False
            or metrics.get("block_weights_applied") is not False):
        raise PolicyError("A complete, unweighted native metrics receipt is required")
    if not isinstance(entries, list) or not 2 <= len(entries) <= 8:
        raise PolicyError("Choose two to eight pinned profiles for an experiment")
    sources, pairs = metrics.get("sources"), metrics.get("pairs")
    if not isinstance(sources, list) or len(sources) != len(entries) or not isinstance(pairs, list):
        raise PolicyError("Measurements must include every selected profile")
    count, identities, versions, vectors, strengths, priorities = len(entries), [], [], [], [], []
    for entry in entries:
        if not isinstance(entry, dict):
            raise PolicyError("Each selection must name a pinned profile and priority")
        for key in ("stable_id", "profile_version_id"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise PolicyError("Each selection must name a pinned profile and priority")
        if entry["stable_id"] in identities:
            raise PolicyError("Repeated LoRAs are not supported by this experiment")
        priority = entry.get("priority")
        if type(priority) is not int or priority not in (0, 1, 2):
            raise PolicyError("Choose an explicit Flexible, Normal or Protect priority")
        identities.append(entry["stable_id"])
        versions.append(entry["profile_version_id"])
        vectors.append(_vector(entry.get("values")))
        strengths.append(_number(entry.get("strength_model")))
        priorities.append(priority)
    diagonals = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or type(source.get("source_index")) is not int or source["source_index"] != index:
            raise PolicyError("Source measurement order must match the selection")
        diagonal = _vector(source.get("block_squared_norms"))
        if any(value < 0 for value in diagonal):
            raise PolicyError("Squared norms cannot be negative")
        diagonals.append(diagonal)
    products = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise PolicyError("Each pair requires indexed block measurements")
        left, right = pair.get("left_index"), pair.get("right_index")
        if type(left) is not int or type(right) is not int or not 0 <= left < right < count or (left, right) in products:
            raise PolicyError("Each measurement pair must appear exactly once")
        products[left, right] = _vector(pair.get("block_inner_products"))
    if len(products) != count * (count - 1) // 2:
        raise PolicyError("Measurements must include every selected pair")
    proposed, changes, blocks = deepcopy(vectors), [], []
    for slot, label in enumerate(ARCHITECTURE_LABELS):
        gram = [[float(diagonals[i][slot] if i == j else products[min(i, j), max(i, j)][slot])
                 for j in range(count)] for i in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                bound = math.sqrt(gram[i][i]) * math.sqrt(gram[j][j])
                if abs(gram[i][j]) > bound * (1 + 1e-10):
                    raise PolicyError("Pair measurements exceed their norm bound")
        # Work in bounded floating arithmetic without changing the exact saved
        # numbers retained in before/proposed vectors for untouched slots.
        coefficients = [_finite(float(strengths[i]) * float(vectors[i][slot])) for i in range(count)]
        before, tolerance = _energy(coefficients, gram)
        energy = [_finite(coefficients[i] * coefficients[i] * gram[i][i]) for i in range(count)]
        factors, interactions = [1.0] * count, []
        for i in range(count):
            for j in range(i + 1, count):
                cross = _finite(2 * coefficients[i] * coefficients[j] * gram[i][j])
                denominator = _finite(energy[i] + energy[j])
                pressure = min(1.0, max(cross, 0.0) / denominator) if denominator else 0.0
                if cross != 0:
                    interactions.append({"left_id": identities[i], "right_id": identities[j],
                                         "signed_cross_term": cross, "positive_pressure": pressure})
                lower = i if priorities[i] < priorities[j] else j if priorities[j] < priorities[i] else None
                if slot and lower is not None:
                    reduction = MAX_REDUCTION * max(0.0, (pressure - PRESSURE_THRESHOLD) / (1 - PRESSURE_THRESHOLD))
                    factors[lower] = min(factors[lower], 1.0 - reduction)
        trial = [_finite(coefficients[i] * factors[i]) for i in range(count)]
        after, trial_tolerance = _energy(trial, gram)
        reverted = after > before + max(tolerance, trial_tolerance)
        if reverted:
            factors, after = [1.0] * count, before
        for i, factor in enumerate(factors):
            if factor == 1 or vectors[i][slot] == 0:
                continue
            value = vectors[i][slot] * factor
            proposed[i][slot] = value
            minimum, maximum = sorted(((1 - MAX_REDUCTION) * vectors[i][slot], vectors[i][slot]))
            changes.append({"stable_id": identities[i], "slot_index": slot, "slot_label": label,
                            "before": vectors[i][slot], "value": value, "min": minimum, "max": maximum,
                            "basis": "Uncalibrated policy trial interval; not a proven safe range",
                            "reason": "Positive parameter alignment with a higher-priority contributor"})
        blocks.append({"slot_label": label, "energy_before": before, "energy_after": after,
                       "reverted": reverted, "interactions": interactions,
                       "reason": "Reduction would increase signed combined energy; original values retained" if reverted else None})
    return {"policy_version": POLICY_VERSION, "status": "experimental_preview", "calibrated": False,
            "constants": {"max_reduction": MAX_REDUCTION, "pressure_threshold": PRESSURE_THRESHOLD},
            "entries": [{"stable_id": identities[i], "profile_version_id": versions[i], "priority": priorities[i],
                         "before_values": vectors[i], "values": proposed[i]} for i in range(count)],
            "changes": changes, "blocks": blocks, "image_quality_verified": False,
            "limitations": ["Parameter-space experiment only; no prediction of semantic compatibility or image quality.",
                            "BASE, equal-priority contributions and negative alignment are not automatically changed.",
                            "Scalar attenuation does not improve per-block cosine alignment.",
                            "The caller must revalidate source identity and export mapping before saving or copying."]}
