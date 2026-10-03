"""Optional CPU measurements of native LoRA parameter updates, never image scores.

For B[out, rank] and A[rank, in], Delta = scale * B @ A. Small Gram
matrices measure norms and signed inner products without materialising Delta.
Torch is imported only when an analysis is explicitly requested.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

from flux_header_coverage import CONTRACT, CONTRACT_ID, CoverageError, read_header, resolve_native_header, verify_local_sources
from inspire_export import ARCHITECTURE_LABELS

ENGINE_VERSION = "effective_native_lora_v1"


class AnalysisError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise AnalysisError("analysis_dependency_missing", "Explicit analysis requires the optional CPU-capable Torch runtime; ordinary app use does not.") from exc
    return torch


@dataclass(frozen=True)
class AnalysisBudget:
    chunk_elements: int = 262144
    max_rank: int = 256
    max_loras: int = 8
    max_modules: int = 4096
    max_tensor_bytes_read: int = 1024 * 1024 * 1024
    max_multiply_adds: int = 5_000_000_000
    max_seconds: float = 180.0

    def __post_init__(self):
        for name in ("chunk_elements", "max_rank", "max_loras", "max_modules", "max_tensor_bytes_read", "max_multiply_adds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise AnalysisError("invalid_budget", f"{name} must be a positive integer.")
        if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise AnalysisError("invalid_budget", "max_seconds must be positive and finite.")
        if self.max_rank > 2048 or self.chunk_elements > 16_777_216:
            raise AnalysisError("invalid_budget", "Requested rank or chunk size exceeds this bounded CPU implementation.")


class Work:
    def __init__(self, budget: AnalysisBudget, cancelled: Callable[[], bool] | None = None):
        self.budget = budget
        self.cancelled = cancelled
        self.started = time.monotonic()
        self.bytes_read = 0
        self.multiply_adds = 0
        self.chunks = 0

    def check(self):
        if self.cancelled and self.cancelled():
            raise AnalysisError("cancelled", "Analysis was cancelled; no complete metrics receipt was produced.")
        if time.monotonic() - self.started > self.budget.max_seconds:
            raise AnalysisError("time_budget_exceeded", "CPU analysis exceeded its time budget between bounded operations.")

    def charge(self, *, bytes_read=0, multiply_adds=0):
        self.check()
        if self.bytes_read + bytes_read > self.budget.max_tensor_bytes_read:
            raise AnalysisError("read_budget_exceeded", "Tensor reads would exceed the analysis byte budget.")
        if self.multiply_adds + multiply_adds > self.budget.max_multiply_adds:
            raise AnalysisError("work_budget_exceeded", "Gram products would exceed the analysis work budget.")
        self.bytes_read += bytes_read
        self.multiply_adds += multiply_adds


class Factor:
    """A shaped CPU tensor slice source. Reader(axis,start,end) loads one slice."""
    def __init__(self, shape, element_bytes, reader):
        self.shape = tuple(shape)
        self.element_bytes = element_bytes
        self.reader = reader
        if len(self.shape) != 2 or any(type(d) is not int or d <= 0 for d in self.shape):
            raise AnalysisError("invalid_factor", "Only non-empty two-dimensional factors are supported.")

    def chunk(self, axis, start, end, work):
        torch = _torch()
        count = (end - start) * self.shape[1 - axis]
        work.charge(bytes_read=count * self.element_bytes)
        tensor = self.reader(axis, start, end)
        expected = list(self.shape)
        expected[axis] = end - start
        if tuple(tensor.shape) != tuple(expected) or tensor.device.type != "cpu":
            raise AnalysisError("invalid_slice", "The factor reader returned an unexpected shape or non-CPU tensor.")
        value = tensor.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(value).all()):
            raise AnalysisError("nonfinite_tensor", "A LoRA factor contains NaN or infinity.")
        work.chunks += 1
        return value


def tensor_factor(tensor):
    """Small in-memory adapter used by tests and explicit analysis callers."""
    return Factor(tensor.shape, tensor.element_size(), lambda axis, start, end: tensor[start:end, :] if axis == 0 else tensor[:, start:end])


def cross_gram(left: Factor, right: Factor, *, axis: int, work: Work):
    """axis=0: B_left.T @ B_right; axis=1: A_left @ A_right.T."""
    torch = _torch()
    if axis not in (0, 1) or left.shape[axis] != right.shape[axis]:
        raise AnalysisError("shape_mismatch", "Compared factors must share the same target module dimension.")
    rank_left, rank_right = left.shape[1 - axis], right.shape[1 - axis]
    if max(rank_left, rank_right) > work.budget.max_rank:
        raise AnalysisError("rank_budget_exceeded", "LoRA rank exceeds the configured Gram-matrix budget.")
    length = left.shape[axis]
    chunk_size = work.budget.chunk_elements // max(rank_left, rank_right)
    if chunk_size < 1:
        raise AnalysisError("chunk_budget_exceeded", "A single rank row exceeds the chunk element budget.")
    result = torch.zeros((rank_left, rank_right), dtype=torch.float64, device="cpu")
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        work.charge(multiply_adds=(end - start) * rank_left * rank_right)
        a = left.chunk(axis, start, end, work)
        b = a if left is right else right.chunk(axis, start, end, work)
        result += a.T @ b if axis == 0 else a @ b.T
        work.check()
        if not bool(torch.isfinite(result).all()):
            raise AnalysisError("numeric_overflow", "A Gram matrix exceeded finite float64 arithmetic.")
    return result


def alpha_scale(alpha, rank):
    """Comfy's absent-alpha behavior is scale ONE, not 1/rank."""
    if type(rank) is not int or rank <= 0:
        raise AnalysisError("invalid_rank", "Rank must be positive.")
    if alpha is None:
        return 1.0
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise AnalysisError("invalid_alpha", "Alpha must be absent or a finite scalar.")
    return float(alpha) / rank


@dataclass(frozen=True)
class Update:
    up: Factor
    down: Factor
    alpha: float | None = None

    def __post_init__(self):
        if self.up.shape[1] != self.down.shape[0]:
            raise AnalysisError("shape_mismatch", "Up and down factors must have the same rank.")
        alpha_scale(self.alpha, self.up.shape[1])


def update_inner_product(left: Update, right: Update, *, work: Work) -> float:
    torch = _torch()
    gram_b = cross_gram(left.up, right.up, axis=0, work=work)
    gram_a = cross_gram(left.down, right.down, axis=1, work=work)
    products = gram_b * gram_a
    raw = float(products.sum().item())
    if not bool(torch.isfinite(products).all()) or not math.isfinite(raw):
        raise AnalysisError("numeric_overflow", "Effective update products exceeded finite float64 arithmetic.")
    if left is right and raw < 0:
        absolute_sum = float(products.abs().sum().item())
        if not math.isfinite(absolute_sum):
            raise AnalysisError("numeric_overflow", "Rounding-error estimation exceeded finite float64 arithmetic.")
        tolerance = 32 * torch.finfo(torch.float64).eps * absolute_sum
        if raw < -tolerance:
            raise AnalysisError("numeric_instability", "Squared update norm became negative beyond rounding tolerance.")
        raw = 0.0
    result = raw * alpha_scale(left.alpha, left.up.shape[1]) * alpha_scale(right.alpha, right.up.shape[1])
    if not math.isfinite(result):
        raise AnalysisError("numeric_overflow", "Alpha scaling exceeded finite float64 arithmetic.")
    return result


def signed_cosine(inner, left_squared_norm, right_squared_norm):
    if left_squared_norm == 0 or right_squared_norm == 0:
        return None
    # Separate roots avoid overflowing the product of squared norms.
    result = (inner / math.sqrt(left_squared_norm)) / math.sqrt(right_squared_norm)
    if not math.isfinite(result) or abs(result) > 1 + 1e-8:
        raise AnalysisError("numeric_instability", "Parameter alignment exceeded its numerical bounds.")
    return max(-1.0, min(1.0, result))


def _slot(target):
    parts = target.split(".")
    if parts[1] == "double_blocks":
        return 1 + int(parts[2])
    if parts[1] == "single_blocks":
        return 20 + int(parts[2])
    return 0


def _modules(tensors):
    resolve_native_header(tensors)  # Reject unsupported adapters before any tensor read.
    modules = {}
    for key, tensor in tensors.items():
        if not key.endswith(".lora_up.weight"):
            continue
        prefix = key[:-len(".lora_up.weight")]
        target = CONTRACT["aliases"][prefix]
        modules[target] = {"up": key, "down": prefix + ".lora_down.weight", "alpha": prefix + ".alpha" if prefix + ".alpha" in tensors else None}
    return modules


def _analyse_files(paths, *, budget=None, cancelled=None, progress=None):
    """Read source LoRAs without modifying them. Return only complete receipts.

    The byte budget counts repeated sliced reads used for pair comparisons.
    Memory is bounded by factor chunks plus rank-sized Gram matrices; no dense
    target update or base checkpoint is allocated. Deadlines are checked between
    bounded matrix operations, not as a promise to interrupt native BLAS calls.
    """
    budget = budget or AnalysisBudget()
    work = Work(budget, cancelled)
    paths = [Path(path).resolve() for path in paths]
    if not 1 <= len(paths) <= budget.max_loras:
        raise AnalysisError("lora_budget_exceeded", "Choose a non-empty set within the configured LoRA count budget.")
    if len({str(path).casefold() for path in paths}) != len(paths):
        raise AnalysisError("duplicate_source", "Each LoRA may be analysed only once in the source set.")
    try:
        sources = verify_local_sources()
        planned = []
        for path in paths:
            work.check()
            tensors, identity = read_header(path)
            modules = _modules(tensors)
            if len(modules) > budget.max_modules:
                raise AnalysisError("module_budget_exceeded", "A source exceeds the module count budget.")
            if any(tensors[module["up"]]["shape"][1] > budget.max_rank for module in modules.values()):
                raise AnalysisError("rank_budget_exceeded", "A source rank exceeds the configured budget.")
            planned.append((tensors, identity, modules))
    except CoverageError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    torch = _torch()
    try:
        import safetensors
        from safetensors import safe_open
    except ImportError as exc:
        raise AnalysisError("analysis_dependency_missing", "Explicit tensor analysis requires safetensors in its Python environment.") from exc

    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for path in paths]

        def get_update(source_index, target):
            tensors, _identity, modules = planned[source_index]
            module = modules[target]
            handle = handles[source_index]

            def factor(key):
                tensor = tensors[key]
                view = handle.get_slice(key)
                element_bytes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}[tensor["dtype"]]
                return Factor(tensor["shape"], element_bytes,
                              lambda axis, start, end: view[start:end, :] if axis == 0 else view[:, start:end])

            alpha = None
            if module["alpha"]:
                work.charge(bytes_read={"F16": 2, "BF16": 2, "F32": 4, "F64": 8}[tensors[module["alpha"]]["dtype"]])
                alpha = float(handle.get_tensor(module["alpha"]).item())
            return Update(factor(module["up"]), factor(module["down"]), alpha)

        file_results = []
        for index, (_tensors, identity, modules) in enumerate(planned):
            squares = [0.0] * 58
            for position, target in enumerate(sorted(modules)):
                update = get_update(index, target)
                squares[_slot(target)] += update_inner_product(update, update, work=work)
                if progress:
                    progress({"phase": "source_norms", "source_index": index, "module": position + 1, "module_count": len(modules)})
            if not all(math.isfinite(value) for value in squares) or not math.isfinite(sum(squares)):
                raise AnalysisError("numeric_overflow", "Aggregated update norms exceeded finite arithmetic.")
            file_results.append({"source_index": index, "path": str(paths[index]), "identity": identity,
                                 "module_count": len(modules), "block_squared_norms": squares,
                                 "block_norms": [math.sqrt(value) for value in squares],
                                 "total_squared_norm": sum(squares)})

        pairs = []
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                inner_products = [0.0] * 58
                shared = sorted(set(planned[left][2]) & set(planned[right][2]))
                for position, target in enumerate(shared):
                    inner_products[_slot(target)] += update_inner_product(get_update(left, target), get_update(right, target), work=work)
                    if progress:
                        progress({"phase": "pair_alignment", "left_index": left, "right_index": right, "module": position + 1, "module_count": len(shared)})
                if not all(math.isfinite(value) for value in inner_products) or not math.isfinite(sum(inner_products)):
                    raise AnalysisError("numeric_overflow", "Aggregated pair products exceeded finite arithmetic.")
                pairs.append({"left_index": left, "right_index": right, "shared_module_count": len(shared),
                              "block_inner_products": inner_products,
                              "block_signed_cosines": [signed_cosine(value, file_results[left]["block_squared_norms"][slot], file_results[right]["block_squared_norms"][slot]) for slot, value in enumerate(inner_products)],
                              "total_inner_product": sum(inner_products),
                              "total_signed_cosine": signed_cosine(sum(inner_products), file_results[left]["total_squared_norm"], file_results[right]["total_squared_norm"])})

    # Re-read bounded headers and verify identity after every tensor operation.
    try:
        for path, (_tensors, identity, _modules_) in zip(paths, planned):
            work.check()
            _fresh_tensors, fresh = read_header(path)
            if fresh != identity:
                raise AnalysisError("file_changed", "A source changed during analysis; discard the incomplete measurements.")
        if verify_local_sources() != sources:
            raise AnalysisError("runtime_source_changed", "The pinned source contract changed during analysis.")
    except CoverageError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    return {
        "schema_version": 1, "status": "complete", "engine_version": ENGINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "target_contract_id": CONTRACT_ID,
        "target_contract_sha256": hashlib.sha256(json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "source_sha256": sources, "runtime": {"torch": torch.__version__, "safetensors": safetensors.__version__, "device": "cpu", "accumulation_dtype": "float64"},
        "slot_labels": list(ARCHITECTURE_LABELS), "sources": file_results, "pairs": pairs,
        "measurement_basis": "effective_native_parameter_update", "alpha_absent_scale": 1.0,
        "outer_model_strength_applied": False, "block_weights_applied": False,
        "tensor_payload_read": True, "tensor_contents_finite": True,
        "full_file_hash_verified": False, "checkpoint_verified": False,
        "semantic_conflict_verified": False, "image_quality_verified": False,
        "recommendation": None,
        "limitations": ["These are parameter-space norms and signed alignments, not image quality or semantic compatibility scores.", "Only standard native two-dimensional LoRA pairs are supported; no DoRA, mid, reshape or arbitrary adapter tensors.", "Source identity uses header hash plus file stat checks; this is not a full-file cryptographic fingerprint.", "No base-model tensor, activations, denoising trajectory or rendered image was evaluated."],
        "work": {"tensor_bytes_read": work.bytes_read, "multiply_adds": work.multiply_adds,
                 "byte_budget_basis": "logical_requested_tensor_slices_not_OS_page_reads",
                 "chunks": work.chunks, "elapsed_seconds": time.monotonic() - work.started,
                 "budget": vars(budget)},
    }


def analyse_files(paths, *, budget=None, cancelled=None, progress=None):
    """Public boundary: tensor I/O failures never expose partial measurements."""
    try:
        return _analyse_files(paths, budget=budget, cancelled=cancelled, progress=progress)
    except AnalysisError:
        raise
    except Exception as exc:
        # safetensors' Rust exception has moved modules between releases. Match
        # its package without importing the optional runtime during app startup.
        if isinstance(exc, (OSError, RuntimeError)) or type(exc).__module__.startswith("safetensors"):
            raise AnalysisError("tensor_read_failed", "A source could not be opened or read consistently during tensor analysis; no complete metrics were produced.") from exc
        raise
