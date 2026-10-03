"""Explicit local CPU analysis. This tool never edits LoRAs, Comfy or the database.

Run from E:\\LoRA Project with an optional Torch-capable analysis Python:
  python -B tools/analyse_effective_lora.py <first.safetensors> <second.safetensors>
      --output .local/effective-analysis/receipt.json

The ordinary app environment can remain dependency-light. Install optional
analysis requirements into a separate project environment, never into Comfy.
"""
from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Database/backend"))
from effective_lora_metrics import AnalysisBudget, AnalysisError, _torch, analyse_files


def save_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure conditional native FLUX LoRA parameter updates on CPU; no automatic balancing.")
    parser.add_argument("loras", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--max-read-mib", type=int, default=1024)
    parser.add_argument("--max-multiply-adds", type=int, default=5_000_000_000)
    parser.add_argument("--max-rank", type=int, default=256)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be between 1 and 64")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must be a JSON receipt path")
    output = args.output.resolve()
    if any(output == path.resolve() for path in args.loras):
        parser.error("The output receipt must not replace an input LoRA")
    try:
        budget = replace(AnalysisBudget(), max_seconds=args.max_seconds,
                         max_tensor_bytes_read=args.max_read_mib * 1024 * 1024,
                         max_multiply_adds=args.max_multiply_adds, max_rank=args.max_rank)
        torch = _torch()
        torch.set_num_threads(args.threads)
        receipt = analyse_files(args.loras, budget=budget)
        code = 0
    except AnalysisError as exc:
        receipt = {"schema_version": 1, "status": "failed", "reason_code": exc.code,
                   "reason": str(exc), "metrics": None}
        code = 2
    except KeyboardInterrupt:
        receipt = {"schema_version": 1, "status": "cancelled", "reason_code": "cancelled",
                   "reason": "Cancelled by the user; no complete metrics were produced.", "metrics": None}
        code = 130
    save_receipt(output, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output),
                      "reason_code": receipt.get("reason_code")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
