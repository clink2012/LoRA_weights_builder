"""Read one local safetensors header; never load tensor payloads or write models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Database" / "backend"))
from adapter_identity import identify_file
from flux_header_coverage import CoverageError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="One local .safetensors adapter file")
    parser.add_argument("--root", type=Path, default=Path(r"E:\models\loras"), help="Read-only library root; path must remain inside it")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("The library root must be a directory.")
        path = args.path.resolve(strict=True)
        relative = path.relative_to(root)
        folder_hint = relative.parts[0] if len(relative.parts) > 1 else None
        result = identify_file(path, folder_hint=folder_hint)
    except (CoverageError, OSError, ValueError) as exc:
        print(json.dumps({"status": "unavailable", "reason_code": getattr(exc, "code", "invalid_path"),
                          "reason": str(exc), "export_verified": False, "tensor_payload_read": False}))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
