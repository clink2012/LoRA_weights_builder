"""Private bounded worker entry point. Only the API writes its input manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Database/backend'))
from effective_lora_metrics import AnalysisError, _torch, analyse_files
from analyse_effective_lora import save_receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.manifest.stat().st_size > 65536:
            raise AnalysisError('invalid_manifest', 'The worker input exceeds its limit.')
        request = json.loads(args.manifest.read_text(encoding='utf-8'))
        paths = request['paths']
        if not isinstance(paths, list) or not 1 <= len(paths) <= 8 or any(not isinstance(p, str) for p in paths):
            raise AnalysisError('invalid_manifest', 'The worker input must contain one to eight source paths.')
        torch = _torch()
        import safetensors
        import numpy
        if str(torch.__version__) != '2.9.1+cpu' or safetensors.__version__ != '0.7.0' or numpy.__version__ != '2.3.5':
            raise AnalysisError('analysis_runtime_mismatch', 'The project analysis runtime does not match the pinned CPU requirements.')
        torch.set_num_threads(2)
        receipt = analyse_files([Path(p) for p in paths])
        receipt['runtime']['numpy'] = numpy.__version__
        code = 0
    except AnalysisError as exc:
        receipt = {'status': 'failed', 'reason_code': exc.code, 'reason': str(exc), 'metrics': None}
        code = 2
    except Exception:
        receipt = {'status': 'failed', 'reason_code': 'worker_failed',
                   'reason': 'The optional CPU worker could not complete analysis.', 'metrics': None}
        code = 2
    save_receipt(args.output, receipt)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
