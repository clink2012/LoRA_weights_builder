"""Stage the pinned maintainer block loader under project .local; never install.

No download, import/execution of upstream code, dependency installation, model
access or write to ComfyUI occurs. Preserve upstream LICENSE and NOTICE.
"""
import argparse
import hashlib
import json
from pathlib import Path

PINS = {
    'h3_block_loader.py': '3d57625d2fb907bbbb8902d293287df7db5d721b4032a544eaa1a971a82f29d8',
    'LICENSE': '48df70bbd4732d47ad021e8dd2203eff33868b31d8d9394c4ff15e9ec67d43a0',
    'NOTICE': '62f3d722285ad4146ffeeff5f199f408759adefc6abdb9da1fd74992ddfbca27',
}
REGISTRATION = b'''# Registration-only wrapper for the pinned Apache-2.0 maintainer node.
from .h3_block_loader import FL_MiniMaxH3LoraBlockLoader

NODE_CLASS_MAPPINGS = {"FL_MiniMaxH3LoraBlockLoader": FL_MiniMaxH3LoraBlockLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"FL_MiniMaxH3LoraBlockLoader": "FL MiniMax H3 LoRA Block Loader"}
'''


def prepare(project_root, source, license_file, notice_file):
    root = Path(project_root).resolve()
    output = root / '.local/h3-loader-trial/lora_builder_h3_loader_trial'
    try: output.resolve().relative_to(root)
    except ValueError as exc: raise ValueError('Staging location escapes the project.') from exc
    files = {'__init__.py': REGISTRATION}
    for name, path in [('h3_block_loader.py', source), ('LICENSE', license_file), ('NOTICE', notice_file)]:
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError('Required pinned source/license file is absent or unbounded.')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != PINS[name]:
            raise ValueError(f'Pinned {name} changed; stage nothing until reviewed.')
        files[name] = content
    manifest = {'package': 'lora_builder_h3_loader_trial', 'upstream_revision': '4e6379770e7f4fc48651b76d03284a7eee7df8a4',
                'source_url': 'https://github.com/filliptm/ComfyUI-FL-MiniMaxH3',
                'files': {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
                'install_performed': False, 'export_verified': False,
                'scope': 'One pinned block loader plus registration only. No routes, web assets or other pack nodes. Uses existing ComfyUI dependencies.'}
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    # Refuse divergent or extra staging content; never delete/overwrite it.
    if output.exists():
        if set(p.name for p in output.iterdir()) != set(files):
            raise ValueError('Existing staging directory differs; no files overwritten.')
        if any(not (output / name).is_file() or (output / name).read_bytes() != content for name, content in files.items()):
            raise ValueError('Existing staging files differ; no files overwritten.')
        return manifest
    output.mkdir(parents=True)
    try: output.resolve().relative_to(root)
    except ValueError as exc: raise ValueError('Staging location escapes the project.') from exc
    for name, content in files.items(): (output / name).write_bytes(content)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--license', required=True, dest='license_file')
    parser.add_argument('--notice', required=True, dest='notice_file')
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.source, args.license_file, args.notice_file), indent=2))
