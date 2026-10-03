"""Build and serve the existing LoRA UI/API on Bender loopback only."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from uuid import uuid4, UUID

PROJECT = Path(__file__).resolve().parents[1]
APP_ID = "lora-comfy-combiner-local"
BUILD_RECEIPT = PROJECT / ".local/runtime/ui-build.json"


class LocalLaunchError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_files(root: Path, paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ui_source_digest(project: Path = PROJECT) -> str:
    ui = project / "Database/UI"
    paths = [path for folder in (ui / "src", ui / "public") if folder.exists()
             for path in folder.rglob("*") if path.is_file() and ".test." not in path.name]
    paths += [path for name in ("index.html", "package.json", "package-lock.json", "vite.config.js")
              if (path := ui / name).is_file()]
    if not paths or not (ui / "package-lock.json").is_file():
        raise LocalLaunchError("The UI source or lockfile is missing. Restore the project checkout before building.")
    return _hash_files(ui, paths)


def source_identity(project: Path = PROJECT) -> dict:
    try:
        revision = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "-C", str(project), "status", "--porcelain"], capture_output=True, text=True, check=False)
        head = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except OSError:
        head, dirty = None, None
    backend = project / "Database/backend"
    paths = list(backend.glob("*.py")) + list((backend / "contracts").glob("*.json"))
    wrapper = project / "tools/serve_local.py"
    if wrapper.is_file():
        paths.append(wrapper)
    return {"git_revision": head, "working_tree_dirty": dirty, "backend_sha256": _hash_files(project, paths)}


def record_build(project: Path = PROJECT, *, expected_source: str | None = None) -> dict:
    source = ui_source_digest(project)
    if expected_source is not None and source != expected_source:
        raise LocalLaunchError("UI source changed while the build ran. Build again before launching.")
    dist = project / "Database/UI/dist"
    if not (dist / "index.html").is_file():
        raise LocalLaunchError("Built UI is missing. Run tools\\Launch-LoRA.ps1 -Build first.")
    files = {path.relative_to(dist).as_posix(): _sha(path) for path in dist.rglob("*") if path.is_file()}
    receipt = {"format": 1, "input_sha256": source, "files": files,
               "created_at": datetime.now(timezone.utc).isoformat(), "source_revision": source_identity(project)["git_revision"]}
    destination = project / ".local/runtime/ui-build.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return receipt


def build_ui(project: Path = PROJECT) -> dict:
    before = ui_source_digest(project)
    node = shutil.which("node")
    if node is None:
        raise LocalLaunchError("Node is unavailable. Restore the project's Node runtime before building; nothing was installed.")
    vite = project / "Database/UI/node_modules/vite/bin/vite.js"
    if not vite.is_file():
        raise LocalLaunchError("UI dependencies are missing. Restore/install the locked dependencies before building; nothing was downloaded.")
    package = json.loads((project / "Database/UI/package.json").read_text(encoding="utf-8"))
    if package.get("scripts", {}).get("build") != "vite build":
        raise LocalLaunchError("The UI build command changed. Update this launcher to match the reviewed project build.")
    # Keep the built UI on this process even if a developer shell has a Vite
    # backend override left over from a preview session.
    environment = os.environ.copy()
    environment["VITE_API_BASE"] = "/api"
    subprocess.run([node, str(vite), "build"], cwd=project / "Database/UI", env=environment, check=True)
    return record_build(project, expected_source=before)


def verify_build(project: Path = PROJECT) -> dict:
    receipt_path = project / ".local/runtime/ui-build.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalLaunchError("A verified UI build is missing. Run tools\\Launch-LoRA.ps1 -Build first.") from exc
    if not isinstance(receipt, dict) or receipt.get("format") != 1 or receipt.get("input_sha256") != ui_source_digest(project):
        raise LocalLaunchError("The UI build is stale. Run tools\\Launch-LoRA.ps1 -Build, then launch again.")
    dist = (project / "Database/UI/dist").resolve()
    files = receipt.get("files")
    if not isinstance(files, dict) or "index.html" not in files:
        raise LocalLaunchError("The UI build receipt is incomplete. Build again.")
    for relative, expected in files.items():
        path = dist / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(dist) or path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise LocalLaunchError("Built UI files have changed or are missing. Build again.")
    if {path.relative_to(dist).as_posix() for path in dist.rglob("*") if path.is_file()} != set(files):
        raise LocalLaunchError("The UI output differs from its build receipt. Build again.")
    return receipt


def validate_database(database: Path, project: Path = PROJECT) -> tuple[Path, str]:
    database = database.expanduser().resolve()
    if not database.is_file():
        raise LocalLaunchError("The selected database does not exist. Choose an existing application database or a restored copy.")
    from local_state_backup import inspect_database
    try:
        info = inspect_database(database)
    except Exception as exc:
        raise LocalLaunchError("The selected file is not a healthy SQLite application database.") from exc
    if "lora" not in info["table_counts"] or "lora_block_weights" not in info["table_counts"]:
        raise LocalLaunchError("The selected database lacks the LoRA catalogue tables. It was not started or changed.")
    kind = "main" if database == (project / "Database/lora_master.db").resolve() else "copy"
    return database, kind


def snapshot_main_database(database: Path, project: Path = PROJECT) -> str | None:
    _, kind = validate_database(database, project)
    if kind != "main":
        return None
    from local_state_backup import snapshot, verify
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = project / f".local/backups/before-launch-{stamp}-{uuid4().hex[:8]}"
    snapshot(project, destination)
    verify(destination)
    return str(destination)


def create_local_app(database: Path, *, project: Path = PROJECT, port: int = 5187,
                     run_id: str | None = None, backup_path: str | None = None):
    """One database per process. Startup runs existing API migrations on that DB."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    build = verify_build(project)
    database, kind = validate_database(database, project)
    os.environ["LORA_DB_PATH"] = str(database)
    backend_path = str(project / "Database/backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    backend = importlib.import_module("lora_api_server")
    backend.DB_PATH = database
    backend._schema_migrations_done = False

    @asynccontextmanager
    async def lifespan(_app):
        async with backend.app.router.lifespan_context(backend.app):
            yield

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    identity = {"app": APP_ID, "run_id": run_id or str(uuid4()), "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(), "host": "127.0.0.1", "port": port,
                "database_path": str(database), "database_kind": kind, "source": source_identity(project),
                "build": {key: build[key] for key in ("input_sha256", "created_at", "source_revision")},
                "pre_start_backup": backup_path}
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        host, origin = request.headers.get("host", "").lower(), request.headers.get("origin")
        if host not in allowed_hosts or (origin is not None and origin.lower() not in {f"http://{item}" for item in allowed_hosts}):
            return JSONResponse({"detail": "This application is available on Bender loopback only."}, status_code=403)
        return await call_next(request)

    @app.get("/local-app/status")
    def status():
        return identity

    # Reuse routes without the development-wide CORS middleware or CDN-backed
    # Swagger pages. Existing startup handlers are retained through lifespan.
    app.router.routes.extend(route for route in backend.app.router.routes
                             if getattr(route, "path", None) not in {"/docs", "/redoc", "/docs/oauth2-redirect"})
    app.mount("/", StaticFiles(directory=project / "Database/UI/dist", html=True), name="built-ui")
    return app


def serve(database: Path, port: int, run_id: str | None = None) -> None:
    import uvicorn
    verify_build()
    database, _ = validate_database(database)
    if not 1024 <= port <= 65535:
        raise LocalLaunchError("Choose a local port from 1024 to 65535.")
    if run_id is not None:
        UUID(run_id)
    # Reserve the loopback socket before snapshot/migration. Occupied ports
    # must not cause any app startup writes to the selected database.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise LocalLaunchError(f"Port {port} is already in use. Stop the recorded app or choose another local port.") from exc
        listener.listen(128)
        listener.setblocking(False)
        backup = snapshot_main_database(database)
        app = create_local_app(database, port=port, run_id=run_id, backup_path=backup)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, reload=False, workers=1,
                                proxy_headers=False, log_level="info", access_log=False)
        uvicorn.Server(config).run(sockets=[listener])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    check = sub.add_parser("check")
    check.add_argument("--database", type=Path, default=PROJECT / "Database/lora_master.db")
    start = sub.add_parser("serve")
    start.add_argument("--database", type=Path, default=PROJECT / "Database/lora_master.db")
    start.add_argument("--port", type=int, default=5187)
    start.add_argument("--run-id")
    args = parser.parse_args()
    try:
        if args.command == "build":
            receipt = build_ui()
            print(json.dumps({"status": "built", "input_sha256": receipt["input_sha256"]}))
        elif args.command == "check":
            build = verify_build()
            database, kind = validate_database(args.database)
            print(json.dumps({"status": "ready", "database_path": str(database), "database_kind": kind,
                              "source": source_identity(), "build_input_sha256": build["input_sha256"]}))
        else:
            serve(args.database, args.port, args.run_id)
    except (LocalLaunchError, subprocess.CalledProcessError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
