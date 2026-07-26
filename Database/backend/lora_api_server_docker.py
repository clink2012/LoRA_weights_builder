from __future__ import annotations

import os
from pathlib import Path

import lora_id_assigner
import lora_indexer
from model_family_integration import apply_model_family_registry


def _path_from_env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return str(Path(value).expanduser())


# Docker/runtime path overrides. The original scripts remain usable on Bender
# with their historical Windows defaults, while this wrapper makes container
# deployment use mounted Linux paths.
RUNTIME_LORA_ROOT = _path_from_env("LORA_ROOT", "/loras")
RUNTIME_DB_PATH = _path_from_env("LORA_DB_PATH", "/data/lora_master.db")

# Apply the reviewed Phase 8.9 family metadata mapping before importing the API.
# This changes folder parsing only; it does not run an index or alter scanner maths.
apply_model_family_registry(lora_indexer)

# Patch module-level script constants before importing the FastAPI app.
lora_indexer.LORA_ROOT = RUNTIME_LORA_ROOT
lora_indexer.DB_PATH = RUNTIME_DB_PATH
lora_id_assigner.DB_PATH = RUNTIME_DB_PATH

# Ensure the persistent DB file has the base schema before FastAPI health checks
# run. This creates tables only; it does not scan or reindex the LoRA library.
_db_path = Path(RUNTIME_DB_PATH)
_db_path.parent.mkdir(parents=True, exist_ok=True)
_schema_conn = lora_indexer.ensure_db()
_schema_conn.close()

import lora_api_server  # noqa: E402
from model_family_router import router as model_family_router  # noqa: E402

# Patch API DB path after import because lora_api_server has its own module-level
# default. The startup backfill and all request handlers then use the mounted DB.
lora_api_server.DB_PATH = _db_path
lora_api_server.app.include_router(model_family_router)

app = lora_api_server.app
