from __future__ import annotations

from fastapi import APIRouter

from model_family_registry import api_model_families

router = APIRouter(prefix="/api", tags=["model-families"])


@router.get("/model-families")
def api_model_family_registry() -> dict[str, object]:
    return {
        "schema_version": "8.9a",
        "families": api_model_families(),
    }
