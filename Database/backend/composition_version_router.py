"""Composition endpoints require an application-owned fresh preparation resolver."""
from contextlib import closing
import sqlite3

from fastapi import APIRouter, Body, HTTPException, Query

from composition_versions import (
    PreparationChangedError, get_composition, list_compositions,
    prepare_composition, save_composition,
)
from profile_versions import ProfileNotFoundError, ProfileValidationError


def create_composition_version_router(connection_factory, preparation_resolver):
    router = APIRouter(prefix="/api/composition-versions", tags=["composition versions"])

    def run(operation):
        try:
            with closing(connection_factory()) as conn:
                return operation(conn)
        except PreparationChangedError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ProfileValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        except sqlite3.OperationalError as exc:
            if "no such table: lora_" in str(exc):
                raise HTTPException(503, "Composition history has not been initialised for this database") from exc
            raise

    @router.get("")
    def history(composition_id: str | None = Query(None)):
        return run(lambda conn: {"versions": list_compositions(conn, composition_id)})

    @router.post("")
    def save(body: dict = Body(...)):
        required = {"name", "entries", "target_contract_id", "expected_preparation_digest"}
        if not required.issubset(body) or set(body) - required - {"parent_version_id"}:
            raise HTTPException(422, "Missing or unexpected composition fields; client snapshots are not accepted")
        return run(lambda conn: save_composition(conn, preparation_resolver=preparation_resolver, **body))

    @router.get("/{version_id}")
    def get(version_id: str):
        return run(lambda conn: get_composition(conn, version_id))

    @router.post("/{version_id}/prepare")
    def prepare(version_id: str, body: dict = Body(default_factory=dict)):
        if body:
            raise HTTPException(422, "Saved composition preparation accepts no replacement entries or values")
        return run(lambda conn: prepare_composition(conn, version_id, preparation_resolver=preparation_resolver))

    return router
