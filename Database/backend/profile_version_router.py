"""Router factory: application owns connections, schema migration and source resolution."""
from contextlib import closing
import sqlite3
from typing import Callable

from fastapi import APIRouter, Body, HTTPException, Query

from profile_versions import (
    ProfileNotFoundError, ProfileValidationError, capture_default, create_revision,
    get_selection, get_version, list_versions, select_version,
)


def create_profile_version_router(
    connection_factory: Callable[[], sqlite3.Connection],
    default_resolver: Callable[[sqlite3.Connection, str], dict],
) -> APIRouter:
    """Use a trusted resolver; browser-supplied fingerprints are never evidence.

    ``default_resolver(conn, stable_id)`` must identify the actual source and
    ordered architecture, then return binding, values, settings and optional ab.
    It must reject missing/unresolved data. Schema initialisation is an explicit
    application migration, never an effect of a history read. Profile capture
    by itself does not verify an export against a ComfyUI loader.
    """
    router = APIRouter(prefix="/api/profile-versions", tags=["profile versions"])

    def run(operation):
        try:
            with closing(connection_factory()) as conn:
                return operation(conn)
        except ProfileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ProfileValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        except sqlite3.OperationalError as exc:
            if "no such table: lora_profile_" in str(exc):
                raise HTTPException(503, "Profile history has not been initialised for this database") from exc
            raise

    def fields(body, required, optional=()):
        if set(body) - set(required) - set(optional) or not set(required).issubset(body):
            raise HTTPException(422, "Missing or unexpected profile fields")
        return body

    @router.get("/{stable_id}")
    def history(stable_id: str, default_id: str | None = Query(None)):
        return run(lambda conn: {"stable_id": stable_id, "versions": list_versions(conn, stable_id, default_id)})

    @router.post("/{stable_id}/defaults")
    def capture(stable_id: str, body: dict = Body(default_factory=dict)):
        # Deliberately accept no identity, layout or default-value claims here.
        fields(body, ())
        return run(lambda conn: capture_default(conn, stable_id=stable_id, **default_resolver(conn, stable_id)))

    @router.get("/{stable_id}/selection")
    def selected(stable_id: str, default_id: str = Query(...)):
        return run(lambda conn: get_selection(conn, stable_id, default_id))

    @router.post("/{stable_id}/selection")
    def select(stable_id: str, body: dict = Body(...)):
        fields(body, ("default_id", "version_id"))
        return run(lambda conn: select_version(conn, stable_id=stable_id, **body))

    @router.post("/{stable_id}/revisions")
    def revise(stable_id: str, body: dict = Body(...)):
        fields(body, ("default_id", "parent_id", "name", "values", "settings"), ("ab",))

        def operation(conn):
            binding = get_version(conn, stable_id, body["default_id"])["binding"]
            return create_revision(conn, stable_id=stable_id, binding=binding, **body)
        return run(operation)

    @router.get("/{stable_id}/versions/{version_id}")
    def version(stable_id: str, version_id: str):
        return run(lambda conn: get_version(conn, stable_id, version_id))

    return router
