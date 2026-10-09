"""Server-bound preferred composition recall and non-destructive original reset."""
from contextlib import closing
import sqlite3

from fastapi import APIRouter, Body, HTTPException

from composition_preferences import choose_preference, resolve_preference, restore_originals
from composition_versions import PreparationChangedError
from profile_versions import ProfileNotFoundError, ProfileValidationError


def create_composition_preference_router(connection_factory, default_resolver, preparation_resolver):
    router = APIRouter(prefix='/api/composition-preferences', tags=['composition preferences'])

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
            if 'no such table: lora_' in str(exc):
                raise HTTPException(503, 'Composition preference history has not been initialised') from exc
            raise

    def fields(body, required):
        if not isinstance(body, dict) or set(body) != set(required):
            raise HTTPException(422, 'Missing or unexpected preference fields; source identities and values are resolved by the server')

    @router.post('/resolve')
    def resolve(body: dict = Body(...)):
        fields(body, ('stable_ids', 'target_contract_id'))
        return run(lambda conn: resolve_preference(conn, **body, default_resolver=default_resolver, preparation_resolver=preparation_resolver))

    @router.post('/choose')
    def choose(body: dict = Body(...)):
        fields(body, ('version_id', 'expected_preparation_digest'))
        return run(lambda conn: choose_preference(conn, **body, default_resolver=default_resolver, preparation_resolver=preparation_resolver))

    @router.post('/originals')
    def originals(body: dict = Body(...)):
        fields(body, ('stable_ids', 'target_contract_id'))
        return run(lambda conn: restore_originals(conn, **body, default_resolver=default_resolver))

    return router
