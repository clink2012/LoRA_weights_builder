from contextlib import closing
import sqlite3

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from composition_versions import PreparationChangedError
from profile_versions import ProfileNotFoundError, ProfileValidationError
from render_trials import MAX_PNG, add_evidence, assess_trial, create_trial, evidence_bytes, get_trial, list_trials


def create_render_trial_router(connection_factory):
    router = APIRouter(prefix='/api/render-trials', tags=['visual trial history'])

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
                raise HTTPException(503, 'Render history has not been initialised for this database') from exc
            raise

    def fields(body, required):
        if not isinstance(body, dict) or set(body) != set(required):
            raise HTTPException(422, 'Missing or unexpected fields; recipe snapshots are server owned')

    @router.get('')
    def history(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        return run(lambda conn: {'trials': list_trials(conn, limit, offset)})

    @router.post('')
    def create(body: dict = Body(...)):
        fields(body, ('name', 'composition_version_id', 'generation', 'criteria', 'baseline_trial_id', 'idempotency_key'))
        return run(lambda conn: create_trial(conn, **body))

    @router.get('/{trial_id}')
    def get(trial_id: str):
        return run(lambda conn: get_trial(conn, trial_id))

    @router.post('/{trial_id}/evidence')
    async def upload(trial_id: str, request: Request, filename: str = Query(..., max_length=250)):
        if request.headers.get('content-type', '').split(';')[0] != 'image/png':
            raise HTTPException(415, 'Choose a PNG image')
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_PNG:
                raise HTTPException(413, 'PNG exceeds 16 MiB')
            data.extend(chunk)
        return run(lambda conn: add_evidence(conn, trial_id, filename, bytes(data)))

    @router.get('/{trial_id}/evidence/{evidence_id}')
    def image(trial_id: str, evidence_id: str):
        data = run(lambda conn: evidence_bytes(conn, trial_id, evidence_id))
        return Response(data, media_type='image/png', headers={'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'none'", 'Cache-Control': 'no-store'})

    @router.post('/{trial_id}/assessments')
    def assess(trial_id: str, body: dict = Body(...)):
        fields(body, ('assessment', 'expected_assessment_id', 'idempotency_key'))
        return run(lambda conn: assess_trial(conn, trial_id, **body))

    return router
