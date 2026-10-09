"""Measurement jobs accept catalogue IDs, never browser-supplied file paths."""
from fastapi import APIRouter, Body, HTTPException
from analysis_job_service import JobError
from profile_versions import ProfileValidationError, ProfileNotFoundError


def create_analysis_job_router(service):
    router = APIRouter(prefix='/api/analysis-jobs', tags=['CPU analysis'])

    def run(operation):
        try:
            return operation()
        except JobError as exc:
            raise HTTPException(exc.status, {'reason_code': exc.code, 'reason': str(exc)}) from exc
        except (ProfileValidationError, ProfileNotFoundError) as exc:
            raise HTTPException(409, {'reason_code': 'selection_changed', 'reason': str(exc)}) from exc

    @router.post('', status_code=202)
    def start(body: dict = Body(...)):
        return run(lambda: service.start(body))

    @router.post('/resolve')
    def resolve(body: dict = Body(...)):
        return run(lambda: service.resolve(body))

    @router.get('/{job_id}')
    def get(job_id: str):
        return run(lambda: service.get(job_id))

    @router.post('/{job_id}/cancel')
    def cancel(job_id: str, body: dict = Body(default_factory=dict)):
        if body:
            raise HTTPException(422, 'Cancellation accepts no additional fields.')
        return run(lambda: service.cancel(job_id))

    return router
