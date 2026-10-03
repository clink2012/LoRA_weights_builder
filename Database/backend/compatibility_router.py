"""Stable-ID-only current-library structural preflight."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from compatibility_preflight import PreflightError


class CompatibilityQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reference_stable_id: str = Field(min_length=1, max_length=200)
    target_contract_id: str = Field(min_length=1, max_length=200)
    view: Literal['eligible', 'excluded', 'all'] = 'eligible'
    base: str | None = Field(default=None, max_length=100)
    category: str | None = Field(default=None, max_length=100)
    search: str | None = Field(default=None, max_length=500)
    limit: int = Field(default=50, ge=1, le=500, strict=True)
    offset: int = Field(default=0, ge=0, strict=True)


def create_compatibility_router(service):
    router = APIRouter(prefix='/api/catalogue', tags=['structural compatibility'])

    @router.post('/compatible')
    def compatible(body: CompatibilityQuery):
        try:
            return service.query(**body.model_dump())
        except PreflightError as exc:
            raise HTTPException(exc.status, {'reason_code': exc.code, 'reason': str(exc), **exc.context}) from exc

    return router
