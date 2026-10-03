"""Local metadata refresh; clients cannot supply filesystem or database paths."""
import sqlite3
from fastapi import APIRouter, Body, HTTPException, Query
from catalogue_refresh import CatalogueError


def create_catalogue_router(service):
    router = APIRouter(prefix='/api/catalogue', tags=['local catalogue'])

    def run(operation):
        try:
            return operation()
        except CatalogueError as exc:
            raise HTTPException(exc.status, {'reason_code': exc.code, 'reason': str(exc)}) from exc
        except sqlite3.Error as exc:
            raise HTTPException(503, {'reason_code': 'catalogue_unavailable', 'reason': 'The local catalogue schema is unavailable.'}) from exc

    @router.post('/refresh')
    def refresh(body: dict = Body(default_factory=dict)):
        if body:
            raise HTTPException(422, 'Library refresh uses the configured local root and accepts no replacement paths.')
        return run(service.refresh)

    @router.get('')
    def search(presence: str = Query('current'), base: str | None = None, category: str | None = None,
               search: str | None = None, limit: int = Query(50, ge=1, le=5000), offset: int = Query(0, ge=0)):
        return run(lambda: service.search(presence=presence, base=base, category=category, search=search, limit=limit, offset=offset))

    return router
