"""Fixed local library background scan; request bodies never accept paths."""
import sqlite3
import os
from contextlib import closing
from fastapi import APIRouter, Body, HTTPException, Query
from catalogue_refresh import CatalogueError
from pydantic import BaseModel, ConfigDict, Field


class LibraryLocationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    root: str = Field(min_length=1, max_length=4096)


def create_library_scan_router(service):
    router = APIRouter(prefix='/api/library-scan', tags=['library scan'])
    provider = service if callable(service) else lambda: service

    def run(operation):
        try:
            return operation(provider())
        except CatalogueError as exc:
            raise HTTPException(exc.status, {'reason_code': exc.code, 'reason': str(exc)}) from exc
        except sqlite3.Error as exc:
            raise HTTPException(503, {'reason_code': 'scan_unavailable', 'reason': 'The local scan database is unavailable.'}) from exc

    def no_options(body):
        if body:
            raise HTTPException(422, 'Library scans use the configured local root and accept no options or paths.')

    @router.post('')
    def start(body: dict = Body(default_factory=dict)):
        no_options(body)
        return run(lambda current: current.start())

    @router.post('/resume')
    def resume(body: dict = Body(default_factory=dict)):
        no_options(body)
        return run(lambda current: current.start(resume=True))

    @router.post('/cancel')
    def cancel(body: dict = Body(default_factory=dict)):
        no_options(body)
        return run(lambda current: current.cancel())

    @router.get('')
    def status():
        return run(lambda current: current.status())

    @router.get('/issues')
    def issues(stable_id: str | None = None, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return run(lambda current: current.issues(stable_id=stable_id, limit=limit, offset=offset))

    @router.get('/freshness')
    def freshness():
        return run(lambda current: current.freshness())

    @router.get('/location')
    def location():
        return run(lambda current: current.location())

    @router.put('/location')
    def choose_location(body: LibraryLocationRequest):
        return run(lambda current: current.choose_location(body.root))

    return router


def install_library_scan(app, service_factory, on_location_selected=None):
    """Resolve paths only at startup, after the application's database selection.

    Tests can disable the entire startup hook before creating TestClient; no
    factory or schema access then occurs. Manual service/router tests remain usable.
    """
    from library_scan_service import initialise_library_scan_schema
    from library_location import initialise_location_schema, selected_location

    class CurrentService:
        def __getattr__(self, name):
            def invoke(*args, **kwargs):
                service = getattr(app.state, 'library_scan', None)
                if service is None:
                    raise CatalogueError('scan_not_started', 'Library scanning is not enabled in this app session.', 503)
                return getattr(service, name)(*args, **kwargs)
            return invoke

    def startup():
        if os.environ.get('LORA_DISABLE_STARTUP_SCAN') == '1':
            return
        service = service_factory()
        with closing(service.connection()) as conn:
            initialise_library_scan_schema(conn)
            initialise_location_schema(conn)
            service.catalogue.root = selected_location(conn, service.catalogue.root)
        if on_location_selected:
            on_location_selected(service.catalogue)
        app.state.library_scan = service
        service.startup()

    def shutdown():
        service = getattr(app.state, 'library_scan', None)
        if service is not None:
            service.shutdown()

    app.include_router(create_library_scan_router(CurrentService()))
    app.add_event_handler('startup', startup)
    app.add_event_handler('shutdown', shutdown)
