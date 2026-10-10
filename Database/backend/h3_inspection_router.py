"""Read catalogue-selected H3 headers under the current library; no client paths."""
from contextlib import closing
from pathlib import Path
import stat

from fastapi import APIRouter, HTTPException

from catalogue_refresh import file_identity, reparse
from flux_header_coverage import CoverageError
from h3_adapter_observations import inspect_file


def create_h3_inspection_router(catalogue):
    router = APIRouter(prefix="/api/h3-inspection", tags=["H3 header observations"])

    @router.get("/{stable_id}")
    def inspect(stable_id: str):
        with closing(catalogue.connection()) as conn:
            row = conn.execute("SELECT filename,file_path,base_model_code FROM lora WHERE stable_id=?", (stable_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "This LoRA is not in the catalogue.")
        if row["base_model_code"] != "MH3":
            raise HTTPException(422, "Choose a catalogued H3 adapter for this inspection.")
        try:
            root = Path(catalogue.root).resolve(strict=True)
            path = Path(row["file_path"]).absolute()
            relative = path.relative_to(root)
            # Reject symlinks/junctions in the root and every catalogue component.
            for parent in (Path(catalogue.root), *Path(catalogue.root).parents):
                if reparse(parent.lstat()):
                    raise CoverageError("outside_library", "Inspection requires a real library folder.")
            current = root
            for component in relative.parts:
                current = current / component
                if reparse(current.lstat()):
                    raise CoverageError("outside_library", "Inspection cannot follow links or junctions.")
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise CoverageError("file_missing", "The catalogued H3 file is unavailable.")
            result = inspect_file(path)
            measured = result["file_identity"]
            observed = (measured["file_device"], measured["file_inode"], measured["file_size"], measured["file_mtime_ns"])
            if file_identity(before) != observed or file_identity(path.lstat()) != observed:
                raise CoverageError("file_changed", "The adapter changed during inspection; retry.")
            # Recheck containment after reading; a parent can change as well as the file.
            if path.resolve(strict=True) != path or Path(catalogue.root).resolve(strict=True) != root:
                raise CoverageError("file_changed", "The library path changed during inspection; retry.")
            current = root
            for component in relative.parts:
                current = current / component
                if reparse(current.lstat()):
                    raise CoverageError("file_changed", "The library path changed during inspection; retry.")
        except (OSError, ValueError) as error:
            return {"stable_id": stable_id, "filename": row["filename"], "status": "unavailable",
                    "reason_code": getattr(error, "code", "file_unavailable"),
                    "reason": str(error) if isinstance(error, CoverageError) else "The catalogued file is missing or outside the current library.",
                    "export_verified": False, "measurements_available": False, "slots": []}
        return {"stable_id": stable_id, "filename": row["filename"], **result}

    return router
