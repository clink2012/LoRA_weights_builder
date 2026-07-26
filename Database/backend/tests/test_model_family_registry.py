from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_family_registry import (  # noqa: E402
    MODEL_FAMILIES,
    base_model_map,
    get_model_family_by_code,
    get_model_family_by_folder,
)
from model_family_router import router  # noqa: E402


def test_registry_contains_full_known_library_ecosystem() -> None:
    assert {family.folder_name for family in MODEL_FAMILIES} == {
        "FLUX",
        "Flux Krea",
        "Flux.2-Klein",
        "Illustrious",
        "LTXV2",
        "PONY",
        "SD",
        "SDXL",
        "WAN2.1",
        "WAN2.2",
        "Z-Image",
    }


def test_codes_are_unique_three_character_identifiers() -> None:
    codes = [family.code for family in MODEL_FAMILIES]
    assert len(codes) == len(set(codes))
    assert all(len(code) == 3 and code.isalnum() and code == code.upper() for code in codes)
    assert get_model_family_by_folder("flux.2-klein").code == "F2K"
    assert get_model_family_by_code("ltx").folder_name == "LTXV2"
    assert get_model_family_by_code("zim").folder_name == "Z-Image"


def test_only_flux_families_claim_block_analysis() -> None:
    block_capable = {family.code for family in MODEL_FAMILIES if family.block_analysis}
    assert block_capable == {"FLX", "FLK"}

    for family in MODEL_FAMILIES:
        if family.code not in block_capable:
            assert family.support_level == "metadata-only"
            assert family.role_aware_orchestration is False
            assert family.architecture_key_normalisation is False
            assert family.comfyui_export is False


def test_base_model_map_matches_existing_indexer_shape() -> None:
    mapping = base_model_map()
    assert mapping["FLUX"] == ("FLX", "Flux")
    assert mapping["Flux.2-Klein"] == ("F2K", "Flux.2-Klein")
    assert mapping["LTXV2"] == ("LTX", "LTXV2")
    assert mapping["Z-Image"] == ("ZIM", "Z-Image")


def test_api_endpoint_exposes_explicit_support_capabilities() -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/model-families")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "8.9a"

    by_code = {family["code"]: family for family in body["families"]}
    assert by_code["FLX"]["block_analysis"] is True
    assert by_code["F2K"]["support_level"] == "metadata-only"
    assert by_code["LTX"]["block_analysis"] is False
    assert by_code["ZIM"]["comfyui_export"] is False
