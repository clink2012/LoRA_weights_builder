from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from lora_path_parser import parse_base_and_category  # noqa: E402
from model_family_integration import apply_model_family_registry  # noqa: E402


def _path(root: Path, *parts: str) -> str:
    return str(root.joinpath(*parts))


def test_new_model_families_parse_as_metadata_codes(tmp_path: Path) -> None:
    root = tmp_path / "loras"

    assert parse_base_and_category(
        _path(root, "Flux.2-Klein", "03 - Utils", "klein.safetensors"),
        str(root),
    ) == ("Flux.2-Klein", "F2K", "Utils", "UTL")

    assert parse_base_and_category(
        _path(root, "LTXV2", "02 - Styles", "ltx.safetensors"),
        str(root),
    ) == ("LTXV2", "LTX", "Styles", "STL")

    assert parse_base_and_category(
        _path(root, "Z-Image", "05 - Body", "zimage.safetensors"),
        str(root),
    ) == ("Z-Image", "ZIM", "Body", "BDY")


def test_existing_family_codes_and_db_names_remain_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "loras"

    assert parse_base_and_category(
        _path(root, "FLUX", "01 - People", "flux.safetensors"),
        str(root),
    ) == ("Flux", "FLX", "People", "PPL")

    assert parse_base_and_category(
        _path(root, "SD", "03 - Utils", "sd.safetensors"),
        str(root),
    ) == ("SD", "SD1", "Utils", "UTL")

    assert parse_base_and_category(
        _path(root, "SDXL", "02 - Styles", "sdxl.safetensors"),
        str(root),
    ) == ("SDXL", "SDX", "Styles", "STL")


def test_wan_mode_folder_parsing_and_db_names_are_preserved(tmp_path: Path) -> None:
    root = tmp_path / "loras"

    assert parse_base_and_category(
        _path(root, "WAN2.1", "T2V", "04 - Action", "wan.safetensors"),
        str(root),
    ) == ("WAN2.1", "W21", "Action", "ACT")

    assert parse_base_and_category(
        _path(root, "WAN2.2", "I2V", "05 - Body", "wan.safetensors"),
        str(root),
    ) == ("WAN2.2", "W22", "Body", "BDY")


def test_unknown_family_remains_visible_without_invented_code(tmp_path: Path) -> None:
    root = tmp_path / "loras"

    assert parse_base_and_category(
        _path(root, "FutureModel", "03 - Utils", "future.safetensors"),
        str(root),
    ) == ("FutureModel", None, "Utils", "UTL")


def test_integration_shim_only_replaces_metadata_parser_contract() -> None:
    sentinel = object()
    indexer = SimpleNamespace(
        BASE_MODEL_MAP={"OLD": ("OLD", "Old")},
        parse_base_and_category=sentinel,
        inspect_lora=sentinel,
        make_flux_layout=sentinel,
    )

    apply_model_family_registry(indexer)

    assert indexer.BASE_MODEL_MAP["Flux.2-Klein"] == ("F2K", "Flux.2-Klein")
    assert indexer.BASE_MODEL_MAP["LTXV2"] == ("LTX", "LTXV2")
    assert indexer.BASE_MODEL_MAP["Z-Image"] == ("ZIM", "Z-Image")
    assert indexer.parse_base_and_category is parse_base_and_category
    assert indexer.inspect_lora is sentinel
    assert indexer.make_flux_layout is sentinel
