from __future__ import annotations

from phase89_relocation_audit import analyse_relocations


def _section(values):
    return {"count": len(values), "sample": values[:20], "all": values}


def test_unique_and_ambiguous_filename_matches_are_separated() -> None:
    report = {
        "stale_db_paths": _section(
            [
                "/loras/WAN2.2/T2V/04 - Action/moved.safetensors",
                "/loras/FLUX/01 - People/duplicate.safetensors",
                "/loras/FLUX/02 - Styles/duplicate.safetensors",
                "/loras/FLUX/03 - Utils/deleted.safetensors",
            ]
        ),
        "mounted_files_missing_from_db": _section(
            [
                "WAN2.1/T2V/04 - Action/moved.safetensors",
                "Z-Image/03 - Utils/duplicate.safetensors",
                "LTXV2/03 - Utils/new.safetensors",
            ]
        ),
        "unresolved_db_paths": _section(
            [
                "/loras/Hunyuna_15/03 - Utils/old-one.safetensors",
                "/loras/Hunyuna_15/04 - Action/old-two.safetensors",
                "/loras/SD/02 - Styles/old-three.safetensors",
            ]
        ),
    }

    result = analyse_relocations(report)

    assert result["audit_mode"] == "read-only"
    assert result["summary"] == {
        "stale_current_family_rows": 4,
        "mounted_files_missing_from_db": 3,
        "unique_filename_matches": 1,
        "ambiguous_filename_matches": 1,
        "unmatched_stale_rows": 3,
        "unmatched_missing_files": 2,
        "legacy_unmounted_rows": 3,
    }
    assert result["unique_matches"] == [
        {
            "filename": "moved.safetensors",
            "old_path": "/loras/WAN2.2/T2V/04 - Action/moved.safetensors",
            "new_path": "WAN2.1/T2V/04 - Action/moved.safetensors",
            "from_family": "WAN2.2",
            "to_family": "WAN2.1",
        }
    ]
    assert result["ambiguous_matches"][0]["filename"] == "duplicate.safetensors"
    assert result["transitions"] == [
        {"from_family": "WAN2.2", "to_family": "WAN2.1", "count": 1}
    ]
    assert result["legacy_families"] == [
        {"family": "Hunyuna_15", "count": 2},
        {"family": "SD", "count": 1},
    ]


def test_filename_matching_is_case_insensitive() -> None:
    report = {
        "stale_db_paths": _section([r"E:\models\loras\Z-Image\04 - Action\Example.SAFETENSORS"]),
        "mounted_files_missing_from_db": _section(["Z-Image/05 - Body/example.safetensors"]),
        "unresolved_db_paths": _section([]),
    }

    result = analyse_relocations(report)

    assert result["summary"]["unique_filename_matches"] == 1
    assert result["unique_matches"][0]["from_family"] == "Z-Image"
    assert result["unique_matches"][0]["to_family"] == "Z-Image"
