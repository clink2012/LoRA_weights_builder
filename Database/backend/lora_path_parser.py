from __future__ import annotations

import os
from typing import Optional, Tuple

from model_family_registry import base_model_map

BASE_MODEL_MAP = base_model_map()

# Preserve established DB-facing names for existing families. The registry's
# display names are intended for UI presentation and may include spacing or
# clarifying labels that should not silently rewrite stored metadata.
INDEX_NAME_OVERRIDES = {
    "SD": "SD",
    "WAN2.1": "WAN2.1",
    "WAN2.2": "WAN2.2",
}

CATEGORY_INDEX_MAP: dict[str, tuple[str, str]] = {
    "01": ("PPL", "People"),
    "02": ("STL", "Styles"),
    "03": ("UTL", "Utils"),
    "04": ("ACT", "Action"),
    "05": ("BDY", "Body"),
    "06": ("CHT", "Characters"),
    "07": ("MCV", "Machines_Vehicles"),
    "08": ("CLT", "Clothing"),
    "09": ("ANM", "Animals"),
    "10": ("BLD", "Buildings"),
    "11": ("NAT", "Nature"),
}

WAN_MODE_FOLDERS = {
    "T2V",
    "I2V",
    "V2V",
    "T2I",
    "I2I",
    "IMG2VID",
    "IMAGE2VIDEO",
}


def normalise_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def parse_base_and_category(
    file_path: str,
    root_dir: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse model family and category from the existing library folder shape.

    This preserves the current indexer behaviour while sourcing model-family
    codes from the Phase 8.9 registry.
    """
    root_dir = normalise_path(root_dir)
    file_path_norm = normalise_path(file_path)

    try:
        rel_path = os.path.relpath(file_path_norm, root_dir)
    except ValueError:
        return None, None, None, None

    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        return None, None, None, None

    base_model_folder = parts[0]

    category_index = 1
    if base_model_folder in ("WAN2.1", "WAN2.2") and len(parts) >= 4:
        mode_folder = (parts[1] or "").strip().upper()
        if mode_folder in WAN_MODE_FOLDERS:
            category_index = 2

    category_folder = parts[category_index]

    base_model_name = base_model_folder
    base_model_code = None
    mapped = BASE_MODEL_MAP.get(base_model_folder)
    if mapped is not None:
        base_model_code, display_name = mapped
        base_model_name = INDEX_NAME_OVERRIDES.get(base_model_folder, display_name)

    category_code = None
    category_name = None
    first_token = category_folder.split(" ")[0].strip()
    mapped_category = CATEGORY_INDEX_MAP.get(first_token)
    if mapped_category is not None:
        category_code, category_name = mapped_category
    else:
        category_name = category_folder

    return base_model_name, base_model_code, category_name, category_code
