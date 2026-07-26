from __future__ import annotations

from types import ModuleType
from typing import Any

from lora_path_parser import BASE_MODEL_MAP, parse_base_and_category


def apply_model_family_registry(indexer_module: ModuleType | Any) -> None:
    """Apply the approved Phase 8.9 family mapping to the legacy indexer module.

    This is intentionally a narrow integration seam. It changes folder-to-code
    metadata parsing only. It does not alter scanner selection, block extraction,
    database schema, stable-ID logic, or indexing execution.
    """
    indexer_module.BASE_MODEL_MAP = dict(BASE_MODEL_MAP)
    indexer_module.parse_base_and_category = parse_base_and_category
