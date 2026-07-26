from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

SupportLevel = Literal[
    "mixed-scanned-fallback",
    "metadata-only",
]


@dataclass(frozen=True)
class ModelFamilyDefinition:
    folder_name: str
    code: str
    display_name: str
    support_level: SupportLevel
    metadata_indexing: bool
    block_analysis: bool
    fallback_layout: bool
    role_aware_orchestration: bool
    architecture_key_normalisation: bool
    comfyui_export: bool
    notes: str


MODEL_FAMILIES: tuple[ModelFamilyDefinition, ...] = (
    ModelFamilyDefinition(
        folder_name="FLUX",
        code="FLX",
        display_name="Flux",
        support_level="mixed-scanned-fallback",
        metadata_indexing=True,
        block_analysis=True,
        fallback_layout=True,
        role_aware_orchestration=True,
        architecture_key_normalisation=True,
        comfyui_export=True,
        notes="Flux scanner is implemented; individual rows may still be scanned or fallback-only.",
    ),
    ModelFamilyDefinition(
        folder_name="Flux Krea",
        code="FLK",
        display_name="Flux Krea",
        support_level="mixed-scanned-fallback",
        metadata_indexing=True,
        block_analysis=True,
        fallback_layout=True,
        role_aware_orchestration=True,
        architecture_key_normalisation=True,
        comfyui_export=True,
        notes="Uses the current Flux scanner path; individual rows may still be scanned or fallback-only.",
    ),
    ModelFamilyDefinition(
        folder_name="Flux.2-Klein",
        code="F2K",
        display_name="Flux.2-Klein",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Known mounted family. Metadata support only until architecture-specific scanner and layout tests exist.",
    ),
    ModelFamilyDefinition(
        folder_name="Illustrious",
        code="ILL",
        display_name="Illustrious",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Metadata-only in the current indexer.",
    ),
    ModelFamilyDefinition(
        folder_name="LTXV2",
        code="LTX",
        display_name="LTXV2",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Known mounted family. Metadata support only until architecture-specific scanner and layout tests exist.",
    ),
    ModelFamilyDefinition(
        folder_name="PONY",
        code="PNY",
        display_name="Pony",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Metadata-only in the current indexer.",
    ),
    ModelFamilyDefinition(
        folder_name="SD",
        code="SD1",
        display_name="SD 1.x",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Legacy mapped family; not present in the current mounted top-level folder audit.",
    ),
    ModelFamilyDefinition(
        folder_name="SDXL",
        code="SDX",
        display_name="SDXL",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Metadata-only in the current indexer.",
    ),
    ModelFamilyDefinition(
        folder_name="WAN2.1",
        code="W21",
        display_name="WAN 2.1",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Mode-aware folder parsing exists; block extraction is not implemented.",
    ),
    ModelFamilyDefinition(
        folder_name="WAN2.2",
        code="W22",
        display_name="WAN 2.2",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Mode-aware folder parsing exists; block extraction is not implemented.",
    ),
    ModelFamilyDefinition(
        folder_name="Z-Image",
        code="ZIM",
        display_name="Z-Image",
        support_level="metadata-only",
        metadata_indexing=True,
        block_analysis=False,
        fallback_layout=False,
        role_aware_orchestration=False,
        architecture_key_normalisation=False,
        comfyui_export=False,
        notes="Known mounted family. Metadata support only until architecture-specific scanner and layout tests exist.",
    ),
)


def iter_model_families() -> Iterable[ModelFamilyDefinition]:
    return MODEL_FAMILIES


def get_model_family_by_folder(folder_name: str) -> ModelFamilyDefinition | None:
    key = str(folder_name or "").strip().casefold()
    return next((family for family in MODEL_FAMILIES if family.folder_name.casefold() == key), None)


def get_model_family_by_code(code: str) -> ModelFamilyDefinition | None:
    key = str(code or "").strip().upper()
    return next((family for family in MODEL_FAMILIES if family.code == key), None)


def base_model_map() -> dict[str, tuple[str, str]]:
    """Return the folder mapping shape expected by the existing indexer."""
    return {
        family.folder_name: (family.code, family.display_name)
        for family in MODEL_FAMILIES
        if family.metadata_indexing
    }


def api_model_families() -> list[dict[str, object]]:
    return [asdict(family) for family in MODEL_FAMILIES]
