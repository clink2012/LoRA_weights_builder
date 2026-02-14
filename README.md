# LoRA Weights Builder
A structured LoRA indexing, inspection, normalization, and API system.

This project provides:
- Deterministic `stable_id` assignment
- Block-weight extraction (Flux)
- Layout normalization and taxonomy
- API-safe responses
- Layout-aware UI rendering
- Database invariant protection

The system is designed to move LoRA management from loosely inferred strings to a consistent architectural model.

---

# Architecture Overview
The system consists of:
Database/backend/
- lora_indexer.py
- delta_inspector_engine.py
- block_layouts.py
- lora_api_server.py
- lora_id_assigner.py
- phase2_smoke_check.py

Database/UI/
- React frontend (layout-aware display)

Data is stored in:
- Database/lora_master.db

---

# Phase 1 – Core Indexing & Stability (Completed)

Phase 1 established database integrity and stable identity guarantees.

## Objectives
- Deterministic `stable_id` assignment
- Eliminate duplicate IDs
- Ensure consistent indexing behavior
- Prevent schema drift
- Establish API baseline behavior

## What Phase 1 Implemented
### 1. Stable ID System

Each LoRA receives a structured `stable_id`:
- FLX-CLT-057
- FLK-UTL-004

This ensures:
- IDs remain consistent across re-indexing
- No duplicates exist
- Downstream references remain stable

### 2. Duplicate Protection

The system enforces:
- No duplicate `stable_id` groups
- Safe reassignment during repair
- Deterministic ID generation

### 3. Schema Self-Healing

The backend includes:
- Safe SQLite migration handling
- Protection against missing columns
- Defensive API validation

### 4. API Baseline

Endpoints guarantee:
- Safe fallback behavior
- No crashes on missing block weights
- Deterministic response structure

---

# Phase 2 – Block Layout Architecture (Completed)

Phase 2 formalized block layout handling across the stack.

This phase ensures:
- Flux LoRAs always store a normalized `block_layout`
- Layout taxonomy is centralized
- API responses are layout-consistent
- UI renders layout-aware block counts
- Database invariants are enforced

---

## 1. Block Layout Registry

New authoritative registry:
- Database/backend/block_layouts.py

This file defines:
- Canonical layout IDs
- Layout normalization rules
- Expected block count logic
- Flux layout inference

Example layouts:
- flux_transformer_<N>
- flux_double_<N>
- flux_te_<N>
- flux_unet_57
- flux_fallback_16


All layout logic must pass through this registry.

---

## 2. Flux Normalization Rules

During indexing:
- Flux LoRAs with block weights receive a normalized layout.
- 57-block UNet double+single LoRAs are mapped to:

flux_unet_57
- Flux LoRAs without block weights are assigned:

flux_fallback_16
No Flux row may have:
block_layout IS NULL

---

## 3. API Guarantees

`lora_api_server.py` ensures:
- Layout values are validated
- Flux fallback behavior is deterministic
- Block count mismatches are handled safely
- Search responses always return normalized layout

---

## 4. UI Improvements

The UI now:
- Displays layout-aware block counts
- Differentiates transformer vs UNet layouts
- Uses registry logic to determine expected blocks

---

## 5. WAN Scaffold

WAN (W21 / W22):
- Detected safely
- Indexed without crashing
- Returns placeholder metadata
- Block extraction not yet implemented

This prepares the system for Phase 3 expansion.

---

# Database Invariants

The system guarantees:
1. No Flux rows have `block_layout IS NULL`
2. No duplicate `stable_id` groups exist

These are verified using the smoke check tool.

---

# Smoke Check

Run from repo root:
python Database\backend\phase2_smoke_check.py

Expected output:
[CHECK] Flux rows with null block_layout
  value=0
  PASS

[CHECK] Duplicate stable_id groups
  value=0
  PASS

If either value is non-zero, system guarantees are broken.

### Development Workflow
## Pull latest main
git checkout main
git pull

## Compile backend sanity check
python -m py_compile Database\backend\block_layouts.py

## Run API server
python Database\backend\lora_api_server.py

### Layout Naming Philosophy
Layouts are structured identifiers:
<family>_<type>_<block_count>

Example:
flux_unet_57

This enables:
- Deterministic filtering
- Layout-based grouping
- Reliable downstream logic
- Architecture-safe expansion

---

### Current Status

- Phase 1 complete (Stable ID + schema safety)
- Phase 2 complete (Layout registry + Flux normalization)
- Phase 3 planned (WAN block extraction and inference)

The system has transitioned from string-based inference to structured architecture.
