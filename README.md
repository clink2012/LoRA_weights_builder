# LoRA Weights Builder

![Python](https://img.shields.io/badge/backend-Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/UI-React-61DAFB?logo=react&logoColor=black)
![SQLite](https://img.shields.io/badge/database-SQLite-003B57?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/status-active_development-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A layout-aware LoRA inspection, analysis, and deterministic multi-LoRA configuration platform built specifically for advanced ComfyUI users.

---

## Overview

LoRA Weights Builder is a full-stack application designed to:

- Index large LoRA libraries (thousands of models)
- Extract structural block-level weight data
- Classify layout types (Flux, UNet-style 57-block, etc.)
- Visualize per-block strength distribution
- Build, save, and reload custom block override profiles
- Deterministically calculate safe multi-LoRA configurations
- Output values ready for ComfyUI’s **LoRA Loader (Block Weight)** node

This tool does **not** merge LoRAs into a single synthetic LoRA.

Instead, it calculates properly scaled values for **each LoRA individually**, so they can be safely stacked in ComfyUI without destructive interference.

---

## Why This Exists

Most LoRA workflows rely on:

- A single global strength slider
- Blind stacking of multiple LoRAs
- Trial-and-error tuning
- “Vibes-based” merging

This tool exposes what actually matters:

- Block-level weight distribution
- Layout architecture behind each LoRA
- Compatibility constraints between LoRAs
- Deterministic scaling math
- Controlled per-LoRA contribution

The goal is simple:

> Combine LoRAs safely — without melting your output.

---

## Target Users

This project is built specifically for:

- ComfyUI users
- Users of the **LoRA Block Weight** node
- Advanced Flux / SDXL / UNet-based workflows
- Users managing large LoRA libraries (1k+ models)
- Image and video generation pipelines

---

## Architecture

### Backend

- Python + FastAPI
- SQLite with safe, additive schema migrations
- Layout taxonomy engine
- UNet 57-block extraction
- Flux layout support
- Deterministic multi-LoRA scaling engine
- Persistent combined configuration storage
- Safe fallback exclusion logic
- Stable ID system for deterministic referencing

### Frontend

- React (Vite)
- Layout-aware block visualization
- Profile editing + override UI
- Large-library support (pagination, search)
- Composition-ready structure (Combine tab in progress)

---

## Current Feature Set

### 1. Library Indexing

- Scans LoRA folders
- Assigns deterministic stable IDs
- Classifies base model family + category
- Stores layout type per LoRA
- Extracts block weights where supported
- Safe reindex (no ID collisions)

### 2. Block Layout Awareness

- Automatic layout classification
- UNet 57-block extraction
- Flux layout detection
- API-consistent block ordering
- CSV export for reproducibility

### 3. Profiles

- Save block override profiles
- Edit profiles with validation
- Load profile into active LoRA
- Export to CSV
- Deterministic storage (no silent transforms)

### 4. Multi-LoRA Configuration Engine (Phase 6 – Backend Complete)

- Multi-LoRA compatibility validation
- Enforces same base model + layout
- Excludes fallback LoRAs safely
- Deterministic scaling math per LoRA
- Returns structured configuration for **each selected LoRA**
- Save combined configuration set as persistent profile
- List and reload saved configuration sets
- Zero recompute on load
- Fully test-covered

Each LoRA in a configuration receives its own:

- strength_model
- strength_clip
- Block weight vector

Ready to paste directly into separate LoRA Loader nodes in ComfyUI.

---

## Project Status

### Completed Phases

- Phase 1 – Backend stability
- Phase 2 – Layout taxonomy
- Phase 3 – UNet block extraction
- Phase 4 – UI enhancements
- Phase 5 – Profiles + UX refinement
- Phase 6 – Deterministic Multi-LoRA Configuration Engine (Backend)
- Phase 7 – Combine UI

Backend engine is now stable and test-covered.

---

## Upcoming Phases

## Phase 8 – Role-Aware Stacking Engine

Phase 8 transitions the system from simple weighted combination to spatially-aware orchestration.

The objective is not UI expansion.  
The objective is deterministic stacking intelligence.

This is a personal power tool. Folder structure is authoritative.

There is:
- No heuristic guessing
- No manual dropdown overrides
- No naive equal scaling
- No dynamic architecture changes mid-phase

All orchestration decisions must derive from deterministic signals.

---

### Phase 8 Structure

8.1 – Folder-Derived Role (Foundation)
    - Role is inferred strictly from folder path
    - No override mechanism
    - Stored and exposed via API
    - Deterministic and testable

8.2 – Clip Contribution Awareness
    - Distinguish UNet-only vs CLIP-contributing LoRAs
    - Combine engine becomes aware of clip contributors
    - Clip aggregation rules become explicit and deterministic
    - No auto-scaling heuristics

8.3 – Block Energy Analysis
    - Introduce measurable block energy metrics
    - Compute spatial distribution of LoRA influence
    - Expose structured block-energy data via backend
    - No UI change until backend is validated

8.4 – Role-Aware Orchestration
    - Combine engine respects:
        - Role
        - Clip contribution
        - Block energy distribution
    - Introduce deterministic stacking policies
    - Maintain backward compatibility of combine schema

---

### Architectural Principle

Phase 7 = weight arithmetic  
Phase 8 = structural orchestration

Backend-first.  
Deterministic logic only.  
Tests must pass before UI layering.

---

### Current Status

- Phase 7 complete and stable (response_schema_version = 7.1)
- Phase 8.1 implemented or in progress
- Phase 8.2 is the active development target

### Phase 9 – Expanded LoRA Type Support

- Improve layout detection coverage
- Support Pony
- Support WAN
- Support additional UNet-style families
- Reduce fallback classification rate
- Increase block extraction coverage

### Phase 10 – Quality of Life

- Improved filtering
- Tagging
- Composition history
- Preset templates
- Export helpers for ComfyUI pipelines

---

## Installation

### Backend

cd Database/backend  
pip install -r requirements.txt  
python lora_api_server.py  

Backend runs on:  
http://127.0.0.1:5001

### Frontend

cd Database/UI  
npm install  
npm run dev  

Frontend runs on:  
http://127.0.0.1:5174

---

## Design Philosophy

- Layout-aware first
- Deterministic over “magic”
- No silent fallback logic
- Explicit compatibility enforcement
- Designed for large libraries (10k+ LoRAs)
- Built for real production workflows

---

## Long-Term Vision

LoRA Weights Builder becomes:

- A structural LoRA engineering cockpit
- A compatibility-aware configuration planner
- A deterministic scaling assistant
- A safety layer for complex ComfyUI pipelines

Not just a browser — but a control system.

---

### License

MIT License

---

### Maintainer

Developed and maintained by Clink
