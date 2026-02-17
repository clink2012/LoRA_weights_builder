# LoRA Weights Builder

![Python](https://img.shields.io/badge/backend-Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/UI-React-61DAFB?logo=react&logoColor=black)
![SQLite](https://img.shields.io/badge/database-SQLite-003B57?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/status-active_development-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A layout-aware LoRA inspection, analytics, and profile-building tool for advanced block-level control.

---

## Overview

LoRA Weights Builder is a full-stack application designed to:

- Index large LoRA libraries
- Extract block-level weight information
- Classify layout types (Flux, UNet-style, etc.)
- Visualize per-block strengths
- Build and store custom block override profiles
- Prepare for compatibility-aware multi-LoRA composition

Built for power users working with:

- Flux (Transformer-based)
- UNet-style LoRAs (57-block extraction)
- WAN / additional architectures (planned)

This project prioritizes correctness, layout-awareness, and extensibility.

---

## Why This Exists

Most LoRA workflows treat models as opaque strength sliders.

This tool exposes:

- The structural block weights (what the LoRA actually does per block)
- The layout architecture behind each LoRA
- Compatibility constraints between LoRAs
- User-defined tuning profiles and overrides

It enables informed combination instead of blind stacking.

---

## Architecture

### Backend
- Python + FastAPI
- SQLite (self-healing schema + safe migrations)
- Layout taxonomy engine
- Block extraction engine (Flux + UNet 57)

### Frontend
- React (Vite)
- Layout-aware rendering + block visualization
- Profiles (create/edit/load/export)
- High-performance interaction for large layouts

---

## Current Feature Set

### Block Layout Awareness
- Automatic layout classification
- 57-block UNet extraction
- Flux transformer layout support
- API-consistent layout return

### Visualization + Editing UX (Phase 5)
- Interactive block weight bars (pointer drag)
- Inline numeric editing (with clamping/validation)
- Dirty state tracking + reset controls
- Stable scrolling/layout (no overlapping controls)
- Copy weights + CSV export (including fallback LoRAs)
- UI smoke test (Vitest + jsdom)
- Production build verified (`npm run build`)

### Profiles
- Save and edit block override profiles
- Load profile onto active LoRA
- Copy weights to clipboard
- Export weights to CSV

---

## Project Roadmap

See: the `Project Roadmap` issue for full tracking.

### Completed
- Phase 1 – Backend stability
- Phase 2 – Layout taxonomy
- Phase 3 – UNet block extraction
- Phase 4 – UI enhancements
- Phase 5 – Profiles + analytics foundation + UX refinement (5.1 + 5.2 complete)

### Upcoming
- Phase 6 – Multi-LoRA composition engine
  - Multi-select LoRAs
  - Compatibility validation (same base model + layout required)
  - Weighted combination math
  - Combined weights visualization
  - Safety warnings for incompatible combinations
  - Export combined profile as a new custom profile

---

## Installation

### Backend
cd Database/backend  
pip install -r requirements.txt  
python lora_api_server.py  

### Frontend
cd Database/UI  
npm install  
npm run dev  

### Useful Dev Commands
npm run lint  
npm run test  
npm run build  

---

## Database Notes
- Schema self-heals on startup
- Safe for older DB versions
- Concurrent-safe migrations
- Layout stored per LoRA
- Block weights stored in `lora_block_weights`

---

## Design Philosophy
- Layout-aware first
- Never assume compatibility
- No silent fallback logic
- Visual clarity over clutter
- Scalable for 10k+ LoRA libraries

---

## Future Vision

Phase 6 turns this into a structural LoRA engineering platform:

- Select multiple LoRAs
- Enforce base + layout compatibility
- Combine weights using defined math (not vibes)
- Visualize the combined block graph
- Flag risky merges with clear warnings
- Export the combined result as a reusable profile

---

### License
MIT License

---

### Maintainer
Developed and maintained by Clink
