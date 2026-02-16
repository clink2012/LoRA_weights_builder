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
- Prepare for advanced compatibility-aware multi-LoRA composition

The system is built for power users working with:

- Flux (Transformer-based)
- UNet-style LoRAs (57-block extraction)
- Future WAN / additional architectures

This project prioritizes correctness, layout-awareness, and extensibility.

---

## Why This Exists

Most LoRA workflows treat models as opaque strength sliders.

This project exposes:

- The actual structural block weights
- The layout architecture behind each LoRA
- Compatibility constraints between LoRAs
- User-defined tuning profiles

It enables informed combination instead of blind stacking.

---

## Architecture

### Backend
- Python
- FastAPI
- SQLite
- Safe migration system
- Layout taxonomy engine
- Block extraction engine (Flux + UNet 57)

### Frontend
- React (Vite)
- Layout-aware rendering
- Dynamic block visualization
- Profile creation / editing system

---

## Current Feature Set

### Block Layout Awareness
- Automatic layout classification
- 57-block UNet extraction
- Flux transformer layout support
- API-consistent layout return

### Visualization
- Per-block weight bars
- Real vs fallback detection
- Block count display
- Clean layout-specific rendering

### Profiles (Phase 5.1)
- Save block override profiles
- Edit existing profiles
- Load profile onto active LoRA
- Copy weights to clipboard
- CSV export
- Background indexing improvements

---

## Project Roadmap

See: `Project Roadmap` issue for full tracking.

### Completed
- Phase 1 – Backend stability
- Phase 2 – Layout taxonomy
- Phase 3 – UNet block extraction
- Phase 4 – UI enhancements
- Phase 5.1 – Profiles + analytics foundation

### In Progress
- Phase 5.2 – Usability refinement & visualization polish

### Upcoming
- Phase 6 – Multi-LoRA combination engine
  - Compatibility checking
  - Weighted average math
  - Combined visualization
  - Safety warnings

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

---

### Database Notes
- Schema self-heals on startup
- Safe for older DB versions
- Concurrent-safe migrations
- Layout stored per LoRA
- Block weights stored in lora_block_weights

---

### Design Philosophy
- Layout-aware first
- Never assume compatibility
- No silent fallback logic
- Visual clarity over clutter
- Scalable for 10k+ LoRA libraries

---

### Future Vision
- Phase 6 introduces structural LoRA combination:
- Multi-select LoRAs
- Enforced base + layout compatibility
- Weighted combination math
- Combined block graph
- Risk indicators for incompatible merges

---

This turns the tool into a structural LoRA engineering platform.

---

### License
MIT License

---

### Maintainer
Developed and maintained by Clink

---
