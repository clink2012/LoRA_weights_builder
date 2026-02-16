# LoRA Weights Builder – Project Roadmap

This issue tracks the overall development roadmap for the LoRA Weights Builder project.  
Individual tasks should be broken into separate Issues and linked here.

---

## ✅ Phase 1 – Backend Stability (Foundation)

- [x] API fallback for non-block LoRAs (neutral block profile)
- [x] API-side schema self-healing (ensure missing columns are auto-created)
- [x] Concurrency-safe migration (duplicate column name race handled)
- [x] Ensure DB migrations are safe across older databases
- [x] Add block_layout validation on API responses
- [x] Add reindex-single-LoRA endpoint (quality-of-life)

---

## ✅ Phase 2 – Block Layout Architecture

- [x] Define block_layout taxonomy (flux_transformer_19, unet_57, etc.)
- [x] Store layout type reliably during indexing - FLUX
- [x] Store layout type reliably during indexing - All
- [x] Return layout consistently via API
- [x] Update UI to render layout-aware block counts

---

## 🔲 Phase 3 – UNet-Style Block Extraction

Phase 3 implementation is now tracked in Issue #11:

- [x] Design UNet layer-to-block mapping strategy
- [x] Implement 57-block extraction for UNet-style LoRAs
- [x] Store computed weights in lora_block_weights
- [x] Mark has_block_weights = 1 for extracted LoRAs
- [x] Reindex existing LoRAs
- [x] Validate results with test cases

---

## 🔲 Phase 4 – UI Enhancements

- [x] Show fallback badge when fallback=true
- [x] Display lora_type clearly in UI
- [x] Visual differentiation between real and fallback blocks
- [x] Add sorting/filtering by layout type

---

## 🔲 Phase 5 – Advanced Features

### Phase 5.1 (Implemented) – Profile Editing & UX Refinement  
Tracked in Issue #14.

- [x] Block strength analytics (mean/variance groundwork)
- [x] Export block weights to CSV
- [x] User override profiles (create/edit/delete)
- [x] Copy weights functionality
- [x] Improved details panel layout
- [x] Performance improvements for large libraries

---

### Phase 5.2 – Usability & Interaction Refinement (In Progress)

Focus: Improve editing workflow and prepare UI for multi-LoRA features.

- [ ] Inline block weight editing (numeric input + slider sync)
- [ ] Real-time validation (range enforcement + visual feedback)
- [ ] Unsaved changes detection & warning
- [ ] Reset-to-original weights button
- [ ] Compact block spacing option (UI density mode)
- [ ] Sticky analytics header (mean/variance always visible)
- [ ] Minor layout polish and alignment consistency

---

## 🔲 Phase 6 – Multi-LoRA Composition Engine (Major Feature)

Heavy feature set, separated intentionally from Phase 5.

- [ ] Multi-LoRA selection
- [ ] Compatibility validation (same base model + layout required)
- [ ] Weighted average combination math
- [ ] Combined weights visualization
- [ ] Safety warnings for incompatible combinations
- [ ] Export combined profile as new custom profile

---

## Future Ideas (Parking Lot)

- [ ] Multi-model support beyond Flux/WAN
- [ ] Auto-detect LoRA compatibility with checkpoints
- [ ] API auth layer (if exposed externally)
- [ ] Web dashboard improvements

---

This roadmap is evolving and will be updated as development progresses.
