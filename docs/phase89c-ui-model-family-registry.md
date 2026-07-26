# Phase 8.9c registry-backed UI model filter

This slice replaces the UI's hard-coded base-model dropdown with the backend model-family registry exposed by:

```text
GET /api/model-families
```

## Behaviour

- The UI requests the registry when the application loads.
- The backend registry controls the model codes and display order.
- Families whose declared support level is `metadata-only` are labelled with `· metadata only` in the dropdown.
- `FLX` and `FLK` remain unqualified because they have mixed scanned/fallback support.
- `All Models` remains the final option.

## Staggered-deployment fallback

The UI contains a complete fallback snapshot of the approved registry. If `/api/model-families` is unavailable during a staggered deployment, the dropdown remains usable and still includes:

- Flux.2-Klein (`F2K`)
- LTXV2 (`LTX`)
- Z-Image (`ZIM`)

All non-Flux fallback options are visibly marked metadata-only.

## Safety

This phase does not:

- run a scan
- write to the database
- assign stable IDs
- remove stale rows
- add block layouts
- change scanner or orchestration maths
- imply block analysis for metadata-only families

The dropdown can expose a family before it has indexed rows. Until a controlled indexing plan is approved and run, selecting such a family may legitimately return zero results.

## Validation

The UI smoke test mocks `/api/model-families` and verifies that the dropdown contains API-provided F2K, LTX and ZIM options with explicit metadata-only labels.
