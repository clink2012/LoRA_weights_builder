# GUI preference discovery

Status: revised design discovery, 3 October 2026. No production GUI design is approved. The earlier Guided Studio / Mixing Desk / Library Workbench previews are superseded: they did not express the visual design the owner wants and over-emphasised scalar strength. A guided interaction remains an option, not approval of those previews.

## Purpose and fixed behaviour

The shared workflow is **Choose compatible LoRAs → review roles and block contributions → adjust individual block weights → copy a full vector per loader → save a recipe**. Overall strengths are supporting settings. Block weights must be visible and central, not relegated to an advanced screen.

- Show each LoRA's complete ordered vector with clearly labelled block groups, exact loader target, BASE semantics and input-slot count. Architecture, adapter coverage and loader contract determine the mapping; never hard-code one universal length.
- Offer per-block inspection/editing and an understandable comparison of multiple LoRAs. Overlapping or neighbouring block activity is structural evidence, not a proven semantic conflict or guaranteed predictor of image quality.
- Keep full numeric export available even when the owner uses A/B placeholders. Named experiments must show affected blocks, current values, explicit minimum/maximum values and why those limits were chosen. Suggested trial bounds, loader limits and visually tested bounds are different things.
- Default → named personal variant → exact saved composition. Edits and A/B experiments create separate entries; returning to Default or an earlier version must not delete personal history.
- One authoritative calculated result powers screen values, copied vectors and saved recipes. Show which result belongs in each node in the chosen chain order.
- Explain compatibility and support limits before export. Unknown or unsupported is not a green status. Recognise LoRAs by stable identity and relative path, not filename alone.
- Keyboard access, readable text, visible focus/selection states and useful empty/loading/error states remain essential in every theme.

## Temporary local questionnaire

The owner requested a multiple-choice web interface with visual examples instead of a prose-only questionnaire. It is a design-discovery tool, separate from the application and its database.

- Source: `tools/gui-questionnaire`.
- Bender launcher: `tools/gui-questionnaire/Launch-Questionnaire.ps1`.
- Local address: `http://127.0.0.1:5186` (loopback only).
- Answer storage: `.local/gui-questionnaire/`, excluded from Git. Use the exact saved-result path reported by the tool; opening the questionnaire does not prove that answers were submitted or saved.
- Startup and browser validation must be checked when launching; these documented paths alone are not a claim that the service is running.

The seven supplied references suggest several qualities to explore: deep navy/purple panels with luminous accents, translucent layered surfaces, rounded modular cards, a bright green/white alternative, and dense coloured analysis displays. They are style references, not a request to reproduce unrelated dashboards, accounts, calendars or decorative metrics.

Show distinct finished-looking examples using LoRA-specific content. Let the owner compare the same block-weight task under different palettes, surface treatments and layouts. Demonstrate what each choice changes; avoid making them infer the design from names such as 'modern' or 'clean'. All example weights and experiment limits must be labelled illustrative, not real LoRA recommendations.

The questionnaire should cover:

1. Overall visual character and colour palette, with selectable examples.
2. Surface treatment: solid, softly layered or more pronounced glass/glow; keep legibility demonstrable.
3. Workspace layout: guided flow, block comparison workspace or library-led composition; show the full-vector output in each.
4. Density, navigation and the usual/maximum stack size.
5. Block presentation and editing: colour matrix, curves/bars and precise numeric controls, with a way to inspect exact values.
6. How A/B experiments, suggested ranges and Default/personal variants should appear.
7. Library thumbnails, starting view and copy/paste presentation.
8. Optional dislikes and a final review of the selections before saving.

Already decided: Bender only; FLUX.1 first without abandoning other model families; full per-LoRA vectors; preserved Default and variant history. LTX/MiniMax analysis and comparison foundations remain in scope, with existing/newer node research and a focused companion node authorised if needed. Do not reopen these as unanswered questions.

## Acceptance material needed later

The supplied old FLUX.2/FLUX.1 workflow establishes the daisy-chained loader-bank pattern; see `workflow-reference-2026-10-03.md`. It is an example, not a current acceptance fixture. Prepare a separate ordinary FLUX.1 test with current compatible LoRAs, ideally identity plus clothing/pose/style. Record the base model, LoRA files/versions, intended result, prompts, seed and sampling settings. The owner's judgement of controlled renders establishes usefulness; mockup charts and a numerical score do not.
