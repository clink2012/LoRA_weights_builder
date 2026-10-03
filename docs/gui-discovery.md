# GUI preference discovery

Status: design options only, 3 October 2026. No application GUI changes have been made. All mockup values are illustrative, not recommendations for actual LoRAs.

The shared workflow is **Choose compatible LoRAs → review roles and priorities → balance → copy settings per node → save a recipe**. Default and personal variants remain visibly distinct, with reset/rollback available. Technical block detail should be reachable without dominating every screen.

## Three genuinely different directions

| Direction | Everyday experience | Trade-off |
| --- | --- | --- |
| Guided Studio | A calm step-by-step composition screen, with a small selection tray and plain-language explanations. | Easiest to follow, but more steps when doing repetitive expert work. |
| Mixing Desk | Selected LoRAs appear as channels, with role, priority, strength and variant beside each other; output stays close by. | Best for comparing several LoRAs, but can become busy with large stacks. |
| Library Workbench | A searchable library table and a persistent composition/inspector pane. | Fastest for a large collection and keyboard use; more information is visible at once. |

Suggested starting point: Mixing Desk for composing, with a restrained searchable library drawer. This is a recommendation to compare, not an approved design.

## Short questionnaire

1. Which direction feels closest: Guided Studio, Mixing Desk or Library Workbench? Mixing parts is fine; say which parts.
2. Should the first screen open your library, resume your last composition, or offer a new composition?
3. Prefer dark, light, or follow Windows? Any accent colours to use or avoid?
4. Prefer a compact interface with more visible, or a roomier interface with advanced details tucked away?
5. How many LoRAs do you normally stack, and what is your realistic maximum?
6. Are preview thumbnails useful for choosing LoRAs, or are names, roles and folders enough? Use local images only.
7. Should individual block controls live in a side inspector or a separate advanced view?
8. For copy/paste, do you prefer one clearly labelled result card per loader node, with individual copy buttons, or a compact output table?

Already decided: FLUX.1 first; manual changes become their own saved variants; Default remains recoverable; Bender only. LTX/MiniMax analysis and comparison foundations remain in scope, with existing/newer node research and a focused companion node authorised if needed. Do not reopen these as unanswered questions.

## Behaviour that every direction must preserve

- Show compatibility and support limits before export; unknown or unsupported is not a green status.
- Recognise each LoRA by stable identity and relative path, not filename alone.
- Default → named personal variant → exact saved composition; show what is active.
- One authoritative calculated result powers screen values, copy/paste and saved recipe.
- Explain the main adjustment plainly; expose maths and block details on demand.
- Reset is non-destructive and does not delete saved variants.
- Remember local preferences and let the owner save a composition without losing its exact settings.
- Keyboard access, readable labels and useful empty/loading/error states; no technical infrastructure jargon in the normal workflow.

## Acceptance material needed later

Choose one ordinary FLUX.1 workflow and a small compatible set, ideally identity plus clothing/pose/style. Record the base model, LoRA files/versions, intended result, prompts, seed and sampling settings. We can inspect local files once selected. We need the owner's visual judgement for 'better', rather than pretending a numerical score establishes it.
