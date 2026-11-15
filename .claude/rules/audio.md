---
paths:
  - "audio/**"
  - "tools/sound_clips.py"
  - "tools/sound_audition.py"
  - "assets/sounds/**"
---

# Audio

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Sound layer
- Presentation only: it reads each sim step's `tick_events` (fed after EVERY step, beside `renderer.consume_events`) and a read-only view of the sim, writes nothing to it, draws from its own seeded generator, and is imported by main.py alone (never a headless path; muted in a hidden window).
- Decisions are the model's, as data (`Start` / `Stop` commands); the device decides nothing. A sound is a CUE row and a BINDING row: never a file name or a per-weapon branch in code; an unbound weapon or payload is silent.
- Every voice's gain comes from `acoustics`: a new propagation effect is a term there, never a second path. `reach = "loudness"` is a STOPGAP (Erik, 2026-11-12: loudness is a source level, not a distance; #201).
- raylib 5.5's pan runs 1.0 = left: converted in the device and nowhere else.
- ACCEPTED GAPS (v1): no occlusion; a one-shot keeps the gain and pan it started with; fades are stepped per frame.

## Sound samples
- Every game sample is written by `tools/sound_audition.py --export` from a generated clip and has its generator, source clip and prompt in `assets/sounds/SOURCES.md`: never a hand-cut file. The audition window is where a clip is judged by ear; it is not a second audio layer.
