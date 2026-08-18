# B1 build log — blackbody ramp + brightest-K fire lights (as-built)

Fire & Heat Beauty arc, beat B1. Built 2026-07-21 (Opus, autonomous-patch-
workflow) from the locked design `docs/fire_b1_blackbody_fire_lights_design_
2026-07-21.md`. Branch `fire-b1-blackbody-lights` (off main), pushed. **RENDER-
ONLY: zero sim / cpp / digest / golden surface touched.** FEEL-ADJACENT →
awaiting Erik's HUMAN-TEST before any merge (never auto-merge).

## What shipped (commits)

- **P0** `e55700c` — merged `fire-lit-search` (docs-only: `docs/fire_rendering_
  research.md`). ⚠ NOTE: that branch is ANCIENT — its tip-vs-main diff shows
  ~202k deletions, but the true delta vs the merge-base is ONLY that one doc
  (verified), so the merge is clean docs-only. Nothing else changed.
- **P1** `bfa3be1` — `renderer/blackbody.py`: `BlackbodyRamp` bakes a 1D LUT,
  game-ΔT → pseudo-Kelvin (`kelvin_ambient + k_temp_to_kelvin·T`) → Tanner
  Helland/Bartlett normalized chroma (max channel = 1) + a SEPARATE Macklin
  reference-exposure T⁴ intensity. `chroma_intensity(field)` (overlay) and
  `light_color(t)` (lights) index the SAME LUT the SAME way (float64 both) so
  they agree bit-for-bit. `[render.blackbody]` config. 11 tests.
- **P2** `4e479fd` — `HeatFieldOverlay` rewired to `chroma·intensity`, ACES
  tone-mapped (same Narkowicz curve as `lighting.fs`), split into peak-normalized
  hue (RGB) + brightness (alpha) so the additive draw reconstructs the tone-
  mapped emissive. Deleted the 5-stop ramp + `temp_display_max`. Overlay A/B
  defaults: black-body overlay ON (`T`), flat-orange `FireOverlay` demoted OFF
  (`F3`). +5 tests.
- **P3** `b3e9473` — `renderer/fire_lights.py`: candidates (`T ≥ t_light_min`) →
  3×3 non-max suppression (numpy sliding-window) → brightest `max_lights` by
  temperature → omni `LightSource` param dicts, colour+intensity from the shared
  ramp. **Structural `heat=0.0`, `jitter=0.0`** — render lights NEVER write the
  synced heat channel (the sim casts fire heat separately). Wired into `main.py`
  via the existing `_build_light_source` loop. `[render.fire_lights]`. 15 tests.
- **P4** `ac391e1` — HUD `Fire lights: kept/peaks cap N` (amber + `CAP!` when the
  brightest-K cap truncates — no silent caps), live **`L`** toggle, legend line.

## Gates (all green)

- `pytest tests -q` green at every patch — **1012 passed / 22 skipped** after
  P3/P4 (was 997 pre-B1; +31 new B1 tests). All digest/golden tests pass, so
  **goldens are byte-untouched** (as guaranteed by construction — render-only).
- No `renderer/` → `src/simulation/` import added.
- E2E: the game boots into the main loop clean (no traceback) with the overlay
  rewire + fire selection + HUD executing every frame.

## §6 ACES precondition check — PASSES (no shader change)

Read `shaders/lighting.fs`: `aces_tonemap` operates component-wise on a `vec3`
(each channel through the same Narkowicz rational — NOT a luma-only map), and
`incoming_rgb` stays unclamped HDR (sampled from a 16F texture) right up to the
tonemap. Both preconditions hold, so the black-body HDR intensity tone-maps
correctly with no per-channel clip / hue-shift. No shader edit needed. The
Python overlay mirrors the identical curve (`blackbody.aces_tonemap`).

## Perf spot-check

Headless CPU raycaster (the render caster), baseline = flashlight + 4 statics:
adding **16 fire lights (range 18)** costs only **~0.25 ms** of extra marching
@128² and @256² (baseline sub-0.1 ms) — comfortably in budget; ~1824 rays/frame,
matching the design's ~1810 estimate. `max_lights` is the budget lever, not range.

## ★ Tuning flag for the human test (read first)

At the design-default `k_temp_to_kelvin = 2.0`, the intensity is aggressive:
`light_color(3000)` already white-saturates (intensity = `i_max` = 8), while
`light_color(400)` is deep-red but nearly invisible (intensity ≈ 3e-4). The
visible deep-red→orange band sits around **T_game ≈ 1000–2500**. Real flame-zone
fields are ~kK-scale, so with defaults a fire will read quite white.
**First dial to reach for: lower `k_temp_to_kelvin`** (pushes the white end out,
reserving white for extremes per §7) — or raise `kelvin_ref`. All dials live in
`config.toml` `[render.blackbody]` / `[render.fire_lights]`.

## Human-test controls

- `T` — black-body emissive overlay (default ON). `F3` — old flat-orange overlay
  (default OFF). Toggle both to A/B the overlay look.
- `L` — fire light sources on/off (A/B the ray-traced fire glow lighting rooms /
  walls / marines).
- HUD shows the live fire-light count / NMS-peaks / cap.
- Full human-test script: design §7. Suggested dials: `k_temp_to_kelvin`,
  `intensity_*`, `t_light_min`, `max_lights`, `light_range`, `light_gain`.

## Housekeeping notes

- Fresh worktree had no built `breach_physics` — copied the main-tree
  `cp312-win_amd64.pyd` into `cpp/build/Release` (render-only beat → cpp is
  byte-identical to main). Not committed (gitignored build artifact).
- After Erik blesses + merges: delete the worktree
  `.claude/worktrees/fire-b1-blackbody-lights` and the local+remote branch;
  fold the as-built into the canon chapters at arc close (engine/06 fire-as-
  light, engine/08 ray engine).
