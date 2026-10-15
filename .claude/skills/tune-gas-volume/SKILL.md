---
name: tune-gas-volume
description: How to tune the volumetric smoke (tilt view, issue #90) — the dials in [render.gas_volume], what each one does to the picture, the order to turn them, the debug views, and how to bake and compare a new detail volume. Use when the smoke look needs fine-tuning before a release.
---

# Tuning the volumetric smoke (`[render.gas_volume]`)

The system: `renderer/gas_volume.py` + `shaders/gas_volume.fs` raymarch a
baked detail volume above the ground plane of the tilt view. The SIM owns
the macro shape (per-tile optical depth `tau` from `renderer/gas_medium.py`),
the BAKED VOLUME owns everything below tile scale, the PHASE FIELD
(`renderer/volume_phase.py`) gives every tile its own moment in the loop.
Design and history: `docs/volumetric_smoke_design_2026-10-14.md`.

## The loop

1. Run the studio from the worktree with the conda `data` python:
   `python main.py --level smoke_light_studio --tilt` (Space runs the sim;
   add `--warp 3.0` for a frozen, reproducible frame).
2. Edit `[render.gas_volume]` in `config.toml`, save, press **Ctrl+R** in
   the game. Every dial applies next frame; only `asset` reloads a volume.
3. **F12** cycles the debug views: 0 normal, 1 tile checker on the floor
   (camera alignment), 2 the movie alone in flat grey (judge the bake and
   the repeat here, not in the lit view), 3 the lamp-direction field
   (brightness = how directional the light is), 4 the heat input (the
   blackbody overlay the fire glow multiplies).
4. Keep a reproducible before/after: `tools/e2e_drive.py --level
   smoke_light_studio --control gamepad --warp 3.0 --tilt --zoom 40 --pan
   2,14 --frames 30 --shot west.png@25` (writes into the CWD; the env var
   `BREACH_VOLUME_DEBUG=2` selects a debug view headless).
5. Commit the tuned values on the branch; feel-adjacent → HUMAN-TEST gate.

## Dials, in the order to turn them

**Shape (how much smoke the columns show)**
- `h_max_tiles` slab height = tallest column. `tau_h` = the optical depth
  at which a column reaches it. Lower `tau_h` → thin smoke already stands
  tall. `top_var` raggeds the tops (0 = flat lid).
- `tau_solid` = the optical depth where erosion stops. `erode` = how deep
  the detail carves thin smoke (0 = none, 1 = only the volume's cores stay).
  More erosion = wispier edges but LESS opacity; compensate with `k_ext`.
- `k_ext` extinction multiplier: the eroded volume shows ~1/3 of its mean
  density, so 3 restores the sim's per-tile opacity. If the floor shows
  through where the 2D view was opaque, raise it. `detail_gamma` = contrast
  on the smoke channel (>1 sharper, <1 softer).
- `step_texels` = march step in volume texels — THE cost dial. 1.5 is the
  sharpness/cost sweet spot; `steps` is only the cap.

**Light**
- The smoke's lit colour is the `[gases.smoke] scatter_albedo` row (0.22
  since #91: soot's real single-scattering albedo, charcoal-grey when lit).
  `albedo_floor` (0 = honest) only lifts darker gases artistically.
- `k_shadow` vertical self-shadow (dark bottoms when lamps are low);
  `light_from_below` = lamps at floor height (bright bases) vs sky-lit.
- `dir_light` lamp-facing side ×(1+a), back ×(1−a); `dir_eps_texels` the
  normal's step (bigger = smoother shading); `hg_g` / `hg_mix` the
  forward-scatter halo when a lamp is behind the smoke. Subtle on soot
  (its look is mostly the floor darkening), visible on lit plumes.
- `emit_gain` / `emit_pow` fire glow through the fire channel; `exposure`
  before ACES.

**Motion and the repeat (the phase field)**
- `phase_noise_amp` (loops) + `phase_noise_tiles` (wavelength): static
  de-tiling. 0.5 / 6 is the default; shorter wavelength hides the repeat
  better but shears the movie more within a tile.
- Sources: `phase_temp_amp` (ref `phase_temp_ref` game ΔT),
  `phase_press_amp` (ref `phase_press_ref` atm above the room median),
  `phase_wind_amp` (tamed wind speed, 0..1). In `offset` mode an amplitude
  is loops of shift per saturated source; in `speed` mode it is the rate
  deviation (loops per loop) and `phase_relax_loops` bounds the drift.
- `rise_turns` upward scroll per loop (whole heights, seamless);
  `loop_seconds` 0 = the asset's own.

## The bake (when the dials cannot get there)

`python tools/bake_smoke_volume.py --name NAME --res 128 128 64 --frames 48
--set KEY=VAL ...` → `assets/volumes/NAME.npz` (gitignored; ~12 min at 128;
`--res 64 64 32 --frames 24` ≈ 1 min for a parameter sweep). Judge with
`tools/view_smoke_volume.py NAME` (B&W movie) and the F12-2 view in game.
- The band floor `force_kmin` / `rho_kmin` / `phi_kmin` kills the
  box-scale blobs that repeat every period (the lattice, 2026-10-15); the
  forcing band `force_kmin..force_kmax` sets the billow size
  (3–5 per box ≈ a tile across with period 4).
- `rho_kmax` / `rho_spectrum` / `scalar_diffusion`: finer wisps vs softer
  smoke; `vort_eps` swirl; `buoyancy` rising motion; `period_tiles` /
  `height_tiles` the box (the vertical period MUST be as tall as the
  columns or tilted rays smear it into streaks).
- Sweep two or three parameters on a small grid at 64-res, pick by the
  mosaic, rebake the winner at 128, set `asset`, commit the dials.

## Pitfalls

- A dial only in `renderer/gas_volume.py::VolumeDials.DEFAULTS` is a
  fallback; `config.toml` is the source of truth. Add new dials to BOTH.
- Never judge the repeat in the lit view: the sim's macro shape hides it.
- The directional light costs three extra volume samples per lit sample;
  `dir_light = 0` removes them (laptop).
- A new 2D sampler in the shader must get its own GL unit through
  `gl3d.bind_texture_2d`: raylib places only four samplers, a fifth
  silently reads the quad's own texture (the red-smoke incident, #91).
