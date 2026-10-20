---
name: smoke-tuning
description: Breach's procedure for fine-tuning how smoke LOOKS: which dial lives where (render-only relief/medium dials vs the physics rows that feed the light engine), the live loop with Erik at the screen (Ctrl+R, debug views, the old-look A/B), the look-shot before/after, what each relief dial does and the traps found while building it (G1, 2026-10-19). Load before any smoke-look tuning session, before changing a [render.gas_relief] / [render.gas_medium] / [render.gas_detail] / [smoke] / [gases.*] smoke dial, and after #88 (smoke-unit rescale), which owes a retune.
---

# Tuning the smoke look

*Status: PROVISIONAL. Written 2026-10-20 from the G1 build (#93: 15 steps, each judged on the
look shots) and its merge. No tuning session has used it yet. Every run amends it before merging
(see "Keeping this skill honest").*

The smoke you see is two things stacked:

- **The simulation's smoke field.** How much soot is where, carried by the air. This decides
  WHERE smoke is and how thick.
- **The smoke relief.** A render-only shader that turns that field into a lit 3D cloud seen
  from above. This decides how it LOOKS.

Tuning the look means the second. Touch the first only in one of Erik's physics sessions.

## 1. Which dial is which: render or physics

| Section in `config.toml` | Side | Moves a digest/golden? | Who decides |
|---|---|---|---|
| `[render.gas_relief]` | render: shape, edge, height, lighting, layers | never | Claude tunes; Erik judges by eye |
| `[render.gas_medium]` | render: `tau_curve_a/b` (opacity curve), `glow_gain` (god-ray brightness) | never | as above. **`tau_curve_a` is 1.6 to keep the old look; 1.0 is "opacity == the light engine's extinction". Which one is Erik's call** |
| `[render.gas_detail]` | render: the noise clock (`cycle_seconds`), `adv_gain`; the rest is the old erosion look | never | as above |
| `[smoke]` | feeds the light engine's columns and the medium's base scale | light is not digested (until #81) | Erik |
| `[gases.smoke]` and other gas rows | physics: `absorption`, `scatter_albedo`, `heat_absorb` (DERIVED, re-derived by `tests/test_gas_heat_absorb.py`), `decay`, `diffusion` | `heat_absorb` / `decay` / `diffusion` move the sim | Erik's physics session; never in a look session |

If the fix you want is "less smoke" or "smoke that lasts longer", that is physics (`soot_yield`,
`decay`, #88), not a look dial. Say so; don't fake it with `cov_tau_*`.

## 2. The live loop (Erik at the screen)

```powershell
cd C:\Users\steen\projects\breach                 # or the session's worktree
C:/Users/steen/anaconda3/python.exe main.py --level smoke_light_studio
C:/Users/steen/anaconda3/python.exe main.py --level fire_studio
```

- **Ctrl+R reloads `config.toml` live.** `[render.gas_relief]` is read every frame, so a dial
  change shows instantly. Give Erik the exact line to edit and the value to try. Change one
  dial at a time.
- **A/B:** `enabled = false` in `[render.gas_relief]` draws the pre-G1 erosion look.
- **Debug views:** `debug_view` = 1 coverage, 2 height, 3 N·L, 4 self-shadow, 5 AO, 6 boil.
  Use them to answer "which stage makes this ugly?" before you touch a dial. Set it back to 0
  before committing.
- Erik needs exact copy-paste commands (python path, level name). He has stumbled on both.

## 3. Before/after pictures (Claude's own eyes)

```powershell
C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py --shots                       # all
C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py --shots smoke_lamp_puff,fire_plume
C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py --shots smoke_lamp_puff --shots-out scratchpad/look/try1
```

- Output goes to `scratchpad/look/<name>_<git-sha>.png`. **The file name carries the COMMIT,
  not your dials.** Two uncommitted tries overwrite each other, so give each try its own
  `--shots-out` folder.
- The smoke shots are `smoke_lamp_puff` (a lit puff: the main judge), `smoke_plume_noheat`,
  `fire_plume`, `smoke_light_studio`, `steam_flashlight` and `steam_flashlight_game`,
  `smoke_over_squad` (the layers) and `explosion_t40_noheat`.
  - **Judge steam on `steam_flashlight_game`.** The studio's `light_gain` 10 over-exposes it
    since G0/G1.
- Look at every PNG with Read, side by side with the before. Say plainly if it got worse.
- A new situation to judge gets a row in `tools/look_shots.toml`, not a one-off script.

## 4. What the relief dials do (`[render.gas_relief]`)

The pipeline is coverage → height → lighting → self-shadow → powder/AO → rim → in-scatter →
layers. Tune in that order: a later stage cannot fix an earlier one.

- **Coverage: where the cloud's edge is.** `cov_tau_*` is the optical depth the edge sits at:
  higher shrinks the cloud. `cov_edge_tiles` is the edge sharpness where the field has a
  gradient; `cov_soft_*` is the edge softness where it is flat. `cov_push_tiles` is how far
  billows bulge the edge. `haze_*` is how much thin smoke survives outside the edge.
- **Shape:** `billow_cell_tiles` is the lump size (0.75 read as crumpled rock, 1.5 as billow
  masses). `billow_amp` is how deep the lumps carve. `height_tiles` / `height_tau` are how tall
  a thick column stands. `warp_tiles` is the lumps' wobble.
- **Motion:** `boil_t0..t1` / `boil_speedup` make hot smoke churn faster; check them against
  the soot's actual T range in the scene. `drift_cap_tiles` stops blast wind shearing the
  billows into streaks.
- **Lighting:**
  - `lamp_z_*` is the light's elevation. Lower means longer self-shadow and more relief;
    `lamp_z_ceiling` 0.8 cast almost none.
  - `wrap` keeps thin smoke from going black.
  - `top_light_*` lights a soot column's top from the brightest light nearby. It exists
    because, before #88, the light engine's light dies inside a plume.
- **Detail:** `shadow*`, `powder` (bright, fluffy thin edges), `ao` (dark lanes), `rim` /
  `rim_g` (silver lining against a lamp) and `glow_texture` (billow texture on the god rays).
- **Layers:** `ceiling_gases` are drawn above the units; every other gas is a floor layer below
  them (Erik's ruling, 2026-10-18).
- **Shadows and glow-through (G2):** `cast_shadow_*`, `cast_parallax`, `cast_lift_ceiling` (the
  smoke's shadow on the floor and the units; `debug_view` 7 shows it), `glow_through_*` and
  `own_light_cut` (the fire under the smoke), `low_band_*` (the synthetic ~1 m band; height carve
  off). Doc §12 says what each try looked like.
- **Soot brightness:** the surface uses thick-cloud reflectance from the gas's
  `scatter_albedo` (soot 0.22 → about 0.06, steam 0.92 → about 0.56). "Soot looks too pale or
  too dark when lit" is therefore either `[gases.smoke] scatter_albedo` (physics, Erik) or the
  lighting dials above (render). Say which before changing anything.

## 5. Traps found while building it (G1, 2026-10-19)

- **MUST:** when the billows vanish, check texture binding first. raylib binds only 4 extra
  textures and silently drops the 5th onward, so the billow field read as zero. The pass binds
  its own units 5–10. *Incident: G1 step 3, a whole step judged on a blank billow field.*
- **MUST:** never judge "the lighting does nothing" without checking the light inside the
  plume. Pre-#88 soot absorbs about 138 per unit density per tile, so the planar light there
  is about 0. *Incident: G1 step 3, hence the top light.*
- Soot right over a burning tile is lit by that tile's own light, as if from above. That's
  G2's glow-through, not a dial to fight.
- The look-shot child ignores keyboard and mouse. Before that fix, a stray wheel turn moved the
  camera mid-shot.

## 6. Owed retune after #88 (smoke-unit rescale)

These were set on a field that goes opaque and then vanishes; retune them once #88 lands:
`cov_tau_*`, `cov_soft_*`, `haze_*`, `height_tau`, `top_light_*` (possibly to 0). Also
recheck the boil range after #12's fire tuning. Doc: `docs/graphics_direction_2026-10-18.md`
§11.

**#88 landed 2026-10-20: this retune is now DUE.** Smoke now thins along the configured
decay (125 s e-fold for soot) and diffuses at visible densities, so plumes live roughly ten
times longer than the field these dials were set on. Re-shoot `plume_motion_t48..t96` for
the §2.5 B call. The density is still in real units (only its resolution changed), so
absorption coefficients and the gas door's numbers keep their meaning; raw trace counts
are 256x larger: read them through `gas_fixed.trace_*`, never `/65536`.

## 7. Landing a tuning

- Dials only: commit `config.toml` with each changed value's reason in its comment (what it
  looked like before, Erik's words). In a shared tree, read `git diff config.toml` and stage
  only your own hunks: Erik may be tuning there too.
- It's feel-adjacent, so HUMAN-TEST: Erik plays it before it merges.
- Run `pytest tests -q`. `tests/test_gas_relief.py` refuses missing, mistyped or out-of-range
  dials.
- A look Erik loves but that doesn't fit the main smoke is kept, not lost: record its recipe and
  picture as a candidate smoke type (precedent: #96, "r1b",
  `docs/graphics_direction_2026-10-18/smoke_look_r1b.md`). **Commit or save every try you show
  Erik.** r1b was never committed and had to be rebuilt from the implementer's memory.

## Keeping this skill honest

Each run that uses it adds one dated line below before merging: what was skipped, what failed,
what was learned, what took longer than expected. A step becomes a MUST only with its incident
beside it, and is demoted the same way. The skill stays PROVISIONAL until three runs in a row
needed no amendment.

- 2026-10-20: written from G1's build; not yet used by a tuning session.
- 2026-10-20 (G2 build): followed §3 for every step (one `--shots-out` folder per try, every judged try committed). Learned: give each A/B its own folder by flipping the dial in config, not a git stash (a stash also hid the new look-shot rows); a long Python edit goes through a script file (Bash truncates past ~8 KB); a `str.replace` with an empty match once corrupted config.toml -- check `git diff config.toml` after every scripted edit. Pre-#88 soot hides most of G2's cast shadow: judge it after #88.
- 2026-10-20 (#88, not a tuning run): §6 marked due; raw trace counts are now 2^24 per unit.
