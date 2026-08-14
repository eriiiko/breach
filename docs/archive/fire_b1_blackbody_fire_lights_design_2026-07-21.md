# B1 design — blackbody ramp + two-tier fire lights (render-only)

Arc: Fire & Heat Beauty + Tuning (living plan: `docs/plan-for-tuning-and-graphics.md`,
bottom section). Beat B1 of B1–B4. Written by Fable 2026-07-21 for an Opus build
session (autonomous-patch-workflow). Status: DESIGN — Erik has approved the arc
plan; this doc is the build spec.

Research base: `docs/fire_rendering_research.md` (branch `fire-lit-search`,
unmerged — MERGE IT as patch 0 of this build, it's a docs-only commit).
Executive pick ① there ("temperature-is-color, everywhere, physically") IS this
beat. Prior design threads: `docs/blackbody_smoke_and_rendering_brainstorm.md`
(§8 item 2 = the two-tier light decision, DECIDED 2026-07-05),
`docs/architecture/engine/06_temperature_and_fire.md` § "Fire as a light
source", `docs/architecture/engine/08_ray_engine.md` (fire-as-light: designed,
unbuilt).

## 0. Iron constraints

- **RENDER-ONLY. Zero sim writes, zero C++ changes, zero new sim state.**
  Everything here reads `gmap.temperature` (and optionally `gmap.fire`) and
  writes only render-side structures. Digests/goldens are untouched BY
  CONSTRUCTION — if any patch in this beat needs to touch `src/simulation/`,
  `cpp/`, or recorder/digest surface, STOP and escalate to Erik/Fable.
  (Exception: the read-only ACES check in §6 may propose a shader fix —
  shaders are render-layer, allowed.)
- **Traffic:** own worktree, branch `fire-b1-blackbody-lights`, branched off
  main. Files touched: `renderer/*`, `main.py` (the sources block ~:375-402
  and overlay construction), `config.toml` (NEW `[render.*]` sections only),
  `shaders/lighting.fs` (only if the §6 check fails), `tests/`. S8c merged
  2026-07-21 (9eb47c0) before this build starts; the only concurrent session
  is Arc B (logic layer, own worktree) — its files (`src/simulation/`
  logic/entities, SignalBus) are disjoint from this beat. Do not touch them.
- **Feel-adjacent → HUMAN-TEST gate.** Build, gate, push; Erik plays before
  merge. No auto-merge.
- **Credit the source** (repo rule): the blackbody module's header cites
  Tanner Helland's Kelvin→RGB, Neil Bartlett's refinement, and Macklin's
  blackbody-rendering exposure treatment (links in
  `fire_rendering_research.md` §4).

## 1. The blackbody primitive — `renderer/blackbody.py`

One module, one job: **game-temperature → (linear RGB chroma, HDR intensity)**.

- **LUT, not per-call polynomial** (research Q1 recommended default): at
  renderer startup, bake a 1D LUT over pseudo-Kelvin `[kelvin_floor,
  kelvin_ceil]` (default 800–10000 K, N=256) using the Tanner
  Helland/Bartlett piecewise fit for chromaticity. Chroma is normalized
  (max channel = 1) — brightness is carried SEPARATELY (below). Float,
  render-layer, determinism-exempt.
- **Game-T → pseudo-Kelvin mapping** (the honest dial for "white reserved
  for extremes"): `kelvin = kelvin_ambient + k_temp_to_kelvin * T_game`
  where `T_game` is the dequantized `gmap.temperature` (ΔT, game units;
  wood ignites at 300, `T_MAX_PHYS` rail = 16000). Defaults:
  `kelvin_ambient = 293`, `k_temp_to_kelvin = 2.0` — a 300–600-unit wood
  fire lands ~900–1500 K (deep red→orange), and only extreme events
  (future B3 explosions) reach the 6500 K+ white end. **Tune-by-eye with
  Erik; these are config, not constants.**
- **Intensity (the T⁴ half the fits don't give you):**
  `intensity = clamp(((kelvin - kelvin_glow_min) / (kelvin_ref -
  kelvin_glow_min))^p, 0, i_max)` with defaults `kelvin_glow_min = 800`,
  `kelvin_ref = 3000` (Macklin's reference-temperature exposure move),
  `p = 4.0` (Stefan-Boltzmann flavor), `i_max = 8.0` (HDR headroom — the
  16F light pipeline + ACES handle >1 values; NO bloom pass, per Erik).
- Config: NEW `[render.blackbody]` section carrying all of the above.
- API: vectorized `chroma_intensity(temp_field) -> (rgb (h,w,3), inten
  (h,w))` for the overlay, and scalar `light_color(t_game) -> (r,g,b),
  intensity` for light sources. One code path builds both (the "two wiring
  points agree by construction" invariant).

## 2. Wiring point (a) — the emissive overlay (tier 1, ray-free)

Rework `HeatFieldOverlay` (`renderer/overlays.py:317-407`) to draw
`ramp.chroma * ramp.intensity` (tone-mapped into the overlay's additive
draw; alpha from intensity as today, curve preserved). This replaces the
5-stop LUT AND `temp_display_max=300` (delete the latter — the kelvin
mapping + intensity curve subsume it). This is brainstorm §8-item-2's
"per-tile in-march glow is ray-free" tier: the whole plume glows for a LUT
read per tile, no rays.

Toggle policy (Erik decides at HUMAN-TEST): keep `T` as the toggle, but the
build should add a config default `[render] blackbody_overlay_on = true` so
the blessed look can become the shipped default, and demote the flat-orange
`FireOverlay` (F3) to default-OFF *in the same human-test build* so Erik
compares both in one session.

## 3. Wiring point (b) — fire light sources (tier 2, brightest-K)

New render-side helper `renderer/fire_lights.py`, called from the `main.py`
sources block (after beacons, before the flashlight):

1. **Candidates:** tiles with `T_game >= t_light_min` (config, default 250).
2. **Non-max suppression, 3×3:** keep a candidate only if it is the maximum
   of its 3×3 neighbourhood (numpy maximum-filter compare — this is Erik's
   "3×3 march" as *selection*; widen to 5×5 via config `nms_window` if
   blazes still oversubscribe).
3. **Cap:** take the brightest `K` by temperature (config `max_lights`,
   default 16). Log-once (debug HUD counter) when the cap truncates, so
   tuning sessions can SEE saturation — no silent caps.
4. **Per light:** omni (`angle_spread = 2π`), `x,y` = tile center,
   `max_range` = config `light_range` (default 18 — deliberately LONG, per
   Erik: geometry caps range naturally, big-room illumination is the point;
   cost note below), `color` = `ramp.light_color(T).rgb`, `intensity` =
   `ramp.intensity * light_gain` (config, default 1.0), `jitter = 0.0`
   (flicker is a later dial, not v1).
5. Marines/zombies light up for free — the marine shader samples the same
   light-field textures (`renderer/marine_shader.py`).

**Cost model** (verified `raycaster.h:65-70`): omni ray count =
`ceil(2π·range)`; K=16 × range=18 ≈ 1,810 rays ≈ 11 flashlights' worth of
marching — comfortably inside the render budget the flashlight already set.
Selection cost is a couple of numpy passes over the field — negligible.
Rays from separate sources DON'T merge (cost is the sum) — the light FIELD
accumulates additively, which is why brightest-K + NMS is the budget lever
and range is not.

**Explicitly out of B1:** detonation flash lights (transient timed sources —
that's B3's explosion rework, where explosions gain real temperature; a
scripted flash before then would be fake data), fire-noise/particles (B2+),
per-gas glowing smoke (B2), any promotion of these lights into the sim's
heat cast (heat stays a strict one-way channel — brainstorm §8 item 2,
decided).

## 4. Config summary (all NEW, all render-layer)

```toml
[render.blackbody]
kelvin_floor = 800.0      # LUT low end
kelvin_ceil = 10000.0     # LUT high end
lut_size = 256
kelvin_ambient = 293.0
k_temp_to_kelvin = 2.0    # game-ΔT → Kelvin slope  (tune-by-eye)
kelvin_glow_min = 800.0   # below this: no visible glow
kelvin_ref = 3000.0       # exposure reference (Macklin)
intensity_exponent = 4.0  # Stefan-Boltzmann flavor
intensity_max = 8.0       # HDR headroom pre-tonemap

[render.fire_lights]
enabled = true
t_light_min = 250.0       # game units, candidate threshold
nms_window = 3            # 3 or 5
max_lights = 16           # brightest-K cap
light_range = 18.0        # LONG on purpose (Erik ruling)
light_gain = 1.0
```

## 5. Patch plan (autonomous-patch-workflow)

- **P0** — merge `fire-lit-search` (docs-only) + commit THIS doc to the
  build branch.
- **P1** — `renderer/blackbody.py` + `[render.blackbody]` + unit tests
  (LUT endpoints red→white, chroma monotonically desaturating with K,
  intensity monotone in T, vectorized == scalar on samples).
- **P2** — overlay rewiring (§2) + FireOverlay default flip behind config.
  Screenshot harness: same level, before/after PNG pair for Erik.
- **P3** — `renderer/fire_lights.py` + `main.py` wiring + `[render.
  fire_lights]` + tests (NMS correctness on a synthetic field, cap
  respected, source params in range, zero sources when field cold).
- **P4** — §6 ACES check + perf spot-check: `--level` with a scripted big
  blaze, frame-time before/after, HUD light-count counter. Then the
  HUMAN-TEST build for Erik (both toggles comparable live).

Gates per patch: `pytest tests -q` green; goldens byte-untouched (run the
digest gate even though construction says it can't move — cheap insurance);
no imports from `renderer/` into `src/simulation/` (add a test asserting
the dependency direction if one doesn't exist).

## 6. Precondition check (brainstorm §8 item 8, folded in)

Read `shaders/lighting.fs`: confirm ACES is applied PER-CHANNEL (not on
luma) and that light values stay unclamped HDR until the tonemap. If
luma-only, propose the per-channel switch (or Stephen Hill's fit) as part
of P4 — shader-only, still render-layer. Report the finding either way in
the build log.

## 7. Human-test script (Erik's session)

Wood-room fire: watch glow rise from dull red through orange as it grows —
white should be RARE. Sealed-room fire: fire dims/reddens as O2 starves
(color now shows real chemistry — research pick ②'s first visible payoff).
Fire in a dark room: walls and a marine standing nearby must pick up
firelight, flickering as tiles ignite/die. Big blaze: HUD light counter at
the cap, frame time acceptable, room still reads well (NMS spreading the
K lights). Wind: plume stretch + light drift downwind. Compare old vs new
overlay live (T / F3). Tuning dials for the session: `k_temp_to_kelvin`,
`intensity_*`, `t_light_min`, `max_lights`, `light_range`, `light_gain`.

## 8. Kickoff prompt (paste into a fresh Opus session)

---

You are building **B1 of the Fire & Heat Beauty arc** on the breach project:
the blackbody temperature→color ramp + brightest-K fire light sources.
**Render-only — zero sim writes, zero C++ changes.**

The design is LOCKED: `docs/fire_b1_blackbody_fire_lights_design_2026-07-21.md`
(this doc — read ALL of it; §5 is your patch plan P0–P4, §4 the config
schema, §0 the iron constraints). Context: the arc's living plan is the
bottom section of `docs/plan-for-tuning-and-graphics.md`; the research base
is `docs/fire_rendering_research.md` on branch `fire-lit-search` (P0 merges
it — docs-only). Workflow: autonomous-patch-workflow.

Environment: Python = conda env `data` (see docs/dev_setup.md /
docs/lenovo_dev_setup.md for this machine); pytest = `pytest tests -q`;
no C++ build needed for this beat. Work in your OWN worktree on branch
`fire-b1-blackbody-lights` off main. Arc B (logic layer) is concurrently
in flight in its own worktree — do not touch `src/simulation/` at all.

Gates: unit tests per patch; goldens byte-untouched (run the digest gate as
insurance even though the beat is render-only by construction); screenshot
before/after pairs at P2; perf + HUD light-counter at P4. The beat is
FEEL-ADJACENT: finish = built, gated, pushed, HUMAN-TEST build ready —
**Erik plays before any merge. Never auto-merge.**

Escalate (STOP, bring Erik/Fable) if the work seems to require: touching
`src/simulation/`, `cpp/`, recorder/digest surface, or the sim tick; any
golden moving; the §6 ACES check failing in a way a shader-only fix can't
cleanly address; or the light-source seam in `main.py` colliding with
something unexpected.

---
