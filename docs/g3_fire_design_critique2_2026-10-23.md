# G3 Fire F0: round-2 critique (resolution check), 2026-10-23

Reviewed: `docs/g3_fire_design_2026-10-23.md` v2 at `88950a7`, against round 1
(`docs/g3_fire_design_critique_2026-10-23.md`) and the code at `cfcba40`.
Lens: are round 1's blockers and majors actually resolved, and did the v2 fixes
add new ones? Erik's §1 decisions are not relitigated.

Extra evidence: one headless probe (scratch script, main checkout's build at
`cfcba40`). It ignited a 3×3 patch of air on `fire_smoke_studio` at (17,29),
using the studio's own `_inject_fire` (`tools/lighting_demo.py:900-913`), and
ran 15 s. The burning centre tile measured **283–294 K**, and the 3×3 maximum
stayed ≤ 306 K. Over the same 15 s, `fire` fell from 1.0 to 0.10.

**Verdict: APPROVE-WITH-FIXES.** One new blocker (N-B1), with a cheap fix, plus
seven new majors. All are doc-level fixes that need no ruling from Erik.

---

## 1. Round-1 items: resolution table

| # | Status | Evidence |
|---|---|---|
| B1 | **RESOLVED** | §3.3 sets `T_src` = the burning tile's own T: the over-tile source for permeable fuel and the strip for wood. Furniture is `thermal_solid` (`gamemap.py:1451-1455`, `game_renderer.py:1007-1011`), so the tile holds the hot object temperature (the probe in round 1: 1061–1331 K). The vacuum gate is kept. Residual: an *air* fire's tile stays at ~290–306 K (probe above), so it injects R that emits nothing. That feeds N-M7. |
| B2 | **PARTIAL** | The ceiling keeps 0.2 through a compensation term, and the gas glow is drawn after the soot without dimming. Both are correct in intent. But the floor's 0.3 is *not* kept (N-M2), and the compensation is only approximate and under-specified (N-M3). |
| B3 | **PARTIAL** | The prescribed divergence, the p = 0 open edge, the chosen sign and the F0b prototype are all in. But the problem is singular for any enclosed fluid pocket inside a box (**N-B1**), and the velocity scale is not anchored (N-M5). |
| M1 | **PARTIAL** | Sim clock plus α plus fixed substeps is the right design. But α is not exposed anywhere, and the substep cap breaks shot/game parity (N-M1). |
| M2 | **PARTIAL** | The flame is composited as an emitter (the over operator), the CIELAB metric is written down, and the toggle is redefined. But the gate runs on the exact conversion G0 measured failing, and the fallback re-breaks M4 (N-M4). The opacity is also not tied to emission (N-M7). |
| M3 | **PARTIAL** | A dedicated [0,1] input from `blackbody.py` is the right shape. But "normalised as today" means the ACES display alpha (`overlays.py` `brightness` = packed alpha / max_alpha). Today `aces_curve` is allowed only through `CPU_TWIN_HOME`, whose stated reason is "until G3 fire" (`tests/test_one_tonemap.py:79-100`). Name the function, either `1−exp(−I/I₀)` or the twin kept with a new reason, and gate it: `smoke_over_squad` and `fire_thin_smoke` keep their own-light cut. |
| M4 | **PARTIAL** | Covering the burning tile is in. The radiance-bound property test that round 1 asked for is missing from §8. Covering the *lit floor* can darken it (N-M7). |
| M5 | **RESOLVED** | 1024 entries, linear filter, the upload in `fire_vfx.py`, and Temperature scale in Systems (a). Minor: `BlackbodyRamp`'s arrays have `lut_size` = 256 from config (`blackbody.py:231-236`, `config.toml:1475`). Build the 1024 texels from `emission_at_kelvin` (`blackbody.py:255-277`). Do not raise `lut_size`, which would move the heat layer's nearest-index pack and so the hue-gate baseline. |
| M6 | **PARTIAL** | The CPU/GPU split, the warm start and the pass count are in. Gaps: |
|  |  | • "view **∪** active region" (§4) should be ∩. A union runs passes over the view with no fire, which contradicts "no fire → zero cost". |
|  |  | • Clipping the world-RT draws (flame, body glow, compensation, gas glow; 48 px/tile) to view ∩ boxes is not stated. |
|  |  | • Up to 4 boxes × ~46 passes ≈ 184 passes per frame (≈ 4–7 ms CPU), unless each stage draws all boxes as quads in **one** shader/texture-mode bracket. |
|  |  | • Caching uniform locations is not stated. |
|  |  | • VRAM is not counted: a world-anchored M = 8 grid on 128×256 is 1024×2048 per field, about 9 fields at RGBA32F ≈ 300 MB. |
| M7 | **RESOLVED** | Per-tile mask, ≤ 4 boxes, clear on leave, zero cost with no fire. The p = 0 seam is covered by N-B1. Minor: boxes must stay disjoint by ≥ margin (merge otherwise), and margin = lifetime × the *saturated* tamed speed (33.6 tiles/s) can reach ~15 tiles, so cap it. |
| M8 | **RESOLVED** (one error) | "`BLEND_ADD_COLORS` with ONE/ZERO" contradicts itself: raylib's `BLEND_ADD_COLORS` *is* ONE/ONE, and the relief uses it as an additive sum (`gas_relief.py:742`). Use `BLEND_CUSTOM` (ONE, ZERO) or disable blending. The odd/even identity-pass readback gate named in §9 is not listed in §8. |
| M9 | **RESOLVED** | The menu is corrected and D goes back to Erik before F4. |
| M10 | **RESOLVED** | Minor: the scene RT is POINT-filtered (`world_composite.py:70-72`), so a sub-texel UV offset steps in whole world pixels. Bilinear-filter the offset tap by hand in the resolve. |
| M11 | **RESOLVED** | Minor: say "one batched draw" for the sprites. |
| M12 | **RESOLVED** | M_u = 8 and a flame-noise function in `noise_bake` are in. Minor: `fss_lone` is not named as a gate shot. |

---

## 2. New BLOCKER

### N-B1. The pressure problem is singular for any enclosed fluid pocket inside a box.

**Claim.** §3.3 and §5 put p = 0 only at the box edge, and solids are no-through (Neumann).

**Evidence.**
- Picture a room enclosed by walls, or by closed doors (which are `solid`, `gamemap.py:1446-1449`), lying wholly inside a box. A wood wall burning into a small room, or any fire with `margin_tiles` ≥ the room size, does this.
- That fluid pocket never touches a p = 0 cell. It is a pure-Neumann problem with net `Σq > 0`, which has no solution: round 1's B3 point, come back one level down.
- Jacobi with a warm start makes p grow every substep, without bound. That ends in fp16 overflow, Inf, then NaN.
- Ship rooms are exactly this shape. F0b's open 64² scenes would not catch it.

**Fix.** Make the solve **screened**: `(∇² − 1/ℓ²) p = ∇·u* − q`.
- It is the honest top-down physics: the plume leaves the plane upward, a sink into the column above that §3.3 already names.
- It is always SPD and needs no p = 0 cells, so the box edges stop being special: abutting boxes no longer leave a seam.
- Jacobi also converges faster, because the diagonal dominates.
- ℓ ≈ a few tiles is a structure dial.
- Alternative: remove the mean of `q` per connected component. That is noisier, and it models entrainment.

**Tests.**
- F0b adds a 6×6 sealed-room scene.
- §8 adds a property test: "p and u stay bounded over 10⁴ substeps in a sealed room". It breaks if the screening is removed.

---

## 3. New MAJORs

### N-M1. The clock: α is not plumbed, and the substep cap makes shots run slower than the game.

**Evidence.**
- `main.py:599-616` keeps `tick_accum` local. `upload_state` takes no α (`game_renderer.py:448-449`, called at `main.py:637`).
- Look shots and the studio run through a **second** glue, `tools/lighting_demo.py`: one tick per frame under `--auto` (`:1708-1717`), its own accumulator otherwise (`:1720-1727`), plus `--perf` (`:1120`).
- In shots, `t_vfx` advances 1/24 s per frame, so 5 substeps of 1/120 s are due per frame. "Cap 4" either drops 20 %, so the flame runs at 0.8× in shots, or carries a debt that grows without bound. Either way it breaks §3.4's "same speed in shots and in the game".
- `tick_accum` can be ≥ one tick after the catch-up cap (`max_catch_up = 5`) or after an auto-pause mid-batch (`:613-614`).
- There is no determinism risk: it is a pure read of a wall-clock float.

**Fix.**
- Add an `alpha=` keyword to `upload_state`, passed from both glues and from `--perf`.
- Clamp α to [0, 1) on the render side, and never let `t_vfx` step backward.
- Set the cap to at least `ceil(1/(tps·dt_s)) + 1` (6), or use dt_s = 1/96.
- **Test:** substeps per tick are equal in shot mode and at 60 fps. It breaks on a cap below 5.
- Specify the idle clock for `idle_when_paused`: wall dt, a separate counter, never written into `t_vfx`.
- Key the curl-noise and edge-noise phase on `t_vfx + t_idle`, not on the tick-locked `advection_phase`. Otherwise the noise freezes on a flame that is still moving.

### N-M2. Drawing the body glow before the floor gas gives `glow_through_floor` = **1.0**, not the configured 0.3.

**Evidence.**
- The floor relief draws premultiplied over (`gas_relief.py:718-720`, `game_renderer.py:764-770`), so a glow already in the scene RT is multiplied by `(1−α_floor)`. That is full occlusion.
- G2 dims the body share by `(1 − 0.3·cov.b)` (`overlays.py:347-352`, `config.toml:1702`).
- Round 1's minor 2 was itself wrong here, and v2 §3.6 repeats the claim.

**Fix.** Extend the one compensation pass with the floor term: add back `glow·α_floor·(1 − glow_through_floor)·below` from `cov.b`.
- Alternatively, draw the body glow *after* the floor gas with G2's floor factor `(1 − 0.3·cov.b)`. That is exact, and it keeps one fewer term.
- Also say where the body glow sits relative to the water pass (`game_renderer.py:741-753`).

### N-M3. The compensation term is approximate and under-specified.

**Evidence.**
- The cast product is 8 px/tile (`gas_relief.py:81`). It is bilinear-sampled, while the drawn ceiling α is 48 px/tile with billow detail.
- Same formula, different resolution (`gas_relief.fs:305-314` vs `:388-390`). So `(1−α_drawn) + 0.8·cov₈` equals `1 − 0.2·α` only where the soot is smooth. "Net occlusion = Erik's 0.2" is approximate.
- The term also needs the flame's own radiance `Le·E` at the pixel, so it is a **second flame draw** in "compensation mode", not a re-read of what is in the RT. And `below` is unspecified for the flame, which has no per-tile body/gas split.

**Fix.**
- State that the term is a second evaluation of body glow + flame, additive, weighted `α·(1−g)` from `smokeCoverAt`.
- Set `below_flame = 1`.
- Gate it with a measurement: on `fire_thin_smoke` and `fire_thick_smoke`, the masked fire-region luminance of v2 with the flame off must sit within G0's ±4 % of `cfcba40`.
- Exact alternative, if the gate fails: add the term inside the ceiling relief shader, which knows its own α. That needs the glow as a texture.

### N-M4. The hue gate tests the conversion G0 already measured failing, and its fallback re-breaks M4.

**Evidence.**
- §3.6 runs the gate with the flame **off**. The body glow is then plain additive linear: exactly G0's conversion, chroma 0.219 → 0.166/0.179 (`graphics_direction_2026-10-18.md:545-556`).
- §3.6 itself says the emitter-cover is what keeps chroma, and that is absent in the gated configuration. So "chroma ≥ old" almost certainly fails, and F3 starts in the fallback.
- In the fallback the heat layer goes back after the resolve, additive and above the flame. The flame then no longer covers the burning tile's glow, which is M4's double count again. The draw order and the units-above-glow change also go away.

**Fix.**
- Gate the configuration that ships: the flame on, covering the body tiles.
- Or give the body share the same over operator, with ε from its own emission.
- Define a fallback that keeps M4: the display-space heat shader samples the flame's `E` texture and draws `glow·(1−E)`.
- Write down the metric:
  - hue = atan2 of the **chroma-weighted** mean (a*, b*);
  - the mask comes from the burning tiles, dilated and projected through the shot's camera, and is the same for both runs;
  - the halo is measured against a heat-off twin row (`hide = ["heat"]`);
  - the chroma tolerance is G0's ±4 %, not a strict ≥.

### N-M5. The velocity scales are unanchored, the wind's units differ, and vorticity confinement has nothing to amplify.

**Evidence.**
- `q ∝ I·(T_src/T_amb − 1)` has no constant and no unit. For one source tile, the outflow is `u(r) = q·A/(2πr)`.
  - Physically, in-plane expansion is weak: a 500 kW fire's ≈ 1.4 m³/s over a 2.5 m column gives ~0.2 tiles/s at the tile edge.
  - Tongues 1–2 tiles long within a ~0.4 s lifetime need `q` ≈ 20–30 /s, about 100× that.
  - So `q` is a look dial, and the doc should say so.
- The tamed wind is in **tiles per tick** (`gas_detail.py:76-80`), while `u_vfx` is in tiles/s, a ×24 conversion. Near a fire it can reach 1.4 tiles/tick, i.e. 33.6 tiles/s, which overwhelms any `q`.
- The projection removes the tamed wind's divergent part every substep, so the two do not fight on divergence (worth stating). But they compete in magnitude.
- Expansion plus `−∇P`-based wind are both irrotational, and free-slip solids generate no vorticity. So "vorticity confinement on the expansion's shear against the tamed wind and solids" (§3.3) has almost nothing to amplify. Round 1's point stands: the curl noise is the structure source.

**Fix.**
- State `u_wind = tamed·tps`.
- Give `q` as `q₀·I·(T_src/T_amb − 1)`, with `q₀` in 1/s, named as a look dial.
- Name the structure sources honestly: curl noise scaled by local `I`, and a pulsing `q`/`s` at the pool-fire puffing frequency `f ≈ 1.5/√D` Hz (Cetegen & Ahmed, *Combust. Flame* 1993). Confinement only amplifies what those seed.
- F0b drives its "with wind" scene from the **real** tamed-wind field dumped from `fire_smoke_studio` (the probe), not a constant.

### N-M6. GL parity on a chaotic loop cannot be a trajectory comparison.

**Evidence.** §7 F2 and §8 compare the GL loop with the twin "within tolerance on 3 scenes". With vorticity confinement and curl noise, fp32/fp16 GPU against float64 numpy diverges within a few substeps. A long-run tolerance is either meaningless or always red.

**Fix.**
- **One-step operator parity:** upload the same state, run one stage (advect, confine, divergence, Jacobi sweep, project, inject, decay), read back, compare per stage.
- Over long runs, check only the invariants and statistics: `R ≥ 0`, `T ≤ T_src`, total R within x %.

### N-M7. The emitter-cover opacity is not tied to emission, so the flame can draw dark.

**Evidence.**
- §3.5's `out = Le·E + under·(1−E)` uses `E = E(R)` alone.
- R decays over `lifetime_s` while T cools toward the local gas T, which sits below 800 K (round 1: 457–761 K). The tongue tips then have `E > 0` and `Le → 0` (`blackbody.py:240-253`): dark tips over the lit floor.
- Air fires inject at `T_src` ≈ 300 K (probe), which gives black patches.
- Near the fire the floor is lit by the fire itself, and wherever `Le < under` the cover darkens it.
- Cold opacity is smoke, and smoke belongs to G1/G2.

**Fix.**
- Use `ε = E(R)·w(T_vfx)`, with `w` rising from 0 at `kelvin_glow_min`. For example, `w = clamp(I(T)/I(T_src))` or a smoothstep over 800–900 K.
- Add no source where `T_src < kelvin_glow_min`.
- **Property test:** "the flame pass never lowers masked luminance below the flame-off render by more than a tolerance". It breaks if ε is decoupled from emission.
- Add round 1's M4 radiance bound to §8.

---

## 4. MINOR

1. **Cooling can heat.** "Cool T_vfx toward the local gas T" heats the flame wherever the gas is hotter than `T_src` (an explosion next to a fire). That breaks the invariant and the `max T_vfx ≤ max T_src` test. Use `min(T_vfx, relax(...))`.
2. **Vacuum.** R advects into vacuum tiles (only solids zero it), and a premultiplied over writes dstA, which feeds the blit over the backdrop. Zero R in vacuum as in solids, or keep dstA with a separate blend.
3. **Tamed wind.** The array exists only on the relief branch (`game_renderer.py:494-516`). With the smoke hidden or the relief off there is none, and props call `tame_wind` with the default dials (`:546-548`). Make one per-frame tamed product, with the relief's dials, computed when any consumer needs it.
4. **Speckle.** It modulates the display pack (`game_renderer.py:563-572`). Say whether the linear heat layer keeps it.
5. **Cold overlay.** It stays display-space "under the heat" (`game_renderer.py:826-829`). After F3 it draws *over* the linear heat. Check halo overlap.
6. **One-tone-map test.** `test_one_tonemap` scans `shaders/*.[fv]s` plus two Python FS strings only (`tests/test_one_tonemap.py:41-48`). Any Python-composed fire or compensation shader must be added. Remove `CPU_TWIN_HOME` (`:79-84`), or re-justify it if M3 keeps the twin.
7. **Source upload.** Upload `src` and `solid` per tile and sample them nearest. Do not upsample to M on the CPU.
8. **`idle_when_paused` scope.** Nothing in the shots pauses (`lighting_demo.py:1704-1705` un-pauses every frame), so shots stay reproducible. The game's VFX state after a pause depends on wall time. Say that it is not replay-reproducible.

---

## 5. Would F0b or F1 mis-start?

- **F0b:**
  - N-B1 (add the sealed-room scene and screen the solve);
  - N-M5 (the real wind field, `q₀` as a dial, the honest structure sources);
  - minor 1 (cooling clamp).
  
  Without these, F0b can pass on open grids and hand a singular solver to F2.
- **F1:**
  - N-M1 (α plumbing in *both* glues, the substep cap);
  - M6 gaps (∩ not ∪, batched boxes, VRAM);
  - the M8 blend correction.
- **F2:** N-M6, replacing the trajectory parity.
- **F3:** N-M2, N-M3, N-M4, N-M7.

## 6. Ranked summary

| # | Severity | One line |
|---|---|---|
| N-B1 | BLOCKER | A sealed fluid pocket inside a box gives a singular Neumann solve with Σq > 0: p diverges. Screen the Poisson solve (the plume leaves upward) |
| N-M1 | MAJOR | α is not exposed (`main.py` and `lighting_demo.py`); cap 4 < 5 substeps per tick makes shots slower than the game; α clamp; idle clock |
| N-M2 | MAJOR | Body glow before the floor gas = `glow_through_floor` 1.0, not 0.3 |
| N-M3 | MAJOR | Compensation: product at 8 px/tile vs drawn α at 48 px/tile, a second flame draw, `below` unset. Gate it against G2 |
| N-M4 | MAJOR | The hue gate tests G0's failing conversion (flame off); its fallback re-breaks M4 |
| N-M5 | MAJOR | `q` unanchored (a look dial); tamed wind is tiles/**tick**; confinement has no vorticity to amplify |
| N-M6 | MAJOR | GL-vs-twin parity must be per-operator, one step at a time, not trajectories |
| N-M7 | MAJOR | `ε = E(R)` ignores T: cool tongue tips and ~300 K air fires draw dark over the lit floor |

**Overall: APPROVE-WITH-FIXES.**
- Fold N-B1 and the seven majors into v3 as specified. None needs Erik: N-M2 honours his existing 0.2/0.3 values.
- No third critique round is needed if they are folded as written.
- F0b must demonstrate N-B1 (the sealed room stays bounded) and N-M5 (tongues under the real wind) numerically before F1 starts.
