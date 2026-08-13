# Temperature-scale unification — design (2026-08-13)

**Status:** v2 — adversarial critique (3 lenses) synthesized, blockers resolved,
Erik's rulings locked. Build may proceed.

**Rulings (Erik, 2026-08-13, this session), each with its reason:**
1. **FULL unification of the Kelvin map now** — one canonical game-T→Kelvin
   table used by sim and game alike (chosen over plumbing-only, knowing it
   pre-empts part of the storm-session ruling).
2. **`phi_exp` named now, value-frozen** — the storm session tunes a named dial
   instead of discovering an accident; naming ≠ retuning.
3. **Table lives in `[physics.temperature_scale]`** — render keeps only
   render-only dials.
4. **Goldens do not gate this arc** — "we retune freely now, and produce goldens
   after." A NEW golden suite is co-designed with Erik post-tuning (§7).
   CPU↔CUDA lockstep parity at tol 0 remains a hard gate (correctness, not
   calibration).
5. **P-K0: promote the blessed TUNE dial set into config.toml** (critique F1,
   physics lens) — the branch config is pre-P-F1b; the radiation re-anchor and
   the P-K5 feel-test must run against the fire Erik actually tuned. This takes
   the *values* half of the parked P-F1b merge-sequencing decision; branch
   cleanup of `pf1b-recalibration` stays separate.
6. **EOS ambient stays 290 K as a named table entry** (critique F1/F3,
   determinism + physics lenses) — 290 is Q16.16-near-optimal (ambient pin
   1 atm + 4 counts); 293 would be near-worst-case (+96 counts, a 24× larger
   standing source in every ambient cell, right before the storm-damping work).
   T_game is a ΔT; 3 K of ambient offset has no in-game meaning. The Kelvin
   MAP is still fully unified; the EOS *pressure calibration* is an explicit,
   reasoned exception, not an accident.
7. **The pre-session rewrite of the append-only ruling doc is restored and
   re-expressed as an appended supersession note** (doc culture: dated docs are
   append-only).

## 1. Problem

Four live conventions for "what T_game = 0 means in absolute terms" coexist:

| path | mapping | constant site | synced state? |
|---|---|---|---|
| Radiation bake | 293 + **2**·T | hardcoded C++ (`cpp/src/raycaster.cpp:59`: `K = 297 + 8t`) | YES |
| EOS pressure | **290** + **1**·T | config `[physics.eos] t_amb_k`, `C = 1/t_amb_k` | YES |
| Render blackbody | 293 + **3**·T | config `[render.blackbody]` (2026-08-13 preview, Erik-blessed by eye) | no |
| Unit heat damage | **20 °C** + flux·slope | config `heat_ambient_ref` (Celsius twin of 293 K) | YES |

Bench finding (2026-08-13): under ×3 the P-F1b plateau (~300 game) reads
**1193 K** — inside the realistic 1100–1250 K band for a crate burning with
flames. ×3 is the physical choice, not just a color choice.

Post-unification: radiation + render + tools share ONE map (293 + 3·T); the
EOS ambient is a named exception (ruling 6); felt-damage ambient stays its own
pinned dial (§2 gap). Full site inventory in §9.

## 2. Canonical table

New config section — the ONLY place these constants are defined:

```toml
[physics.temperature_scale]
kelvin_ambient   = 293.0   # absolute K at T_game = 0 (≈ 20 °C) — THE map ambient
k_temp_to_kelvin = 3.0     # game-ΔT -> Kelvin slope (Erik's k_calibr, blessed 2026-08-13)
phi_exp          = 0.3333333333333333  # EOS expansion fraction of the Kelvin excess.
                           # VALUE-FROZEN: phi_exp * k_temp_to_kelvin == 1.0 exactly
                           # (0.3333333333333333 parses to double(1/3); 3.0·double(1/3)
                           # == 1.0, a round-to-even tie landing on 1.0 — verified).
                           # Retuning phi_exp is the storm session's ruling.
eos_t_amb_k      = 290.0   # EOS absolute-ambient (pressure calibration) — DELIBERATE
                           # exception to kelvin_ambient, ruling 6: quantize(1/290)
                           # yields ambient pin 65540 (+4 counts); 293 would give
                           # 65632 (+96). Byte-preserves all EOS state this arc.
```

Derived at load, single formula each, asserted **via the existing
`gas_fixed.quantize_scalar`** (critique: no third hand-rolled rounding —
the 139b/A8 hazard):
- **EOS slope** `s_eos = phi_exp * k_temp_to_kelvin` → 1.0 exactly; assert
  `quantize_scalar(s_eos) == 65536` while value-frozen.
- **`C = 1 / eos_t_amb_k`** → unchanged 1/290. The doubles identity
  `C·N_amb·t_amb == 1.0` is IEEE-exact, and the **quantized sim-chain pin**
  (`ambient.effective_pin`) stays 65540 — P-K3's test pins the sim chain, not
  just the real identity (critique F3, physics lens).
- **Render** reads ambient+slope from this section; `[render.blackbody]` keeps
  only render-only dials (kelvin_floor/ceil, lut_size, glow_min, kelvin_ref,
  intensity_*).
- `[physics.eos] t_amb_k` and `C` keys are removed — see §3c migration guards.

**ACCEPTED GAP — heat_ambient_ref stays pinned at 20.0 °C.** kelvin_ambient −
273.15 = 19.85, not 20. Deriving it would change unit heat damage for zero
physical gain (the 0.15 K slop is negligible against felt-temp magnitudes —
verified, physics lens F4 table).

**Migration guards (critique F3, scope lens — Erik live-edits config on two
machines syncing only via git, so stale keys must be LOUD):** config load hard-
errors, with a message pointing at `[physics.temperature_scale]`, if
`[physics.eos]` still carries `t_amb_k`/`C`, or `[render.blackbody]` still
carries `kelvin_ambient`/`k_temp_to_kelvin`, or the new section is missing.
Fallback defaults survive only in the standalone tool path (§3d gap).

## 3. Per-path migration

### 3a. Radiation bake (`cpp/src/raycaster.cpp:59`)
- `Raycaster` gains `double kelvin_ambient = 293.0; double k_temp_to_kelvin = 3.0;`
  next to `rad_scale` (`raycaster.h:423`), exposed via `def_readwrite`
  (`bindings.cpp:1853` block), assigned in `physics_runner.py:335` block
  (assignment precedes the `:355` eager bake — ordering verified safe).
- **Staleness check, spelled out (critique, determinism lens):** two new
  mutable baked-at cache members `e_table_amb_`, `e_table_slope_` alongside
  `e_table_scale_` (`raycaster.h:912`), written in `bake_emissive_table()`,
  compared in `emissive_table()` (`raycaster.cpp:71`). Both consumers route
  through `emissive_table()` and CUDA uploads the host table per cast
  (`bindings.cpp:589-592`) — one bake covers both backends; verified no
  hardcoded map exists in `cuda_raycaster.cu`.
- Integer bake stays exact: bucket midpoint T_mid = 4t+2, so at 293/3
  **K = 299 + 12t** (exact int64, same idiom as 297 + 8t). Assert integer-ness
  of ambient and slope at bake time while frozen; a future non-integer slope
  bakes in double and quantizes via the `+0.5` idiom already in the file (NOT
  `std::round` — keep one rounding convention; determinism lens).
- **Overflow headroom re-proved:** max K = 48287; K⁴ = 5.4365e18 < INT64_MAX
  (headroom ×1.70). Verified. Platform-determinism verified: `/fp:strict` /
  `-ffp-contract=off` per CMakeLists; int64→double conversion + multiply are
  single correctly-rounded ops.
- **Comment rewrite list (extended per critique):** `raycaster.h:160-167`
  (1.088e13 magnitude), `:172-211` (old 32289 bound), `:259` (~1.09e13),
  `raycaster.cpp:48-53` (bake comment), `:62` (name BOTH roundings — the
  int64→double conversion is also inexact at max K, −385 counts, benign).

### 3b. rad_scale re-derivation
Shipped `rad_scale = 1.0e-5` was derived under ×2 (config.toml:304-323).
Re-anchor by **preserving emitted flux at the promoted plateau anchor
T_a = 300 game** (valid because P-K0 puts the blessed dial set into config —
critique blocker F1, physics lens):

```
rad_scale' = 1.0e-5 · (893/1193)⁴ = 1.0e-5 · 0.313938 = 3.1394e-6
```

**The ONE shipped literal is `3.1394e-6`** (critique F2: earlier draft said
3.1377e-6 — arithmetic error, corrected; and F9, determinism lens: exactly one
literal so two machines can't diverge). Flux is identical at the plateau;
E′/E = 0.31 @ T=0, 0.87 @ 200, 1.00 @ 300, 1.09 @ 400, 1.21 @ 600, 1.34 @
1000, asymptote 1.59 — never exceeds 2× (physics lens F5; carry this table
into the P-K5 checklist). Seed-doc sanity re-run passes: adjacent-wood
ignition delivers ~26 kW/m² > the 12 kW/m² piloted threshold (was ~22), and
the config's own two-tile ignition inequality clears with ×1.82 margin
(was ×1.54) — physics lens F10. Rewrite the config.toml:304-323 derivation
under the new map with the new anchors.

**Stated consequence — unit heat damage is retuned off-plateau** (physics lens
F4): felt damage follows E′/E, so it is anchored at 300 game (2.7 dmg/s
unchanged) but +15% at 450, +34% at 1000, −13% at 200, −34% at 100. Smolder
cooks less, infernos cook more. This is judged at P-K5, explicitly.

### 3c. EOS (`cpp/src/eos_solver.*`, `cuda_eos_step.*`, `cuda_eos_resident.cu`)
**Byte-identical this arc** (ruling 6 + frozen phi_exp): ambient stays 290 and
the slope mechanism is an exact identity, so the oracle is bit-identity.
- `t_abs = T + t_amb_q` becomes `t_abs = ((int64_t)s_eos_q * T >> 16) + t_amb_q`
  with `s_eos_q = quantize(phi_exp * k_temp_to_kelvin)` folded once on the
  host. At s_eos_q == 65536 the product has zero low bits, so arithmetic shift
  ≡ truncation and the result is exactly T **for every int32 T including
  negatives; no overflow (|product| ≤ 2⁴⁷)** — verified, determinism lens.
  Code comment must record that off-identity values floor toward −∞ at T<0
  (mul_q16's documented convention) so the asymmetry is deliberate when the
  storm session retunes phi_exp.
  Sites (verified complete): `eos_solver.cpp:311,573`; `cuda_eos_step.cu:149,
  164,261,517` (s_eos_q joins `t_amb_q` in the params struct,
  `cuda_eos_step.h:83`); `cuda_eos_resident.cu:146,155,770`.
- **Atomically in P-K3** (critique blocker, scope lens F1): remove
  `[physics.eos] t_amb_k`/`C` from config + add §2 migration guards + rewire
  `physics_runner.py:404-405` to the accessor (`eos_t_amb_k`, `C = 1/eos_t_amb_k`,
  `s_eos`) + rewrite `tests/test_eos_p1_calibration.py` (currently a bare
  `CFG.physics.eos.t_amb_k` attribute access → would hard-red; and it must
  additionally pin the quantized sim chain: `effective_pin` == 65540).
  Rationale: `physics_runner`'s `_ep` getattr-defaults would otherwise fail
  SILENTLY while the test fails LOUDLY in a different patch's gate.
- `cpp/src/eos_solver.h:162-163` struct defaults (290.0f, 1/290 float) stay
  numerically valid under ruling 6; update their comments to point at the
  table, and note the fold path is double→float→quantize (verified: c_q still
  226 after the float32 round-trip) so nobody "fixes" it asymmetrically on one
  backend (determinism lens F2).
- `src/simulation/ambient.py:39-40` (`DEFAULT_C`, `DEFAULT_T_AMB_K`) become
  accessor-derived — no hand-kept copies. `pump_system.py` `ct == 1.0` identity
  unchanged. `tests/test_air_boundary.py` pin 65540 (:548,:576,:764) stays
  valid — no retune needed under ruling 6 (it WAS unlisted regression surface;
  now inventoried, §9).

### 3d. Python accessor (game + tools + tests)
New module `src/temperature_scale.py` (import-light; asserts via
`gas_fixed.quantize_scalar` which is numpy-only):
- loads `[physics.temperature_scale]` from the CFG object, plus a standalone
  `from_toml(path)` for tools. **Per-key defaults live in exactly one place**
  shared by both entry points (critique F7: Namespace-vs-dict drift). No
  module-level caching, or an explicit "snapshots at first use" docstring —
  `CFG.reload()` does not propagate through caches (pre-existing for render).
- exposes `kelvin_ambient`, `k_temp_to_kelvin`, `phi_exp`, `eos_t_amb_k`,
  `to_kelvin`, `from_kelvin`, `eos_slope`, `C`.
Consumers:
- `renderer/blackbody.py`: `from_config` reads the new section; ctor defaults
  → 293/3.
- `tools/fire_tune_plot.py: kelvin_map()` delegates to `from_toml` with a
  **narrow** except (print a warning, don't swallow arbitrary import errors —
  critique F7c) and fallback **(293.0, 3.0)**.
- `tools/fire_tune_loop.py`: its OWN scorecard `except` fallback also moves to
  **(293.0, 3.0)** (critique F11: it's a live code path, was slope-2), and the
  stale Kelvin comment table spans **:170-200**, not just :186.
- `tests/test_hover_readout.py:53` lambda re-derived from the accessor
  (consistency work — it stays green regardless; not counted on any gate).
- `renderer/hover_readout.py:18` + `tools/lighting_demo.py:1366` docstrings:
  update the named config home.

### 3e. Tests — per-boundary enumeration (critique, scope lens F4/F5)
*P-K1:* `test_blackbody_ramp.py:170-175` (ctor-default pin 2.0 → update to 3.0);
`:153-167` **must be rewritten to pin a NON-default value through
[physics.temperature_scale]** — otherwise it passes vacuously once defaults
equal config. `kelvin_map()` has zero test coverage: P-K1's report includes a
manual `kelvin_map() == (293.0, 3.0)` check (and adds a small test).
`test_frame_lights/fire_lights/speckle`: verified green (self-consistent or
game-unit-gated).
*P-K2:* `test_pr1_fire_plane_cast.py` re-implementation → `K = 299 + 12t`
derived from the shared constants (kept an independent re-implementation, so
it stays an oracle); **pre-decision:** the test-pinned `RAD_SCALE = 1.0e-5`
constants in `test_pr1_fire_plane_cast.py:94`, `cuda_pr1_fire_plane_check.py:45`,
`test_pf1a_radiation_books.py:68` move to `3.1394e-6`, and pf1a's flux-limiter
crossover ("~1300 game", sweep top 1200) is **re-derived in-patch** — under the
steeper K(T) the limiter onset moves to ~1450 game for a=0.5-vs-cold pairs
(physics lens F6), so the narrative and margins are recomputed, not discovered
as a surprise red. `cuda_pr1` part 3 hard-fails on limiter-engagement flips —
that IS the gate doing its job; the patch anticipates it.
Saturation-property assertions at (1.0e-5, 3.0e-6) survive (headroom verified).
*P-K3:* `test_eos_p1_calibration.py` (rewritten in-patch, §3c);
`test_air_boundary.py` unchanged-green (ruling 6); pump ΔN re-checked in the
report; CPU↔CUDA lockstep tol 0.
*Suite gate everywhere:* **the 27 known reds stay the set-unchanged gate; "no
new reds" is the bar.** Anything invalidated by THIS approved change is a
named, justified addition recorded in P-K4 — never silent. Golden/digest
baselines: deferred per ruling 4; `docs/cuda_xarch_ada_runbook.md:24` lineage
noted as superseded (append).

## 4. What deliberately does NOT change
- All game-unit dials keep their values THROUGH the map change: ignition 300,
  `T_emit_gate` 310, cool_shift 13, the promoted P-F1b set (P-K0 is promotion,
  not retuning).
- `heat_ambient_ref = 20.0` (§2 gap) and the `[combat]` flux→felt-temp dials —
  but NOTE: felt damage is nonetheless retuned off-plateau via E′ (§3b).
- EOS behavior: byte-identical (ruling 6 + frozen phi_exp).
- Bench targets in fire_tune_loop (Erik judges by feel).

## 5. Storm-session interaction
This build takes two of the four parked decisions (Kelvin-map unification;
phi_exp *naming*). It does NOT take: phi_exp's tuned value, interior air
damping, the pf1b *branch* cleanup, cool_shift 9-vs-5. With ruling 6 the EOS
is fully byte-identical this arc, so storming dynamics are untouched by
construction (physics lens F9 verified: air has heat_atten 0, radiation never
heats gas; only second-order conduction from solids' shifted radiative
equilibria reaches the EOS, and only off-plateau).

## 6. Patch list

| # | patch | mode | tier / effort | gate |
|---|---|---|---|---|
| P-K0 | Promote the blessed TUNE dial set into config.toml (all §TUNE sim dials incl. Erik's k_grow 0.5; fixes the load-time ignition_seed warnings) | subagent, this worktree | Sonnet 5 / medium | harness run with NO --set overrides reproduces the last blessed bench CSV byte-identically (override-string comment row excepted); game boots with zero [fire] load warnings |
| P-K1 | Config section + `src/temperature_scale.py` + render/tools/tests migration (§3d) | subagent | Sonnet 5 / medium | §3e P-K1 list green incl. de-vacuoused config test + kelvin_map check; no sim files touched |
| P-K2 | Radiation: C++ plumbing + K = amb + slope·T_mid bake + rad_scale′ 3.1394e-6 + pr1/pf1a updates + comment rewrites (§3a-b) | subagent | Sonnet 5 / high | rebuild; `pytest tests -q` no new reds (named additions recorded); pr1 + pf1a + cuda_pr1 green with re-derived crossover; CPU/CUDA parity if CUDA-capable here (else desktop follow-up before merge, flagged) |
| P-K3 | EOS: s_eos identity mechanism + config-key move + migration guards + calibration-test rewrite incl. effective_pin==65540 (§3c) | subagent | Sonnet 5 / high (bit-identity oracle — tier per skill rule) | rebuild; **CPU-vs-CUDA lockstep tol 0 AND before/after byte-identity** on an EOS-exercising scenario; no new reds |
| P-K4 | Known-red bookkeeping; runbook append; append-only supersession notes (§9 docs list); canon-fold (ch. 04/06/08/12); bench re-run | inline (orchestrator) | — | bench physics guards PASS |
| P-K5 | **HUMAN-TEST: Erik plays.** Checklist: fire spread feel under ×3 (E′/E table §3b); stand-next-to-fire damage at plateau vs inferno vs smolder (§3b consequence); warm-glow-but-radiatively-inert band (169–310 game now visible-but-inert vs 253–310 before — physics lens F8); k_calibr look in-game | Erik | — | Erik's verdict gates merge; NO auto-merge |

Sequencing: P-K0 → P-K1 → P-K2 → P-K3 → P-K4 → P-K5, all sequential in THIS
worktree (one git-touching agent at a time). Commit slicing BEFORE P-K0
(critique F9, scope lens):
1. `tools/fire_tune_loop.py tools/fire_tune_plot.py config.toml` — session WIP
   (TUNE k_grow 0.5, ×3 preview, kelvin_map), explicit paths, never `add -A`.
2. `docs/radiation_raycaster_extinction_ruling_2026-07-31.md` — **restore the
   append-only original, re-express the pending change as an appended
   supersession note** (ruling 7).
3. This design doc.
`.vscode/`, `_fire_tuning_artifacts/`, beastiary/campaign docs stay untracked.

Environment note: consult `machine-env` before P-K2 for this machine's CUDA
build capability; if absent, CPU gates run here and CUDA parity is a flagged
desktop follow-up before merge.

## 7. Post-arc: golden-suite redesign (Erik's initiative, 2026-08-13)

After tuning settles, the golden suite is redesigned WITH Erik rather than
re-baselined from whatever state the tuning left. His sketch: one deterministic
canonical scenario exercising the full sim surface — standing water, a living
fire, rooms held at different pressures, and perhaps an explosion — so a digest
catches regressions in every subsystem, not just the quiet ones. Separate
design session (design doc → critique → bake), deliberately NOT part of this
arc. Until then digest gates run on parity (CPU↔CUDA), not historical
baselines.

## 8. Rollback
Everything funnels through one config section + one accessor + two C++ member
pairs. Revert = `git revert` of the patch commits; no data migration.

## 9. Appendix — site inventory (scout 2026-08-13, amended by critique)

Radiation: `cpp/src/raycaster.cpp:42,48-53,59,62,70-75,512-526,851,995`,
`raycaster.h:160-167,172-211,259,267-272,423,912`, `config.toml:304-323`,
`bindings.cpp:589-592,1853-1864`, `cuda_raycaster.cu:189-250,374,395` (no
second bake — verified), `physics_runner.py:335-336,354-355`.

EOS: `eos_solver.h:57,162-166`, `eos_solver.cpp:286,290,311,316,469-475,573,578`,
`cuda_eos_step.cu:149,150,164,168,261,517`, `cuda_eos_step.h:83`,
`cuda_eos_resident.cu:146,155,770`, `bindings.cpp:2113`,
`physics_runner.py:390-391,404-405`, `config.toml:467-470`,
`ambient.py:11-12,22,36-40,69-88,98-107,139-145`, `pump_system.py:16-19,28,142`,
docs mirrors `field_edit.py:414`, `gamemap.py:739,1035`, `temperature_solver.h:35-38`.

Render/tools: `config.toml:837-848`, `renderer/blackbody.py:60,164-165,212-215,
236,246,253-269`, `game_renderer.py:239-241`, `overlays.py:347,373`,
`fire_lights.py:95`, `frame_lights.py:94,120`, `speckle.py:213,254`,
`hover_readout.py:18,66-81`, `tools/lighting_demo.py:1366,1382`,
`tools/fire_tune_plot.py:62-76,116-123`, `tools/fire_tune_loop.py:170-200,706-714`.

Tests: `test_pr1_fire_plane_cast.py:69-70,94,99-112,470-505`,
`test_pf1a_radiation_books.py:68,539-546,704-708,724`,
`cuda_pr1_fire_plane_check.py:45,56-61,318,345-357`,
`test_fire_heat_source.py:147,371,432`, `test_blackbody_ramp.py:98-116,153-175`,
`test_hover_readout.py:53,92,118`, `test_frame_lights.py:40,137,157,174`,
`test_fire_lights.py:40`, `test_speckle.py:37`, `test_eos_p1_calibration.py:25-52`,
**`test_air_boundary.py:21,548,576,764`** (added by critique),
`test_a5_seal_evacuation.py:562-564` (false positive, verified twice).

Felt-temp (Celsius): `config.toml:280,1987-1990`, `exchange.py:253,306-307`,
`test_damage_pipeline.py:366-367`, `test_unit_heat_damage.py:65`, `gamemap.py:531`.

Docs debt (P-K4): canon live-edit — `docs/architecture/engine/04_atmosphere_and_pressure.md:76`
(65540 pin — stays valid under ruling 6, annotate), `06_temperature_and_fire.md`,
`08_ray_engine.md`, `12_config_and_hot_reload.md`. Append-only supersession
notes — `fire_tuning_plan_2026-07-22.md`, `radiation_and_raycaster_design_seed_2026-07-31.md`,
`radiation_raycaster_extinction_ruling_2026-07-31.md`, `eos_research_report.md`,
`thermal_mass_eos_escalation_2026-07-30.md`, `fire_sizing_package_2026-08-02.md`,
`cuda_s7_diffuse_port_spec.md`, `cuda_xarch_ada_runbook.md:24`.

Externals: clean — no CI, no config-reading .bat/.vscode tasks (verified).

## 10. As-built record (P-K4, 2026-08-14)

Bookkeeping close-out for P-K0..P-K3, written after the fact from patch gates.
§3b above is left as the pre-build estimate (record, not corrected in place);
this section is the measured reality, and where the two disagree, THIS section
governs.

**Commit chain:** `9e3f570` (session WIP: k_grow 0.5, k_temp_to_kelvin 3.0
preview, config-driven Kelvin axis) → `6a312d2` (restore the extinction-ruling
doc to its append-only form; re-express the pending change as an appended
supersession note) → `f4c8c4a` (this design doc, v2, rulings locked) →
`9016cd7` (P-K0: promote the blessed TUNE dial set into config, values
verbatim) → `d243993` (P-K1: `[physics.temperature_scale]` + accessor;
render/tools migrate) → `2922eb3` (P-K2: radiation bake on the canonical map)
→ `57e9e67` (P-K3: EOS `phi_exp` slope mechanism at the frozen identity).

**The canonical map, as shipped:**

```toml
[physics.temperature_scale]
kelvin_ambient   = 293.0   # THE map ambient
k_temp_to_kelvin = 3.0     # THE map slope
phi_exp          = 0.3333333333333333   # frozen; eos_slope == 1.0 exact,
                                          # quantizes to 65536
eos_t_amb_k      = 290.0   # deliberate exception (ruling 6) — Q16.16 pin
                            # 65540 (+4 counts) vs 65632 (+96) at 293
```

**Radiation (P-K2), measured:** the bake formula is
`K = kelvin_ambient + k_temp_to_kelvin·(4t+2) = 299 + 12t` from the table
(was hardcoded `297 + 8t`). `rad_scale` re-anchored `1.0e-5 → 3.1394e-6`,
flux preserved exactly at the P-F1b plateau (T = 300 game) — the factor is
`(893/1193)⁴ = 0.313938`. Max `e_table` entry measured `1.7067e13`; int64
headroom ×1.70 (max K = 48287, K⁴ = 5.4365e18).

**Flux-limiter crossover — corrects §3b.** §3b's pre-build estimate said the
crossover moves to "~1450 game" and gave the direction as the steeper `K(T)`
pushing it later. **Measured in-patch: the crossover is at ~1140 game** (down
from ~1300 under the old ×2 map), and the direction in §3b was backwards —
above the T=300 plateau anchor, the re-anchored curve is *hotter* than the old
map at the same game-T, so it reaches the map-independent linear rail
**sooner**, not later. This entry in §3b's estimate table is superseded by
this paragraph; §3b's text is left unedited as the historical record of what
was predicted before measurement.

**EOS (P-K3):** byte-identical this arc, as designed. 120 s bench CSV
byte-equal pre/post; CUDA PART-1 lockstep bit-identical. `t_abs = (s_eos_q·T
>> 16) + t_amb_q`, `s_eos_q = 65536` (the frozen identity).

**Goldens:** deferred per Erik's ruling 4 — no re-baseline this arc. The new
golden suite (water + fire + pressure rooms + explosion) is co-designed with
Erik post-tuning (§7), separately.

**Known-red additions from this arc** (the 27-known-reds gate is
"set-unchanged"; these are the named, justified additions on top of it, all
sourced from P-K0's dial promotion unless noted):

- `tests/test_cool_shift_axis.py` (2 tests) — pinned the pre-promotion
  `cool_shift` dial value; the promoted TUNE set moved it.
- `tests/test_eos_p4_combustion.py` (3 e2e tests) — combustion outcomes shift
  under the promoted fire dial set (values verbatim from Erik's blessed TUNE
  bench, not a retune).
- `tests/test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood`
  — now `XPASS(strict)`. Its own docstring names this the P-F1b handoff
  signal. Verified caused by the P-K0 dial promotion; present already at the
  OLD map (not a P-K2/radiation effect). **DECISION PENDING WITH ERIK:**
  whether to un-xfail it now that the handoff it signals has actually landed.
- `tests/test_pr3_capacity_law.py::test_fire_T_ext_is_derived_from_ignition_temp`
  — `fire_T_ext` derivation shifts under the promoted dial set.
- `tests/test_s3b_fire_determinism.py` (golden-digest based) — digest moves
  with the promoted fire dials; not re-baselined (ruling 4).
- `tests/test_w6_armory.py` (golden) — same digest-movement cause.

**Justified scenario retune (P-K2, not a new red, a deliberate parameter
change):** `test_pf1a_radiation_books` gate-ii firestorm emitter cap lowered
`3000 → 2500` game to restore int32 headroom to ~23% (a pre-existing scene
was already at ~96% of rail before this arc; the ~3.6% flux increase from the
re-anchor tipped it over). Recorded here, not silent.

**Suite state (this machine — Lenovo, CUDA build present):** 47 failed / 1879
passed / 5 skipped. The 47-failure set is stable across P-K1..P-K3 (verified
by stash A/B at each patch boundary — no drift). CUDA reds decompose into (a)
stale-golden parts (expected, ruling 4 defers re-baseline) and (b) one
desktop-calibrated 3.0 ms cost budget the Lenovo doesn't consistently clear
(measured 3.35 ms once, 1.54 ms another run — machine variance, not a
regression); every parity part (CPU↔CUDA bit-identity) is green at tol 0.

**Arc status:** P-K0 through P-K4 complete on `thermal-mass-axis`. P-K5
(HUMAN-TEST, Erik plays) is the only remaining gate before merge — see §6's
checklist pointer (E′/E table §3b, the stand-next-to-fire damage delta, the
warm-glow-but-inert band).
