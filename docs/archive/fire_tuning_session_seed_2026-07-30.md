# SEED — the fire tuning session (Erik's manual loop, 2026-07-30)

**For: a fresh session whose whole job is to help Erik tune the fire, at the bench.**
Self-contained. Written by the Opus session that built the thermal-mass arc.

**This is not a design session.** The model questions that remain are queued elsewhere
(`docs/fire_model_design_seed_2026-07-30.md`, with Fable). This session turns dials,
reads the scorecard, and helps Erik converge. If a dial cannot reach a target, say so and
stop — do not redesign the model to hit a number.

---

## 1. Where to work

```
cd C:\Users\steen\projects\breach.worktrees\thermal-mass-axis
conda run -n data python tools/fire_tune_loop.py
```

Branch `thermal-mass-axis` (pushed, **unmerged**, HUMAN-TEST gate). It is the only tree
with the fire work; `o2-continuous-law`, `fire-o2-integration` and `sky-exchange` are
ancestors and carry nothing live. **Do not switch the main checkout's branch to look at
this** — open the worktree folder directly.

Edit the `TUNE` block in `tools/fire_tune_loop.py`, run, read the scorecard, repeat.
The loop drives `tools/fire_timing_harness.py` via `--set` overrides; **`config.toml` is
never written by the loop.** When a combo is blessed, its values go to config in a
close-out commit.

Suite baseline on this branch: **39 failed / 1771 passed / 5 skipped**. The 39 are
pre-existing by-design reds inherited from the o2-continuous-law line (`FireSimulation.step`
missing the `n_total` arg — enumerated in `docs/continuous_o2_law_p3_handoff_2026-07-24.md`).
They are owed test updates before merge; they are not breakage.

## 2. What changed under the tuning surface (why old numbers are meaningless)

Four things moved. **Every pre-2026-07-30 dial value predates all of them.** Erik has
never set a fire number himself; treat nothing in `config.toml` as blessed.

1. **A crate's temperature is now an object's.** It used to be advected away by the fire's
   own plume — you were tuning against a number the engine was erasing. (`f5e9aa3`,
   `6f57762`.) `temperature[]` on a `thermal_solid` tile is owned by the TemperatureSolver;
   every other system is a reader.
2. **`cool_shift` is per-material** (`344f3ed`). e-fold = `2^cool_shift / 24` s:
   5 → 1.3 s · 8 → 10.7 s · 10 → 43 s · 12 → 171 s. All rows seeded at 5. Furniture's
   `conductivity = 0`, so **`cool_shift` is the crate's ONLY loss channel** — one clean
   dial. The steady state is exactly
   `T* = k_fire_heat · I · 2^(cool_shift − heat_inv_shift)`, `heat_inv_shift = 3` for
   furniture (verified to ±1% at equilibrium).
3. **The O₂ law no longer normalizes by ambient** (`b340bba`):
   `o2f = clamp01((X − 0.13)/(o2_frac_full − 0.13))` with `o2_frac_full = 1.0` (pure O₂).
   Ambient air now gives `o2f = 0.092`, not 1.0 — so local O₂ enrichment finally raises
   fire intensity, which was Erik's intent all along.
4. **Consequence of (3): `k_die/k_grow` must be rescaled ~10×.** Measured: **0.0506** puts
   a normal-air fire at `I_eq = 0.50`, giving 0.68 @ X=0.25, 0.79 @ X=0.30, 1.00 at pure
   O₂. **Start there.** The shipped 0.5 ratio now means extinction at every X.

## 3. Erik's design intent, in his words

> *"A normal fire in air burns at I ≈ 0.5 — the higher values, I = 1, are for fires which
> have more than normal O₂, or wind that feeds it more O₂, or radiation from neighbouring
> burning tiles."*

So `I` ∈ [0,1] is the tile's normalized burn rate; **0.5 is a healthy normal fire** and the
upper half is headroom for enhanced conditions. §9.3's existing "peak ~0.5" target already
agrees.

## 4. The dial order (fire_tuning_plan §9.3) and the targets

Tune in this order; each stage's dial is roughly orthogonal to the next.

1. **STRUCTURE (set once, don't tune):** `fire_T_ext = 250` (must be BELOW `ignition_temp`
   280/300 — §9.2's cold-start fix), `fire_T_span = 100`.
2. **THERMAL:** `k_fire_heat` vs `materials.furniture.cool_shift` → flame T **400-500**
   game units. Use the §2 analytic; it is exact.
3. **RAMP:** `k_grow` / `k_die` → peak **I ≈ 0.5** at a realistic time (§9.3 says ~3 min).
   Hold the ratio near 0.0506 and move the magnitude to change ramp speed — the ratio sets
   peak height, the magnitude sets how fast it gets there.
4. **LIFETIME:** `wall_damage` → fire death in **6-8 min**, leaving charred `wall_hp`.

**ANCHORED — verify, never tune:** `burn_rate = 0.02` (Huggett 1980); `o2_frac_ext = 0.13`
(Peatross-Beyler 1997); `o2_frac_amb = 0.21` (it is the atmosphere, not a fire dial — and
it is no longer read by the fire laws at all); `fuel_per_o2 = 0.7` (wood stoichiometry).
**Verify every run: far-field X ≈ 0.21 throughout an open-field burn.**

## 5. Traps that will waste your time

- **★ `--set physics.thermal.COOL_SHIFT=N` is now SILENTLY INERT** for any material with an
  explicit `cool_shift` column. Use `materials.furniture.cool_shift`. The loop was migrated;
  anything else you write by hand must be too.
- **★ `cool_shift` is per-material but the vacuum offset is global**: a vacuum-exposed tile
  cools at `max(SHIFT_MIN, cool_shift − 2)`. Raising a material's `cool_shift` also slows
  its space-exposed cooling.
- **The `I_crit` cliff.** Deposit is linear in I and loss linear in T, so the `hot` gate
  only opens above `I_crit = I_peak · fire_T_ext / T_flame`. If a fire snaps out at the
  ignition seed, that is the cliff, not a dial you have not found. Per-material `cool_shift`
  is the honest lever (a slow wood e-fold lets a young fire survive on borrowed heat);
  a *global* `COOL_SHIFT = 12` also "works" but only by making every material in the game
  ~128× more sluggish. **Do not reach for that.** If the cliff still bites at a physical
  wood cooling time, stop and hand it back to the design session — it is Q3 there.
- **The bench report's §3.1 alternate operating points (175/7, 75/8, 35/9) do not
  free-run** — they snap out at the seed. Reproduced on a pristine build, so it is the
  cliff, not a regression.

## 6. Two rules that are not yours to break

- **DO NOT REBASE ANY GOLDEN.** The arc carries exactly ONE deliberate rebase, to be spent
  with Erik when the fire is blessed, with written rationale.
- **★ BEFORE that rebase: give the golden scenario fuel.**
  `tests/field_ab_harness.default_scenario_sim` has `gmap.flammable.sum() == 0` — it seeds
  fire on interior AIR, and `fire_simulation.cpp:143` early-outs (`if (!flammable[i])
  continue`). **No golden in the suite can move when a fire or O₂ law changes.** Rebasing
  against it would re-baseline digests that never watched the thing being tuned. Add
  flammable material under the seeded fire first. (Bench report §8 item 26.)

## 7. What is still open (do not fix here)

With Fable, `docs/fire_model_design_seed_2026-07-30.md`: whether `hot` also uncaps so
neighbour radiation can push I above normal (it is `clamp01` today, same shape the O₂ law
just shed); whether the plume→T shim (`fire_simulation.cpp:265-293`) should convert through
the tile's `heat_inv_shift` or is correctly substrate-independent; and where the `I_crit`
cliff should sit. **Watch the plume shim while tuning**: it was <1% of steady state at
`k_fire_heat = 1600`, but it does not scale with `k_fire_heat`, so if you tune that far
down it could quietly become the dominant heat path. If the §2 analytic starts missing by
more than ~20%, suspect it and report the numbers.

## 8. When it feels right

Fire is **feel-adjacent**: Erik's eyes are the gate, not the scorecard. When he blesses a
combo: values into `config.toml` in a close-out commit, then the deliberate golden rebase
(after §6's fuel fix), then merge `main` in (the branch is ~22 behind), fix the 39 inherited
reds, and merge once — retiring `o2-continuous-law`, `fire-o2-integration`, `sky-exchange`
and `fire-tuning` together.

Next after that, per Erik: **revisit smoke** (`fire-b2-smoke-honesty`, already in this
branch's history at `2875408`, never play-tested). He could not tune the look of smoke
because the fire was too intense to see it — which is exactly what this arc fixed.
