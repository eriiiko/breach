# Breach on the Lenovo — dev setup + handoff

Seamless guide for making the **Ada Lenovo** the breach dev machine (vacation move).
Lenovo specs (from the attestation): RTX 1000 Ada (sm_89), MSVC 14.44, CUDA 12.9,
miniconda `data` env (py3.12, has torch for the future RL loop).

## 1. Get current
```
cd C:\Users\steen\projects\breach
git pull            # main
```

## 2. Copy the breach MEMORY (the one thing not in the repo)
The Claude memory lives outside git, at:
`C:\Users\steen\.claude\projects\c--Users-steen-projects-breach\memory\`
**Copy the whole `memory\` folder wholesale to the SAME path on the Lenovo.** The
project-hash folder name (`c--Users-steen-projects-breach`) is identical as long as the
repo sits at `C:\Users\steen\projects\breach` on both (same username → same hash). No
distilling needed — it's a curated `MEMORY.md` index + one-fact files.
- Easiest transfer: ask desktop-Claude to zip it into your Google-Drive `ClaudeSync`
  folder (it'll sync to the Lenovo), then unzip into the path above.
- Permanent fix (optional, later): symlink the memory dir through Google Drive like
  `CLAUDE.md` already is → cross-machine memory auto-syncs, no more manual copies.

## 3. Build + run
- **CPU (default, unchanged):** `python main.py`
- **CUDA (whole engine on the Ada GPU):**
  ```
  cmd /c "cpp\build_cuda_lenovo.bat"      # your Lenovo CUDA build recipe
  <lenovo-py> main.py --cuda              # 7/7 solvers on the GPU
  <lenovo-py> main.py --cuda --res 2      # crank the physics grid (2/3/4...)
  ```
  (`<lenovo-py>` = the miniconda `data` python, cp312.)

## 4. Tests
```
<lenovo-py> -m pytest tests/ --ignore=tests/test_main_smoke.py --ignore=tests/test_renderer_smoke.py -q
```
(The CUDA gates `tests/test_cuda_s*.py` need the CUDA build; they skip cleanly without it.)

## 5. THE cuda-breached CONFIRM RUN (Q2-lift landed 2026-07-04 — do this)
The **Q2-lift is merged** (`4b2d0d7`): a pure-integer trig kit replaced every libm
transcendental in the synced path (unit facing, combat bullet trig, HP-delta snapping,
the raycaster's cos/sin), and the golden re-baselined `60bd331f…` → `453829a6…`.
**2026-07-04 SECOND fix (after this run found the spawn-stat hole — §8/§8b): the
spawn-stat pin** replaced `rng.multivariate_normal` (LAPACK — cross-machine
nondeterministic) with Q16.16-quantized species means; golden re-baselined again
`453829a6…` → `ae1164ca…`. This run is the cross-machine proof:

1. `git pull`  (HEAD must be at/after `4b2d0d7`).
2. **REBUILD the CPU build — do not skip.** The lift changed C++ (`fixed_point.h`,
   `raycaster.cpp`, `bindings.cpp`); a stale `.pyd` reproduces the OLD golden = a
   false RED. Rebuild the same way as during the attestation:
   `cmake --build cpp/build --config Release`
   (if the cache fights you, reconfigure the same recipe as before: VS BuildTools
   MSVC 14.44 + the miniconda `data` py3.12 interpreter).
3. Run: `<lenovo-py> tests/_xarch_perfield_digest.py`
   → it auto-diffs vs the committed Ampere baseline (`tests/_xarch_perfield_ampere.txt`).
   - **ALL GREEN** (aggregate digest == `ae1164ca163b4bf49a86694ba78ea5319f86cfff46301c6aa59190207e6c1a12`,
     no diverging (field, tick)) ⇒ **cross-machine determinism PROVEN** — tell Claude,
     who tags `cuda-breached` and pushes it. Done, for good.
   - Any diverger ⇒ the tool now names the exact unit sub-field (hp / facing / pos /
     life+events) — send that line; it localizes the culprit precisely.
4. *(Optional but gold-standard, while you're there)*: rebuild CUDA
   (`cmd /c "cpp\build_cuda_lenovo.bat"`) and run the full suite
   (`<lenovo-py> -m pytest tests/ --ignore=tests/test_main_smoke.py
   --ignore=tests/test_renderer_smoke.py`) — 392 expected green, incl. all CUDA gates
   vs the new golden ⇒ the COMPLETE Beat-B re-attestation on Ada in one go.

## 6. Python-version note
The Lenovo is py3.12; the desktop golden is py3.11. **After the Q2-lift this won't
matter** (the last cross-interpreter float is gone). You do NOT need to match versions
for the engine — the physics is version-independent. (Bumping the desktop to 3.12 is an
optional env-parity nicety, not a determinism requirement.)

## 7. Next work (the plan — see docs/roadmap_2026-07.md)
- ~~Q2-lift~~ DONE (`4b2d0d7`) — §5 above is the remaining confirm → tag `cuda-breached`.
- Next arc: the **game-design discussion** (units, weapons, damage/health, combat,
  game rules, unit-position representation) — Claude leads, decisions together.
- Next week (fresh tokens): the EOS literature research + rung-A/B prototype
  (roadmap Phase 1).

## 8. Confirm-run RESULT — 2026-07-04 (Ada Lenovo, `erik_lenovo`)

Ran §5 on the Lenovo after the Q2-lift, CPU **rebuilt** first (fresh `.pyd`), then
also the §5-optional full re-attestation (CUDA rebuilt too). Toolchain: MSVC 14.44,
CUDA 12.9, miniconda `data` py3.12.

**Outcome: RED — but hugely narrowed, and precisely localized.**

- `tests/_xarch_perfield_digest.py` → aggregate `08cc4dff…` ≠ golden `453829a6…`.
- **First (and only) divergence: `__unit_hp__` at tick 0.**
  - Lenovo: `299deeac…`  vs  Ampere baseline: `097a9ba4…`
  - Facing / pos / life-events and ALL 17 gmap fields (incl. `heat`, `temperature`)
    are **identical** cross-machine at tick 0 — so the Q2-lift **trig/raycaster fix
    worked**: the divergence went from the *whole trajectory* (June 29) to *one
    sub-field*.
- Full suite: **380 passed, 8 failed, 4 skipped.** The 8 fails are exactly the CUDA
  golden-gated trajectory tests (s1, s2b, s3, s4a, s4b, s5, s6, s7) — each fails ONLY
  on the aggregate golden (which `unit_hp` poisons); their on-machine `GPU==CPU`
  halves are bit-identical. `test_fixed_trig` ✅ and `test_unit_heat_damage` ✅ pass.

**Diagnosis (strong inference, to confirm):** the tick-0 HP change comes from
`combat.apply_environmental_damage`, whose chain (`phi = peak_raw/HEAT_SCALE` →
linear `*`/`+`/`-` → `quantize_hp_delta` snap) is **pure IEEE float64 with an integer
input** — bit-identical across x86 machines and Python versions. So the arithmetic
cannot be the cause; the diverging input must be **`peak_raw` — the max *live* `heat`
on the unit's footprint, sampled before the end-of-tick heat clear.** The captured
`heat` field is post-clear (~zeros), so it can't expose this. **Prime suspect:
residual float in the CPU raycaster's `heat` deposit differing across MSVC versions**
(cross-machine raycaster-heat identity was never proven — S2b only proves GPU==CPU
on one box).

**Next step (proposed):** add the *pre-clear* `heat` (and per-unit `peak_raw`) to the
per-field dump, diff cross-machine to confirm the raycaster-heat hypothesis, then
integerize the offending deposit step. Then re-run §5 → expect all-green → tag
`cuda-breached`.

_(Portability note: this run used `BREACH_CUDA_PYTHON=<data py>` + `CUDA_PATH=…\v12.9`
for the CUDA gates, per the `cuda_harness.py` env override.)_

## 8b. ROOT CAUSE FOUND — same day (desktop, 2026-07-04). NOT the raycaster.

§8's raycaster-heat hypothesis is **dead** — the C++ side is fully exonerated. The
culprit is **unit SPAWN-STAT SAMPLING**: `generation.sample_unit_attributes` draws the
10-dim stat vector with `rng.multivariate_normal` (numpy → **LAPACK SVD** of the species
covariance), and `simulation.py` sets `unit.current_hp = float(base_stats.vitality)` —
so tick-0 HP is **BLAS/LAPACK-derived synced state**. The RNG *stream* (seeded PCG64)
is cross-machine exact; the *transform* of that stream is not:

1. **Exact repro**: a standalone `default_rng(20260615).multivariate_normal(mean, cov)`
   with the human species params reproduces the in-sim tick-0 `hp_before` **bit-for-bit**
   (`0x1.750ed3a6c7124p+6` ≈ 93.26 — the spawn draw is the sim's first RNG use, and no
   damage touches HP before the tick-0 env-damage call). Tool:
   `tests/_xarch_liveheat_dump.py` (live pre-clear heat + per-unit peak_raw + hp in hex).
2. **numpy 2.4.6 vs 1.26.4, same CPU**: bit-identical — version alone didn't flip it here.
3. **`OPENBLAS_CORETYPE=NEHALEM` (forcing a different CPU-dispatched BLAS kernel)**:
   **7 of 10 components change — some by whole σ-fractions** (mass 96.2 → 63.7!), two
   components swap places. The species covariance has repeated variances + 6 correlation
   terms → (near-)degenerate singular subspaces → different LAPACK kernels legitimately
   pick different bases → **O(σ) differences, not ULP noise**. (In this particular flip
   vitality happened to hold — its σ=15 direction is the unique dominant one — but on the
   Ada box's CPU/BLAS the hp record diverged, so its draw moved there.)

So: different CPU (kernel dispatch) and/or BLAS build ⇒ different spawn stats ⇒
`__unit_hp__` (and silently mass/base_speed — invisible for a static unit) diverge at
tick 0, with every field + facing/pos identical. Matches every observation, including
GPU==CPU green on both boxes and the June compiler exoneration.

**FIXED same day — the spawn-stat pin (Erik green-lit):** spawn attributes are now the
species MEANS, Q16.16-quantized at the boundary (ingress door 2) — `generation.py`'s
`predefined_unit_attributes`; NO RNG in the spawn path (the unseeded fallback in
`unit.py` is gone too). Spawn hp = exactly 100.0. Golden re-baselined `453829a6…` →
`ae1164ca…` (only `__unit_hp__` moved, from tick 0 — every field trajectory identical).
Stat VARIATION returns with the units/stats redesign as a deterministic sampler
(integer stream → pure-algebraic transform → Q16.16 snap). Also landed: the
**number-ingress lint** (`tests/test_ingress_lint.py`) — AST-scans `src/simulation/`
for libm transcendentals / BLAS-LAPACK / RNG distribution methods / unseeded RNGs;
`ingress-exempt:` pragma for audited exceptions (see materials.py). **§5 is live
again: re-run it → all-green vs `ae1164ca…` ⇒ tag `cuda-breached`.**
