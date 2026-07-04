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
py3.11-vs-3.12 should now be IRRELEVANT. This run is the cross-machine proof:

1. `git pull`  (HEAD must be at/after `4b2d0d7`).
2. **REBUILD the CPU build — do not skip.** The lift changed C++ (`fixed_point.h`,
   `raycaster.cpp`, `bindings.cpp`); a stale `.pyd` reproduces the OLD golden = a
   false RED. Rebuild the same way as during the attestation:
   `cmake --build cpp/build --config Release`
   (if the cache fights you, reconfigure the same recipe as before: VS BuildTools
   MSVC 14.44 + the miniconda `data` py3.12 interpreter).
3. Run: `<lenovo-py> tests/_xarch_perfield_digest.py`
   → it auto-diffs vs the committed Ampere baseline (`tests/_xarch_perfield_ampere.txt`).
   - **ALL GREEN** (aggregate digest == `453829a67a38d79e0befd01d591cb19bdeb19f49d9234fb4d27a5083d126501a`,
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
