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

## 5. Determinism state — FYI (resolved 2026-06-29)
The **physics engine is fully cross-machine deterministic** — every field is integer
Q16.16, proven bit-identical cross-compiler (MSVC 14.50≡14.44) AND cross-arch (your Ada
run: GPU≡CPU). The ONLY cross-machine wobble is the **Q2-fenced combat HP/facing float**
in `src/simulation/combat.py` (Python 3.11↔3.12 round it a hair differently → the
unit-state digest flips). That's the determinism boundary we drew on purpose, and it's
the next thing we fix (the **Q2-lift**). Optional 1-command confirm any time:
```
<lenovo-py> tests/_xarch_perfield_digest.py
```
→ it auto-diffs vs the Ampere baseline and prints the first diverging (field, tick) —
should be the **unit-state** hash, every physics field matching. Send me that line.

## 6. Python-version note
The Lenovo is py3.12; the desktop golden is py3.11. **After the Q2-lift this won't
matter** (the last cross-interpreter float is gone). You do NOT need to match versions
for the engine — the physics is version-independent. (Bumping the desktop to 3.12 is an
optional env-parity nicety, not a determinism requirement.)

## 7. Next work (the plan)
- **Q2-lift** (integerize combat HP/facing) → cross-machine fully green → tag
  `cuda-breached`. We do it via the same delegate→gate→review→merge flow as the solver
  ports; I'll bring a plan + the decisions before building.
- Then your roadmap (ML; and the explosion/black-body as a parallel beauty track).
