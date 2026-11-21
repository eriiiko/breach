# Affected card checks: the evidence (2026-11-21)

`tools/affected_cuda.py` is the device twin of `tools/affected_tests.py`: given a
change, it selects the card checks (the `test_cuda_*` wrappers and the
`cuda_*_check.py` scripts they launch) the change can reach. It may over-select,
it must never under-select, and it is never the merge gate. This doc records how
it was proven, what it saves today, and what a sharded run could still miss.
Measured on the Work Desktop (RTX 3060, 16 GB), at `cd07866a` + the tool.

## How it selects

- **The card set**: every test file with a test that cuda_green's own `-k`
  collects (read from `tools/cuda_green.py` through the staging gate's
  `cuda_green_select`) and whose source launches the harness. 64 files today.
- **Python**: `affected_tests.closure` (the wrapper, the check script it
  launches, everything imported). A Python or data path goes through
  `affected_tests.reach`, the one per-path rule.
- **Bindings**: every name a closure file uses (attribute, Name, identifier
  string; an f-string as a pattern) that is a binding of `bindings.cpp`. A
  module-level binding counts only in a file that names the module object
  (`bp`, `_bp`, `breach_physics`); a member counts anywhere, gated by its class
  being live.
- **C++**: each binding's body, the module's import-time code and every TU's
  dynamic initialisers seed a walk over C++ DEFINITIONS (function, class with its
  inline members, namespace-scope variable, macro, enum). A name reaches a
  header's definition inside the naming file's include closure, a TU's function
  or variable anywhere, an out-of-line member `C::m` only while class `C` is live.
  A changed C++ file selects the checks whose reach holds one of its definitions.
- **Full card set**: `bindings.cpp`, the build files, the kits, a header more than
  half the TUs include, `config.toml`, a deleted or unknown file.
- **Twins**: a twin's `files` / `device` path also selects the checks that launch
  a kernel in every kernel file of that twin.

File grain was tried first and failed: utility functions live in orchestrator
TUs (`tstream`, `counted_sync` in `cuda_resident_tick.cu`), so every TU reached
every other and every change selected all 64 checks. Two more leaks were found
and closed by measurement: a local variable `h` reached every struct with a field
`h` (members are no longer names of their own), and `ui/draw.py`'s `obs_masks`
field reached every obs kernel (module-level bindings now count only where the
module is named; 27 checks became 1 for an obs change).

## The method: execution, not agreement

A red/green run per mutation shows only the checks that happen to notice one
particular change. The stronger oracle is EXECUTION: a check that never runs the
changed code cannot be changed by it, and every check that runs it can be. So:

1. **Tripwires** at six real sites, each logging once per process when its code
   runs (device code through an atomic flag and `printf`, Python through a file),
   no behaviour changed. One full card run (`--run` over all 64) recorded, per
   check, which sites ran: the baseline green, the per-check timing and the oracle
   in one pass. The rule checked: every check that ran a site is selected by a
   change to that site's file.
2. **Semantic one-line mutations** at the same sites, each run on the checks that
   executed its site (a check that never runs the line is green by construction,
   so it is not re-run), with `-x`. The rule checked: every red check is selected.

The tripwire result holds for ANY mutation at those sites, not only the one run.

## Results

Full card set: 64 checks, all green, 6159 s = 102.7 min (Work Desktop, RTX 3060).

| | site (file) | selected | sel. min | ran the site | went red (semantic mutation) | all red selected |
|---|---|---|---|---|---|---|
| M1 | EOS `kick_kernel` gx shift 16 -> 15 (`cuda_kick_compression.cu`) | 62 | 102.7 | 26 | not run (see below) | n/a |
| M2 | unit lib `clearing_drain` m -> m+1 (`cuda_unit_lib.cuh`) | 25 | 99.0 | 4 | not run (see below) | n/a |
| M3 | vents `sweep` `L.o2 += rem[o2] + 1` (`cuda_unit_vents.cuh`) | 24 | 98.9 | 6 | not run (see below) | n/a |
| M4 | obs `k_air` air value + 1.0f (`cuda_obs_windows.cu`) | 1 | 14.2 | 1 | not run (see below) | n/a |
| M5 | conductor `_advance` tick += 2 (`conductor.py`) | 25 | 99.0 | 3 | 2: conductor, obs | yes |
| M6 | unit store `_flat` `(v + 1,)` (`unit_store.py`) | 25 | 99.0 | 22 | 22: all that ran it | yes |

Every check that ran a site was selected, at all six sites: **no
under-selection**. The semantic red runs for the four C++ sites (M1-M4) were
not run: each needs its own CUDA rebuild, and the machine was busy with
another agent's full suite and cuda_green when the run was cut. They are
implied by the tripwire column (a check that never runs the line cannot go
red; every check that runs it is selected), so they add illustration, not
proof; the Python sites M5 and M6 were run in full. M5's `conductor_batch` ran the line and stayed green (it
compares an env in a batch with the same env alone, both off by the same +2), the
reason red/green alone is a weaker oracle than execution.

### What it saves today

- A change confined to the observation (M4) runs 1 check, 14 of 103 min.
- A physics-only change spares the unit side, but the unit checks are the long
  ones, so a unit-side change (M2, M3, M5, M6) still costs ~99 of 103 min: the
  25 selected checks are the 25 longest.
- An EOS / resident-chain change (M1) selects 62 of 64: nearly every check
  constructs a `PhysicsEngine`, and its `run_substeps` reaches the EOS kernels.

The gap between "selected" and "ran" (M2: 25 vs 4) is the Python grain: a unit
check imports `unit_kernels`, `conductor` and the device tables, so every binding
those modules name counts, whether the check calls it or not. Closing it needs a
Python call graph (which function a check runs, not which module it imports);
not attempted, and the dynamic dispatch of the lockstep harness makes it costly.

## What a shard can miss, and why

`.cuda_green` == HEAD before a Class-P merge stays the rule; this tool does not
touch it. Whether a sharded run (the selection only) could satisfy Class P is a
change to a determinism gate and to git-workflow: Erik's ruling. What a static
reach cannot see:

1. **A change that runs nowhere but still changes the build.** A compile or link
   failure, an ODR clash (two inline definitions of one name), a static
   initialisation order change, a new `__constant__` table pushing a TU past a
   device limit. The module either fails to build (cuda_green's build-id check
   catches it) or changes code no check reaches, which the reach cannot rule out
   for undefined behaviour already latent elsewhere.
2. **Dispatch the reader does not see.** The member rule rests on cpp/src having
   no `virtual` (guarded: `test_member_gating_rests_on_no_virtual_dispatch`). A
   binding looked up by a computed name, or the module object travelling under a
   name other than `bp` (guarded: `test_a_binding_is_never_looked_up_by_a_computed_name`,
   `test_the_module_never_reaches_a_file_that_does_not_name_it`). Function
   pointers are reached by the name written where they are taken; a pointer made
   from a cast or a `void*` table would not be.
3. **Data reached by a path built at runtime.** A level or golden whose name no
   closure file spells (levels are matched by name; an unmatched data path falls
   back to affected_tests' rule or the full set).
4. **The GPU's own state.** Every check runs in its own process, so no state
   carries between checks; a change in driver, toolkit or card is outside any
   source diff.
5. **A red that is not the change's.** A flaky or timing check
   (`cuda_kick_drag2_timing`, `cuda_tick_capture`) can go red under load on code
   the change never touched; a shard would not run it and would not see it.

## What would make a shard safe

- Keep the build in the proof: a shard is valid only for a module built at HEAD
  (`build_id()` == HEAD, the non-empty `.pyd` check), so class 1's failures
  cannot hide.
- Keep the guards that pin the reader's assumptions (no `virtual`, no computed
  binding names, the module named wherever it goes) and add one per new blind
  spot found.
- Re-prove by execution whenever a reach rule changes: tripwires at a few sites
  across regions, one full run, every executing check selected (this doc's
  method; the tripwire patch is small and reverts cleanly).
- Run the full card set at a lower cadence regardless (arc close, nightly), so a
  class 1 or 5 miss surfaces within a day.
- Decide what a shard's stamp would mean: today `.cuda_green` names HEAD and the
  full set; a shard stamp would have to name the selection and the base it was
  computed from.

## Reproducing

    python tools/affected_cuda.py --files cpp/src/cuda_obs_windows.cu
    python tools/affected_cuda.py --run          # the selection, a log each
    python tools/affected_cuda.py --why tests/test_cuda_obs.py

The tripwire patch (six sites + a harness capture keyed on
`PYTEST_CURRENT_TEST`, `TRIPWIRE_DIR` set) and the mutation script were
experiment-only and are not committed.
