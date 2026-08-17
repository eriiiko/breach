# Session handoff — 2026-08-17 (energy-books arc closed; pressure arc next)

> **SUPERSEDED 2026-08-17 (same day) — read
> `docs/pressure_arc_root_cause_2026-08-17.md` first.** The "★ Lead worth
> chasing first: the peak cell is at a corner" below was **measured and
> falsified**: restricted to true interior gas, the anomaly is volume-filling
> with no edge or wall preference (the corner reading came from counting
> solid/vacuum cells, which sit at P=0). The root cause is the pressure solve
> running under-converged at `mg_cycles = 2`. Everything else below still
> stands. Kept unedited as the dated record.

**Why this file exists:** Erik is switching to the office PC. Everything a fresh
session needs must live in the repo, not in machine-local memory. This is the
"how to pick up" note; the arc's own record is
`docs/energy_books_arc_close_2026-08-17.md`, and the item-level queue is
`docs/TODO.md` (§ "PICK UP HERE").

## Where things stand

- **The energy-books arc is CLOSED, merged, and pushed.** `origin/main` =
  `11b359c`. Canon folded into `architecture/engine/04`, `/05`, `/06`; the
  arc's nine working docs are in `docs/archive/`.
- Blessed suite state: **48 failed / 2186 passed / 5 skipped**. The single new
  red is `tests/test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing`
  — declared, owned by the retune pass (a spray cone rides the wind and shipped
  damping shortens its throw).
- Shipped dials: `k_drag = 0.5` (Erik: a *starting* value, not tuned),
  `k_drag_heat_frac = 0.0014`, `n_floor_heat = 0.01`, `n_work_ref = 0.25`.
- The `storm-damping` branch is pushed and fully merged. Per repo hygiene it can
  be deleted (local + remote) plus its worktree — **left alive deliberately** so
  the office PC can see the arc's history; delete when convenient.

## Setting up the office PC

Standard per `docs/dev_setup.md` / `docs/lenovo_dev_setup.md`. The two things
that bite: Python is **always** the conda env `data` (bare `python` fails on
breach imports with a misleading ModuleNotFoundError), and the compiled module
is **not** in git — build before anything else:

```
cpp\build_cpu_data.bat          # CPU module -> cpp/build/Release/
cpp\build_cuda*.bat             # the per-machine CUDA script
conda run -n data python -m pytest tests -q      # expect the 48-red state above
conda run -n data python main.py --level playground
```

## Next arc: pressure / momentum — AUDIT FIRST

Erik's ruling (2026-08-17): this is **its own arc**, and it starts with a
measured audit, not a dial — the same discipline that made the energy-books arc
work. The thermal question is closed and confirmed both by the gates and by
Erik at the HUMAN-TEST ("temperature is under control"). What remains is
pressure.

### What is already known (measured, from the 2026-08-17 play session)

Analysed with the committed tool:
`conda run -n data python tools/analyze_blowup_dump.py <dump.npz>`

From `debug_blowup_20260817_051006` (drag OFF — so this is *not* a drag artifact):

- **Temperature is innocent.** Peak T over the whole run = **740.7**, normal
  fire range; **zero** cells anywhere near the 16000 ceiling. The old thermal
  runaway signature (×1.4957/tick geometric climb) is absent. The arc worked.
- **Pressure is not.** `P_max` = **98.075 atm**, and `P_min` reaches
  **−40.3 atm** — deeply negative, which is unphysical on its face and is
  probably the more informative half of the anomaly.
- **It is a density event.** At the peak-pressure cell T = 0.0 game, so
  `p* = C·N·T_abs` gives **N ≈ 98× ambient** in a single cell.
- **★ Lead worth chasing first: the peak cell is at (y=3, x=3)** — a *corner* of
  the 70×100 map, at ambient temperature — not in the play area. That points at
  a boundary/ring/vacuum interaction rather than an explosion or a fire. Treat
  as a LEAD, not a conclusion: this session twice reached a confident wrong
  explanation by reasoning ahead of measurement (see the retraction in design
  §2.7 and the "it was just explosions" correction), so measure before believing.

### Evidence status — read this before hunting for files

The dumps used above are **untracked and local to the desktop PC**
(`debug_blowup_20260817_051006.npz`, `debug_blowup_20260817_051730.npz`, and
Erik's original `debug_blowup_20260814_015714.npz` in the `thermal-mass-axis`
worktree). They will **not** be on the office PC, and that is fine — they are
**superseded**. Since `df088f1` the recorder captures `wind_x`, `wind_y` and
`inert_n2` in `DEFAULT_FIELDS`, so any *new* pop is strictly more diagnosable
than these are: the old dumps cannot answer a momentum question at all, because
**wind is not recoverable from the pressure field** (the gradient is the
per-tick acceleration; `u` is its accumulated history, and the two run ~90° out
of phase in the Helmholtz mode). Just play until it pops and analyse that.

Cost note: the instrumented ring buffer is ~500 MB at the default 2400 slots
(was ~336 MB). Reduce `capacity` if that is inconvenient.

### Suggested opening moves

1. Play `playground` (or the level where it pops most) until the recorder trips;
   run the analyzer on the fresh, wind-bearing dump.
2. Reproduce headlessly if possible — `levels/bench_two_room` + `tools/storm_ledger.py`
   is the committed storm fixture, but note it peaks at ~7.7 m/s and **did not**
   reproduce the drag detonation, so it may be too gentle for this too. A new
   fixture may be needed; that is audit work.
3. Extend the ledger to momentum: it already seams every pass and reports
   `ke`/`eth`; the energy-books arc added per-stage energy counters. The
   momentum equivalent (who writes `u`, who removes it) is the natural next
   instrument — the storm audit §4.1 already established that only the EOS
   writes velocity and that at shipped dials there was no interior sink, which
   `k_drag` has now changed.

## The other two queued items (order per `docs/TODO.md`)

- **T_abs compression-work patch** (archived design §2.9, RULING R1): step 4c
  multiplies ambient-*relative* T, so below ambient it **inverts** the physics —
  compression *freezes* cold gas, which is the cold-rail window's engine. Honest
  form `T_new = (T + 290)·(1±w) − 290`. Own design + critique + HUMAN-TEST;
  feel-adjacent (breach rarefaction becomes genuinely cold, ~97 game-deg at the
  clamp versus 0 today).
- **Post-pressure retune pass**: fire anchors (`peak time` 2.29 → 2.00 min fell
  out of band), `k_drag`'s real value, and the declared spray-cone red.

## Reported-not-fixed, awaiting an Erik ruling

Three consumers read raw, N-unguarded gas temperature (P-E2b's inventory). The
serious one is **sim-affecting**: the EOS CFL sound-speed max-reduction
(`eos_solver.cpp:347-351` + CUDA twin) takes an unweighted MAX of gas `t_abs`,
and that maximum steers `n_sub` — the substep count for the whole tick — so one
thin-N cell with a rounding-dominated T can change integration everywhere.
(Contrast `p*` in the same function, which is N-weighted and benign.) The others:
the `temperature` sensor's area-mean (`sensor_accessor.py:154-175`, wire-able by
a level author, covered by no test) and render fire-light selection (cosmetic).

## Debt noted, not paid

A few code/test comments still cite arc docs at their pre-archive paths
(`src/simulation/physics_runner.py:518-519`; module docstrings of
`tests/test_e1_cold_rail.py`, `test_p_e3_drag.py`, `test_p_e4_reversible_work.py`).
Harmless, one sweep when convenient.
