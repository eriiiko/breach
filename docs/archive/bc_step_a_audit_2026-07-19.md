# BC STEP-A audit — `is_vacuum` consumer table (2026-07-19)

Deliverable of B1 (boundary-conditions build, `boundary_conditions_spec_2026-07-19.md` §2).
Repo-wide: loader + GameMap init + Python sim + C++ + CUDA. This table is the B2–B4 build
checklist; classifications reflect the v2 spec decisions (shift trick, per-substep reset,
AMBIENT≡vacuum for u/T).

Legend: **(a)** stays vacuum-only · **(b)** widen to `is_vacuum | is_ambient` ·
**(c)** mode-valued (resolved per spec v2 §1) · **(w)** new-writer site (reset/shift/σ/rail).

## Python (`src/simulation/`, `src/`)

| site | what | class / v2 resolution |
|---|---|---|
| `gamemap.py:357` | load-time `is_vacuum[vac]=True` | THE HOOK — ambient branch routes vac→`is_ambient` wholesale; runs BEFORE door stamp |
| `gamemap.py:369-372` + `door_system.py:97` | door stamp + span-on-ring validation | (b) door validation widens |
| `gamemap.py:395` | water seed mask `(~solid)&(~is_vacuum)` | (b) — ring = space-ring for water (Decision 3) |
| `gamemap.py:418,440` | air_init override mask | (b) — ring rules win over air_init |
| `gamemap.py:442-446` | air_init O2 split (hardcoded 0.21) | (c) — uses `o2_frac`; pattern reused for N_amb split |
| `gamemap.py:518` | t=0 atmosphere `where(solid\|is_vacuum,0,FP_ONE)` | (c) — dial-aware: interior (P_amb), ring P_amb, vacuum 0 |
| `gamemap.py:533-535` | t=0 gas O2/N2 seed | (c) — dial-aware N_amb split |
| `gamemap.py:768,873` | freed-wall refill excludes vacuum | (a) — ambient neighbors count naturally (hold real gas) |
| `gamemap.py:955` | `_neighbor_mean` excludes `solid\|is_vacuum` | (a) — ambient neighbors included naturally; explicit note |
| `gamemap.py:1019,1042` | `find_burst_walls`: vacuum side = P 0 | (c) — ambient side reads effective P_amb (walls with ambient both sides don't burst) |
| `gamemap.py:1082,1096-1109` | `destroy_wall` was_hull/exposes → `is_vacuum=True` | (c) — joins-AMBIENT twin on ambient maps |
| `gamemap.py:1395,1400-1413` | `unseal_tiles` vacuum-join | (c) — joins-AMBIENT twin |
| `combat.py:419,449,466` | fire-O2 open-neighbor mean | (a) — ring participates naturally (breathes) |
| `physics.py:116,130` | explosion deposit skips `solid\|is_vacuum` | (b) |
| `field_edit.py:182,191` | FieldEdit skip mask | (b) |
| `physics_runner.py:474,500,511,548` | C++ call plumbing | plumbing — thread `is_ambient`, N_amb/P_amb, σ grid |
| recorder / renderer | — | NONE read is_vacuum (verified); render tint is B5-new |

## C++ / CUDA live path (`cpp/src/`)

| site | what | class / v2 resolution |
|---|---|---|
| `eos_solver.cpp:351` + `cuda_sl_advection.cu:76` | SL cmask barrier | (b) — ring is a still boundary |
| `eos_solver.cpp:402,1099` + `cuda_sl_advection.cu:222` | SL write `T = is_vacuum ? 0 : sample` | (b) — ΔT=0 IS ambient; vacuum code verbatim |
| `eos_solver.cpp:435` + `cuda_eos_step.cu:367` | Helmholtz RHS `div_u=0` skip | (b) — pin owns ring |
| `eos_solver.cpp:464` + `cuda_eos_step.cu:394` | `pstar=0` at vacuum | NO EDIT — ring (not vacuum) computes p*(N_amb,0)=P_amb naturally after per-substep reset |
| `eos_solver.cpp:511,1183` + `cuda_kick_compression.cu:119,205` | velocity zero at `solid\|is_vacuum` | (b) — ring u ≡ 0 |
| `eos_solver.cpp:600` (step 4c) | compression-work T write skips vacuum | (b) — ring skipped like vacuum |
| `eos_solver.cpp:715-718` | MG excl build (vacuum→Dirichlet) | (c) — ambient→excl=1 too; pin VALUE unchanged (=0) under the shift |
| `eos_solver.cpp:736,744` (`mg_build_levels`, shared host-side) | rhs + warm start | **(w) THE SHIFT** — subtract P_amb here; both paths consume it (`cuda_eos_step.cu:412`) |
| `eos_solver.cpp:1038` + `cuda_mg_solve.cu:266` | post-solve pin apply P=0 | NO EDIT (P′=0 ≡ P=P_amb) |
| `eos_solver.cpp:822-844,925-973` + `cuda_mg_solve.cu:132-178` | smoother/residual/coarse anchor (zero-Dirichlet) | NO EDIT — the shift's whole point |
| step-5 store (both paths) | P materialization | (w) — add P_amb back |
| `bulk_transport.cpp:182-188` + `cuda_bulk_transport.cu:163-173` | vacuum mass sink `N=0` per substep | **(w) THE RESET** — `else if ambient: N=N_amb[plane]`; rail accumulates Σ(N_pre−N_amb), int64, per plane, per substep (atomicAdd on device) |
| `bulk_transport.cpp:36-118` | donor-cell flux vacuum reads | (a) — flux unchanged; reservoir via clamp |
| `smoke_dynamics.cpp:66-72,94,131` + `cuda_smoke.cu:79-82,121-125,161,234,250,284` | trace zero at vacuum | (b) — traces absorbed at ring |
| `fire_simulation.cpp:54,148` + `cuda_fire.cu:118,155,329` | fire O2 excludes vacuum | (a) — ring breathes |
| `combustion.cpp:122` + `cuda_combustion.cu:84` | skip `solid\|is_vacuum` | (a) |
| `cuda_temperature.cu:62` + `temperature_solver.cpp:159-161` | breach T wipe (`is_vacuum && !solid`) | (b) — ring wiped to ΔT=0 |
| `cuda_temperature.cu:205,279` (`cool_shift_vacuum`) | vacuum-adjacent 0 K fast-cool | (a)/accepted — ring-adjacent solids naturally lose the fast-cool (T_amb bath is correct planetside); behavioral delta on ambient maps only |
| Helmholtz level-0 row build (pre-`mg_build_levels`) | row mass m | (w) — σ(d) added at band cells (rung 1, if gate demands) |
| `physics_engine.cpp` (261-360), `bindings.cpp`, headers (`eos_solver.h:209,264,319,355,386` etc.) | signatures | plumbing — many TUs touched, harmless to digests; list per-TU in the B3/B4 patch notes |

## Dead path (do not edit)

`atmosphere_solver.cpp` — pre-EOS solver, not on the live tick (no live caller; bound in
bindings only). Its `vac_dist` BFS + ramp (`:480-542`) is the ALGORITHM REFERENCE for the
σ/sponge grid. Confirm exclusion; never edit for BC.

## Fact-checks (all verified, feeding spec v2)

1. u at vacuum: hard-zeroed + flux-excluded (sites above) → ring mirrors via (b).
2. MG pin: build `:715-718`, apply `:1038`/`cuda_mg_solve.cu:266`; coarse anchor
   `:822-844` is zero-value-only → SHIFT adopted (v2 §1), zero kernel edits.
3. Reset slot: the per-substep clamp pass IS the vacuum idiom; p* then materializes
   P_amb at ring naturally (v2 changes 2–4).
4. Water: solver never reads is_vacuum at runtime; only the seed mask (`gamemap.py:395`).
5. Traces die at the smoke-SL vacuum sites → (b) widening.
6. Renderer/recorder: zero is_vacuum reads.
