# Atmosphere & Pressure

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md), [State & Ownership](02_state_and_ownership.md)

---

> ## EOS refactor (2026-07) — as-built
>
> **This chapter below documents the pre-EOS two-field model (`atmosphere` + `wave_p`,
> IMEX). That model has been replaced.** The refactor shipped end-to-end (P1–P6, closed on
> `main`); read the rest of the chapter for the design *rationale* still in force (implicit
> stability, sealed-vs-breach masks, permeability, wind-as-interface), but treat the specific
> two-field mechanics as superseded by the following. Canon design lives in
> `docs/eos_refactor_design.md` (v2) and `docs/eos_refactor_decisions.md` (locked decisions);
> the GPU end-state in `docs/eos_p6_gpu_alignment_review.md`.
>
> **What actually ships now:**
>
> - **One derived pressure, not two fields.** Pressure is a *genuine compressible ideal gas*:
>   `P = C · N_total · T`, evolved by a **Kwatra semi-implicit solver** (a red-black Gauss–Seidel
>   / multigrid Helmholtz solve — the same stability story the chapter argues for, now on one
>   field). Bulk equilibration *and* acoustic fronts fall out of that single pressure; the old
>   `atmosphere`/`wave_p` split existed **only** as a numerical workaround for running implicit
>   diffusion and an explicit wave on one field, and is obsolete.
> - **`atmosphere` is now a zero-copy ALIAS of the derived `P`.** `P` is **materialized once
>   per tick** — right after the `(N, T)` update and *before any consumer* — into a stored
>   Q16.16 field; every reader (wind, water head, `find_burst_walls`, unit push) sees that one
>   consistent `P`. Nothing "feeds pressure" any more: every old writer became a **heat/energy
>   feed (T)** or a **gas-mass feed (N)**, and `P` follows from physics.
> - **`wave_p` / `wave_v` / `wave_source` are RETIRED as acoustic fields.** `gmap.wave_p` is
>   repurposed as the **`P_prev` store** (last tick's materialized `P`); the per-tick pressure
>   *transient* `|P − P_prev|` is what drives the ripple splash and the blow-up trigger. The
>   transient-buffet-vs-sustained-dome distinction consumers relied on now survives as the single
>   field's **time evolution** (the front passes, the dome lingers) — more physical, not bolted on.
>   (`physics_runner.py` documents `gmap.wave_p` as "the repurposed P_prev buffer".)
> - **Two bulk species + traces.** The bulk air is two explicit, *conserved* gases —
>   **O₂ + inert-N₂** — transported by **donor-cell conservative flux** (the water-solver pattern),
>   with the traces (smoke / poison / …) riding on top. `N_total = Σ species` (Dalton),
>   `P = C · T · N_total`. Mass is exact (LSB-level, no silent decay); sealed rooms are airtight
>   by construction.
> - **Wind = −∇P** off the single materialized field (computed before consumers read it).
> - **Native breach venting.** Venting emerges from real `−∇P` toward a true-vacuum (`N = 0`)
>   cell; the geometric `sink_hop` *atmosphere* hack is gone and breach→vacuum generalises beyond
>   the old edge-hull-only rule.
> - **Consumers repointed:** `apply_wave_push` reads `grad(P)` (`k_push` recalibrated);
>   `find_burst_walls` reads the `P` spread through the `atmosphere` alias; the water pressure-head
>   reads the derived **integer** `P` (the old float bridge removed); combustion/ignition O₂ gates
>   read real `N_O2` (see ch.06). The over-pressure relief valve, sealed/breach masks, and
>   permeability model (§2.3, §2.7) all carry over unchanged.
> - **Ported bit-identical to CUDA (P6).** The whole tick — pressure materialization, Helmholtz
>   solve, species flux, compression-work, combustion — runs on the GPU exactly matching the CPU
>   reference (`cuda_eos_step`, `cuda_mg_solve`); the CPU path is permanent as the bit-identity
>   reference.

> ## Boundary conditions — planetside AMBIENT ring (2026-07, as-built)
>
> The literal grid edge is closed/reflective everywhere; a map's *outer* boundary is made in
> LEVEL DATA by its border tiles. Two modes, chosen by the top-level `boundary` field:
>
> - **`"space"` (default):** a ring of SPACE tiles → `is_vacuum`. The MG solve pins them
>   Dirichlet **P = 0**; bulk transport treats them as a mass sink (native venting to vacuum);
>   traces are absorbed. This is the original behavior; unchanged.
> - **`"ambient"` (planetside):** the SAME SPACE tiles route **wholesale** to a new `is_ambient`
>   mask instead (no `is_vacuum` on an ambient map; an interior SPACE tile is a legal "sky
>   shaft"). The ring is an **infinite ambient reservoir** — air exits and enters freely.
>
> The AMBIENT ring is symmetric to SPACE, all local per-tile edits (no new solve structure):
> - **Pressure via a change of variable.** The ring pins **P = P_amb**, implemented as the
>   shift **P′ = P − P_amb**: subtract P_amb from the RHS + warm-start in the shared host-side
>   `mg_build_levels`, solve the *unchanged* zero-Dirichlet multigrid, add P_amb back at the
>   store (masked to `!solid`). Legal because an ambient map has no P = 0 pins to coexist. The
>   coarse Galerkin anchor is only exact for pin value 0 — the shift is what keeps the whole MG
>   byte-identical.
> - **N-primary dials.** `[ambient]` carries `p_amb`/`o2_frac`; **N is primary** —
>   `N_total := quantize(p_amb)`, split O2/inert-N2 — and the **effective pin** is the sim's own
>   `p*(N_amb, ΔT=0)` chain (`src/simulation/ambient.py`). At Earth defaults that is **65540 raw
>   (1.000061 atm), not 65536**: every reachable `p*` is a multiple of ~T_AMB_K raw counts, so
>   1.0 atm has no integer preimage. Seeding the interior + ring to the effective pin is what
>   makes a sealed planetside room's interior trajectory flat. **(2026-08-14, temperature-scale
>   unification):** `C` and the EOS ambient now live in `[physics.temperature_scale]
>   eos_t_amb_k` (290 K), not `[physics.eos]` — a *deliberate* exception to the sim's unified
>   Kelvin map (`kelvin_ambient = 293`), kept because 290 gives this exact 65540 pin (+4 counts)
>   while 293 would give 65632 (+96 counts), a 24× larger standing source in every ambient cell.
>   The pin is test-pinned: `tests/test_eos_p1_calibration.py` asserts `effective_pin == 65540`.
>   See `docs/temperature_scale_unification_design_2026-08-13.md` §2 ruling 6.
> - **Reservoir + bath.** The bulk clamp resets ring `N` to `N_amb` **every substep** (mirroring
>   the vacuum sink `N=0`), rail-counted by the int64 per-plane `boundary_flux`. `u` and `T` at
>   the ring are the vacuum code verbatim (still boundary, ΔT = 0 ≡ ambient — no separate ambient-T
>   dial). Traces are absorbed at the ring (the smoke solver's vacuum idiom widened to
>   `is_vacuum | is_ambient`) — smoke vents into the sky.
> - **Absorber.** Outgoing acoustic fronts are damped by a **velocity-damping band** (a graded
>   `|u| *= (1−k(d))` over a BFS distance band from the ring; `sponge_u_damp`, k_max = 0.9·FP_ONE).
>   A σ *pressure* sponge was tried and **reflects** (a pin hardened is a pressure-release wall) —
>   it ships off (`sponge_strength = 0`). **Un-absorbable residual:** because the pressure solve is
>   elliptic (whole-domain coupling per tick), a near boundary changes the interior at tick 0 —
>   correct finite-domain venting, not a bounced wave, and removable only by a radiation BC we
>   rejected. At ~0.02 atm transients this residual is a feel matter, deemed acceptable.
> - **No water BC** (oceans are an authored indestructible reservoir; the ring is a water sink like
>   SPACE). **Wind-in-from-boundary** stays a source term, not a boundary mode. **Structural edits**
>   (`destroy_wall`/`unseal_tiles`) join `is_ambient` (not vacuum) on ambient maps.
>
> Bit-identical CPU == CUDA (tol 0). Existing space-map goldens are byte-untouched (every ambient
> branch is gated on a live `is_ambient` — dormancy by branch). Design + as-built:
> `docs/archive/boundary_conditions_spec_2026-07-19.md` (v2.4) + `bc_step_a_audit_2026-07-19.md`.
>
> ### Sky exchange — the sky as a VOLUMETRIC composition reservoir (2026-07-24, as-built)
>
> The ring is an **edge** reservoir: a 2-D slice has no sky, so an open planetside field is
> refilled only by diffusion/advection *from its border*, and one crate's volumetric O2 draw
> outruns that edge refill (a room suffocates in ~5 min even outdoors — `fire_tuning_plan` §7 Q2).
> The missing third dimension is added as a **local source term**: every **sky-connected** interior
> air tile slowly relaxes its gas **COMPOSITION** toward ambient at **fixed local N_total** — the
> vertical mixing the slice cannot resolve.
>
> - **The sky mask** (`GameMap.sky_mask`): a flood fill from the `is_ambient` ring through open air
>   (`~solid`) — the exact sponge-BFS reachability, no distance cap — minus the ring itself and all
>   solid/vacuum. Sealed rooms are unreachable → excluded (no sky). Order-free (reachability), so
>   cross-machine identical. Rebuilt **lazily** via a `_sky_mask_dirty` flag set at the
>   `on_tile_changed` structural seam, at a **fixed tick-order point** (top of the sky pass): a wall
>   breach EXPANDS the mask, so a newly opened room starts breathing next tick. *(Accepted
>   approximation: a roofed room with an open door is "sky-connected" and gets refill — authored
>   roof mask later only if it bites.)*
> - **The exchange** (once per TICK, after combustion — `sky_exchange_step`, host pass on the
>   mirror): `target = mul_q16(o2_frac_q, N_total[i])`; `dN = round_signed(λ·(target − N_O2[i]))`;
>   `N_O2 += dN`, `N_inert = N_total − N_O2` (clamped `[0, N_total]`, complement restated exactly).
>   **N_total per tile is invariant by construction → zero pressure/wind footprint** (the next-tick
>   `p* = C·N_total·T` sees no change — the load-bearing property). `o2_frac_q` is the same quantized
>   ambient mole fraction the ring N-split uses (one source of truth). `λ = quantize(dt_tick /
>   sky_tau_s)`, the per-level `[ambient] sky_tau_s` dial (vertical-mixing timescale, s; **0/absent =
>   dormant**, the back-compat default — recommended ~60 s, blessed at the fire re-tune). Conserved
>   O2/inert pair only; smoke's upward-removal λ and temperature are out of scope (`COOL_SHIFT` is
>   the vertical heat channel).
> - **Conservation rail.** Per-plane totals now change volumetrically (O2 up, inert down); the int64
>   per-plane `sky_flux` records the actual applied Δ (a pure transfer: `sky_flux[O2] ==
>   −sky_flux[inert]`), closing the open-system budget alongside `boundary_flux`.
> - **Quantization deadband** (as-built property): the round-to-nearest relaxation has a deadband of
>   ≈ `0.5·N_total/λ` counts — the far field rests slightly *below* ambient (≈0.199 mole frac at
>   τ=60, ≈0.188 at τ=120, under the 0.19 breathable floor). It grows with τ; the re-tune weighs it
>   (τ=60 recommended). Recovery time-constant ≈ τ, confirmed end-to-end (`tools/sky_exchange_bench.py`).
> - **Runs HOST-side on the mirror in BOTH the normal and GPU-resident ticks** (combustion is itself
>   a host bracket on the mirror in the resident tick, so the pass rides it) → CPU↔CUDA-resident
>   bit-identical by construction, no device kernel (gate e, `tests/cuda_sky_exchange_check.py`).
>   Dormant by default → space + existing planetside goldens byte-untouched. Design:
>   `docs/sky_exchange_design_2026-07-24.md`; τ menu: `docs/sky_exchange_p3_results_2026-07-24.md`.

> ## Energy-conservative thermal transport (energy-books arc, 2026-08-17) — as-built
>
> **The EOS no longer *copies* temperature; it moves ENERGY.** Every thermal exchange in the
> pressure path is denominated in energy, and temperature is what that energy looks like through
> a cell's actual mass. This closed a measured mint: the old semi-Lagrangian T-*sample* wrote
> whatever T it found — including the enormous values a collapsing denominator produces in a
> near-empty cell — onto real mass downstream, creating energy from nothing every tick. Arc
> record: `docs/energy_books_arc_close_2026-08-17.md`; design + per-patch as-builts under
> `docs/archive/` (`energy_transport_design_2026-08-16.md` v2.2 is the contract).
>
> **The four rules the pass obeys.** *Transport conserves* — thermal energy rides the same
> conservative donor-cell face fluxes bulk mass rides, so mixing is mass-weighted by construction
> and phantom T carries ~no energy and dilutes on contact. *Conversions are endpoint-local* —
> energy→T at any endpoint divides by that endpoint's own capacity. *One-way guards, all counted,
> in energy units* — floors and wipes may only destroy, and every such site carries an int64
> energy-sum counter, not a hit count. *Determinism unchanged* — Q16.16/int64, order-pinned loops,
> CPU↔CUDA bit-identical, and the energy accumulator is transient scratch, never synced state.
>
> - **Transport (the substep loop).** The fused SL sample is **u-only**: its `.t` slot is retired
>   at every site, live and reference and both GPU dispatch paths, and the A2 `t_occlude`/`tcmask`
>   machinery retires with it. In its place, each substep builds an exact unshifted int64
>   accumulator `e[i] = n_bulk[i] · T[i]` over participating gas cells (`!solid && !ts &&
>   !is_vacuum && !ring`), moves `φ_e = (Σ conservative dq) · T[donor]` across every face using the
>   **post-limiter, post-`scale_mag`** flux the mass books actually move, and recovers
>   `T[i] = floordiv_q(e[i], n_bulk_new[i])`.
> - **Floor division is load-bearing, not a style choice.** Both C++ and CUDA integer `/` truncate
>   toward zero, which on a sub-ambient cell rounds T *upward* — a mint. Both backends agree with
>   each other, so **no parity gate can catch it; only an energy ledger can.** The shared FP_HD
>   helper `floordiv_q` (`fixed_point.h`) is the one transcription, used by the recovery, by
>   conduction's endpoint divide, and by the expansion branch of compression work.
> - **Rule (d) — the crate face.** Relative energy never crosses a face touching a `thermal_solid`
>   tile; mass still moves. Air→ts debits the donor **at its own temperature**, so the donor's
>   recovered T is exactly invariant as mass leaves (no concentration mint), and the debited amount
>   lands in the signed counter `e_ts_residual`. Ts→air delivers **zero** relative energy — mass
>   emerging from an object dilutes the receiver toward ambient, the same born-at-ambient class rule
>   vacuum and ring emergence follow. Ts→ts moves no energy at all. Physically: gas transiting an
>   object sheds its excess relative heat, counted in the ledger rather than delivered to the object.
>   The honest gas↔object convective exchange is a named future upgrade, not built.
> - **Guards, each with an energy counter.** `n_bulk_new < N_EPS` wipes T to 0 with the residual in
>   the signed `e_wipe_sum`; the T_MIN clamp on recovery is a **creator** and is counted in
>   `e_floor_sum`; vacuum and ambient-ring cells keep their per-substep `T := 0` wipe verbatim; the
>   ring's N reset stays a named boundary channel. `n_active_flux` / `n_bulk_active_sum` measure the
>   active-flux fraction the truncation bound is scaled by, rather than assuming it.
> - **The property, and how it is gated.** The transport pass's Σ eth contribution is **one-way
>   non-positive per tick**, and on a sealed map it closes as an *identity*:
>   `eth_transport_delta = −e_ts_residual − e_wipe_sum + e_floor_sum + trunc`, with
>   `trunc ∈ (−n_bulk_active_sum, 0]`. Asserted every tick on both backends
>   (`cuda_bulk_flux_check` PART 3), with all five counters bit-identical CPU↔GPU.
> - **`digest_advect` moved** across the flux call — it hashes `(wx, wy, T)` and the T it must hash
>   now only exists after recovery. A declared, one-time digest-stream reorder.
>
> **Compression work (step 4c) gained a trust gate and became reversible.**
>
> - **Trust gate (`n_work_ref`, default 0.25).** The work term is faded by the cell's own bulk N:
>   the factor is 0 below `n_work_ref/2`, ramps linearly to 1 at `n_work_ref`, and is applied
>   magnitude-first (`scale_mag`) **before** the ±`T_WORK_CLAMP` compare, so a negative k fades
>   toward zero and never past it. The input is the existing `n_total` plane — no new reduction in
>   any twin. This is what ended the hot-rail runaway: a starved, evacuated pocket is no longer
>   trusted to do compression work on itself.
> - **Reversible work.** The multiplicative update was not self-inverse — a full oscillation gave
>   `T·(1+k)(1−k) = T·(1−k²)`, bleeding a proportional slice of T per cycle with no counterparty.
>   The **compression branch is kept verbatim** (so the hot rail stays bit-identical to its measured
>   history and `sat_add_q16`'s wrap protection survives) and the **expansion branch is its exact
>   inverse**, `T ← floordiv_q(T << 16, FP_ONE + w)`, with `k == 0` pinned to the expansion branch.
>   The pair is exactly self-inverse in reals; in integers the residual is ≤1 LSB, one-way, and
>   measured (exactly 0 at the clamp in both cycle orders and both signs of T). Under asymmetric
>   oscillation a proportional term survives but is far smaller — the worked case loses 2.78 %/cycle
>   where the retired law lost 10.4 %. The virtue claimed is **reversibility, not adiabatic
>   fidelity**; expansion cooling is ~33 % weaker at the clamp as a consequence.
> - **Named accepted gap:** the work term multiplies **game-T, not T_abs**. Below ambient this does
>   not merely omit physics, it inverts it — compression *freezes* sub-ambient gas — and that is the
>   cold-rail window's engine. The honest form (`T_new = (T + 290)·(1±w) − 290`) is specified and
>   queued as its own patch with its own HUMAN-TEST.
>
> **Interior momentum drag — the storm's honest grave.** The engine had no interior momentum sink
> at shipped dials, and the undamped door-neck Helmholtz mode *is* the storming. Damping without an
> energy destination is what created the old 0.002–0.01 `wave_absorb` rectifier window; real
> viscosity deposits wave KE as heat, which makes the dial safe at every value instead of only above
> a threshold.
>
> - **Placement: per TICK, in the step-4 kick loop, after the |u| cap and before the store**, in all
>   four kick twins. Per-tick (not per-substep) dissolves the substep-count trap — `n_sub` varies
>   1–8 with CFL state, so any per-substep form would make total damping depend on flow speed — and
>   removes any need for a `pow`, which would have been the codebase's first scalar fold not built
>   from IEEE-exact ops. `kd_q = quantize(k_drag·dt)` is folded once per tick beside the other
>   scalars, and the mechanism branches on the **quantized** fold, so a `k_drag` too small to
>   quantize is exactly dormant rather than float-branch-live.
> - **The shrink is component-wise magnitude-first** (`u ← u·(1−kd_q)`, the sponge idiom). This is
>   load-bearing beyond style: it makes `|u_old|² − |u_new|² ≥ 0` **structurally**, so the heat
>   deposit can never go negative from rounding and needs no clamp and no signed oracle term.
>   "Improving" this to a magnitude-based scale silently reintroduces that term.
> - **The counterparty.** `ΔE_cell = (|u_old|² − |u_new|²)/2` is a *specific* (per-unit-mass) kinetic
>   energy — `u` is intensive — so `ΔT = k_drag_heat_frac · ΔE_cell / c_v` needs **no per-cell N
>   divisor**; N enters only in the n-weighted oracle counters that convert to the extensive N·T
>   currency the rest of the books use. The deposit lands in the same cell (shear heating at a door
>   neck is physically placed), before 4c, identically in every twin. Ts cells skip both the drag and
>   the deposit; a phantom-T guard suppresses the T *write* below 1 raw count of bulk N while the
>   accounting still runs.
> - **Its oracle is an identity, checked per tick:** `ke_drag_removed = 2·c_v·(e_drag_deposit +
>   e_drag_drop_sum + e_drag_rail_clipped)`. The rail-clip term is not optional bookkeeping — with
>   `c_v = 1` a capped jet reaches `T_MAX_PHYS` in ~14 ticks, so clipping is an expected regime and
>   every clipped LSB is KE destroyed without counterparty unless counted.
> - **Two dials, both in `[physics.eos]`.** `k_drag` (per-second rate; **shipped 0.5** — a *starting*
>   value Erik picked at the HUMAN-TEST, with real tuning deferred to the retune pass after the
>   pressure arc) and `k_drag_heat_frac` (**shipped 0.0014**, the physical-air anchor). The fraction
>   matters more than it looks: Q16 game units put air's heat capacity ~700× below physical
>   (`c_v = 1` by convention), and because the deposit scales with **u²**, a fully honest fraction
>   detonates at blast velocities — an explosion's own wind self-immolates into heat. Any
>   non-deposited remainder is the counted, named destruction channel `e_drag_drop_sum`. A load-time
>   tripwire warns when `wave_absorb` sits in the forbidden `(0, 0.02)` band with `k_wind_strip > 0`:
>   `k_drag` is the intended replacement for that lever (`wave_absorb`@0.02 ≡ `k_drag` 0.0067).
> - **KE↔eth is HALF-coupled, by decision.** The kick still mints KE with no eth debit, and the drag
>   launders that mint into eth. Bounded and small at bench scale; named here because it is a
>   positive-feedback path at blast scale.
>
> **Dalton is bulk-only.** Traces carry **zero** pressure weight — `trace_mass_scale` is retired from
> every Dalton sum (both live-step sites, the P6.4 kick-reference family, and every CUDA twin), so
> `n_total ≡ n_bulk = O₂ + inert-N₂` everywhere, and the trace-decay→N₂ credit is deleted. See ch.05
> for what traces still are.

## 1. What this system is

Breach's atmosphere is the air that fills the ship: a scalar pressure field over the
tile grid, plus a propagating acoustic shockwave, plus the **wind** (pressure gradient)
that the field induces. It is the central physical medium of the game. Almost everything
interesting is downstream of it:

- A grenade pumps a pressure spike into a room. The spike radiates outward as a shockwave
  and equalises by diffusion.
- An explosion shatters a hull tile. The hole exposes vacuum, the room's pressure drains
  through it, and the resulting gradient blows air — and the smoke and fire riding on it —
  toward the breach.
- A sealed compartment holds its pressure; an over-pressured one can rupture its own walls.

None of these are scripted. They fall out of one solver reading and writing shared fields
(`gmap.atmosphere`, `gmap.wave_p`, `gmap.wave_v`, `gmap.wave_source`, `gmap.wind_x`,
`gmap.wind_y`) under a single set of boundary rules. The smoke, fire, and unit systems are
**consumers** of the atmosphere fields; they never reimplement the air physics.

The solver is `AtmosphereSolver` in `cpp/src/atmosphere_solver.{h,cpp}`, a single-step
C++ class. Python (`src/simulation/physics_runner.py`) owns the substep loop and orchestrates
it against the smoke and fire solvers each tick.

### The two-field model

The air is represented by **two** coupled scalar fields rather than one:

| Field | Meaning | Character |
|-------|---------|-----------|
| `atmosphere` | bulk air pressure (1.0 = standard atm) | slow: diffuses, drains, holds gradients |
| `wave_p` | zero-mean acoustic shockwave | fast: a damped wave that radiates and dies |

with `wave_v` as the wave's velocity auxiliary, and `wave_source` as a staging buffer where
explosions deposit energy before it is fed into `wave_p`. The **total pressure** a downstream
system sees is `atmosphere + wave_p`, and **wind is the negative gradient of that total**.

Splitting the field is a deliberate choice, justified below (§4). The short version: the slow
bulk field gives the satisfying decompression/venting behaviour and the gradients that drive
smoke; the fast zero-mean field gives the sharp blast front. Keeping them separate lets each be
tuned — and integrated — on its own terms, and is what makes the solver unconditionally stable.

---

## 2. How it works

### 2.1 The IMEX scheme

The solver advances one substep with an **IMEX** (implicit-explicit) update: the wave part is
**explicit**, the diffusion part is **implicit**. Per substep, for step size `dt`:

```
1. Feed sources:        wave_source → wave_p   (rate-limited)
2. Explicit wave kick:  v += dt·(c²·Δwave_p − γ·v)
                        wave_p += dt·v
                        wave_p, v ← 0 on walls / vacuum / obstacles
3. Transfer anomaly:    atmosphere += (wave_p − mean(wave_p))·transfer·dt
4. Implicit diffusion:  solve (I − D·dt·Δ) atmosphere_new = atmosphere
                        via red-black Gauss-Seidel (gs_iters sweeps)
5. Boundary conditions: vacuum relaxation + 2-tile sponge layer
6. Wind:                wind = −∇(atmosphere + wave_p)
```

`Δ` is the standard 4-neighbour discrete Laplacian with **Neumann** boundaries: where a
neighbour is an obstacle, its value mirrors the centre (the wall reflects). This single stencil
gives reflection off walls, diffraction through doorways, and channeling along corridors for
free. The `obstacles` mask is walls **and** units, so units block waves and airflow.

**Step 1 (source feed).** Explosions stage energy in `wave_source` rather than poking `wave_p`
directly. Each substep feeds out a rate-limited fraction — `min(wave_source·feed_rate·dt,
wave_source, max_source_per_step)` — so a five-grenade stack injects over many substeps instead
of as one grid-scale impulse that would ring.

**Step 2 (wave).** The wave is integrated in kick-drift form: velocity is kicked by the
Laplacian and damping, then pressure drifts by the velocity. This is the damped wave equation
`u_t = v, v_t = c²Δu − γv` and behaves far better than naive forward Euler on the second-order
form. After the kick, `wave_p` and `wave_v` are zeroed on walls, vacuum, and obstacles — the
acoustic field has no business existing inside solids.

**Step 3 (transfer).** The wave is zero-mean by construction; only its *anomaly* relative to the
current mean is bled into the bulk field. This is the coupling that lets a blast leave a lasting
pressure bump (and therefore a lasting wind) after the acoustic ring has decayed.

**Step 4 (diffusion).** The bulk field equalises by solving `(I − μΔ)atm_new = atm` implicitly,
`μ = D·dt`, with red-black Gauss-Seidel. Walls and obstacles are skipped (Neumann); vacuum tiles
are left to the boundary pass.

**Step 5 (boundaries).** See §2.3.

**Step 6 (wind).** Wind is the negative gradient of total pressure, central-differenced, computed
as a byproduct of the same pass — no separate gradient solve. It is zero inside solids and vacuum.

### 2.2 Why implicit diffusion — the stability result

The reason diffusion is implicit and not explicit is a hard numerical result, not a preference.

For a Fourier mode of the 5-point Laplacian, write the mode magnitude as
`σ = 4(sin²(θx/2) + sin²(θy/2)) ∈ [0, 8]`, with the worst case `σ_max = 8` (the checkerboard
mode). An **explicit** diffusion step multiplies that mode by `1 − μσ`. The classical stability
bound is `μ ≤ 1/4`; but the *monotonicity* (no-sign-flip) bound is the stricter `μ ≤ 1/8`.

The catch is the coupling. When wave and diffusion are applied as **separate** passes on the same
field, the combined per-mode update is unstable the instant `1 − μσ < 0` — i.e. as soon as the
diffusion factor flips sign on the high-frequency modes, the wave coupling amplifies them. Damping
softens this but does not remove it. An earlier split scheme ran with `μ ≈ 0.24`, giving
`μ·σ_max ≈ 1.92` and a sign-flipping factor of `−0.92` on the sharpest mode — exactly the regime
that blows the solver up, even though the wave alone and the diffusion alone were each stable.

The **implicit** diffusion factor is instead `1/(1 + μσ)`, which is **positive and ≤ 1 for any
μ ≥ 0**. There is no sign flip, no high-frequency oscillation, and **no diffusion CFL limit at
all**. The only timestep restriction left is the wave CFL:

```
max_dt = 0.5 / c
```

This is why `D` (diffusion coefficient) can be set aggressively (default 200) without affecting
stability or substep count, and why the substep count is governed purely by the wave speed `c`.
The full Fourier analysis lives in `docs/atmosphere_solver_analysis_and_patch_plan_20260319.md`;
this chapter records the conclusion the code is built on.

### 2.3 Boundary conditions — the sealed/breach distinction

The single most important behavioural rule is the distinction between a **sealed hull** and a
**breach**, and it is encoded entirely in the masks:

- **Sealed border / hull** tiles are `is_vacuum` **and** `obstacles` (now sourced from
  `permeability == 0`). Because they are obstacles, the Neumann reflection blocks waves and diffusion:
  air cannot leak through an intact hull. A sealed ship holds its pressure indefinitely.
- **A breach** is a `is_vacuum` tile that is **not** an obstacle (its wall was destroyed). Waves
  and diffusion propagate *into* it, and it drains the room.

This is what makes "explosion breaks hull → room vents" emergent: `destroy_wall` flips an edge
hull tile to exposed vacuum, and the same solver that was holding pressure now drains it. No
special venting code exists.

The drain itself is **not** a hard Dirichlet `p = 0` (which is numerically sharp and rings).
Instead:

- **Vacuum relaxation:** on exposed vacuum tiles, `atmosphere *= (1 − η)`, `η = clamp(breach_rate·dt, 0, 1)`
  (`breach_rate` is a *vacuum-drain* rate, **not** wall failure — a misleading name; see §3).
  A smooth ramp toward zero that drains a room's **pressure** over ~1–2 s. It raises the near-breach
  gradient that drives wind, but the deep-interior gradient flattens, so this does **not** by itself
  carry *smoke* out of a deep room — clearing lingering smoke is a smoke-side concern (ch.05), not the
  atmosphere's.
- **2-tile sponge layer:** a short absorbing region seeded **only from exposed vacuum**. The inner
  ring (distance 1) strongly damps `wave_v`, relaxes `atmosphere`, and zeros `wave_source`; the
  outer ring (distance 2) damps moderately. This kills reflections and grid-scale ringing at the
  opening. Crucially, the sponge's distance field does not propagate through walls, so it cannot
  reach through a sealed hull to drain the interior — the sealed-ship guarantee survives.

When a wall is destroyed *between two air tiles* (not a hull edge), the new air tile is filled
with the **neighbour mean** of `atmosphere` rather than zero, so opening a door does not punch an
artificial vacuum pulse into the room. This fill lives in `GameMap.destroy_wall` /
`_neighbor_mean`, on the Python side, because it is a state-topology edit, not a solver step.

### 2.4 Source injection

Explosions are discrete events, handled in `src/simulation/physics.py:apply_explosion`, not in
the solver loop. Within an explosion radius, per tile (with linear `falloff`):

- The shockwave deposit goes into `wave_source` spread over a **3×3 kernel** `[1,2,1; 2,4,2; 1,2,1]/16`
  with total energy preserved. Smoothing a single-tile impulse over the kernel keeps the sharp blast
  feel while not exciting the worst grid-scale modes.
- A **direct** `atmosphere += pressure·falloff` boost lays down the sustained pressure bump that
  drives smoke transport. This direct deposit is safe precisely because IMEX implicit diffusion
  absorbs the spike — under the old explicit scheme it would have been a blow-up risk.

The same call damages walls in radius (destroying them at 0 HP), clears smoke in the inner 40%,
and ignites flammable tiles within 70%. Unit blast damage is deliberately separate
(`combat.apply_blast_damage`) so the physical event and the gameplay event stay decoupled.

### 2.5 Orchestration

`PhysicsRunner.step(gmap, sim_time)` advances all physics for one game tick
(`sim_time = 1 / ticks_per_second`):

```
dt = atmos.max_dt()              # = 0.5/c, wave CFL only
n  = ceil(sim_time / dt)         # ~ substep count
dt_actual = sim_time / n
for _ in range(n):
    atmos.step(...)              # wave + transfer + implicit diffusion + BCs + wind
    smoke.step(dt_actual·dt_scale)   # diffusion + advection by the just-computed wind
destroyed = fire.step(sim_time)  # one step at full sim_time
return destroyed                 # [(y,x), …] tiles where fire burned through a wall
```

Atmosphere and smoke are **interleaved** per substep so smoke rides the shockwave in real time
rather than seeing only the end-of-tick state. Fire runs once per tick on the settled atmosphere.
The runner does not touch the material grid: it returns the burned-through wall coordinates and
the caller runs `gmap.destroy_wall` on each, keeping all state-topology edits on the Python side.

### 2.6 Wind as the shared interface

Wind is the system's public output. It is consumed by:

- **Smoke** — advected and turbulently diffused by `wind` (see the Smoke chapter); the dominant
  consumer today.
- **Fire** — wind biases spread direction and modulates intensity.
- **Units** — designed to be pushed along the gradient: decompression suction drags entities
  toward a breach, a shockwave shoves them outward. See §5 for status.

Because wind is just `−∇(total pressure)`, every one of these effects is the *same* gradient read
differently. That is the design intent: one field, many readers.

### 2.7 Permeability boundary, wave absorption, and the over-pressure relief valve

Three coefficient-model upgrades have landed; all three are bit-identical to the old behaviour for
the current materials and open air, and only differ where a partial coefficient is present.

- **Bulk permeability boundary.** The diffusion/wave boundary is now a per-cell *permeability* rather
  than the bare boolean `obstacles` mask: the solvers gather flux across each face as
  `face = min(perm[self], perm[neighbor])` over `gmap.dyn_permeability` (rebuilt each tick in
  `stamp_units`). Walls stay `0` (sealed — no wind or smoke through an intact hull), while units carry
  a partial value (soft bodies) and a grill would pass gas without sealing the cell. The boolean is
  just permeability ∈ {0, 1}, so this is a strict generalisation.

- **Per-material wave absorption (4a).** At a wall/unit face the wave update damps per cell by
  `dyn_wave_absorb` (material `wave_absorb` + units via `[physics] unit_wave_absorb`): **units absorb
  blasts** instead of mirroring them, and soft materials damp while hull rings. The wave still never
  enters solids, so the IMEX stability is untouched; this is energy-out only. (Through-wall
  *transmission*, 4b, is still deferred — §5.)

- **Over-pressure wall failure — an opt-in pressure-relief valve.** A sealed room that keeps absorbing
  grenades would build pressure without limit; the relief is emergent, not a cap.
  `GameMap.find_burst_walls(max_pops)` scans wall tiles and, where the **pressure spread across the
  wall's neighbours** exceeds its per-material `burst_threshold`, `Simulation.step` calls `destroy_wall`
  on it — after the fire burn-through pass, capped by `[physics] burst_max_per_tick`, gated by
  `[physics] burst_enabled`. An interior wall becomes air, a hull-edge wall becomes exposed vacuum → it
  then vents. Over-pressured clusters self-breach in a chain until the gradient relaxes.
  - **Opt-in per material.** `burst_threshold <= 0` means *never* (the default, and `air`). You set a
    positive threshold only on the walls you *want* collapsible.
  - **The spread is a true differential** (fixed 2026-07-17). A *solid* neighbour is not a side at all —
    it is skipped, not counted as 0 — and an *exposed-vacuum* neighbour is a real side holding 0. So a
    wall between two equal-pressure rooms holds regardless of how high both climb (the old rule counted
    the along-wall solid neighbours as 0 and burst it at absolute pressure), an interior wall bursts on
    `|P_A − P_B|`, and a hull membrane against space bursts on `P_room − 0`. A threshold is therefore a
    plain differential (interior walls of a normal ship hold ~0).
  - **The hull never pressure-collapses.** This valve was designed for *collapsible interior walls*, not
    the hull — the hull breaches from damage/explosions, so it ships at `burst_threshold = 0`. Weak
    bulkheads (wood/glass) carry low thresholds; steel is high.
  - **Only 1-tile-deep membranes can burst** (2026-07-17, replaces the old layer-by-layer erosion). A
    tile of a ≥2-thick slab has at most one open side → spread 0 → holds ANY differential. Thickness-as-
    strength for free, no baked thickness field: thick walls yield only to damage/explosions — though a
    blast that thins a slab to one layer re-arms the valve there. Deliberate (Erik, 2026-07-17: "only
    1-deep bursts simplifies lots").
  This is *why we keep* the wave→atmosphere deposit (§2.3–2.4): building pressure is physically correct,
  so the answer is an emergent relief valve, not removing the deposit (which would make blasts feel
  weak). Reuses `destroy_wall` + neighbour-mean; no new field.

---

## 3. Parameters

All tunables live in `[physics]` of `config.toml` and are bound onto the C++ solver at init
(`PhysicsRunner.__init__`). `gs_iters` is a solver-side default.

| Parameter | Config key | Default | Meaning |
|-----------|-----------|---------|---------|
| `c` | `wave_c` | 66.0 | Wave speed (tiles/s). Sets the CFL and substep count. |
| `damping` | `wave_damping` | 3.0 | Wave velocity damping (1/s). |
| `transfer` | `wave_transfer` | 0.5 | wave_p → atmosphere anomaly transfer rate (1/s). |
| `feed_rate` | `source_feed_rate` | 200.0 | wave_source → wave_p feed rate (1/s). |
| `d_atm` | `d_atm` | 200.0 | Diffusion coefficient. Free of CFL (implicit). |
| `breach_rate` | `breach_rate` | 5.0 | **Vacuum-drain** rate (1/s) on exposed-vacuum tiles — *not* wall failure (misleading name; cf. `burst_threshold`). |
| `max_source_per_step` | `max_source_per_step` | 10.0 | Cap on wave energy fed per substep. |
| `gs_iters` | — | 8 | Red-black Gauss-Seidel sweeps per substep. |

`dt = 0.5 / c`. The `c = 66` config value (≈ real `c/3` in tile units) gives a larger `dt` and
fewer substeps than the solver's own `300` default; the solver default is the analysis reference,
the config value is the shipped game-feel tuning.

---

## 4. Why the field is split

The split into `atmosphere + wave_p` was adopted as the long-term architecture for three reasons:

1. **Separation of timescales.** Venting and decompression are slow, bulk effects; a blast front
   is a fast, zero-mean acoustic effect. One field forced a single integrator and a single set of
   boundary rules to serve both, which is what created the instability the split avoids.
2. **Independent tuning.** Grenades excite `wave_p`; decompression acts on `atmosphere`; the
   shockwave and the long-term leak can be tuned separately without one ruining the other.
3. **Stability.** Coupling a zero-mean explicit wave to an implicit bulk diffusion is exactly the
   IMEX structure that is unconditionally stable in the diffusion part.

The trade-off the analysis warned about — that splitting too early could weaken the satisfying
single-field decompression feel — is handled by the **direct atmosphere deposit** (§2.4) and the
**relaxation BC** (§2.3): the bulk field still receives blast energy and still drains hard through
a breach, so the game feel survives the split.

A **face-flux breach law** is a *forward* generalisation of the drain — **not** required for
correctness (vacuum relaxation is adequate; the sponge stays for wave anti-ringing). Its appeal:
outflow `∝ max(p_inside − p_ext, 0)` across each breach *face* (`p_ext = 0` for vacuum) **scales
venting with opening size** physically, and it generalises — a *face flux* `= f(p_a, p_b, face)` is one
primitive serving **breaches** (face to vacuum), **vents/ducts** (a throughput cap), **fans/pumps** (a
forced flux), and **cracked doors** (a small conductance).

**Resolved — lingering smoke is a *smoke* problem, not an atmosphere one.** Face-flux as a pressure
*sink* was tried and reverted: with the aggressive `d_atm = 200`, diffusion flattens the interior
gradient faster than the face drains it, so wind → 0 and the smoke never leaves. The venting
experiments concluded the **atmosphere needs no new mechanism** — its pressure drain is adequate, and
adding a *second wind* to the atmosphere is the wrong layer. The real cause was the **smoke advection
stencil** (central-difference → checkerboard); the fix lives in **ch.05 (smoke v2): semi-Lagrangian
advection plus a dial-able smoke-side sink-pull** toward the nearest breach. That sink-pull is a bias
inside *smoke* advection — **not** a second wind, and it never touches the pressure field. Face-flux
therefore stays the forward idea above, valued for size-scaled venting and the vent/fan/airlock
primitive — not as the lingering-smoke fix.

---

## 5. Forward design (not yet built)

These are coherent extensions of the same field-and-gradient model, recorded so they land
consistently when implemented.

- **Decompression suction on units.** Wind is computed but **not yet consumed by unit movement**.
  The intended effect: an entity in a wind field is pushed along `−∇p`, so a breach sucks units
  toward the hole and a shockwave shoves them outward — the same `wind_x/wind_y` the smoke already
  reads. This is the gameplay payoff of the wind field and is the natural next consumer.

- **Through-wall wave transmission (4b — deferred).** The lossy *reflector* (4a — units absorb,
  soft materials damp) has landed; the remaining flourish is *transmission through walls* (sound
  crossing into a wall and out the far side, thick walls blocking more). It needs the wave to
  propagate inside solids with a per-material speed, which (a) demands re-deriving the IMEX stability
  for variable coefficients and (b) raises `c_max` → smaller `dt` → more substeps everywhere — and we
  already run the wave *below* real sound speed because the dt is punishing, so a *higher* in-wall `c`
  is infeasible for now. Bulk flow (wind/smoke) stays sealed at walls regardless; only the acoustic
  wave would ever cross, and that is the deferred flourish.

- **Water → atmosphere coupling.** Once a fluid layer exists, a rising water surface displaces
  air volume; with `P = nRT/V`, shrinking volume raises pressure and drives airflow and smoke.
  This couples the fluid system into the atmosphere through the same pressure field, no new solver.

- **Fuel as a directed gas field.** A flamethrower or teargas can be modelled as a second field
  reusing the atmosphere's diffusion + advection, with momentum injection at the nozzle — a
  directed gas carried by (and adding to) the wind. Atmosphere-adjacent, same machinery.

- **CUDA residency.** The Laplacian-and-gradient passes are 2D stencils and the Gauss-Seidel solve
  is local; both are direct GPU stencil/iteration targets when the field memory moves GPU-resident
  (see the State chapter's residency model). The IMEX structure ports unchanged.

---

## 6. Implementation status

Audited against `cpp/src/atmosphere_solver.{h,cpp}`, `src/simulation/physics_runner.py`,
`src/simulation/physics.py`, `src/simulation/gamemap.py`, and `config.toml`.

**Built and shipped:**

- Two-field IMEX solver (`AtmosphereSolver`): explicit kick-drift wave on `wave_p`/`wave_v`,
  anomaly transfer into `atmosphere`, implicit red-black Gauss-Seidel diffusion (8 iters),
  wind = `−∇(atmosphere + wave_p)`. All per-substep.
- Wave-CFL-only timestep (`max_dt = 0.5/c`), Python-orchestrated substep loop interleaving
  atmosphere and smoke (`PhysicsRunner.step`).
- Sealed-vs-breach boundary distinction via masks; vacuum relaxation drain; 2-tile sponge layer
  seeded only from exposed vacuum; rate-limited + 3×3-smoothed source injection.
- Neighbour-mean atmosphere fill on `destroy_wall` (no artificial vacuum pulse).
- Direct `atmosphere` deposit + `wave_source` 3×3 deposit in `apply_explosion`.
- All parameters config-bound; `gs_iters` is a solver default (not in config).
- **Bulk permeability boundary** — the diffusion/wave boundary reads per-cell `dyn_permeability`
  (`face = min(perm[self], perm[neighbor])`), `obstacles` now sourced from `permeability == 0`.
  Behaviour-identical for the current materials; units carry a partial value (soft bodies).
- **Per-material wave absorption (4a)** — the wave update damps per cell by `dyn_wave_absorb`
  (material `wave_absorb` + units via `[physics] unit_wave_absorb`); units absorb blasts, open air
  bit-identical. Energy-out only.
- **Over-pressure wall failure** — `GameMap.find_burst_walls(max_pops)` + `Simulation.step` destroys
  walls over their per-material `burst_threshold`, capped by `[physics] burst_max_per_tick`, gated by
  `[physics] burst_enabled`. The emergent pressure-relief valve.

**Designed but not built:**

- **Decompression/shockwave suction on units** — `wind_x/wind_y` are produced and consumed by
  smoke (and fire), but **no unit-movement code reads them**. The marquee gameplay effect of the
  wind field is unwired.
- **Lingering-smoke venting (decided — smoke-side).** The atmosphere drain (vacuum relaxation) is
  adequate and unchanged; the lingering smoke was a *smoke* problem (the central-difference advection
  stencil), fixed in **ch.05 (smoke v2): semi-Lagrangian advection + a dial-able smoke sink-pull**.
  No atmosphere continuity-wind / second wind (§4). Face-flux as a pressure *sink* was attempted and
  reverted; it survives only as a forward vent/fan/airlock idea (§4–§5).
- **Through-wall wave transmission (4b)** — the lossy reflector (4a) shipped; transmission across walls
  is deferred (variable-coefficient IMEX + higher `c_max`).
- **Water→atmosphere volume coupling**, **fuel/directed-gas field**, and **CUDA residency** — forward
  ideas only.

**Gaps / known issues:**

- **Config vs solver-default drift.** `config.toml` ships `wave_c = 66`, while the solver's own
  default and the analysis reference are `c = 300`. Several architecture-doc parameter tables also
  list `c = 300`. The config value wins at runtime; the doc tables are stale.
- **Two distinct `dt` notions in config.** `[physics]` also carries legacy `physics_dt = 0.001`
  and `physics_substeps = 8`; these are **not** used by the IMEX runner, which derives its own
  `dt` and substep count from `max_dt()`. They are dead keys for this solver.
- **Mixed responsibility for spike safety.** `apply_explosion` deposits directly into `atmosphere`
  relying on IMEX to absorb the spike; this is intentional but couples the event code to the
  solver's stability properties — worth noting if the integrator ever changes.
- **No double-buffering in the Gauss-Seidel / Laplacian loops.** Red-black ordering makes
  Gauss-Seidel order-independent within a colour, so this is correct as written; but the broader
  "double-buffered propagation" concern (architecture.md §6.10) is unaddressed and should be
  revisited in any raw-loop C++ rewrite of other passes.
