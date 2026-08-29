# Gas energy conservation — design v1 (2026-08-29)

Arc for issue **#54** (sealed rooms heat toward ignition under any
perturbation). Replaces the per-cell temperature-form compression-work step
(EOS step 4c) with Kwatra's conservative energy update on a **stored gas
energy field**. Source: Kwatra, Su, Grétarsson, Fedkiw (2009), *A method for
avoiding the acoustic time-step restriction in compressible flow*, J. Comput.
Phys. — archived as `docs/papers/ADA492343.pdf`. Our EOS already implements
the paper's pressure half (eq. 10/14, the Helmholtz solve, face velocities,
u-only CFL); this arc ports its energy half (eq. 3 with eq. 13/15 faces).

Rulings by Erik, 2026-08-29 (recorded here; not re-litigated below):
- The adiabatic term is physics we KEEP — explosives will become physical,
  not injected; blast/breach heating and cooling must survive.
- **Store energy** for gas cells (not derive-per-tick): exact conservation
  across ticks, not just within one.
- **Thermal solids stay on T** (an object has heat capacity and no flow).
- Drag / clamp heat become **structural**; `k_drag_heat_frac` is deleted.
- Scope stays tight: no feature creep beyond what closes the books.
- Sequence: design → adversarial critique → patches with gates → CUDA
  lockstep → re-baseline → HUMAN-TEST.

---

## 0. Evidence (measured 2026-08-27..29, `tests/_sealedbox_bisect_bench.py`)

| run (18 s, no fire, only a glass box sealed at t=0) | box ΔT | bunker ΔT | arena ΔT | u_max |
|---|---|---|---|---|
| work term ON (HEAD) | **+121** | +72 | −20 | **21 m/s** |
| work term OFF (`T_WORK_CLAMP=0`) | +0.0 | +0.0 | +0.0 | 1.7 |

- **No mass crosses sealed walls** (box N ×1.000 in every variant). **No
  pressure leak**: P relaxes to N·T exactly (P/N → 0.998); the flat solver
  matches multigrid. Residual cross-wall pressure contamination ≈ 0.4%.
- Per-tick energy books (`eth_compression_delta`): step 4c's net
  contribution over 18 s is **−160k cell·atm·deg** — it *destroys* thermal
  energy in the open hall (−20°) while *pumping* it into cavities (+121°).
  Both signs are impossible for a reversible adiabatic term.
- Every "forcing" in #54's table (fire, vents, hot plate, seal event) is only
  a trigger; the term is the instability.

## 1. The defect, stated precisely

Step 4c updates temperature **per cell**: `T_i ← T_i·(1+w_i)` on compression,
`T_i ← T_i/(1+w_i)` on expansion, `w_i = (γ−1)·div(u_new)_i·dt`. Per cell this
is exactly reversible (energy-books arc, proven). But the *books* quantity is
`Σ_i N_i·T_i`, and its change under 4c is

    Δ(books) = −(γ−1)·dt · Σ_i N_i·T_abs,i·div_i

which does **not** telescope: over a sealed region `Σ_i div_i = 0` but
`Σ_i N_i T_i div_i ≠ 0` whenever N or T is non-uniform (after any wave, they
are). The per-cell form conserves nothing at the region level. Kwatra's
flux form does:

    E_i^{n+1} = E_i^* − dt · Σ_faces (p û)_f · n̂_f / dx        (paper eq. 3)

Each face flux is subtracted from one cell and added to its neighbour, so
`Σ_{sealed region} ΔE ≡ 0` — as an integer sum, to the LSB. Wall faces have
no flux. Storms in sealed rooms become structurally impossible.

A second, smaller defect: **kinetic energy is untracked**. Expansion cools
gas into motion; drag/projection remove that motion silently; compression
heats from motion never debited. Magnitude: `T_ke = |u|²/(2·c_v)` ≈ 0.3 K at
20 m/s — tiny, but one-directional. Kwatra's E is *total* energy, so this
closes for free once E is the state.

## 2. The law (the whole change)

### 2.1 State
- **New field `gas_energy`** (int64 per cell, raw units = the books' raw
  units: `N_raw × T_abs_raw`, i.e. Q32 "atm-equivalent × game-deg-absolute").
  Defined on **gas cells** (`!solid && !thermal_solid`); zero and ignored
  elsewhere. It is the conserved TRUTH for gas thermal energy.
  `E_int,i = N_i · T_abs,i` (with the existing convention `c_v ≡ 1` per unit
  N in books units — the constant is absorbed; `T_abs = T_rel + T_AMB_K`).
- `temperature[]` stays the stored, digested, universally-read field. For
  **thermal solids it remains the truth** (unchanged, thermal solver owns
  it). For **gas cells it becomes a derived mirror**: refreshed from
  `gas_energy` at the end of the EOS energy step (§2.4), and at every seam
  write (§2.5). Readers change nothing.
- Total energy is never stored: `E_tot,i = E_int,i + KE_i(u)`, with
  `KE_i = N_i · T_ke,i`, `T_ke = |u_i|²·k_ke`, `k_ke = 1/(2·c_v_gas)` in
  game-deg per (m/s)². `c_v_gas` is the existing EOS `c_v` dial (bindings
  `kick` signature) — no new constant.

### 2.2 The energy step (replaces step 4c, once per tick, post-correction)
On the corrected velocity `u_new` and solved pressure `p^{n+1}`:

    for each open–open face f between cells i (left/up) and j (right/down):
        p_f  = (p_j·N_i + p_i·N_j) / (N_i + N_j)         (paper eq. 15, N as ρ)
        û_f  = face velocity from the momentum update    (paper eq. 13; we
               use the arithmetic face mean of u_new — the same face value
               the divergence stencil already implies)
        flux_f = k_work · p_f · û_f · dt / dx             (int64, ONE evaluation)
        E_tot,i −= flux_f ;  E_tot,j += flux_f            (telescoping)

`k_work` = the units constant turning `p·u·dt/dx` into books units. With
`p = C·N·T_abs` (eos_solver.cpp:199, `C = 1/T_AMB_K`) and `E = N·T_abs`, the
adiabatic identity gives `k_work = (γ−1)/C` — i.e. `flux_f = (γ−1) ·
(N T_abs)_f · û_f · dt/dx`, reusing the folded `gamma_m1_q`, `dt_q`,
`inv_dx`. (Face value `(N T_abs)_f` is p_f/C by construction, so the
solved pressure and the books stay on one scale.)

Faces touching `solid`, `thermal_solid`, `is_vacuum`, or the ambient ring:
**no face** (mirror BC — paper: "reflect p and ρ at walls"). The ambient ring
and vacuum keep their existing pins; energy that leaves through a breach
face into vacuum is booked by the existing wipe counters (§7).

### 2.3 Kinetic exchange (structural drag / clamp heat)
Before the kick: `E_tot,i = gas_energy_i + N_i·T_ke(u*)`. After kick +
staged drag + velocity clamp + the §2.2 fluxes: `gas_energy_i = E_tot,i −
N_i·T_ke(u_new)`. Whatever momentum drag or the clamp removes lands as heat
automatically; whatever the kick converts to motion leaves the thermal
books. `k_drag_heat_frac` and its plumbing are **deleted** (Erik's ruling).

### 2.4 Recovery (the ONE divide)
`T_rel,i = floordiv(gas_energy_i, N_i) − T_AMB_K` on gas cells, using the
existing `floordiv_q` recovery idiom (P-E1). Rails: `T_MIN` floor and
`T_MAX_PHYS` ceiling stay (they now clamp the *mirror*, and clamp the stored
energy consistently: `gas_energy_i = N_i·(T_clamped + T_AMB_K)`, counted by
the existing `energy_floor_hits` / `t_max_phys_hits`). Thin-N cells: no
trust gate — `E/N` for tiny N is the honest temperature of tiny mass; the
existing `n_floor_solver` guards the divide.

### 2.5 The single writer seam
All writes of GAS temperature go through **one primitive**:
`gas_energy_deposit(i, ΔE)` (C++; Python via the FieldEdit "heat"/"temperature"
combine) which updates `gas_energy` and refreshes the mirror. Sites
migrated (inventory from the 29 C++ write sites, 2026-08-29 grep):
- EOS step 4c → replaced by §2.2 (this patch).
- Bulk transport (`bulk_transport.cpp` / `cuda_bulk_transport.cu`, P-E1
  "energy rides the mass flux"): moves `gas_energy` directly with the mass
  flux instead of moving `N·T` and recovering per substep — **removes the
  per-substep recovery divide** (a rounding drip the energy-books arc fights
  with counters today). Bit-identity is NOT expected here; goldens
  re-baseline (§6).
- Combustion (`combustion.cpp` / `cuda_combustion.cu`): fire heat → deposit.
- Thermal solver (`temperature_solver.*` / `cuda_temperature.cu`): conduction
  and radiation INTO gas cells → deposit; solids side unchanged.
- FieldEdit heat/temperature policies (Python) → deposit on gas cells; T-write
  on solids unchanged.
- `seal_tiles` close-T / `destroy_wall` seeding: new gas cell seeded with
  `gas_energy = N·(T + T_AMB_K)` (P-M3 gate 6's "T:=0 moves the books by
  nothing" still holds: `N·T_AMB_K` is the ambient baseline, and the books
  sum is `Σ N·T_rel` = `Σ gas_energy − N·T_AMB_K`).

### 2.6 Books
`eos_energy_books_sum` becomes `Σ_gas (gas_energy_i − N_i·T_AMB_K)` — the
same quantity as today, now read off the stored field with no products.
Counters kept: `eth_transport_delta`, `eth_compression_delta` (now the §2.2
net, expected ≡ 0 over sealed regions and small elsewhere), wipe/floor sums.

## 3. What changes physically (named before measured)
- Sealed rooms can no longer gain or lose energy except through conduction
  with their walls. #54's signature disappears structurally.
- Blast cores heat adiabatically on compression and cool on expansion with
  the books closed — the physical basis Erik wants for non-injected
  explosives.
- Breach rarefaction cools (kept), sized by the flux form rather than the
  clamped first-order form (expect the same order as the T_abs arc's −97°).
- Drag heat is honest and tiny (≈0.3 K per 20 m/s of stopped wind).
- The open-hall cooling (−20° from nothing) is gone.

## 4. Design decisions
- **D1 stored E, derived T mirror** — Erik's ruling; exactness across ticks.
- **D2 solids stay T** — Erik's ruling; two representations are physically
  honest (heat capacity vs flowing gas); the interface is the existing
  conduction exchange, booked as ΔE on the gas side.
- **D3 face value = eq. 15 density-weighted p** (equivalently `(N T_abs)_f`
  N-weighted), not upwind: symmetric ⇒ telescoping; matches the paper.
- **D4 arithmetic face velocity** — the divergence stencil's implied face
  value; avoids a second face-velocity definition.
- **D5 trust gate + work clamp retired** with the term they compensated.
  `T_WORK_CLAMP`, `n_work_ref`, `work_clamp_hits` deleted; `T_MIN`/
  `T_MAX_PHYS` rails stay.
- **D6 int64 everywhere in the energy step**; face flux computed once;
  128-bit intermediates via the existing `mul128_shr` helpers where
  `N·T·u` products exceed 63 bits (bound: N ≤ 200 atm-eq, T_abs ≤ 16290,
  u ≤ U_MAX → raw ≤ ~1e19 — needs the 128-bit stage; pinned in the
  overflow-budget comment like the v2.2 solve).
- **D7 KE advection is NOT conservative** (KE rides u's semi-Lagrangian
  step). ACCEPTED GAP: bounded by `KE/E ≈ |u|²/(2 c_v T)` ≈ 1e-3 at 20 m/s.

## 5. Twin sites + ABI
- `EOSSolver::step` (CPU): step 4c block + `eos_kick_compression_reference`
  twin; `cuda_kick_compression.cu` K2 `compression_kernel` + folds;
  `cuda_eos_step.cu` / `cuda_eos_resident.cu` call sites (P-W1a plumbing of
  `t_amb_q`, `recip_n_work_ref` retired).
- New device/host buffer `gas_energy` (int64, `(N,h,w)`-shaped per the
  RL-batch habits row — N=1 today); residency via `GameMap.enable_residency`.
- `bindings.cpp`: `run_substeps`/`kick` signatures gain `gas_energy`, lose
  `k_drag_heat_frac`, `n_work_ref`, `T_WORK_CLAMP`.
- Python: `GameMap.gas_energy` field; `Recorder.DEFAULT_FIELDS` += gas_energy
  (additive); `tests/field_digest_spec.toml` → **version 5** (gas_energy
  joins the digest; ENTITY/field goldens regenerated in the same commit).
- `config.toml`: delete `k_drag_heat_frac`, `n_work_ref`, `T_WORK_CLAMP`
  rows; `c_v` documented as the KE constant.

## 6. Patch plan
Gate vocabulary: **SB** = sealed-box bench (`_sealedbox_bisect_bench.py`
`nofire`: box/bunker/pen ΔT within ±2 game-deg, u_max < 3 m/s, books
`|Σ| < 1e-6` relative over sealed regions); **HP** = hot-plate bench (rooms
≈ 0, only the plate's neighbourhood warms); **AS** = all-systems scenario
(P3 PASS; P1a/P4/P6 stay PASS); **AB** = `tests/field_ab_harness.py` CPU vs
CUDA lockstep tol 0; **SUITE** = `pytest tests -q`.

| # | Patch | Mode / tier | Gate |
|---|---|---|---|
| P-G0 | `gas_energy` field + digest v5 + recorder + Python plumbing; `eos_energy_books_sum` reads the field; init `= N·T_abs` at map build / seal / destroy. NO physics change yet. | subagent, Sonnet 5 | SUITE green; books identity `Σ(gas_energy − N·T_AMB) == old sum` asserted on 3 levels |
| P-G1 | CPU energy step: §2.2 fluxes + §2.3 KE exchange + §2.4 recovery replacing 4c; trust gate/clamp/drag-heat deleted (CPU path). | subagent, **Opus 4.8** (no oracle — the physics lands here) | **SB, HP** green; AS P3 PASS |
| P-G2 | Writer seam §2.5: transport moves `gas_energy`; combustion, thermal solver, FieldEdit deposits. | subagent, Opus 4.8 | SUITE (property gates), SB/HP unchanged, books counters ≈ 0 drift on a 60 s quiet run |
| P-G3 | CUDA twins (K2 rewrite, bulk transport, combustion, temperature kernels) + resident buffer. | subagent, Sonnet 5 (oracle) | **AB tol 0** on playground + 2 levels, CUDA harness green → auto-merge on green |
| P-G4 | Golden re-baseline with written rationale (`docs/gas_energy_rebaseline_<date>.md`); float ratchet + ingress lint green. | inline, Haiku 4.5 | SUITE green |
| P-G5 | **HUMAN-TEST**: Erik plays (fires, grenades, breach; T overlay). | Erik | feel |

Each patch: own worktree + branch; re-plan at boundaries; checkpoint memory.

## 7. Accepted gaps
- ACCEPTED GAP: KE advection non-conservative (D7).
- ACCEPTED GAP: the ~0.4% cross-wall pressure contamination stays (a
  separate, small solve-accuracy item; re-measure after this arc).
- ACCEPTED GAP: no separate `c_v` per gas species — one gas `c_v` (the
  existing dial); species heat capacities are a later table column.
- ACCEPTED GAP: KE is computed from cell-centred u (not face-staggered);
  consistent with the arithmetic face velocity (D4).

## 8. What Erik must know before playing (P-G5 brief seeds)
- Rooms stay at ambient unless something heats them. Hot corners gone.
- Grenades: the core gets hotter on compression than today, the ring cooler
  on expansion; "cold grenades" from the T_abs close should look different —
  judge the shape, not the numbers (retune is #5/#8).
- Breach: chill still present, now sized by physics.

## Systems (rules lifecycle)
**(a) Existing canonical systems used**: PhysicsRunner/PhysicsEngine (solver
callers; no Python glue), fixed-point kits (`mul128_shr`, `floordiv_q`,
`recip_mul`), Q16 boundary modules (a `gas_energy_fixed.py` would be a new
one — see (b)), FieldEdit (the only Python write path — its heat policy
becomes the deposit), GameMap (field store; `destroy_wall`/`seal_tiles`
seed the field), field digest + GOLDEN_AGGREGATE + A/B harness + CUDA
harness (the gates), Recorder (additive), Config (dials bound in
PhysicsRunner), RL-batch habits (`(N,h,w)`-born buffer), temperature scale
(`T_AMB_K`, `c_v`), the sealed-box / hot-plate / all-systems benches (the
instruments — extended, not forked).

**(b) New systems (draft rules → CLAUDE.md at implementation)**:
- **Gas energy field + deposit seam** (`gamemap.gas_energy`,
  `gas_energy_deposit`): *the only way heat enters or leaves gas is a
  `gas_energy` deposit or a face flux; nothing writes gas `temperature`
  directly — it is a mirror.* Solids keep writing T through the thermal
  solver.
- **Conservative face-flux energy step** (`eos_solver.cpp` energy step +
  CUDA twin): *every energy exchange between gas cells is a per-face flux
  evaluated once and applied with opposite signs — never a per-cell source
  term.*
- `gas_energy_fixed.py`: the Q32-raw boundary module for the field.
