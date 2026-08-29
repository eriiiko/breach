# Gas energy conservation — design v2 (2026-08-29)

Arc for issue **#54**. Replaces EOS step 4c (per-cell temperature-form
compression work) with Kwatra's conservative flux-form energy update on a
**stored gas energy field**, and closes the kinetic-energy channel. Source:
Kwatra, Su, Grétarsson, Fedkiw (2009), *A method for avoiding the acoustic
time-step restriction in compressible flow*, J. Comput. Phys. — archived as
`docs/papers/ADA492343.pdf`. Our EOS already implements the paper's pressure
half (eq. 10/14 Helmholtz solve, eq. 13 face velocity, u-only CFL); this arc
ports its energy half (eq. 3 with eq. 15 face pressure).

**v2 (same day)**: v1 survived none of four critique lenses (numerics/twins,
physics-vs-Kwatra, scope/seam inventory, systems-reuse). Blockers resolved on
paper here: the KE constant (v1 was ~554× off), the operand (`p^{n+1}`, not
`p*`), outflow faces at vacuum/ring, positivity (sub-cycling), the
born-at-ambient rule in absolute currency, the N-writer seam (combustion,
pumps, seal/unseal), the accountable set, the patch split (P-G1 cannot land
without the writers), and the gate metrics.

Rulings by Erik, 2026-08-29 (recorded; not re-litigated below):
- The adiabatic term is physics we KEEP — explosives will become physical,
  not injected; blast/breach heating and cooling must survive.
- **Store energy** for gas cells (not derive-per-tick).
- **Thermal solids stay on T.**
- Drag heat becomes **structural**; `k_drag_heat_frac` goes away (v2 refines
  this: the *constant* is derived, not deleted — §2.3, D5).
- Scope stays tight. Sequence: design → critique → patches with gates →
  CUDA lockstep → re-baseline → HUMAN-TEST.

v2 asks Erik for two confirmations (flagged ★ in §4): the velocity-cap /
wave-sponge KE is **exported and counted, not heated** (D6); the energy step
is **sub-cycled** inside the tick (D8).

---

## 0. Evidence (2026-08-27..29, `tests/_sealedbox_bisect_bench.py`)

| 18 s, no fire, only a glass box sealed at t=0 | box ΔT | bunker ΔT | arena ΔT | u_max |
|---|---|---|---|---|
| work term ON (HEAD) | **+121** | +72 | −20 | **21 m/s** |
| work term OFF (`T_WORK_CLAMP=0`) | +0.0 | +0.0 | +0.0 | 1.7 |

No mass crosses sealed walls (box N ×1.000); no pressure leak (P → N·T
exactly; flat solver = MG); ~0.4% cross-wall pressure contamination only.
Step 4c's net over 18 s: **−160k cell·atm·deg** — it destroys energy in the
hall and pumps it into cavities. Every "forcing" in #54's table is a trigger.

## 1. The defect

Step 4c: `T_i ← T_i(1+w_i)` / `T_i/(1+w_i)`, `w_i = (γ−1)·div_i·dt`. Reversible
per cell; but the books quantity is `Σ N_i T_i` and its change,
`−(γ−1)dt·Σ N_i T_abs,i div_i`, does not telescope (`Σ div_i = 0` over a
sealed region, `Σ N_i T_i div_i ≠ 0` once N or T is non-uniform). Kwatra's
form `E_i ← E_i − dt·Σ_faces (p û)_f/dx` applies each face flux with opposite
signs to the two cells, so `Σ_region ΔE ≡ 0` over any region whose faces
are all interior or wall — as an integer sum, to the LSB.

Second defect: **kinetic energy is untracked** (expansion → motion; drag /
projection remove motion silently; compression heats from motion never
debited). Correctly sized (§2.1) it is small at wind speeds and dominant at
blast speeds — which is exactly why it must be a *derived* constant, not a
dial (the energy-books arc's `k_drag_heat_frac = 1.0` detonation, close doc
§4, was this term at the wrong constant).

Splitting check (physics lens): Kwatra eq. 1 splits the energy flux as
advective `E·u` (ours: P-E1 energy rides the mass flux) plus non-advective
`p·u` (this arc). Sum = enthalpy flux. **No double count.**

## 2. The law

### 2.1 Units and the two derived constants
The engine's pressure is `p_code = C·N·T_abs`, `C = 1/T_AMB_K`
(eos_solver.cpp:199); the kick uses `K = c_max²/γ`, i.e. `p_phys = K·p_code`.
Consistency with the ideal gas `p = (γ−1)·ρ·e` fixes, in the books' own
units (energy ≡ `N·T`, "atm-equivalent × game-deg"):

    R_books     = K·C = c_max²/(γ·T_AMB_K)
    c_v_phys    = R_books/(γ−1) = c_max²/(γ(γ−1)·T_AMB_K)      ≈ 554  (c_max=300, γ=1.4, T_AMB=290)
    k_work      = (γ−1)/C = (γ−1)·T_AMB_K                       (flux constant, independent of K)
    k_ke        = 1/(2·c_v_phys) = γ(γ−1)·T_AMB_K/(2·c_max²)    ≈ 9.0e-4 game-deg per (m/s)²

So `T_ke(20 m/s) ≈ 0.36 game-deg`, `T_ke(300 m/s) ≈ 81`. The shipped
`[physics.thermal] c_v = 1.0` is a *convention* dial for the radiation
deposit, NOT this constant (config.toml:154, :660 — "c_v=1 by convention …
~700x below physical"). `k_drag_heat_frac = 0.0014` was the hand-rolled
stand-in for `1/c_v_phys ≈ 0.0018`; v2 replaces it with the derived `k_ke`
(D5). Both constants fold host-side once per tick from `c_max`, `γ`,
`T_AMB_K` (the P-W1a fold site), with a build-time consistency assert.

### 2.2 State
- **`gas_energy`** — int64 per cell, **the exact unshifted product
  `N_raw × T_abs_raw`** (Q32 raw; no `>>16`). Defined on the **accountable
  set** = the one canonical skip-set complement `!(solid || thermal_solid ||
  is_vacuum || is_ambient)` (`eos_energy_books_sum`, `e_participates()` in
  bulk_transport.cpp:386 — reused verbatim, not re-derived). Zero elsewhere
  and never read there; the two existing wipes (bulk_transport.cpp:392,
  temperature_solver.cpp:135) zero it and book `−(E − N·T_AMB)` to
  `e_vac_wipe_sum` / `e_ring_pin_sum` (existing signed counters).
- Shape: host `(h, w)` like every field; device `(N, h, w)`, N=1 (RL-batch
  habits row). Digest hashes the host shape → **digest spec v5**.
- `temperature[]` stays stored, digested, universally read. **Truth for
  thermal solids** (thermal solver unchanged). **Mirror for gas cells**,
  refreshed by the recovery (§2.6) at every seam write (§2.7). No reader
  changes; `p*` is built **from `gas_energy` directly** (`p* = C·E`, one
  narrow) so pressure never carries the floored-mirror bias.
- Overflow budget of the field: the enforced map invariant is `N_raw <
  2^30` (eos_solver.cpp:948) and `T_abs_raw ≤ (T_MAX_PHYS + T_AMB)·2^16 ≈
  2^30` ⇒ `E ≤ 2^60`. **Absolute sums over > 8 cells may overflow int64 —
  forbidden**; all books are relative (`Σ (E − N·T_AMB_raw)`), computed as
  the existing books do (int64 per-cell difference, then sum).

### 2.3 Tick placement and the kinetic brackets
The kick loop (eos_solver.cpp ~807-990; K1 on device) applies per cell, in
order: ∇p kick → `dyn_wave_absorb` → B3c `sponge_udamp` → cap clamp
(`cap2_plane`/`U_MAX`) → staged drag L/Q. Every stage changes |u| and each
is ruled **individually** (D6), inside the loop, per cell, on the cell's own
`N_i` — no `u*` snapshot buffer:

| stage | ΔKE = N_i·k_ke·(|u_after|² − |u_before|²) goes to | counter |
|---|---|---|
| ∇p kick | `gas_energy_i −= ΔKE` (reversible exchange, eq. 2/3) | `e_kick_ke_sum` |
| `dyn_wave_absorb` | **export** (numerical damper) — `gas_energy` untouched | `e_absorb_export_sum` |
| B3c sponge band | **export** (models energy leaving to infinity) | `e_sponge_export_sum` |
| cap clamp | **destroyed, counted** (rail; pre-clamp |u| unbounded) | `e_clamp_destroyed_sum` |
| drag L + Q | `gas_energy_i += −ΔKE` (structural drag heat, D5) | `e_drag_heat_sum` (replaces `e_drag_deposit`) |

`T_ke` per cell uses `mul_q16(|u|²-terms, k_ke_q)` with `|u|² ≤ U_MAX²`
(cap applied first for the drag stages; the kick stage bound is
`RAD_SAFE`, so its product runs through `mul128_shr`). This retires the
P-E3 deposit formula (`recip_cv`, `heat_frac_q`, `e_drag_drop_sum`,
`e_drag_rail_clipped`) — see §5.

### 2.4 The face-flux energy step (replaces 4c; a separate pass after the kick loop)
Operands: the **absolute solved pressure `p^{n+1}`** (on an ambient map the
solve runs shifted, `P' = P − p_amb`, restored at step 5 — the energy step
runs **after step 5**, or adds `p_amb` itself; pinned: after step 5) and the
corrected velocity `u_new`. Negative `p^{n+1}` (measured `P_min = −0.98` in
the energy-books close): `p_f` is floored at 0 with a hit counter
`p_face_floor_hits` (a negative absolute pressure is unphysical; flooring
never breaks telescoping since both cells see the same `p_f`).

For each face f between cells i and j, in **canonical orientation** (i =
lower linear index; east and south faces owned by i), evaluated **once**
from identical inputs on both sides (gather form, no atomics — §2.5):

    if both i, j in the accountable set:                       (interior face)
        p_f   = floordiv( p_j·N_i + p_i·N_j , N_i + N_j )      (eq. 15; skip face if N_i+N_j == 0;
                                                                p ≤ 2^30, N < 2^30 ⇒ products ≤ 2^60, int64)
        u_f   = u_i + u_j            (the ½ is folded into k_flux — no separate >>1 bias)
    elif exactly one side (i) is accountable and the other is is_vacuum / is_ambient:   (OUTFLOW face)
        p_f   = p_i ;  u_f = u_i     (ring/vacuum u ≡ 0, so (u_i+0)/2·2 = u_i — the same value the
                                      divergence stencil reads there, since mirror_idx keys on solid only)
        flux booked to i only, and to `e_work_export_sum` (breach rarefaction, §3)
    elif the other side is solid:                              (WALL: reflected u ⇒ û_f = 0)
        no face
    elif the other side is thermal_solid (furniture — permeable, carries N/u/p):
        p_f, u_f as interior using the ts cell's SOLVED p^{n+1} and u; flux applied to the
        gas side only and booked to `e_ts_work_sum` (rule (d)'s precedent: energy crossing a
        ts face is shed and counted). Conservation claim is scoped to furniture-free regions.

    mag   = mul128_shr( mul128_shr( p_f , |u_f| , 16 ) , k_flux_q , 16 )   (int64, two stages)
    flux  = sign(u_f)·mag                    (sign applied AFTER truncation — exact cancellation)
    E_i −= flux ;  E_j += flux               (interior) 

`k_flux_q = quantize( k_work · dt_s / (2·dx) )` folded once per tick
(includes the ½ of the face mean and the sub-cycle `dt_s`, D8).

**Positivity / sub-cycling (D8):** the tick's material Courant number is not
bounded by 1 (`n_sub ≤ N_SUB_MAX = 8` rails during venting; `|div|·dt` up to
~8, i.e. 6× past the retired `T_WORK_CLAMP` rail). The energy step therefore
runs **`n_sub` sub-cycles** with `dt_s = dt/n_sub`, holding `p^{n+1}` and
`u_new` fixed (telescoping exact per sub-cycle; per-sub-cycle outflow
`≤ (γ−1)·E_i·|div|·dt_s` with `|div|·dt_s ≤ ~1` ⇒ positive). Last-resort
rail: if `E_i` would drop below `N_i·(T_MIN + T_AMB)_raw`, the cell's
outgoing fluxes for that sub-cycle are all scaled by one factor `s_i ∈
[0,1)` **and the same factor is applied to each neighbour's credit** (both
sides recompute `s_i` from identical inputs), counted in `e_energy_floor_sum`
(signed energy, not a hit count).

### 2.5 Parity-safe kernel shape
Per-cell **gather**: every cell recomputes the flux of each of its four
faces in canonical orientation from the same inputs, truncates the
magnitude once, applies the sign, and sums — identical to the S1 water-flux
idiom (fixed_point.h:22-25). No face buffer, no atomics; the CPU twin is
the same loop. `mul128_shr` is promoted into `fixed_point.h` as ONE
`FP_HD` primitive (today three copies: eos_solver.cpp:46, `mul128_shr_signed`
in cuda_fixedpoint_device.cuh, `mul128_shr_host` in
cuda_kick_compression.cu:77) — P-G0, bit-identical.

### 2.6 Recovery (mirror refresh)
`T_rel,i = floordiv(gas_energy_i, N_i) − T_AMB_K_raw` on accountable cells;
divide policy reused from bulk_transport.cpp verbatim (`N_EPS_RAW = 1`:
below it, wipe to ambient and book `e_wipe_sum`; never divide by 0 — the
same on both backends). Rails `T_MIN` / `T_MAX_PHYS` clamp the mirror **and
write back `gas_energy = N·(T_clamped + T_AMB)` ONLY when a rail binds**,
booking the delta to `e_rail_sum` (signed energy). **Never** write back
otherwise (`N·floordiv(E,N) ≤ E` would drain up to N−1 raw per cell per
tick — the drip class this arc kills). The CFL max-T scan
(eos_solver.cpp:434-440) gains the same `N_EPS_RAW` guard (thin-N cells
cannot set `c_LOCAL`/`n_sub` for the grid).

### 2.7 The writer seam — energy AND mass
Under stored E every writer of **N** is a writer of T. One primitive,
`gas_energy_move(i, ΔN, T_abs_src)` / `gas_energy_deposit(i, ΔE)` (C++, with
the Python-visible twin on GameMap), applies: mass leaving cell i carries
`ΔN·T_abs,i` out; mass arriving carries the donor's `T_abs`; mass **born**
(seed / donor split / ambient / ring / vacuum / ts-face inflow) is born
**at ambient = credited `ΔN·T_AMB_K_raw` absolute** (the P-E1 "born carrying
zero *relative* energy" rule restated in absolute currency — v1 would have
born it at 0 K). Every site, with its rule:

| site | today | v2 |
|---|---|---|
| bulk transport donor-cell (bulk_transport.cpp:265 + CUDA) | moves `N·T_rel` as int64 `e[]`, per-substep recovery divide | moves `gas_energy` with the face `dq` at the donor's price (single evaluation per face — debit/credit telescope); non-participating donors credit `dq·T_AMB`; the per-substep recovery is deleted (floor error `< N` raw/cell → `< 1` raw/face); counters `e_ts_residual`, `e_wipe_sum`, `e_floor_sum` re-derived in absolute convention; identity `test_no_transport_mint` / `cuda_bulk_flux_check` PART 3 re-stated (§6 table) |
| combustion O2 consumed (combustion.cpp:600) / products to flame cell (:778) | N moves, T untouched | consumed O2 leaves with `burn·T_abs(donor)`; products arrive carrying that energy at the flame cell (mass and its energy travel together; global N conservation ⇒ global E conservation); fire heat `heat_saturating_add` → `gas_energy_deposit` on gas, unchanged on solids |
| thermal solver: heat→T deposit (:323), conduction swap (:414), radiation (:222) | writes T | gas side: `de` (already int64) → `gas_energy_deposit`, endpoint divide deleted; solids side unchanged; wipes (:135) zero E and book |
| pump primitives `inject_gas_n(_vec)` / `extract_gas_n(_vec)` (gamemap.py:2410-2540) | vec inject mixes T; extract leaves T | extract removes `ΔN·T_abs(cell)`; inject credits `ΔN·T_abs(t_dep)`; the vent plenum ledger (`vent_system.py`, relative currency, ENTITY_SECT-digested) is **kept relative** and converted at this seam (`E = e + n·T_AMB`); the primitive's direct `temperature[...] =` write (gamemap.py:2540) is replaced by the seam's mirror refresh |
| `seal_tiles` receivers (:2192), `unseal_tiles` (:2213, writes no T today), `destroy_wall` seed (:1909/:1944) | N moves / seeds; T := 0 or untouched | receivers credited at the donor's `T_abs`; a new gas cell (unseal / destroy) is born at ambient `N·T_AMB`; a gas cell becoming solid/ts retires its `gas_energy` to `e_retire_sum` (counted). Seam = the callers of `on_tile_changed` (gamemap.py:1187) — one row per caller |
| FieldEdit `atmosphere` policy (O2/N2 deposits, field_edit.py:536), `temperature`/`wave_source` (:597), `heat` (:589) | direct writes | `atmosphere` deposits credit at ambient; `temperature` combine on gas cells → deposit; a `FIELD_POLICY` row for `gas_energy` with a **dual-write combine** (store + mirror) |
| tests/tools seeding `gmap.temperature[...] = ` on gas cells (**154 sites / 73 files**, incl. `_hotplate_heating_bench.py:73`) | direct | `GameMap.seed_gas_temperature(sel, T_q)` writes both planes; harnesses migrated in P-G0 (the HP gate would otherwise pass vacuously) |
| `sky_exchange.cpp:46` (O2↔N2 swap) | conserves n_tot per tile | nothing |

### 2.8 Books
`eos_energy_books_sum = Σ_accountable (gas_energy_i − N_i·T_AMB_raw)` — same
quantity as today, read off the field. `eth_compression_delta` is retired
(structurally 0 by telescoping — it can no longer detect anything); the
detectors are the new signed energy counters: `e_kick_ke_sum`,
`e_drag_heat_sum`, `e_absorb_export_sum`, `e_sponge_export_sum`,
`e_clamp_destroyed_sum`, `e_work_export_sum`, `e_ts_work_sum`,
`e_energy_floor_sum`, `e_rail_sum`, `e_retire_sum`, plus the existing wipe /
ring / ts counters in absolute convention. Closure identity (the new
`test_no_transport_mint`): `Δbooks == Σ(deposits) − Σ(exports) − Σ(destroyed)
± rails`, exact in int64.

## 3. What changes physically (named before measured)
- Sealed, furniture-free rooms cannot gain or lose energy except through
  conduction with their walls (exact). #54 disappears structurally.
- Blast cores: **cooler than HEAD**, not hotter — HEAD's 4c pumped energy
  into cavities; the flux form bounds a cell's heating by its neighbours'
  energy. (Corrects v1 §8.) Fire tuning (#5/#8) will see lower core T.
- Breach rarefaction: the mouth's net cooling comes from the **outflow
  face export + the KE debit** (jet enthalpy → kinetic), not from a per-cell
  expansion factor; expect the same order as the T_abs arc's −97 but a
  different profile (a ring, not a core — the trust gate no longer shapes
  it). Correctly-signed only because vacuum/ring faces are OUTFLOW faces
  (v1 treated them as walls and would have *heated* the mouth).
- Drag heat honest and small (0.36 game-deg per 20 m/s stopped); the P-E5
  detonation (`k_drag_heat_frac = 1.0`) cannot recur because the constant is
  derived (`k_ke`), not dialled.
- Open-hall cooling (−20° from nothing) gone.

## 4. Design decisions
- **D1 stored E (int64), derived T mirror** — Erik's ruling.
- **D2 solids stay T** — Erik's ruling; the gas↔solid interface is the
  conduction exchange, booked as ΔE on the gas side; ts faces shed and count
  pressure work (§2.4).
- **D3 face pressure = eq. 15 literally**: `p_f = floordiv(p_j N_i + p_i
  N_j, N_i+N_j)` — harmonic-flavoured N × arithmetic T_abs. NOT
  "N-weighted (N T)_f" (v1's parenthetical was wrong).
- **D4 wall faces û = 0 is authoritative.** The solve's `div(u*)` keeps its
  mirror stencil (`mirror_idx` ⇒ implied `û_wall = u_i`), so the energy step
  and the solve disagree at wall-adjacent cells by `u_i·p_i`. Conservation
  is unaffected (telescoping); the sealed-box guarantee holds. ACCEPTED GAP
  with a probe: SB reports `Σ_wall-adjacent |u_i·p_i|`; moving the solve's
  divergence to face form (which likely also removes the 0.4% cross-wall
  contamination) is the queued follow-up, not this arc.
- **D5 `k_drag_heat_frac` → derived `k_ke`** (§2.1). Not a deletion of the
  physics, a replacement of a hand dial by the unit-bridge constant.
- **★ D6 velocity cap, `dyn_wave_absorb`, B3c sponge: exported/destroyed
  and counted, never heated.** Deviates from the literal wording of Erik's
  "drag/clamp heat structural" ruling: the cap is a numerical rail whose
  pre-clamp |u| is bounded only by `RAD_SAFE` (16384 m/s) — heating from it
  is #54 through a new door; the absorb/sponge stages exist to remove energy
  (CLAUDE.md scopes them out of the drag rule). Drag L/Q *is* heated. Erik
  to confirm.
- **D7 KE advection non-conservative** (rides u's SL step) — ACCEPTED GAP,
  now correctly bounded: `KE/E = k_ke·|u|²/T_abs ≈ 1.2e-3` at 20 m/s,
  ≈ 0.28 at 300 m/s (blast cores; short-lived).
- **★ D8 energy step sub-cycled `n_sub` times** with `p^{n+1}`, `u_new`
  held (§2.4). Erik to confirm (cost: n_sub × one gather pass; n_sub = 1 in
  calm play).
- **D9 int64 + two-stage `mul128_shr`** for every `p·u·k` product; face
  divide by `floordiv` on non-negative operands; operation order pinned in
  the code comment like the v2.2 solve's budget.
- **D10 counters_out[9] layout kept**; slot 2 (`work_clamp_hits`) retired
  and always 0 — no renumbering of the positional unpacks; new counters are
  new `def_readonly` members.
- **D11 trust gate + work clamp retired** (`T_WORK_CLAMP`, `n_work_ref`,
  `work_fade_clamp01_q`, `recip_n_work_ref`) — their job is done by
  telescoping (no temperature unbacked by energy) and by §2.4's positivity
  rail. `renderer/cold_overlay.py` keeps `COLD_N_MIN_FRAC = 0.25` as its own
  constant (HUMAN-TEST ruling 2026-08-21 stands).
- **D12 no `gas_energy_fixed.py`.** The field is a Q32 product, not a Q16
  quantity; the `(N, T) ↔ E` conversion lives beside its single C++
  transcription (books binding) and as `FP_ONE_SQ` helpers in `gas_fixed.py`.
- **D13 Recorder**: int64 fields get their own branch (float64 ring, scale
  2^32) — the frozen npz contract is extended additively by *dtype class*,
  not just membership. `analyze_blowup_dump.py` gas_energy column: later.

## 5. Twin sites + ABI (deletion blast radius verified by the critique)
- C++/CUDA: `EOSSolver::step` 4c block + `eos_kick_compression_reference`
  twin; `cuda_kick_compression.cu` K1 (per-stage KE brackets) + K2 (rewritten
  as the gather flux pass) + `kick_scalar_folds`; `cuda_eos_step.cu` /
  `cuda_eos_resident.cu` call sites; `bulk_transport.cpp` + `cuda_bulk_
  transport.cu` (gas_energy transport, absolute born-at-ambient);
  `combustion.cpp` + `.cu` (O2/product energy, deposit); `temperature_
  solver.cpp` + `cuda_temperature.cu` (gas-side deposit, wipes);
  `physics_engine.cpp` (buffer plumbing). Promote `mul128_shr` to
  `fixed_point.h` (P-G0).
- Deleted symbols and their consumers (full list in the critique; the
  implementer greps before deleting): `T_WORK_CLAMP`/`t_work_clamp`/
  `work_clamp_q` (~12 test files incl. `test_p_e4_reversible_work.py` ×21,
  5 tools), `n_work_ref` (+ `physics_runner.py:474` stale-key guard),
  `k_drag_heat_frac`/`heat_frac_q`/`recip_cv` (+ `physics_runner.py:483`),
  `e_drag_deposit`/`e_drag_drop_sum`/`e_drag_rail_clipped` (→ `e_drag_heat_
  sum`). pybind `run_substeps` / `kick` signatures lose the three floats
  (~29 Python callers) and gain `gas_energy`.
- Python: `GameMap.gas_energy` (h,w) int64; `seed_gas_temperature`;
  `gas_energy_move/deposit`; FieldEdit `gas_energy` policy row; residency
  upload/D2H lists (physics_runner.py:1164/:1216 — combustion + tail run on
  the host mirror, so the field round-trips every tick: accepted for this
  arc, resident combustion is out of scope); Recorder int64 branch;
  `tests/field_ab_harness.py:83 SIM_FIELDS` += gas_energy; digest spec v5
  (`field_digest.py:64`, `field_digest_spec.toml`) — **bumped in P-G0**.
- Config: delete `k_drag_heat_frac`, `n_work_ref` rows (there is no
  `T_WORK_CLAMP` key — only a comment at :628); document `c_v` as the
  radiation convention dial and `k_ke` as derived.

## 6. Patch plan
Gate vocabulary — **SB**: sealed-box bench `nofire`: (i) `Σ_box gas_energy`
constant to the LSB across the whole run except by counted conduction
deposits (`ΔΣE_box == Σ deposits_box`, exact); (ii) `ΔT_box = Δ(ΣE/ΣN)`
within ±2 game-deg (N-weighted — an unweighted mirror mean is not
conserved by mixing); (iii) u_max < 3 m/s; (iv) the D4 wall probe reported.
**HP**: hot-plate via `seed_gas_temperature`, rooms ≈ 0. **VENT**: a new
venting bench (breach of a 1-atm room to vacuum, 5 s): `gas_energy ≥ 0`
everywhere, `n_sub` stable, mouth cools, `e_work_export_sum` matches the
room's energy loss. **BLAST**: `frag_standard` mid-arena, 3 s: no
`T_MAX_PHYS` hit outside the blast disc, core T *below* HEAD's. **AS**:
all-systems scenario P3 PASS, P1a/P4/P6 PASS. **AB**: `field_ab_harness.py`
CPU vs CUDA tol 0. **SUITE**: `pytest tests -q` against the §6 red table.

| # | Patch | Mode / tier | Gate |
|---|---|---|---|
| P-G0 | Field + plumbing, NO physics: `gas_energy` (init `N·T_abs`), digest v5 + regenerate goldens (spec bump event 1), recorder int64 branch, `seed_gas_temperature` + migrate the 154 harness/test seeds, `mul128_shr` promotion, AB `SIM_FIELDS`. Books identity `Σ(E − N·T_AMB) == old sum` on 3 levels. | subagent, Sonnet 5 | SUITE green (values unchanged; only the spec-bump regeneration) |
| P-G1a | CPU EOS-internal: §2.3 brackets in the kick loop, §2.4 flux pass (sub-cycled), §2.6 recovery, transport moves `gas_energy` (§2.7 row 1), trust gate/clamp/heat-frac retired, CFL scan guard. **Transitional state, stated**: at EOS entry `gas_energy := N·T_abs` from the mirror (T is still the cross-tick truth); at exit the mirror is refreshed. D1 not yet live across ticks. | subagent, **Opus 4.8** | SB (i) within the EOS step, HP, **VENT**, **BLAST**, AS P3 |
| P-G1b | Writers: combustion, thermal solver gas side, pump primitives + plenum conversion, seal/unseal/destroy seams, FieldEdit rows; remove the entry re-sync → **D1 live**. Closure identity test rewritten. | subagent, Opus 4.8 | SB (i) across ticks (exact), SUITE per red table, 60 s quiet-run books drift == counted |
| P-G2 | CUDA twins: K1 brackets, K2 gather flux pass, bulk transport, combustion, temperature kernels; resident buffer + upload/D2H lists. | subagent, Sonnet 5 (oracle) | **AB tol 0** on playground + 2 levels; CUDA harness green → auto-merge on green |
| P-G3 | Golden re-baseline (value-move event 2) with `docs/gas_energy_rebaseline_<date>.md`; float ratchet + ingress lint; docs outside archive amended (list in critique: engine/04, drag_law_v2, storm_audit, …). | inline, Haiku 4.5 | SUITE green |
| P-G4 | **HUMAN-TEST**: Erik plays (fires, grenades, breach; T overlay). Brief: §3. | Erik | feel |

**Red-classification table (agreed before P-G1a):**
STOP (must stay green): `test_destroy_wall_conserves_mass.py` gate 6 (restated
in absolute currency), `test_air_boundary.py:820` (`t_max_phys_hits == 0`),
`test_velocity_clamp_property.py`, all `cuda_*_check` A/B parity checks.
EXPECTED red → regenerate in P-G3: every field/aggregate golden, the 11
CUDA check goldens, inline goldens (`test_b6_logic_golden`, `test_vent_*`,
`test_w6_armory`). RETIRED (delete with rationale in the commit):
`test_p_e4_reversible_work.py`, `test_p_e4_trust_gate.py`, `test_e1_cold_
rail.py`, `test_e1_hot_rail.py` (their subject is gone; their *property* —
no temperature unbacked by energy — is the new closure identity).
REWRITTEN: `test_p_e3_drag.py` (drag identity on `e_drag_heat_sum`),
`test_no_transport_mint` + `cuda_bulk_flux_check` PART 3 (absolute closure).
GRAY: `test_thermal_mass_axis.py:640/690`, `test_eos_p4_combustion.py:53`
(keyword args only — mechanical).

Each patch: own worktree + branch; re-plan at boundaries; checkpoint memory.

## 7. Accepted gaps (decisions with bounds)
- D4 wall-stencil mismatch (bound: reported probe; follow-up queued).
- D7 KE advection non-conservative (≈1.2e-3 at 20 m/s; ≈0.28 at 300 m/s,
  transient).
- ts (furniture) faces shed pressure work — counted in `e_ts_work_sum`;
  conservation claim scoped to furniture-free regions.
- 0.4% cross-wall pressure contamination stays; it can only *redistribute*
  energy inside a sealed region (telescoping), never pump it. Residual
  channels that remain open are the rails and the KE brackets — all counted.
- One `c_v_phys` for all gases (no per-species heat capacity).
- `gas_energy` round-trips host↔device every tick (combustion + tail on the
  host mirror); resident combustion is a later arc.
- `analyze_blowup_dump.py` gains no gas_energy column in this arc.

## 8. What Erik must know before playing (P-G4 brief)
- Rooms stay at ambient unless something heats them. Hot corners gone.
- Grenade cores are **cooler** than before (HEAD was pumping); the ring
  cooler on expansion. Judge the shape; retune is #5/#8.
- Breach: chill present, as a ring around the mouth rather than a core.

## Systems (rules lifecycle)
**(a) Existing canonical systems used**: PhysicsRunner/PhysicsEngine;
fixed-point kits (`floordiv_q`, `recip_mul`, and `mul128_shr` promoted into
the kit); **Bulk/trace transport** (survey A20 — `bulk_flux_energy_
transport_cached` is the face-gather shape the new pass copies, and P-G1a
rewrites its energy half); Q16 boundary modules (`gas_fixed.py` extended,
D12); FieldEdit (policy row with dual-write combine); GameMap +
`on_tile_changed` callers (seed/retire seam); **Gas pump primitives**
(`inject_gas_n`/`extract_gas_n` become energy-aware — the rule is amended,
not bypassed); Vent/duct plenum ledger (kept relative, converted at the
seam); field digest / GOLDEN_AGGREGATE / A/B harness / CUDA harness
(gates); Recorder (dtype-class extension); Config; RL-batch habits
(`(N,h,w)` device buffer); temperature scale (`T_AMB_K`, `c_max`, `γ`
define the two constants); the three benches (extended, not forked).

**(b) New systems / amended rules (→ CLAUDE.md at implementation)**:
- **Gas energy field** (`gamemap.gas_energy`): *the conserved truth for gas
  thermal energy; `temperature` is its mirror on gas cells and the truth on
  thermal solids.*
- **Gas energy seam** (`gas_energy_move` / `gas_energy_deposit`): *every
  change of gas N or gas heat goes through the seam — mass carries its
  source's `T_abs`, born mass is born at ambient; nothing writes gas
  `temperature` directly (tests included: use `seed_gas_temperature`).*
- **Face-flux energy step**: *energy exchange between gas cells is a per-face
  flux evaluated once in canonical orientation and applied with opposite
  signs — never a per-cell source term; boundary faces export to a counter.*
- **Amend** Temperature solver rule (survey A22): "derived here alone" →
  "solids' T derived here alone; gas T is the energy field's mirror".
- **Amend** Gas pump primitives rule: "+ energy-aware: the primitives carry
  `ΔN·T_abs` through the gas energy seam".
- **Amend** Interior drag rule: "drag L/Q stages deposit their removed KE as
  heat via `k_ke`; absorb/sponge/cap stages export or destroy it, counted".
