# Gas energy conservation — design v4 (2026-08-29)

**v4**: round-3 critique of the v3 deltas. Fixed: the pinned ΔKE and flux
shift counts (v3's were off by 2^16 — the KE debit would have been 65 536×;
R3-#3/#4), per-face flux saturation for the int64 corner (R3-#4), RAD_SAFE
guard placement unconditional + load-side clamp (R3-#5), one pressure
definition across sub-cycles (increment form, R3-#6), the rail's `head<<16`
overflow (R3-#7), combustion's two-hop energy ledger + soot shed counter +
deposit-site rail (R3-#8/#9), water-tail signature + export (R3-#10), the
D-3 guard stays where it is (R3-#2), P-G0 digest-gate procedure (R3-#11),
and a **FIRE** gate. Verdict after v4: buildable for P-G0 and P-G1a.

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

v2 asked Erik for two confirmations (flagged ★ in §4): the velocity-cap /
wave-sponge KE is **exported and counted, not heated** (D6); the energy step
is **sub-cycled** inside the tick (D8). **Both CONFIRMED by Erik,
2026-08-29 ("option A" on each).** P-G0 merged `dc35ede`.

**v3 (same day)**: round-2 critique (physics+numerics; seam/gates). All
round-1 blockers confirmed resolved (the §2.1 constants were re-derived
independently and match). v3 fixes: RAD_SAFE guard moved above the KE
brackets + ΔKE operation order (F1); sub-cycles refresh `p_i` from the
running `E_i` (F2); donor-only positivity rail with a two-pass kernel (F3,
F13); **thermal-solid faces are walls** and ts cells export their brackets
(F4, F5); transport prices faces off live `E_i` (F6); recovery + rails once
per tick, pinned (F7); SB gate = closure identity on the box (F8);
always-compiled throw for the constants + the S_EOS guard carried forward
(F9, seam 7); `mul128_shr` promoted as a three-branch primitive (F10);
combustion soot + object-site rows (F11, seam 2); KE-not-a-book stated
(F12); water-displacement evacuation row (seam 1); `unseal_tiles` is a
withdrawal, not a mint (seam 3); MECHANICAL red class + corrected test
list, benches migrate in P-G0 (seam 4-6); GPU flux pass is a new kernel
after `K_store_atm` (seam 8); CUDA parity suspension named (seam 9);
Recorder true-int64 branch (seam 10).

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
`T_AMB_K` (the P-W1a fold site). `C` is **derived from `T_AMB_K`** at the
fold (the two are separate `def_readwrite` members today and can drift);
the check is an **always-compiled `throw`** beside the D-3 guard
(eos_solver.cpp:384-397 explains why `assert` is dead in Release) — the
D-3 guard itself (`S_EOS == 1 && T_MIN > −T_AMB_K`) **stays at :393** (a
pre-flight check before any state mutation — moving it after step 5 would
turn it into a post-mortem on a half-mutated map, R3-#2); the recovery
`T_rel = E/N − T_AMB` assumes a slope-free `T_abs`, which that guard already
enforces. Deriving `C` at the fold is digest-neutral at shipped dials
(`quantize(1/290)` = 226 either way) and `C`'s only consumers are the
`p*` build sites (eos_solver.cpp:740 + CUDA twins). `k_ke` folds as a Q.32 `make_recip`
reciprocal (a Q16 constant would be 59 counts — 0.22% bias, ~6 bits).

### 2.2 State
- **`gas_energy`** — int64 per cell, **the exact unshifted product
  `N_raw × T_abs_raw`** (Q32 raw; no `>>16`). Defined on the **accountable
  set** = the one canonical skip-set complement `!(solid || thermal_solid ||
  is_vacuum || is_ambient)` (`eos_energy_books_sum`, `e_participates()` in
  bulk_transport.cpp:386 — reused verbatim, not re-derived). Zero elsewhere
  and never read there; the two existing vacuum/ring temperature wipes
  (bulk_transport.cpp:392, temperature_solver.cpp:135) zero it and book
  `−(E − N·T_AMB)` to `e_vac_wipe_sum` / `e_ring_pin_sum` (existing signed
  counters). **Thermal-solid cells carry N and u but no `gas_energy`**:
  every energy they would exchange is exported to counters (§2.3 F5, §2.4
  walls, §2.7).
- Shape: host `(h, w)` like every field; device `(N, h, w)`, N=1 (RL-batch
  habits row). Digest hashes the host shape → **digest spec v5**.
- `temperature[]` stays stored, digested, universally read. **Truth for
  thermal solids** (thermal solver unchanged). **Mirror for gas cells**,
  refreshed by the recovery (§2.6) at every seam write (§2.7). No reader
  changes; `p*` is built **from `gas_energy` directly** (`p* = C·E`, one
  narrow) so pressure never carries the floored-mirror bias.
- Overflow budget of the field: the enforced map invariant is `N_raw <
  2^30` (eos_solver.cpp:948) and `T_abs_raw ≤ (T_MAX_PHYS + T_AMB)·2^16 ≈
  2^30` ⇒ `E ≤ 2^60` — which holds only because the recovery's
  `T_MAX_PHYS` rail runs **once per tick on the whole accountable set**
  (§2.6); pressure is int32 (`atmosphere`/`pstar_`), bound 2^31. **Absolute sums over > 8 cells may overflow int64 —
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

**Bounds and order (F1).** Today the `RAD_SAFE` component guard sits at
eos_solver.cpp:864-868, *after* absorb and sponge, and the pre-guard
magnitude is measured at ~2^53 raw (comment :850-863) — squaring it is
2^106. So: (a) the loaded `ux, uy` are **clamped to ±2^27 at load**
(structural, not inductive — FieldEdit / level load / first tick after a
change can supply an unguarded wind; R3-#5b), so `|u_before|²` is bounded
too; (b) the guard **moves to immediately after the ∇p block —
unconditionally, outside the `if (gx != 0 || gy != 0)` (R3-#5a)** — and
tightens to **2^27 raw per component (≈2000 m/s, 2× U_MAX)**; the clipped
KE is folded into the kick bracket (the gas keeps the energy the clip
removed — the no-mint direction) with its own counter `rad_clip_hits`.
Nothing between the old and new guard positions relies on the 2^30
headroom (absorb/sponge shrink through `mul128_shr`). Then every stage sees
`|u|² ≤ 2^55` (two components at 2^27; `du2_raw` is **Q32**, the
`cap2_q32`/`rad` convention) and the pinned order is

    t  = mul128_shr(k_ke_recip_q32, du2_raw, 48)    // Q32·Q32 >> 48 = Q16 ΔT   (≤ 2^22·2^55 >> 48 = 2^29)
    dE = mul128_shr(N_raw, t, 0)                     // Q16·Q16 = Q32 energy    (≤ 2^30·2^29 = 2^59)

(v3 pinned `>> 32`, which lands ΔT in Q32 — a 65 536× debit, R3-#3.)
`mul128_shr` is arithmetic on both branches (floors toward −∞ for either
sign); the counter books the same truncated `dE`, so no asymmetry leaks. Moving
the guard changes absorb/sponge inputs only when it binds —
re-baseline-class, declared. **Thermal-solid cells** are kicked/absorbed/
sponged/capped too (the kick skip-set is `solid || is_vacuum || ring`, not
`ts`; only drag adds `!ts`): their brackets all go to `e_ts_ke_sum`,
never to `gas_energy` (F5). This retires the P-E3 deposit formula
(`recip_cv`, `heat_frac_q`, `e_drag_drop_sum`, `e_drag_rail_clipped`) —
see §5. Counter slots: `counters_out[6] = e_drag_heat_sum`, slots 7/8
retired-and-zero (D10); the device counter array and its D2H grow for
the new `def_readonly` members.

### 2.4 The face-flux energy step (replaces 4c; a separate pass after the kick loop)
Operands: the **absolute solved pressure `p^{n+1}`** (on an ambient map the
solve runs shifted, `P' = P − p_amb`, restored at step 5 — the energy step
runs **after step 5**, or adds `p_amb` itself; pinned: after step 5) and the
corrected-and-damped velocity `u_new` (the stored velocity after absorb /
sponge / cap / drag — NOT the projected `u^{n+1}`; where those bands vary
spatially they do spurious, conservative, redistributive work — §7, same
class as HEAD's 4c). Negative `p^{n+1}` (measured `P_min = −0.98` in the
energy-books close): **`p_i` and `p_j` are floored at 0 individually before
eq. 15** (F15 — keeps every operand non-negative for `floordiv`, identical on
both sides), with a hit counter `p_face_floor_hits`.

Face velocity: the **arithmetic** mean `(u_i+u_j)/2` — a deliberate deviation
from Kwatra eq. 13's density-weighted `û` (F16): it is exactly the face value
our centred `div_u_` stencil implies, so `Σ_f p_f û_f` vanishes where the
solve zeroed the divergence.

For each face f, **canonical orientation fixes the sign**: `i` = lower linear
index, `j` = higher; east and south faces are owned by `i`. Evaluated
**once** from identical inputs on both sides (gather form — §2.5):

    if both i, j accountable:                                        (INTERIOR face)
        p_f   = floordiv( p_j·N_i + p_i·N_j , N_i + N_j )              (eq. 15 literally; skip if N_i+N_j == 0;
                                                                       p ≤ 2^31, N < 2^30 ⇒ products ≤ 2^62, int64)
        u_f   = u_i + u_j                                              (½ folded into k_flux — no separate >>1 bias)
        applied to BOTH cells
    elif exactly one side accountable, the other is_vacuum / is_ambient:   (OUTFLOW face)
        p_f = p_acc ; u_f = u_acc  substituted INTO the fixed i/j orientation (F14 — the accountable
        side may be i OR j; the sign never flips). ring/vacuum u ≡ 0, so (u_acc+0)/2·2 = u_acc — the
        value the divergence stencil reads there (mirror_idx keys on solid only).
        applied to the accountable cell only; counted in `e_work_export_sum` (breach rarefaction, §3)
    elif the other side is solid OR thermal_solid:                   (WALL: û_f = 0 — no face)
        no face.   Furniture is a wall to the energy step (F4): a one-sided flux from a ts
        cell would be an unbounded, uncounted source driven by the OBJECT's temperature
        (eos_solver.cpp:442-447: T on a ts cell is the object's). Consistent with HEAD's 4c
        skip (:1020), transport rule (d), and D2. The lost pressure work through a permeable
        crate is a D4-class accepted gap with a probe (§7).

    mag   = mul128_shr( mul128_shr( p_f , |u_f| , 16 ) , k_flux_q , 0 )    // Q16·Q16>>16 = Q16; ·Q16>>0 = Q32
                                                                            // (shifts total 16 — v3's 32 landed
                                                                            //  in Q16, R3-#4)
    mag   = min(mag, 2^60)  → `flux_sat_hits`                               // int64 corner: p_f ≤ 2^31, |u_f| ≤ 2^28,
                                                                            // k_flux_q ≈ 2^19 ⇒ ≤ 2^62 per face,
                                                                            // 4 faces ≤ 2^64 — saturate per face,
                                                                            // 4·2^60 = 2^62 fits (same on both sides)
    flux  = sign(u_f)·mag                    (sign applied AFTER truncation — exact cancellation)
    E_i −= flux ;  E_j += flux               (interior)

`k_flux_q = quantize( k_work · dt_s / (2·dx) )` folded once per tick
(includes the ½ of the face mean and the sub-cycle `dt_s`, D8).

**Positivity / sub-cycling (D8, F2, F3):** the tick's material Courant
number is not bounded by 1 (`n_sub ≤ N_SUB_MAX = 8` rails during venting;
`|div|·dt` up to ~8, i.e. 6× past the retired `T_WORK_CLAMP` rail). The
energy step runs **`n_sub` sub-cycles** with `dt_s = dt/n_sub`, holding
`u_new` and `N` fixed but **refreshing the cell pressure from the running
energy each sub-cycle — in increment form, ONE definition for all
sub-cycles** (R3-#6: refreshing to `C·E` outright would switch the operand
from the solved `p^{n+1}` to `p*` after sub-cycle 1, the very defect v2
resolved):

    p_i^{(k)} = max(0, p_i^{n+1} + mul128_shr(c_q, E_i^{(k)} − E_i^{(0)}, 32))    // Q16·Q32 >> 32 = Q16

Sub-cycle 1 is exactly `p^{n+1}`; later sub-cycles carry the EOS-consistent
correction for the energy already moved. Without a refresh, sub-cycling
with frozen operands is arithmetically identical to one pass at `dt` (F2);
with it the outflow shrinks as `E_i` shrinks and the bound is geometric.
Telescoping survives (both sides read the same live `E_i`, `E_j`).

Last-resort rail, **donor-only** (F3 — a rail that also scales incoming
credit cannot reach a fixed point in one pass, and incoming credits are all
≥ 0, so ignoring them is safe):

    OUT_i  = Σ over faces where flux leaves i of |flux_f|        (interior + outflow; ≤ 4·2^60)
    head_i = max(0, E_i − N_i·(T_MIN + T_AMB)_raw)
    if head_i >= OUT_i:  s_i = 2^16                              (early-out — the common case)
    else:                s_i = floordiv(head_i, (OUT_i >> 16) + 1)   (no 128-bit divide on device;
                                                                  the +1 keeps s_i·OUT_i/2^16 ≤ head_i
                                                                  strictly — R3-#7: `head<<16` overflows)
    applied_f = mul128_shr(flux_f, s_donor(f), 16)                (truncating; the DONOR's factor only —
                                                                  donorship follows sign(u_f), so pass B
                                                                  reads s at self + 4 neighbours)

so `Σ_f applied_f ≤ head_i` and `E_i ≥ floor_i` unconditionally (incoming
credits are ≥ 0). The rail is a two-pass kernel (§2.5, F13): pass A
computes `OUT_i` and `s_i` into a scratch plane (device `(N,h,w)`, CPU
`(h,w)`, int32 Q16); pass B gathers with each face's donor factor read
from the plane. **The CPU twin needs both passes too** — `s_i` is not
knowable until the cell's whole face set is priced; a fused sweep would
be order-dependent or arithmetically different (AB tol 0). Shortfall
counted in `e_energy_floor_sum` (signed energy).

### 2.5 Parity-safe kernel shape
Two passes per sub-cycle, both per-cell **gather** (5-point), no atomics,
no face buffer; the CPU twin is the same two loops:
- **Pass A** (`OUT_i`, `s_i` → scratch plane `(N,h,w)` int32 Q16): every
  cell recomputes its four face fluxes in canonical orientation from the
  same inputs, truncating the magnitude once and applying the sign after.
- **Pass B** (apply): every cell recomputes the same four fluxes, scales
  each by the *donor's* `s` read from the plane, and sums.
Identical int64 on both sides of a face ⇒ exact cancellation — the S1
water-flux idiom (fixed_point.h:22-25). On device this is a **new third
kernel `K3` launched after `K_store_atm`** (cuda_eos_resident.cu:835 — the
step-5 un-shift), NOT a rewrite of K2 inside the K1→K2 pair (K2 runs
before the un-shift; on a space map the two would agree and AB would pass,
on an ambient map they would diverge — seam finding 8). `mul128_shr` is
promoted into `fixed_point.h` as ONE primitive with **three branches**
(`__SIZEOF_INT128__` host, MSVC `_mul128` host, and a `__CUDA_ARCH__` arm
carrying the existing `mul128_shr_signed` body incl. its `S == 0` case —
F10: under MSVC-host nvcc the device pass has no `__int128`); today three
copies: eos_solver.cpp:46, cuda_fixedpoint_device.cuh, cuda_kick_
compression.cu:77 — P-G0, bit-identical.

### 2.6 Recovery (mirror refresh) — cadence pinned (F7)
`T_rel,i = floordiv(gas_energy_i, N_i) − T_AMB_K_raw` on accountable cells;
divide policy reused from bulk_transport.cpp verbatim (`N_EPS_RAW = 1`:
below it, wipe to ambient and book `e_wipe_sum`; never divide by 0 — the
same on both backends). **The full recovery with both rails runs exactly
once per tick, at the end of the energy pass, over the whole accountable
set** — that is what bounds stored `E` (§2.2) and `p* = C·E` (int32) and
keeps `t_max_phys_hits` meaningful (`test_air_boundary.py:820` STOP). Rails
`T_MIN` / `T_MAX_PHYS` clamp the mirror **and write back `gas_energy =
N·(T_clamped + T_AMB)` ONLY when a rail binds**, booking the delta to
`e_rail_sum` (signed energy). **Never** write back otherwise
(`N·floordiv(E,N) ≤ E` would drain up to N−1 raw per cell per tick — the
drip class this arc kills). Seam writes (§2.7) refresh the **mirror only**
(no rails, no write-back). The CFL max-T scan (eos_solver.cpp:434-440)
gains the same `N_EPS_RAW` guard (thin-N cells cannot set `c_LOCAL`/`n_sub`
for the grid).

### 2.7 The writer seam — energy AND mass
Under stored E every writer of **N** is a writer of T. One primitive,
`gas_energy_move(i, ΔN, T_abs_src)` / `gas_energy_deposit(i, ΔE)` (C++, with
the Python-visible twin on GameMap), applies two rules — **moved mass
carries its source's `T_abs`; minted mass is born at ambient**:
- **moved**: mass leaving cell i carries `ΔN·T_abs,i` out; arriving mass
  carries the donor's `T_abs` (transport, pumps, seal receivers, `unseal`
  withdrawals, combustion products);
- **minted**: mass with no gas donor (`destroy_wall`'s seed, ambient / ring
  / vacuum / ts-face inflow) is credited `ΔN·T_AMB_K_raw` absolute (the
  P-E1 "born carrying zero *relative* energy" rule in absolute currency —
  v1 would have born it at 0 K).
Every site, with its rule:

| site | today | v3 |
|---|---|---|
| bulk transport donor-cell (bulk_transport.cpp:265 + CUDA) | moves `N·T_rel` as int64 `e[]`, per-substep recovery divide | face energy priced **off live state, no mirror** (F6): `phi_f = floordiv(dq_f·E_i, N_i)` for donor i, computed identically by both sides from `(dq_f, E_i, N_i)` — the 5-point read stage 3 already does; `Σ_f phi_f ≤ E_i` since `Σ_f dq_f ≤ N_i`; non-participating donors credit `dq·T_AMB`; the per-substep recovery is deleted; counters `e_ts_residual`, `e_wipe_sum`, `e_floor_sum` re-derived in absolute convention; closure identity `test_no_transport_mint` (in `test_e1_hot_rail.py:206`) / `cuda_bulk_flux_check` PART 3 re-stated (§6) |
| **water-displacement gas evacuation** (`physics_engine.cpp:715/820-834` `step_water_tail`, host-side on BOTH backends — confirmed, no `.cu` twin — before the EOS) — **missed by v1/v2** | flooding cell pushes `(1 − 1/ratio)` of EVERY plane incl. bulk O2/N2 to neighbours selected by `!solid && perm > 0` only (i.e. also into vacuum / ring / ts cells), T untouched; the function has NO temperature/energy args (`:715-721`, pybind `bindings.cpp:3285`) | signature + pybind + Python call site gain `gas_energy`, `gas_conservative`, `thermal_solid`, `is_vacuum`, `is_ambient`; only **bulk** shares move energy: `gas_energy_move(i → nb, share, T_abs,i)` for accountable receivers, and shares into non-accountable cells export to `e_water_evac_export_sum` (R3-#10) |
| combustion O2 consumed (combustion.cpp:590-621) / products to flame cell (:753-779) | N moves two hops (`alloc_slot[slot·n+j]` → Pass B gather per source `i` → `dep_site[t·n+i]` → Pass C gather at `s`), T untouched; `soot_yield = 0.5` (config.toml:774): donor loses `burn` bulk O2, flame gains only `burn − soot` bulk N2 — **bulk N is NOT conserved across a burn** (F11, seam 2) | **a parallel two-hop int64 energy ledger** — `e_slot[n_slots·n]` and `e_dep_site[5·n]` planes replicating Pass B's hop1/remainder split (`:707-722`) with its own exact-conservation rule (R3-#8; an accumulator beside `alloc[]` cannot reach `s`). Delivery: **`(burn − soot)·T_abs` to the flame cell's `gas_energy`; `soot·T_abs` booked to `e_soot_shed_sum`** (the enthalpy the soot carries out of the bulk books; pairs with `n_soot_shed_sum`) — delivering all of it would raise the parcel's `E/N` by `1/(1−soot_yield)` = 2× and, in the R=1 donor==deposit case, compound the cell's T by ~+1.7%/tick with no counterparty (R3-#9: #54 through a new door). If the flame cell is a `thermal_solid` (`object_site` branch) the products' energy is exported to `e_ts_products_sum` (rule (d)). Fire heat `heat_saturating_add` → `gas_energy_deposit` on gas, unchanged on solids. **Combustion runs after the once-per-tick recovery** (physics_runner.py:798 → :828), so `gas_energy_deposit` itself applies the `T_MAX_PHYS` rail at the deposit site (clamp + `t_max_phys_hits`, keeping `test_air_boundary.py:820` STOP and §2.2's `E ≤ 2^60` budget honest) |
| thermal solver: heat→T deposit (:323), conduction swap (:414), radiation (:222) | writes T | gas side: `de` (already int64) → `gas_energy_deposit`, endpoint divide deleted; solids side unchanged; wipes (:135) zero E and book |
| pump primitives `inject_gas_n(_vec)` / `extract_gas_n(_vec)` (gamemap.py:2410-2540) | vec inject mixes T; extract leaves T | extract removes `ΔN·T_abs(cell)`; inject credits `ΔN·T_abs(t_dep)`; the vent plenum ledger (`vent_system.py`, relative currency, ENTITY_SECT-digested) is **kept relative** and converted at this seam (`E = e + n·T_AMB`); the primitive's direct `temperature[...] =` write (gamemap.py:2540) is replaced by the seam's mirror refresh |
| `seal_tiles` receivers (:2192) | N redistributed to receivers, T untouched | receivers credited at the sealed tile's `T_abs` (**moved**); the sealed tile's remaining `gas_energy` retires to `e_retire_sum` |
| `unseal_tiles` (:2292-2311; doors via `door_system.py:253`) | seed = `sum(donors)//(k+1)` **withdrawn from the donors** — grid-total N unchanged to the LSB (:2224) | a conservative **withdrawal, not a mint** (seam 3): each donor's share carries its `T_abs` (**moved**); NOT born at ambient (that would mint/destroy `ΔN·(T_AMB − T_abs,donor)` at every door) |
| `destroy_wall` seed (:1909/:1944, `n_destruction_seed_sum`) | mints an ambient cell | born at ambient (**minted**) — the one true mint |
| any gas cell becoming solid / ts (all `on_tile_changed` callers, gamemap.py:1187: seal, burst walls, bullet cover-chew, furniture flips) | — | `gas_energy` retires to `e_retire_sum`; a ts cell becoming gas is born at ambient (its object T stays on the solids side — the mirror is refreshed so no stale 1300 K survives). Property gate: flip a tile both ways, books move by exactly the named amount (`test_destroy_wall_conserves_mass.py` gate 6 precedent) |
| FieldEdit `atmosphere` policy (O2/N2 deposits, field_edit.py:536), the **`wave_source`** row (which is the T writer, :541/:595 — there is no `temperature` row), `heat` (:589), and the generic **`gas`** policy (:219, any slice by channel) | direct writes | `atmosphere` deposits credit at ambient; the `wave_source` combine on gas cells → deposit; a `FIELD_POLICY` row for `gas_energy` with a **dual-write combine** (store + mirror); the `gas` policy **refuses conservative slices** (assert — trace-only today: combat.py:1370, payloads.py:85) |
| tests/tools seeding `gmap.temperature[...] = ` (re-grep at patch time: ~146 sites / 60 files incl. `_hotplate_heating_bench.py:72`; some seed **wall/ts** tiles, e.g. `test_thermal_mass_axis.py:524-537`) | direct | gas-cell sites → `GameMap.seed_gas_temperature(sel, T_q)` (both planes); ts/wall sites unchanged (solids stay T). **Per-site medium check, not a sed.** The three gate benches migrate in P-G0 (else the HP gate passes vacuously) |
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
- Sealed, furniture-free rooms cannot gain or lose **thermal** energy
  through the flux step (exact); the remaining channels into a sealed box
  are all counted — conduction deposits, the KE brackets (kick debit, drag
  credit), and the rails. KE itself is not a book (F12): `Σ N k_ke|u|²`
  changes with N during transport with no counterparty (bounded by `k_ke`;
  sub-degree at wind speeds; SB probes `Σ N|u|²` drift). #54's +121 / −20
  signature disappears structurally.
- Blast cores: **HOTTER than HEAD** (measured at P-G1a: core +200.6 vs
  HEAD +38.3; outside-disc peak +219.6 vs HEAD +383.2). v2/v3's "cooler"
  prediction was wrong and the measurement explains why: HEAD's core was
  *colder than its own surroundings* because 4c pumped the blast's energy
  away from it and out into the map; the conservative form keeps the
  energy in the disc. Fire tuning (#5/#8) will see hotter cores and cooler
  surroundings. (Corrects v1 §8 and v2 §3.)
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
  conduction exchange, booked as ΔE on the gas side; ts faces are walls to
  the energy step and ts cells export their KE brackets (§2.3/§2.4).
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
- **D7 KE is not a book** — ACCEPTED GAP, two channels: KE rides u's SL
  advection (non-conservative) and `Σ N k_ke|u|²` changes with N during
  transport with no counterparty (F12). Bound: `KE/E = k_ke·|u|²/T_abs ≈
  1.2e-3` at 20 m/s, ≈ 0.28 at 300 m/s (blast cores; short-lived); SB
  probes the drift.
- **★ D8 energy step sub-cycled `n_sub` times** with `u_new`, `N` held and
  `p_i` refreshed from the running `E_i` each sub-cycle (§2.4). Erik to
  confirm (cost: n_sub × two gather passes; n_sub = 1 in calm play).
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
- **D13 Recorder**: int64 fields get a **true int64 ring branch** (a
  float64 ring is exact only to 2^53 and would drop the LSBs SB asserts on;
  recorder.py:102 pre-allocates float32 today) — the frozen npz contract is
  extended additively by *dtype class*, not just membership (a CLAUDE.md
  Recorder-rule amendment, §Systems). `analyze_blowup_dump.py` gas_energy
  column: later.

## 5. Twin sites + ABI (deletion blast radius verified by the critique)
- C++/CUDA: `EOSSolver::step` 4c block + `eos_kick_compression_reference`
  twin; `cuda_kick_compression.cu` K1 (per-stage KE brackets, RAD_SAFE
  guard moved/tightened) — **K2 deleted**; the flux pass is a **new K3
  (two launches, A/B passes) after `K_store_atm`** (§2.5) + `kick_scalar_
  folds`; `cuda_eos_step.cu` / `cuda_eos_resident.cu` call sites;
  `bulk_transport.cpp` + `cuda_bulk_transport.cu` (live-priced energy
  faces, absolute born-at-ambient); `combustion.cpp` + `.cu` (per-donor
  energy, soot/object-site rows, deposit); `temperature_solver.cpp` +
  `cuda_temperature.cu` (gas-side deposit, wipes); `physics_engine.cpp`
  (buffer plumbing + `step_water_tail` evacuation row, host-only). Promote
  `mul128_shr` (three-branch) to `fixed_point.h` (P-G0). **CUDA parity is
  suspended from P-G1a until P-G2** (named, time-boxed: the resident path
  keeps the 4c kernels meanwhile; AB and `cuda_*_check` are marked
  `xfail(reason="P-G2 pending")` in P-G1a and re-armed as P-G2's gate).
- Deleted symbols and their consumers (full list in the critique; the
  implementer greps before deleting): `T_WORK_CLAMP`/`t_work_clamp`/
  `work_clamp_q` (~12 test files incl. `test_p_e4_reversible_work.py` ×21;
  **8 tools**: batch_rails, quiet_room_drift, storm_ledger, storm_probe,
  tabs_pw2_venting_capture, velocity_clamp_pv2_measure, analyze_blowup_
  dump, e2b_floor_reciprocal_probe), `n_work_ref` (+ `physics_runner.py:479`
  stale-key guard), `k_drag_heat_frac`/`heat_frac_q`/`recip_cv` (+
  `physics_runner.py:492`), `e_drag_deposit`/`e_drag_drop_sum`/`e_drag_
  rail_clipped` (→ `e_drag_heat_sum` in `counters_out[6]`; 7/8 zero),
  `eth_compression_delta` (→ removed; `test_destroy_wall_conserves_mass.py:
  501` asserts its type — MECHANICAL). pybind `run_substeps` / `kick`
  signatures lose the three floats (~29 Python callers) and gain
  `gas_energy`; the device counter array grows for the new members.
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
Gate vocabulary — **SB**: sealed-box bench `nofire`: (i) the **§2.8 closure
identity restricted to the box**, exact in int64: `ΔΣ_box E == Σ_box
(conduction deposits − kick KE debit + drag KE credit ± rails)` — the flux
term contributes exactly 0 (F8: a bare "constant except conduction" gate
would fail a correct implementation, since `k_drag = 0.5` moves KE every
tick); (ii) `ΔT_box = Δ(ΣE/ΣN)` within ±2 game-deg (N-weighted — an
unweighted mirror mean is not conserved by mixing); (iii) u_max < 3 m/s;
(iv) probes reported: the D4 wall term `Σ_wall-adjacent |u_i·p_i|`, the ts
(furniture) wall term, and the `Σ N|u|²` drift (D7).
**HP**: hot-plate via `seed_gas_temperature`, rooms ≈ 0. **VENT**: a new
venting bench (breach of a 1-atm room to vacuum, 5 s): `gas_energy ≥ 0`
everywhere, `n_sub` stable, mouth cools, `e_work_export_sum` matches the
room's energy loss. **BLAST**: `frag_standard` mid-arena, 3 s: no
`T_MAX_PHYS` hit outside the blast disc, core T *below* HEAD's. **FIRE**: a
burning crate stack, 10 s (the scenario's crate at (26,41)): closure
identity exact incl. `e_soot_shed_sum`; flame-cell `E/N` bounded (no
per-tick compounding — R3-#9); rooms elsewhere ≈ 0. **AS**:
all-systems scenario P3 PASS, P1a/P4/P6 PASS. **AB**: `field_ab_harness.py`
CPU vs CUDA tol 0. **SUITE**: `pytest tests -q` against the §6 red table.

| # | Patch | Mode / tier | Gate |
|---|---|---|---|
| P-G0 | Field + plumbing, NO physics: `gas_energy` (init `N·T_abs`), digest v5 + regenerate goldens (spec bump event 1), recorder int64 branch, `seed_gas_temperature` + per-site migration of the gas-cell seeds (**the three gate benches first**: `_sealedbox_bisect_bench`, `_hotplate_heating_bench`, `_xarch_perfield_digest` — they also reference the retired dials), `mul128_shr` three-branch promotion, AB `SIM_FIELDS`, the constants fold + throw guard (§2.1, no consumer yet). Books identity `Σ(E − N·T_AMB) == old sum` on 3 levels. | subagent, Sonnet 5 | **every per-field digest except `gas_energy` byte-identical to HEAD; only `GOLDEN_AGGREGATE` moves** (catches a bad seed migration — plain "SUITE green" does not). Procedure (R3-#11): `_xarch_perfield_digest.py` emits per-`(tick, field)` lines; capture the HEAD baseline **before** migrating that bench, then diff **keyed by `(tick, field)` name** — its built-in `first_divergence` compares by line index and would report a false divergence once `gas_energy` is inserted; SUITE green |
| P-G1a | CPU EOS-internal: §2.3 brackets in the kick loop (+ RAD_SAFE move), §2.4 flux pass (sub-cycled, two-pass rail), §2.6 recovery once per tick, transport prices faces off `E` (§2.7 row 1), water-tail evacuation row (host), trust gate/clamp/heat-frac retired, CFL scan guard, S_EOS guard carried. **Transitional state, stated**: at EOS entry `gas_energy := N·T_abs` from the mirror (T is still the cross-tick truth; the tail's writers are absorbed by next tick's re-sync); at exit the mirror is refreshed. D1 not yet live across ticks. CUDA parity suspended (xfail, §5). | subagent, **Opus 4.8** | SB (i) within the EOS step, HP, **VENT**, **BLAST**, **FIRE** (P-G1b re-runs it with the combustion ledger live), AS P3 |
| P-G1b | Writers: combustion (per-donor energy, soot, object-site), thermal solver gas side, pump primitives + plenum conversion, seal/unseal/destroy/`on_tile_changed` seams, FieldEdit rows + `gas`-policy assert; remove the entry re-sync → **D1 live**. Closure identity test rewritten; tile-flip property gate added. | subagent, Opus 4.8 | SB (i) across ticks (exact), SUITE per red table, 60 s quiet-run books drift == counted |
| P-G2 | CUDA twins: K1 brackets, **K3** two-pass flux kernel after `K_store_atm`, bulk transport, combustion, temperature kernels; resident buffer + upload/D2H lists; counter array; xfails removed. | subagent, Sonnet 5 (oracle) | **AB tol 0** on playground + 2 levels incl. one **ambient** map (the un-shift trap); CUDA harness green → auto-merge on green |
| P-G3 | Golden re-baseline (value-move event 2) with `docs/gas_energy_rebaseline_<date>.md`; float ratchet + ingress lint; docs outside archive amended (list in critique: engine/04, drag_law_v2, storm_audit, …). | inline, Haiku 4.5 | SUITE green |
| P-G4 | **HUMAN-TEST**: Erik plays (fires, grenades, breach; T overlay). Brief: §3. | Erik | feel |

**Red-classification table (agreed before P-G1a; verified against the
tree by the round-2 critique):**
- **STOP** (assertion survives verbatim): `test_destroy_wall_conserves_
  mass.py` gate 6 (restated in absolute currency — its `eth_compression_
  delta` type assert at :501 is MECHANICAL), `test_air_boundary.py:820`
  (`t_max_phys_hits == 0`).
- **MECHANICAL** (signature/kwarg/docstring edits, property unchanged —
  the retired dials are required pybind kwargs, so these `TypeError`
  until edited): `test_velocity_clamp_property.py:48`, `test_thermal_mass_
  axis.py:640/690`, `test_eos_p4_combustion.py:53` (docstring only), `test_
  drag2_dead_zone_property.py`, `test_drag2_stage_q_law.py`, every
  `cuda_*_check.py` that passes `t_work_clamp`/`k_drag_heat_frac`.
- **EXPECTED red → regenerate in P-G3**: every field/aggregate golden, the
  11 CUDA check goldens, inline goldens (`test_b6_logic_golden`,
  `test_vent_*`, `test_w6_armory`).
- **SUSPENDED P-G1a→P-G2** (xfail): AB harness + `cuda_*_check` parity.
- **RETIRED** (delete with rationale): `test_p_e4_reversible_work.py`,
  `test_e1_cold_rail.py` (subject gone; their property — no temperature
  unbacked by energy — is the closure identity). (`test_p_e4_trust_gate.py`
  does not exist; the trust gate lives inside the former two.)
- **REWRITTEN**: `test_p_e3_drag.py` (drag identity on `e_drag_heat_sum`),
  `test_drag2_venting_gate.py` (`e_drag_*` counters), **`test_e1_hot_
  rail.py`** (hosts `test_no_transport_mint` at :206 — kept, rewritten to
  the absolute closure; NOT retired) + `cuda_bulk_flux_check` PART 3.
- **BENCHES/TOOLS** (migrate in P-G0): `_sealedbox_bisect_bench`,
  `_hotplate_heating_bench`, `_xarch_perfield_digest`, `_drag2_sweep_bench`;
  the 8 tools in §5.

Each patch: own worktree + branch; re-plan at boundaries; checkpoint memory.

## 7. Accepted gaps (decisions with bounds)
- D4 wall-stencil mismatch (bound: reported probe; follow-up queued).
- D7 KE is not a book — SL-advection and transport-ΔN channels (≈1.2e-3
  at 20 m/s; ≈0.28 at 300 m/s, transient; SB probes `Σ N|u|²` drift).
- ts (furniture) faces are walls to the energy step: the pressure work of
  gas seeping through a permeable crate is lost, D4-class, probed not
  counted; ts cells' KE brackets and products' energy are exported and
  counted (`e_ts_ke_sum`, `e_ts_products_sum`).
- `u_new` is the damped velocity, not the projected one: where
  `dyn_wave_absorb` / `sponge_udamp` vary spatially the flux step does
  spurious but conservative, redistributive work (F17) — HEAD's 4c has the
  same; D4's probe covers it.
- 0.4% cross-wall pressure contamination stays; it can only *redistribute*
  energy inside a sealed region (telescoping), never pump it. Residual
  channels that remain open are the rails and the KE brackets — all counted.
- One `c_v_phys` for all gases (no per-species heat capacity).
- `gas_energy` round-trips host↔device every tick (combustion + tail on the
  host mirror); resident combustion is a later arc.
- `analyze_blowup_dump.py` gains no gas_energy column in this arc.

## 8. What Erik must know before playing (P-G4 brief)
- Rooms stay at ambient unless something heats them. Hot corners gone.
- Grenade cores are **hotter** than before and the surroundings cooler
  (HEAD was pumping the core's energy outward). Judge the shape; retune is
  #5/#8.
- Breach: chill present, as a ring around the mouth rather than a core
  (P-G1a VENT: mouth ring −231…−238, interior −102).

---

## P-G1a results (2026-08-29, branch `gas-energy-arc`, commits 3c3ba66..ddabe8c)

| sealed-box `nofire`, 18 s, term ON | HEAD | P-G1a |
|---|---|---|
| box ΔT | +121.0 | **+6.0** (−0.00 with conduction off — the residual is §2.7 row 3, P-G1b) |
| bunker / pen ΔT | +72.1 / +61.7 | **−0.1 / −0.1** |
| arena ΔT | −19.7 | +0.0 |
| u_max | 21.2 | 5.6 |
| closure identity | n/a | **exact in int64, all 432 steps** |

HP: rooms −0.1, plate's neighbourhood +16.8 (the bench's own FIXED).
VENT: `gas_energy ≥ 0` every tick, identity exact, export counter matches
the room's loss. BLAST: no `T_MAX_PHYS` hit; cores hotter (§3). FIRE:
identity exact, box +3.35 (conduction), bunker/pen 0. AS: **P3 PASS**
(bunker +0.0, pen +0.0, gallery +1.0, basin −0.1); P1b/P5 lab now hold
water and air. Suite: 31 red = 24 pre-existing + 4 EXPECTED goldens + 3
value moves (below).

**Open rulings from P-G1a (Erik):**
1. `test_air_boundary.py` gate 2 (`t_max_phys_hits == 0`, STOP) reads 564
   → 0 by tick 4 on its slam-a-40×40-room-to-0.1-atm-then-open test: the
   MG solve lifts P to ~1 atm acoustically while N is still 0.1, the kick
   hits ~200 m/s, energy Courant ~40 per tick, and the `T_MAX_PHYS` rail
   catches the first-tick overshoot — as §2.2/§2.6 specify. An inflow rail
   was tried and reverted (it turns an open boundary into a refrigerator).
   Ruling: restate the gate as "hits decay to 0 within 4 ticks and are
   counted in `e_rail_sum`" (recommended) vs. design an inflow rail.
2. `test_b5_airlock`: cycle completes at +310 ticks vs cap 300 because the
   evacuated chamber's gas is honestly cold (−133 ≈ adiabatic −150) so the
   pump needs more mass for 0.9 atm. Re-measure after P-G1b's pump seam;
   then a cap retune or a pump-rate dial.
3. `test_water_displacement` settle 0.9953 atm vs band [1.00, 1.12] (band
   encoded the 4c law) and `test_air_boundary` gate 3 reflection 2.48% vs
   2%: value moves → P-G3 re-baseline with rationale.

## P-G1b results (2026-08-30, `gas-energy-arc` 61be37e..d3c6689)

**D1 live**: entry re-sync deleted; `refresh_gas_energy` is the level-load
initialiser only. **Closure identity exact ACROSS WHOLE TICKS** over five
counter groups (EOS, thermal-solver gas side, combustion, Python seams,
water-tail evacuation) — SB, FIRE, QUIET (60 s, 1440 ticks, residual 0;
drift −0.41 game-deg total). Airlock test **green** (pump seam). FIRE:
parcel identity + soot shed exact; flame peak 676.3 (no compounding).
BLAST/HP/VENT/AS unchanged. Two holes the gates found and fixed: VENT
regressed to negative E on near-vacuum cells until the conduction
capacity-floor shrink was applied in energy currency; the water-tail
export term was missing from every identity transcription (zero on dry
scenarios — verified non-vacuously with a water dump).

**Residual, measured — needs Erik's ruling (candidate P-G1c):** SB (ii)
`nofire` ΔT_box = **+5.01** (P-G1a +4.97). Bisected to ONE channel:
**gas ↔ thermal-solid conduction** (`e_gas_cond_sum` = +6.8 box-deg
equivalent; kill ts faces → −0.002; kill gas–gas faces → unchanged; the
arena gains too — a global books *source*, not a redistribution).
Mechanism: conduction diffuses *unweighted* T at `min(cap_gas, cap_solid)`
while the books are N-weighted; in a ringing cell T and N correlate
(hot where dense), the unweighted mean sits below the N-weighted one, and
thermal solids — relaxed to ambient by Pass 3 — top the gas up. P-G1b made
the channel named and exact but cannot remove it (with `c_v = 1`,
`cap_gas = N`, the T-form and E-form bookings coincide). Zero in a quiet
box (QUIET drift −0.41 over 60 s); it is ringing-driven. Options:
(a) thermal solids join the books with finite thermal mass, Pass 3's
relax-to-ambient becomes a counted export through the hull; (b) conduction
priced in books currency (energy flux from `E/N` per face, N-weighted);
(c) count Pass 3's relaxation as a source and accept the bound for now.

Red inventory: 34 = 31 pre-existing/expected + 4 new inline
door-trajectory goldens (EXPECTED → P-G3: `unseal_tiles` now refreshes the
opened tile's mirror) − 1 fixed (airlock). STOP `t_max_phys_hits` 564
unchanged (ruling 1 above). CUDA combustion/temperature kernels do NOT
carry `gas_energy` yet — the resident tick breaks the identity until P-G2.

Agent decisions accepted at review (P-G1a): `p*` keeps reading the mirror
(ts cells' `p*` must read the object's T); rail scales magnitude then
re-applies sign; `price_face` splits `E = N·q + r` to avoid a 128-bit
divide; per-face saturation at the `p·u` stage via a folded `pu_cap`;
sub-cycle pressure refresh ceilinged at `C·N·(T_MAX_PHYS+T_AMB)`
(`p_face_ceil_hits`); retired counters kept as always-0 members (D10);
`test_field_ab_harness` (CPU-vs-CPU) NOT xfailed.

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
(gates); Recorder (dtype-class extension — rule amended in (b)); Config;
RL-batch habits (`(N,h,w)` device buffer + scratch plane); temperature
scale (`T_AMB_K`, `c_max`, `γ` define the two constants); PhysicsEngine's
`step_water_tail` (the water-displacement evacuation joins the seam);
**Entity system / sensor accessor** (`sensor_accessor.py:135` returns gas
`temperature` — a mirror read; the once-per-tick recovery runs before the
sensor sweep, so it is never stale); the **Coupling table** (`exchange.py`
reads `gmap.heat`, not `temperature` — untouched, stated); the three
benches (extended, not forked).

**(b) New systems / amended rules (→ CLAUDE.md at implementation)**:
- **Gas energy field** (`gamemap.gas_energy`): *the conserved truth for gas
  thermal energy; `temperature` is its mirror on gas cells and the truth on
  thermal solids.*
- **Gas energy seam** (`gas_energy_move` / `gas_energy_deposit`): *every
  change of gas N or gas heat goes through the seam — MOVED mass carries
  its source's `T_abs`, MINTED mass is born at ambient.*
- **Gas temperature is a mirror**: *nothing writes gas `temperature`
  directly (tests included: `seed_gas_temperature`); the recovery refreshes
  it once per tick and at every seam write.*
- **Amend** Recorder rule: "extend `DEFAULT_FIELDS` additively" → "+ a new
  dtype class (int64) is a contract extension with its own ring branch,
  never a cast".
- **Face-flux energy step**: *energy exchange between gas cells is a per-face
  flux evaluated once in canonical orientation and applied with opposite
  signs — never a per-cell source term; boundary faces export to a counter.*
- **Amend** Temperature solver rule (survey A22): "derived here alone" →
  "solids' T derived here alone; gas T is the energy field's mirror".
- **Amend** Gas pump primitives rule: "+ energy-aware: the primitives carry
  `ΔN·T_abs` through the gas energy seam".
- **Amend** Interior drag rule: "drag L/Q stages deposit their removed KE as
  heat via `k_ke`; absorb/sponge/cap stages export or destroy it, counted".
