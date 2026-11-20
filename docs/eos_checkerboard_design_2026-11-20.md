# EOS checkerboard: measurement, cause, remedy options (2026-11-20)

Issue #217 finding 1 and #72 item 2. First patch of the arc: measurement and
diagnosis only. No solver behaviour changed, no golden re-baselined, no remedy
chosen. The remedy is a physics-modelling call for Erik.

Instrument: `tests/_eos_checkerboard_bench.py` (harness). Property test:
`tests/test_eos_checkerboard.py` (strict xfail). Every number below comes from
the bench at branch `217-eos-checkerboard` (cut from `7cba57dd`), CPU build,
Work Desktop.

## 1. Problem

Neighbouring gas tiles drift apart in density (and, through step 6, in
temperature) in an alternating every-other-tile pattern, while the stored
pressure `atmosphere` stays flat. #217's comments measured it on
`levels/bench_two_room` and in played matches (rms 4-5 % at 0.333 m all match
long); #72 saw the ±12 K version in quiet sealed rooms in September. Unknown
until now: what makes the pattern grow rather than merely persist.

## 2. Measurements

Run any line with `<breach-py> tests/_eos_checkerboard_bench.py ...`. Fixture:
`bench_two_room` (two sealed 12×12 rooms, one door tile, one crate), only
`tile_size_m` changed, as the issue's `TILE` probes did. Seeds go in before the
first tick through the scenario-builder seam (`reseed_gas_energy` /
`seed_gas_temperature` on the cells written). "gap" is the rms over the
accountable cells of (P_stored − P_state)/P_state, with P_state = C·N·T_abs.
"corr" is its lag-1 neighbour correlation. "cbN" is the relative amplitude of
the (−1)^(x+y) component of bulk N, and "cbT" is that component of T in
degrees.

### 2.1 The issue's numbers reproduce

`--case cb --amp 0.001 --tile T --seconds 20`, shipped dials:

| tile | gap rms (issue) | gap rms (bench) | corr | cbN | cbT | T range |
|---|---|---|---|---|---|---|
| 1.0 m | 0.6 % | 0.58 % | −0.76 | −0.35 % | −0.39 | −8 / +6 |
| 0.5 m | 1.7 % | 1.80 % | −0.91 | −0.98 % | −2.1 | −5 / +5 |
| 0.333 m | 4.0 % | 4.02 % | −0.91 | −1.75 % | −6.0 | −16 / +28 |
| 0.25 m | 7.6 % | 7.16 % | −0.92 | −2.73 % | −12.0 | −25 / +25 |

- Control `--case still`: gap, wind and temperature stay exactly 0 for 20 s.
- `--eos adiabatic_index=1.0001` at 0.333 m: gap 3.88 %, cbN −3.74 %, T −0.01 / +0.03. The density pattern stays and the temperature stays flat, as the issue reported.
- The seed's size and sign do not matter. At 0.333 m with γ = 1.0001, a seed of +0.01 %, +0.1 %, −0.1 % or −1 % ends at cbN −3.2 to −3.7 % (−3.9 % gap at +1 % shipped). **The sign is always negative**: the tiles with x+y even lose air, whatever the seed.
- The parity is absolute. Shifting the whole tilemap one column right (a padding column of wall) leaves cbN at the same negative value on absolute parity (−3.82 % against −3.87 % at tick 24).
- Growth is linear, not exponential: cbN falls about 0.2 % per tick from tick 2 (γ = 1.0001) and saturates at about −4 % by tick 30.

### 2.2 The warm room (#217's first comment)

`--case warm`: the left room +40 degrees, its air thinned to equal pressure.

| setting | T range after 40 s (issue) | bench |
|---|---|---|
| shipped | −13 / +85 | −14 / +81 (peak +110 at 15 s); gap 4.3 %, corr −0.92 |
| `adiabatic_index = 1.0001` | 0 / +40 | 0.00 / +39.3, slowly relaxing |

As the issue reported, nothing moves for 5 ticks. At tick 6 the pattern
appears everywhere at once.

### 2.3 What sets the end level

- **c_max** (`--eos c_max=`), 0.333 m, γ = 1.0001, 20 s: 150 → 1.00 %, 300 → 3.88 %, 600 → 12.3 % (gap rms). The level goes as c², and the tile ladder above as 1/dx². Together that is σ² = (c·dt/dx)², the acoustic Courant number squared: σ² = 156, 625, 1409 and 2500 at 1.0, 0.5, 0.333 and 0.25 m.
- **mg_cycles** 2 / 8 / 32 / 128: gap 3.98 / 3.72 / 3.83 / 3.86 % at 10 s. The V-cycle count does not matter.
- **The solve path** (`--eos use_multigrid=0 --eos S=2000`, the flat red-black Gauss-Seidel A/B reference, converged): the seed **freezes**. cbN stays at exactly ±0.1007 % for 20 s, the wind is exactly 0, and the gap is a constant 0.31 %. The operator and the integer arithmetic are the same as on the multigrid path. Only the multigrid cycle drives the mode.

### 2.4 Per-step attribution (`--steps`)

Each tick is split using only what the engine exposes. Step 1a's u is replayed
by the test-only `bp.eos_sl_advect_ref`. N after step 1, p\* and div u come
from the solver's own `dbg_mg_inputs`. P and the final u are read after the
tick. Step 1d's mass flux is re-derived with the engine's integer face law.
Both re-derivations are checked bit for bit every tick: the face-form
divergence of the replayed u equals the cached div u, and the re-derived N
equals the cached N (column `ok` = `yy` on every row of every run).

The checkerboard of N changes only through the "staggered" part of the face
fluxes. A face flux F (lo → hi) changes Σ s·N by −2·s_lo·F, with
s = (−1)^(x+y). For the central part of step 1d's flux, that is the functional
Φ(u) = Σ_open faces s_lo·(u_lo + u_hi)/2. Φ is linear, so each step's velocity
increment carries its own share. The columns are:

- dcbN: step 1's change of cbN, split into its central and upwind parts.
- Φ(u0): what the tick starts with.
- dΦ_SL, dΦ_4: the shares of step 1a and step 4.
- kick: step 4's pressure kick, recomputed in float from the engine's stored P.
- kick_fP: the same kick from a float64 exact solve of step 3 on the engine's own solve inputs.
- rest: absorb, cap, drag and the integer kick's floors.

γ = 1.0001, 0.333 m, growth phase:

| tick | cbN | dcbN | central | upwind | Φ(u0) | dΦ_SL | dΦ_4 | kick | kick_fP | rest |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +0.101 % | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 | +3.17 | +3.22 | +0.14 | −0.05 |
| 2 | +0.101 % | −0.095 | −0.095 | −0.000 | 3.17 | −0.012 | +1.14 | +1.16 | −0.46 | −0.02 |
| 5 | −0.260 % | −0.176 | −0.178 | +0.002 | 6.16 | −0.048 | +0.73 | +0.86 | −1.08 | −0.13 |
| 10 | −1.262 % | −0.212 | −0.220 | +0.007 | 7.71 | −0.076 | +0.03 | +0.04 | −1.80 | −0.00 |
| 15 | −2.308 % | −0.198 | −0.212 | +0.014 | 7.36 | −0.079 | −0.04 | +0.01 | −2.26 | −0.05 |
| 20 | −3.219 % | −0.152 | −0.172 | +0.020 | 5.92 | −0.064 | −0.42 | −0.48 | −2.42 | +0.06 |

Shipped dials, steady state (ticks 100 to 240): dcbN fluctuates about 0. The
central part (−0.01 to −0.03 % per tick) and the upwind part (+0.015 % per
tick) cancel. The engine kick's Φ is 0 ± 0.2, while the float-solve kick's Φ
holds at −1.4 to −1.5. In the growth phase (shipped dials) cbT is unchanged
by step 1 and falls about 0.45 degrees per tick across steps 6 and 7.

What it says:

1. **The density pattern is built by step 1d's central face flux alone.** The upwind part damps it.
2. **The velocity that carries it comes from step 4's pressure kick.** Step 1a's SL advection and the absorb, cap and drag chain are an order of magnitude smaller, and they damp. The kick with the room-mean N in place of each cell's own N is the same to two decimals. So the 1/N coupling (N checkerboard × smooth gradient) is **not** the driver.
3. **The kick pushes the wrong way.** The kick derived from the exact solve of the same step-3 system pushes Φ negative, which is restoring (it removes the cbN of the tick before). The kick from the engine's integer P pushes Φ positive until cbN reaches about −4 %. At that point the two cancel.

### 2.5 Step 3 in isolation (`--solve-bias`)

The test-only `bp.eos_mg_solve_ref` runs the same `mg_build_levels` +
`mg_run_solve_cpu` as the live step. Here it runs on a synthetic still state:
p\* = N = 1 + a·s, T ambient, div u = 0, warm start at ambient. The table shows
Σ W·P in Q16 counts, where W is the kick-to-Φ kernel (§3.1):

| tile | σ² | solve | +8 % | +4 % | +1 % | +0.1 % | −0.1 % | −1 % | −4 % | −8 % |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 156 | int | −527 | −273 | −99 | −37 | −10.5 | +49.5 | +225.5 | +485 |
| | | float | −507 | −254 | −63 | −6.4 | +6.4 | +63 | +254 | +507 |
| 0.5 | 625 | int | −154 | −96 | −41 | −38.5 | −10 | −3 | +47.5 | +107 |
| | | float | −127 | −63 | −16 | −1.6 | +1.6 | +16 | +63 | +127 |
| 0.333 | 1409 | int | −74.5 | −47 | −36.5 | −11 | −5 | −8.5 | 0.0 | +24.5 |
| | | float | −56 | −28 | −7.0 | −0.7 | +0.7 | +7.0 | +28 | +56 |
| 0.25 | 2500 | int | −60.5 | −43 | −37.5 | −7 | −5 | −10.5 | −1 | +12.5 |
| | | float | −32 | −16 | −4.0 | −0.4 | +0.4 | +4.0 | +16 | +32 |

- The float signal follows the sign of the seed and scales as 1/σ². That is the restoring force.
- The integer signal carries an **offset of one sign, whatever the seed's sign**: −5 to −40 counts through the kernel, about 0.1 to 0.5 LSB per weighted cell. At 0.333 m it is larger than the whole physical signal until the seed is about −4 %.
- Its zero crossing sits where each dynamic run ends: about −0.3 % at 1.0 m, −1.2 % at 0.5 m, −4 % at 0.333 m and −5 % at 0.25 m. The dynamic p\* checkerboards (cbN plus cbT/T) are −0.48, −1.7, −3.8 and −6.8 %.
- The converged flat red-black path (`use_multigrid = false`, `S = 2000`) on the same inputs returns exactly 0 at every seed. Its integer P is uniform: no offset, and no restoring signal either. The same flat path at `S = 200`, not yet converged, still shows −47 at +1 %, so the flat result depends on running it to convergence.

### 2.6 Temperature: step 6 converts, at (γ−1)/dx

cbT/T_amb divided by cbN, from §2.1: 0.39 (1.0 m), 0.73 (0.5 m), 1.17 (0.333 m)
and 1.50 (0.25 m). (γ−1)/dx gives 0.40, 0.80, 1.20 and 1.60. At 1.0 m the
temperature pattern is the adiabatic image of the density pattern
(T ∝ N^(γ−1)). Below 1 m it is 1/dx times stronger, because of the units in
§3.4.

### 2.7 The warm room is more than the checkerboard

With the converged flat solve (diagnostic only) the checkerboard disappears
from the warm case: corr −0.02 to −0.08, cbN about 0. The room still leaves
its starting range: at 0.333 m it reaches −27 / +78 at 40 s, with gap 6.5 % rms
(uncorrelated) and wind up to 8.7 m/s. At 1.0 m it reaches −4 / +47. A second,
non-odd-even component of finding 1 exists, and it also grows as the tile
shrinks. Not diagnosed here.

## 3. Cause

### 3.1 Proved: the stencils, and where the odd-even mode couples at all

| step | stencil | file:line |
|---|---|---|
| 1a SL advection of u | backtrace −u·dt_s in **tiles**, bilinear | `cpp/src/eos_solver.cpp:822-841` (displacement :831-832, sampler :128-200) |
| 1d mass flux | face velocity `v_f = (u_i + u_j) >> 1` (the **arithmetic mean of cell velocities**), upwind donor N, coefficient perm·dt_s ("dx == 1 tile") | `cpp/src/bulk_transport.cpp:133-142`, :16-17; coefficient cache `eos_solver.cpp:785-803` |
| solve rhs div u | face form `(û_E − û_W + û_S − û_N)/dx`, `û = (u_i + u_j)/2`, 0 at a wall | `eos_solver.cpp:916-931`, `face2_u` :102-108 |
| 2 p\* | `C·N·T_abs` per cell | `eos_solver.cpp:960-973` |
| 3 operator | `m_i P_i + Σ_f g_f (P_i − P_j) = m_i (p* − γ p* dt div u)`, `g_f = perm/N̂_f`: **compact 5-point**, mass m = 1/aK, aK = γ p\* K dt²/dx² | `eos_solver.cpp:1821-1882` |
| 3 cycle | red-black GS on `(x+y)&1`, floor-rounded increment; 2×2 aggregation from the origin; **piecewise-constant prolongation**; sum restriction | smoother `:2036-2072` (colour :2043, increment :2066); restrict :2106-2124; prolong :2131-2141; V-cycle :2144-2156 |
| 4 kick | `(P[ir] − P[il]) / (2dx)`: **wide central**, `mirror_idx` returns SELF at a solid neighbour | `eos_solver.cpp:1059-1064`, `mirror_idx` :38-43; `u −= K dt ∇P / N̂` :1079-1088 |
| 6 energy face | `p_f` = eq. 15 (N-weighted cross average of the refreshed p), `u_f = u_lo + u_hi` (the same face mean), `k_flux = (γ−1) T_AMB dt_s / (2 dx)` | `eos_solver.cpp:1442-1464`, `k_flux_q` :741-742 |

The classic collocated arrangement holds:

- The mass flux and the divergence see only the face mean of cell velocities.
- The kick sees only the wide central difference of P.
- The solve's operator is compact.

The odd-even mode is therefore invisible in the interior:

- A wide difference of any P has no (π,π) component.
- The face mean of a (π,π) velocity is 0.
- The compact operator damps P's image of a p\* checkerboard by 1 + 8σ² (about 11 000 at 0.333 m).

The kernel W (P → Φ of its kick, `kick_phi_kernel`) shows the interior
blindness directly. Its interior entries are **exactly 0** (133 of the 289 gas
cells of the fixture). Only the two cell layers along each wall and around the
door carry weight (156 cells, alternating ±1 / ±½ along the wall). There, the
wall mirror in the kick and the missing wall face in the flux break the
cancellation.

**The only path from a density checkerboard back to the flow therefore runs
through the wall layers, at a strength of about p\*_cb/(6σ²).** At 0.333 m,
0.1 % of p\* checkerboard is 0.007 Q16 counts of P per wall cell. That is far
below one LSB.

### 3.2 Proved: what drives it

The forcing is a residue of the **multigrid cycle's integer result** in those
wall layers:

- It has one sign whatever the input (§2.5).
- It is tied to the absolute grid parity, not the geometry (§2.1).
- It does not depend on the cycle count (§2.3).
- It is absent from the converged flat red-black solve of the same operator (§2.3, §2.5).

Through the wall kernel and the kick, it gives a staggered velocity. The drag
barely damps that velocity, so it persists, and step 1d's central flux moves
air from the x+y-even tiles to the odd ones at a steady rate (§2.4). The
pattern grows until the physical wall-layer signal, p\*_cb/(6σ²), cancels the
residue. That is why the end level is the same for every seed, why it scales
as σ² = (c·dt/dx)² (c² and 1/dx² both measured), and why the zero crossings in
§2.5 match the dynamic end states.

### 3.3 Inference (labelled): where in the cycle the residue comes from

The parity-locked part most likely comes from the piecewise-constant
prolongation onto 2×2 blocks anchored at the origin, followed by only `mg_nu2`
= 2 floor-rounded red-black sweeps: a sub-LSB imprint with period 2 that the
post-smoother cannot remove. The flat path shares the smoother, the colouring
and the floor, and is clean. This was not separated further: prolongation,
restriction and the F8 mass anchor's truncation were not knocked out one by
one, since that needs C++ changes.

### 3.4 Proved: a units split between transport and the rest of the EOS

Step 1d moves air `u·dt_s` **tiles** per tick (its coefficient is perm·dt with
no 1/dx). The re-derivation with exactly that coefficient matches the engine's
N bit for bit at dx = 0.333 m. Step 1a's SL backtrace is in tiles as well.
Steps 1 (n_sub's CFL, `eos_solver.cpp:724-728`), 3, 4 and 6 use dx in metres,
and u is m/s by every other reader (the kick's K = c²/γ with a gradient per
metre; the wind-load research rows).

Two consequences are measured:

- The temperature image of a density pattern is (γ−1)/dx instead of γ−1 (§2.6).
- The mass that a velocity moves is dx times what the solve assumed when it set P.

**Inference:** the second consequence is a candidate for #72 item 3 (the
"pressure solve lags") and for the non-odd-even warm-room component (§2.7),
which is milder at 1.0 m, where the units agree. Not proved.

### 3.5 Inference: the resolution floor under every remedy that runs through P

One Q16 LSB of P, through the wide kick, is K·dt/(2·dx·65536) of wind: 0.057
m/s at 0.333 m (γ ≈ 1; 0.041 at γ = 1.4). The still-air winds in the bench are
exact multiples of it (0.054, 0.11, 0.16, 0.22 m/s, varying by 1/N). Any
restoring force carried by the stored P needs P's checkerboard image to reach
about one LSB:

- p\*_cb ≈ 8σ²/65536, about 17 % at 0.333 m, through a compact interior coupling.
- 6σ²/65536, about 13 %, at the walls.

The measured residue is a fraction of an LSB. Since the end level scales as
(residue in LSB) × σ², each extra fractional bit of P should halve it. That is
an extrapolation of the measured scaling, not a measurement.

## 4. Remedy options for Erik (NOT chosen)

The odd-even mode in this solver has two halves. The coupling is missing in
the interior (structural), and the coupling at the walls is too weak to beat a
sub-LSB integer residue (resolution). A cure must address the half it relies
on. In every option below, "goldens" means: `GOLDEN_AGGREGATE` re-baselined
once with written rationale, the pinned matches (`bots`, `bots_v1`)
re-recorded with their pins, and the CUDA lockstep with the twins changed in
the same patch (the Class P change merges only with `.cuda_green` == HEAD).

### (a) Rhie-Chow: a compact pressure term in the face velocity, cell winds kept

- **Changes.** The face velocity of step 1d becomes `(u_i+u_j)/2 − (K dt/N̂_f)·[(P_j−P_i)/dx − ½(∇P_i + ∇P_j)]`. Step 6's `u_f` must use the same face velocity (or Σ p_f û_f no longer vanishes where the solve zeroed the divergence: the D4 lesson in `face2_u`'s header). The solve's div u* should use it too, so the operator and the flux agree.
- **Twins.** `cuda_bulk_transport.cu`, `cuda_eos_step.cu`, `cuda_eos_resident.cu`, `cuda_kick_compression.cu`; the twin ledger rows `[twins.eos]`.
- **Closure identity.** Unchanged in form. Energy still rides the mass through `price_face`, inside `e_transport_net_sum`. Step 6 still telescopes face by face, so no new counter.
- **Wind readers.** Untouched (cell winds unchanged).
- **Expected effect on the bench (inference).** On today's Q16 P, **not a cure**. The interior restoring signal p\*_cb/(1+8σ²) stays below one LSB until p\*_cb is about 8 %, and the multigrid residue would then reach every interior face instead of only the wall layers, which could make the forcing worse. It needs (d) under it.
- **Risk.** Medium. The standard fix, but well-known pitfalls: dt dependence of the RC term, and its interaction with the absorb, sponge and drag stages that modify u after the kick.

### (b) Staggered (face-stored) velocities

- **Changes.** The kick becomes the compact face gradient, the flux and divergence read face velocities, and the odd-even null mode is gone by construction. Steps 1a (SL on two staggered grids), 1d, 3 (rhs), 4 and 6 all change.
- **Twins.** Every EOS kernel and the resident chain.
- **Wind readers.** Every one moves to an interpolation: `GameMap.footprint_wind_load` / `exchange.apply_wind_force`, `renderer/gas_detail.py::tame_wind`, vents and pumps through the solver's face fluxes, sensors, the recorder's `wind_x/wind_y`, the field digest (membership change, so a version bump and all goldens), the unit store and the device tables reading wind.
- **Expected effect (inference).** The same resolution floor as (a): the restoring force still runs through Q16 P. The structural half is cured, the resolution half is not, unless (d) comes with it.
- **Risk.** High. The largest change in the list.

### (c) A damper on the odd-even mode, driven by p\*

- **Changes.** A conservative face mass flux in step 1d proportional to a high-pass of the **state** pressure p\* = C·N·T_abs, for example a fourth difference (Jameson-Schmidt-Turkel style), which acts as k⁴ and so barely touches room-scale acoustics. It needs no new energy counter: the mass rides `price_face` like any other flux.
- **Why p\*.** p\* is full-resolution. A 0.1 % checkerboard is about 65 counts of p\*, against 0.007 of P. So the damper sees the mode at any amplitude, independent of σ and of P's LSB. The interior and the walls are treated alike.
- **Twins.** `cuda_bulk_transport.cu` and the two EOS orchestrations.
- **Expected effect (inference).** It removes the pattern at a rate set by its dial. Against the residue's forcing, the equilibrium amplitude falls as the dial rises.
- **Risk.** It is a dial, so a rate to rule on. It smears real grid-scale structure (a blast front's first ticks, a vent's jet edge), and it treats the symptom of the missing coupling, not the coupling.

### (d) Finer resolution for the solved pressure

- **Changes.** Carry P with more fractional bits through the solve and into the kick, for example int64 at the F8 working scale that the row products already use, with `atmosphere` still materialized as Q16 for its readers. The multigrid's integer ranges and `M_CAP`, the warm start and `cuda_mg_solve.cu` all change. If `atmosphere` itself changes currency, that is a field-digest version bump and every golden is regenerated (`.claude/rules/determinism-gates.md`).
- **Expected effect (extrapolated from the measured σ² scaling).** The end level shrinks by about 2 per extra bit: 8 bits would take 4 % to about 0.016 % at 0.333 m. The still-air wind quantum falls from 0.057 m/s to about 0.0002. Alone it keeps the interior blind: the mode is then only weakly held, through the walls.
- **Risk.** Medium. Range proofs in the multigrid, and the `p_face` ceiling and floor of step 6 read `atmosphere`.

### (e) Remove the multigrid's parity-locked residue

- **Changes.** The cycle's integer residue in the wall layers, for example by symmetric (round-to-nearest) smoother increments, a final fine-level sweep in reversed colour order, or a prolongation that does not imprint the 2×2 blocks. Which of these works is the open half of §3.3 and needs a measurement first.
- **Expected effect.** It removes the forcing. The mode is then **frozen**, not damped, as the converged flat solve shows (§2.3): a seed persists at its size and the stored and state pressures stay apart by it. `test_eos_checkerboard.py` would pass (it asserts "does not grow"). That is necessary, not sufficient.
- **Twins and risk.** `cuda_mg_solve.cu`. Low to medium. The pressure arc's convergence measurements (`mg_cycles = 8`, `docs/pressure_arc_root_cause_2026-08-17.md`) must be re-run.

### (f) The units ruling (§3.4): not a cure, but it decides the temperature factor

- **Changes.** Make steps 1a and 1d move `u·dt/dx` tiles (or declare wind tiles/s everywhere else).
- **Effect.** At 0.333 m air and smoke move 3 times faster for the same wind (visible in play: smoke, vents, fire spread). The temperature image of any density pattern drops by dx. The solve's P and the mass it moves agree again. It does not touch the odd-even forcing: 1.0 m, where the units agree, has the mode.
- **Risk.** Gameplay tuning (smoke look, vent rates, larvae, wind load) was done on today's transport speed. Every golden moves.

A combination is likely. (e) + (c) would stop the forcing and damp what
remains, cheaply, with cell winds kept. (d) + (a) would make the solver able to
see and restore the mode, the structural cure at a larger cost. (f) is
independent of either.

## 5. Open questions for Erik

1. **Is wind m/s?** If so, transport (steps 1a/1d) is 1/dx too slow and step 6's temperature response 1/dx too strong (§3.4). Rule on (f) first: it changes what every later measurement of this arc means. **Ruled 2026-11-20: wind is m/s. Patch U, §7.**
2. **Structural cure or damper?** Should the solver restore the odd-even mode physically, which needs finer pressure resolution plus a compact coupling ((d) + (a) or (b)), or is a p\*-driven damper on the mode acceptable ((c), with (e))? Only the first also fixes the 0.057 m/s still-air wind quantum.
3. **The target property.** Today's test asks "does not grow". Should the remedy also have to **damp** a seed (a stricter test), and at what rate?
4. **Scope.** The warm room has a second, non-odd-even component (§2.7). Does it belong in this arc or in its own?

## 6. Systems

**(a) Existing canonical systems a remedy must use** (CLAUDE.md index, rule
files `energy-books.md`, `fields-gas.md`, `determinism-gates.md`,
`device-residency.md`):

- **Face-flux energy step** (step 6): any new face velocity is the one step 6 prices, evaluated once per face in canonical orientation, telescoping.
- **Energy closure identity**: a new energy writer adds a counter and a term in the EOS group. A mass flux inside step 1d needs none, because its energy rides `e_transport_net_sum`.
- **Gas energy seam** / **Gas temperature is a mirror** / **Gas energy recovery**: nothing writes T. A damper moves mass, never temperature.
- **Interior drag**: no new velocity damping site. A velocity-space damper would have to be a stage of the staged drag block, with its KE booked.
- **Trace transport**: trace rides the bulk face flux, so a new mass flux carries smoke with it through `price_face`, with no second path.
- **PhysicsRunner / PhysicsEngine**: new dials bound in PhysicsRunner only, through `[physics.eos]`.
- **Fixed-point kits**: no new rounding or reciprocal.
- **Field digest**: a currency change of `atmosphere` (option d) is a version bump plus all goldens.
- **GOLDEN_AGGREGATE**: re-baselined once, with rationale.
- **A/B lockstep harness**: the refactor parts (e.g. moving P to finer units with no behaviour change) proved with it, never with whole-grid means.
- **Class P / Class T + the twin ledger**: every twin changed in the same patch, `.cuda_green` == HEAD.
- **CUDA harness**: `cuda_eos_step_check.py` and friends.
- **Pinned match plays**: re-recorded.
- **Benches**: `_sealedbox_bisect_bench.py --trace`, `_quiet_books_bench.py` and this bench before and after.

**(b) New reusable systems** (draft rules, to land in CLAUDE.md and
`determinism-gates.md` only when the arc's remedy patch makes them load-bearing):

- **EOS checkerboard bench** (`tests/_eos_checkerboard_bench.py`). Draft rule: *THE measure of the EOS odd-even mode: any change to steps 1, 3, 4 or 6 runs `--table` (the four tile sizes), `--steps` and `--solve-bias` before and after; never a second probe.* Its pieces `attribute_tick` (per-step split, self-checking bit for bit against the solver's caches), `float_solve` (the exact twin of step 3's operator) and `kick_phi_kernel` are reusable for any later EOS diagnosis.
- **The property test** `tests/test_eos_checkerboard.py`. Its strict xfail is removed by the remedy patch.

## 7. Patch U: transport in metres (wind is m/s), v2

Erik's ruling (2026-11-20): **wind is m/s.** Steps 1a and 1d move air `u·dt/dx`
tiles per tick, as every other reader of u already assumes. This is the arc's
first behaviour patch, before any remedy of §4. It changes no dial, no field,
no currency and no stencil: only the one scalar that turns a velocity into a
displacement in tiles.

v2 (2026-11-20) folds the two critiques of §7.8 into the text. Every decision
that answers a finding carries its id (L1-n = lens 1, L2-n = lens 2) and a
one-line reason. Line numbers are at `d2887dcb`, the branch after main's #221
and #97 W0/W1 came in (L2-5). **Re-grep at `d2887dcb`:** `git diff 017b80a3
d2887dcb -- cpp/ src/simulation` is empty, so every C++ and solver site of v1
holds at the same lines. Main brought two 0.333 m levels: `levels/test_arena`
(the tests' fixture since #221, a byte copy of today's playground) and
`levels/water_studio`. Neither adds a transport site; tests on `test_arena`
fall under the categories of §7.5.

"Proved" means read in the code or measured; "inferred" says so.

### 7.1 Inventory

**The bug: one missing factor, at every site that builds it.** Every transport
displacement and face coefficient in the EOS path is built from one scalar,
today `dt_s_q = quantize(dt / n_sub)`. Proved by grep: `dt_s_q` (and its
copies `dts_q_c` / `dts_q`) has no other reader in `cpp/src/`. The whole fix
is therefore to fold `1/dx` into that one scalar.

| # | site | today | bug? | change |
|---|---|---|---|---|
| 1 | `eos_solver.cpp:785-801`, the per-tick coefficient cache `coeffE_` / `coeffS_` of step 1d | `mul_q16(quantize(min perm), quantize(dt_s))`: no 1/dx, so `u·dt_s` **tiles** | **bug** | `quantize(dt_s)` → the fold (§7.2) |
| 2 | `eos_solver.cpp:806-807, 831-832`, the step-1a SL backtrace `bx_q = -mul_q16(u, dt_s_q)` | displacement `u·dt_s` **tiles** | **bug** | the same fold, computed once per tick beside the cache (the per-substep re-quantize of the same double goes) |
| 3 | `eos_solver.cpp:2189-2237`, `eos_sl_advect_reference` (`bp.eos_sl_advect_ref`) | tiles; the signature has no dx | test twin of 2 | takes `dx`, REQUIRED with no default (L1-5), uses the fold |
| 4 | `cuda_eos_step.cu:264-281, 362-368, 420, 577`, the per-call pre-stage (`eos_prestage_finish` sets `pre.dt_s_q`, which feeds the coefficients and the SL launch) | tiles | **bug** (device twin of 1 and 2) | the fold, in `eos_prestage_finish` (`solver.dx`) |
| 5 | `cuda_eos_resident.cu:923`, the fold-table row `r.dt_s_q`, read by `K_coeffs` (:230-257) and `cuda_sl_advection.cu:221` | tiles | **bug** (the resident and batched twin) | the fold, INSIDE the row's poisoning `try` (L1-4: a refused fold fails the env, the `k_flux_q` pattern). The table key (:910) already holds `solver.dx` |
| 6 | `cuda_sl_advection.cu:250-269`, the standalone per-call `eos_sl_advect` (`bp.cuda_eos_sl_advect`). Not on a game path: `sl_advection_backend_is_cuda()` is read only at `cuda_eos_step.cu:115` | tiles | test twin of 3 | takes `dx`, REQUIRED, uses the fold |
| 7 | `cuda_eos_step.h:92` and `cuda_resident_tick.h:81`: the field, documented as "the substep dt"; the parity check's printed name (`cuda_eos_resident.cu:1477, 1500`) | name and comment | none | renamed `dtdx_s_q` ("substep displacement per m/s, in tiles"), so no reader can take it for seconds. Kept although cosmetic: L1-1 offered dropping it to shrink the twin debt, but `eos_solver.h` changes anyway and is in the same twins' files, so dropping it saves nothing |
| 8 | `tests/_eos_checkerboard_bench.py:289-300` (`mass_flux_faces`, `dt_q = _q(dt)`; the comment at :251) and :414 (the `eos_sl_advect_ref` call) | the Python twin of 1 and 3 | twin | the fold as the C++ computes it: `float32` dt and dx, `/ n_sub`, `max(dx, 1e-6)` (L1-5); passes `dx=eos.dx`. Its `ok` column checks it bit for bit against the solver's N cache every tick |

**Callers of the two SL entries** (re-grepped; each passes a CHOSEN dx, L1-5):

- `tests/_eos_checkerboard_bench.py:414`, `tests/cuda_kick_check.py:818`: the solver's own `eos.dx` (they replay a live tick).
- `tests/test_thermal_mass_axis.py:764, 900-902, 1006-1008` and `tests/cuda_thermal_mass_eos_check.py:111-118`: `dx=1.0`. They compare two calls with the same scalar (mask on vs off, CPU vs GPU); 1.0 keeps their numbers bit for bit, so they keep testing what they tested.
- `tests/cuda_p62_check.py:117-123`: a parameter of `_run_pair`; its cases run at `dx=1.0` plus one at 1/3 m, so the CPU-vs-GPU lockstep covers a non-unit fold.

**Riding the fix, with no change of their own.** Proved by reading:

- The face-flux kernels (`bulk_transport.cpp:132-142`, `cuda_bulk_transport.cu:85-95`) consume the coefficient and do not depend on its units.
- Energy rides the mass through `price_face(dq, …)` (`bulk_transport.cpp:401-407`).
- Trace transport's stage-3b ride is `price_face(q, s, n)` (`bulk_transport.cpp:484-490`, `cuda_bulk_transport.cu:300-318`). **Smoke moves at the corrected speed automatically:** "trace moves only by riding the bulk face flux" holds, and there is no second coefficient.
- The SL sampler `eos_backtrace_sample3_q` (`eos_solver.cpp:128-200`) and its device twin march any displacement tile by tile and stop at walls. A longer displacement costs only march steps (§7.4) and is never out of range.

**Already in metres: not the bug, unchanged.** Proved:

- n_sub's CFL: `cfl_dx_q = quantize(CFL_ADV·dx)` (`eos_solver.cpp:724-728`; device `cuda_eos_step.cu:335, 358-361`, `cuda_eos_resident.cu:760`).
- The solve's rhs div u: `inv_2dx_q` (:536-537).
- The operator: `aK ∝ dt²/dx²` (:1746).
- The kick (:1063-1064).
- Step 6's `k_flux_q = (γ−1)·T_AMB·dt_s/(2dx)` (:742; `energy_flux_scalar_folds` on the device). This is the pdV work of an m/s velocity across a face of a tile measured in metres. It is right as written: the transport is what disagreed with it (§2.6). `step()` keeps `dt_s_d` for it (L1-6).

**Deliberately per tile, unchanged.**

- **Trace face diffusion**, `trace_diffusion_dd_q = quantize(d·dt)` (`bulk_transport.cpp:662-668`; the resident tail folds through the same function). The `[gases.*] diffusion` dial is **declared** "1/s, per tile²" (`config.toml:747`), and its stability door `d·dt ≤ 0.25` (`gases.py:169-200`) is in tile units. It is not the wind. **Settled for this patch: it stays per tile²** (§7.7).
- **The legacy entries `bp.bulk_flux_transport` / `bp.cuda_bulk_flux_transport`** (`bulk_transport.cpp:38-72`, `cuda_bulk_transport.cu:650-684`), which say "dx == 1 tile". Their only callers are `test_eos_p1_species_transport.py` and `cuda_bulk_flux_check.py`, both kernel seams, not the game. They stay unchanged: a caller that wants metres passes `dt/dx`. The header comment at `bulk_transport.cpp:16-22` is re-worded to say so, and gains the `flux_to_dq` range bound of §7.4.

**Wind readers outside the EOS.** All of them already read u as m/s. Proved by reading and by grepping `wind_x|wind_y` over `src/`, `renderer/` and `cpp/src/`:

| reader | units | effect of patch U |
|---|---|---|
| `GameMap.footprint_wind_load` (`gamemap.py:2423`) → `exchange.apply_wind_force`; on the device, `cuda_unit_impulse.cu` (`[twins.wind_impulse]`, class T) | m/s (`N·u²`) | The law is unchanged. The winds a unit meets change, because the flow changes |
| `fire_simulation.cpp:257` (W = \|u\| in the `k_wind_fan` / `k_wind_strip` terms) | m/s | The law is unchanged |
| `renderer/gas_detail.py::tame_wind` / `wind_tiles_per_tick` (:117-160): the smoke relief, prop sway, flame VFX, particles and (since #97 W1) the water's flow take its one array | m/s, converted honestly as `v / tile_size_m / tps` (#51) | The law is unchanged. **Its calibration `_WIND_VREF = 1.4`, `_WIND_VSENS = 3.0` (:90-91) goes stale:** it was fitted with `tools/wind_drift_probe.py` (2026-10-19) to a plume that drifted at about 0.35-0.5 × the honest \|u\| (:89). The re-fit is a separate render-only follow-up patch (L2-7), not this one |
| `obs_spec_v1.py:147-164` (the observation's `wind_x`/`wind_y`, `cap_q`) | m/s | none |
| swarm (`swarm.cpp`, `swarm.h`); pumps and vents (`pump_system.py`, `vent_system.py`, the device PUMPS and VENTS stages) | do not read wind (grep) | none. Vent mass enters through the pump primitives; how it then mixes through the room is the corrected transport |
| `atmosphere_solver.cpp` | the pre-EOS wave solver | not on the tick (`physics_runner.py:500-508`); out of scope |

### 7.2 The fold and the oracle

**One definition.** The scalar is computed by one function, declared in
`eos_solver.h` and defined once out of line in `eos_solver.cpp`, on
`/fp:strict`. This follows the `trace_diffusion_dd_q` precedent: one
definition, both backends. All three device TUs already include the header
(`cuda_eos_step.cu:31`, `cuda_eos_resident.cu:48`, `cuda_sl_advection.cu:23`).

```cpp
// Substep displacement per (m/s), in tiles: dt_s/dx, quantized ONCE.
q16 eos_transport_step_q(float dt, int n_sub, float dx) {
    const double dx_d   = std::max((double)dx, 1e-6);   // == step()'s dx_d
    const double dt_s_d = (double)dt / (double)n_sub;   // == step()'s dt_s_d
    const q16 r = quantize(dt_s_d / dx_d);
    if (r < 0 || r > FP_ONE) throw std::runtime_error(...);   // range guard
    return r;
}
```

**Who calls it.** All six fold sites: `step()` (once per tick, replacing
`dts_q_c` and the per-substep `dt_s_q`), the SL reference, the per-call
pre-stage (`eos_prestage_finish`), the fold table (inside its `try`), and the
standalone SL entry.

**Why this form.**

- It is the house idiom: a per-tick host double fold, quantized once, like `k_flux_q` and `K_raw`.
- It is not `mul_q16(dt_s_q, quantize(1/dx))`, which would round twice at every dx ≠ 1.
- It carries an always-compiled range guard in the `k_flux_q` idiom (:751): a result above `FP_ONE` (one tile per m/s per substep: a tile under about 4 cm at 24 tps) is refused. **At 1.0 m it is a behaviour change only for dt_s > 1 s** (L1-4), which no caller reaches: the game's dt is 1/tps and the largest test dt is 1/24 s (grep). Accepted: it is the range proof's price, and a refusal is loud.
- Zero stays legal: `test_eos_div_face_form.py`'s `TINY_DT` relies on a zero displacement, and `quantize(1e-6/0.333) == 0`.
- (L1-6, L2-10) v1's float-ratchet argument is dropped: these TUs are not in `SIM_TUS`, so the ratchet does not count them either way.

**Bit identity at dx = 1.0 m.** Proved by arithmetic:

- `(double)1.0f == 1.0`, and `std::max(1.0, 1e-6) == 1.0`.
- IEEE division by 1.0 is exact, so `dt_s_d / 1.0 == dt_s_d`, and the one `quantize` returns today's integer.
- Every downstream kernel consumes only that integer, through unchanged code.
- So every field, book and digest at 1.0 m is unchanged bit for bit.

**The oracles that must not move.**

- `tests/field_ab_harness.py::_scenario_level` is a 16×16 room at `tile_size_m = 1.0`; `default_scenario_sim` puts fire, smoke, water, a wave pulse, a unit and a hull breach to vacuum in it, so wind is non-zero and venting runs. It is the scenario `tests/_xarch_perfield_digest.py` hashes as `GOLDEN_AGGREGATE`. **`GOLDEN_AGGREGATE` must not move:** a changed hash at 1.0 m means a fold site was transcribed wrong, and the code is fixed, never the pin.
- The box match pins `_PRE_122_BOX` (`test_zombie_control.py:105`), `_STAGE1_BOX` (`test_horde_behaviours.py:199`) and `_RE_PIN_1_BOX` (`test_turning_law.py:728`, `test_unit_impulse.py:1049`) all play `test_zombie_control._box`, which writes `tile_size_m=1.0` (:74, proved; L2-1 had it inferred for three). They are 1.0 m oracles with physics on and must not move either.
- The swarm scenarios use the same 1.0 m level, and about 160 more test files build theirs at `tile_size_m = 1.0`.
- (L2-10) `test_b1_signal_bus` / `test_b2_nodes` stay green because they run no physics, not because they are 1.0 m oracles; v1 miscounted them.

### 7.3 What moves at 0.333 m

**Goldens and pins.**

- **Field digest:** no version bump. Membership, dtypes and currencies are unchanged.
- **`GOLDEN_AGGREGATE`:** unchanged (§7.2).
- **The ship_demo match pins** (0.333 m) move for the right reason: air, smoke and the winds on units change. Eight of them (L2-1), re-pinned once each with a dated "#217 patch U (2026-11-20)" comment: `test_bots_v1.py:503` (`_PRE_G8P_SHIP_DEMO`), `test_zombie_control.py:225` (`_PRE_122_SHIP_DEMO`, also `tools/match_pin_check.py`'s), `test_exposure.py:588`, `test_grenade_kinds.py:667`, `test_horde_behaviours.py:52`, `test_turning_law.py:563`, `test_turn_rate.py:317`, `test_unit_impulse.py:883`. The shared pinned plays' self-checks (the bots_v1 G8p pin and the ship_demo controller pin) follow with them.
- **Not moved in this patch** (L2-8): `docs/rl_agent_rung1_1d_pins/*.json` are tournament records that no test reads (only a docstring of `test_team_picture.py` names them), so they go stale as records, not red.
- **Other 0.333 m tests** (on `ship_demo`, `test_arena`, the studios): §7.5's categories decide.

**CUDA.** This is a Class-P change. The patch touches `eos_solver.cpp`/`.h`,
`cuda_eos_step.cu`/`.h`, `cuda_eos_resident.cu`, `cuda_resident_tick.h`,
`cuda_sl_advection.cu`/`.h`, `bulk_transport.cpp` (comment) and
`bindings.cpp`, all in one patch.

- **Twin debt** (L1-1, L2-2): `eos_solver.h`, `cuda_eos_step.h`, `cuda_resident_tick.h` and `bulk_transport.cpp` are `files` of many twins, so the debt is not `[twins.eos]` alone. The critics counted 12, then 14 (eos, water, trace_tail, div_source, energy_flux, radiation_sweep, temperature, fire, stamp, water_tail, swarm_larva, combustion, edit_queue as P; stamp_rows as T). Decision: **trust `tools/twin_debt.py`** and mark every twin it lists, in the LAST commit (`--mark-level`).
- **The order** (L2-3): code → re-pins → `tools/twin_debt.py --mark-level` (last commit) → `cpp/build_cuda.bat` in this worktree → `tools/cuda_green.py` → the full suite → `tools/match_pin_check.py --backend cuda` and `--backend resident` against the new pin. A commit after `cuda_green` voids the stamp and restarts at mark-level.
- The lockstep gates stay as they are, because both sides change together: `cuda_s8a_check` (N = 1 == CPU, tol 0, built at 1/3 m), `cuda_batch_invariance_check`, `cuda_eos_step_check`, `cuda_p62_check` (now with a 1/3 m case).
- `cuda_prestage_parity_check` compares the scalar by its C++ printed name, which follows the rename.
- The device conductor (class T) runs slot 7 through `_resident_chain` and needs nothing of its own.

**The checkerboard bench** (`--table`), information only.

- The 1.0 m row is identical to §2.1 (proved, by the arithmetic of §7.2).
- Below 1 m, cbT/T_amb ÷ cbN should fall from (γ−1)/dx to γ−1 ≈ 0.40 at every tile size (inferred, §2.6).
- Whether the cbN end level moves is not predicted.

**Dials tuned against the old transport at 0.333 m** (listed, NOT changed;
the material for one tuning session with Erik after this patch, §7.7):

- `[physics.eos] k_drag = 0.01`, `k_drag2 = 0.225`: tuned by eye in `smoke_light_studio` at 0.333 m (Erik, 2026-10-06) for how far a puff coasts. A puff now covers about 3× the distance for the same wind. `test_drag2_venting_gate.py` leg 2's strict xfail belongs to this dial's note (L2-4): if it XPASSes, that is Erik's.
- `[gases.smoke] decay = 0.0011` (Erik, 2026-10-22, the #12 fire + smoke session), and the `decay` of steam, poison, teargas and fuel_gas: same lifetime, about 3× the travel.
- `[physics.combustion] soot_yield = 0.5` and the `[payloads.*] soot_yield` rows: the same smoke, spread wider.
- (L1-3) `[physics.combustion] burn_rate` and `draw_r`: P-O2b widened the O2 draw while resupply ran at a third of its speed.
- (L1-3) `[blast] expansion_m3_per_strength` and `heat_mj_per_strength`: calibrated at 0.333 m; after U the same divergence moves three times the mass at the same u (up to the 64 m/s ceiling of §7.4).
- `[physics.fire] k_wind_fan = 0.5`: the law is in m/s, but the winds at a fire change.
- `[levelgen.vents] ach = 10.0` and the vent entities' rates: the mass rate is unchanged, the room mixes faster.
- `[physics] smoke_sink_strength` and `smoke_vent_hops`: the breach-sink hops, if still live.
- `[render.gas_detail] adv_gain = 1.0`, `[render.props] wind_ref = 0.08`, and the code constants `gas_detail._WIND_VREF` / `_WIND_VSENS`: re-measured with `tools/wind_drift_probe.py` in the render follow-up (L2-7), not tuned.
- `[loadouts.*] wind_push_mult` (the winds on units).

### 7.4 Tick cost

**For a given wind field, cost is unchanged by construction.** Proved:

- The n_sub formula is already in metres (§7.1) and is untouched.
- Each substep does the same operations with a different scalar.
- Today the count is in surplus. Below the cap the transport moves at most `CFL_ADV·dx` = 0.167 tiles per substep at 0.333 m, so the count was 3× what tile-unit transport needed. After the fix each substep moves at most 0.5 tiles, the CFL the count was designed for. The fix spends the surplus and adds no substeps.

**The cap regime** (n_sub = `N_SUB_MAX` = 8, reached when u_est > 4·dx/dt: 32 m/s at 0.333 m, 96 m/s at 1.0 m):

- The per-substep displacement grows by 1/dx: up to 15.6 tiles at U_MAX, against 5.2 today.
- The SL march costs `ceil(|b|)` steps, only in those cells.
- The donor-cell flux is limited to one cell's mass per substep. Real mass speed therefore saturates at 1 tile per substep, 64 m/s at 0.333 m. **That is the same ceiling as today**: it is now reached at u = 64 m/s instead of u = 192. Above it (blasts) the solve/mass mismatch of §3.4 survives (L1, noted).
- A per-substep Courant number above 1 under the outflow limiter is not a new regime: 1.0 m levels already run it above 192 m/s. Positivity and mass exactness hold by the limiter.
- Decision: `N_SUB_MAX` stays 8 (Erik's frozen perf dial).

**Range** (inferred). `flux_to_dq` narrows to int32. At the cap, |v·dt_s/dx|
≤ U_MAX·dt/(8·dx): 15.6 at 0.333 m and 20.8 at 0.25 m, against 5.2 today at
any dx. It therefore wraps only above a donor N of about 2100 (0.333 m) or
1575 (0.25 m) ambient densities, against 6300 today. Bulk N in play is
O(1-10). The bound goes into `flux_to_dq`'s header beside the guard of §7.2.

**Measured** (information): ship_demo's CPU "7 EOS run_substeps" ms
(`tools/bench_resident_scaling.py --run --backend cpu`, 24 warm-up + 96
ticks), before and after, in the As-built paragraph.

### 7.5 The property test, the reproduction, the categories and the gates

**The E2E reproduction** (L2-6). `tools/wind_drift_probe.py` on
`smoke_light_studio` (168 ticks) and `fire_studio` (frag at (24,16) on tick
24, 120 ticks), before and after. The probe's honest velocity is the m/s plane
converted at `1/tile_size_m/tps`; the plume's centroid follows it by
continuity where no source adds smoke. The figure of merit is the median over
source-free windows of centroid drift ÷ |plume-weighted mean u| (a scratch
reader over the probe's own `probe` / `report`): today about 1/3 (dx), after U
about 1. Measured before:

| level | windows | median drift ÷ \|mean u\| | IQR |
|---|---|---|---|
| smoke_light_studio | 125 | 0.451 | 0.321-0.972 |
| fire_studio (frag) | 90 | **0.339** | 0.333-0.344 |

The fire_studio plume, a clean drift after the blast, sits at 0.333-0.344:
the 1/dx of §3.4, reproduced as a player sees it (smoke moves at a third of
the wind).

**New test: `tests/test_transport_in_metres.py`** (L1-2). Both legs measure
through the LIVE `step()` (`PhysicsRunner.step`, then the solver's own
`dbg_mg_inputs` caches, written after step 1 and before the kick), at tile
sizes 1.0 and 0.5 m in a sealed room. No reference entry is involved, so
dropping the fold at either live site (the step-1d coefficient cache or
`step()`'s own SL backtrace) fails it.

- **Mass leg:** uniform air, uniform wind U along x. The mass crossing a mid-room face line, per metre of line per second, Σ_downstream(N1−N0)·dx²/(L_m·dt), is N·U at both tile sizes (within Q16 rounding and the substep split). Breaks if the 1/dx is dropped or doubled at the coefficient: the flux then scales with dx.
- **SL leg:** u_x = U + a·step(x) (a small front, a ≪ U). Step 1a's SL moves the front by U·dt **metres**; it is located, at sub-tile precision, as the centroid of the solver's own `div_u` cache (∂u_x/∂x) over a window around it. Breaks if the fold is dropped or doubled at `step()`'s backtrace: the front then moves U·dt tiles.
- It pins no count and no hash.

**Re-pin categories** (L2-4). A red after the change is classified into one of:

| cat. | what | members (named by the critics; others classified as they turn red) | action |
|---|---|---|---|
| A | match pins that move for the right reason | the eight ship_demo pins of §7.3 | re-pin once, dated comment, rationale |
| B | thresholds measured at 0.333 m with margins | `test_drag2_venting_gate.py` legs 4/5, `test_air_boundary.py` reflection / refill gates, others on `test_arena` / studios | re-measure, restate the threshold with a dated rationale; the property stays |
| C | Erik-ruled properties | `test_wind_doorway_e2e.py`, `test_wind_studio.py:101`, `test_gen_ship_vents_air.py` `PRESSURE_BAND` | NEVER re-pinned: red = a finding for Erik, left red and reported |
| D | strict xfails that may XPASS | `test_drag2_venting_gate.py` leg 2, `test_eos_checkerboard.py` | left as they are, reported to Erik |
| E | transport-independent, must stay green | `test_trace_resolution`, `test_sky_exchange`, `test_explosion`, `test_fire_vfx_ref`, every 1.0 m oracle of §7.2 | red = a fold site is wrong: fix the code |

Any red that fits none is left alone and reported.

**Before merge:** the full suite; `_xarch_perfield_digest.py` with
`GOLDEN_AGGREGATE` unchanged; the box pins unchanged; `tools/cuda_green.py`;
`match_pin_check.py --backend cuda` and `--backend resident`; the energy
gates of `energy-books.md` (`test_no_transport_mint`,
`test_gas_energy_tile_flip`, `_sealedbox_bisect_bench.py`,
`_quiet_books_bench.py`); this bench and the probe, before and after.

**What goes stale by design** (L2-7, L2-8; recorded, not fixed here):

- Replays recorded before U at 0.333 m hit `ReplayDivergence` (the format is unchanged, `FORMAT_VERSION` stays 8: the re-simulation differs, which is what the check is for).
- RL runs, saved soldiers and imitation records made before U are on the old physics: a resume needs `--allow-code-change`, and the game-ai-lead role is told.
- `tests/data/fire_smoke_studio_flame_inputs.npz` (a dump of the studio's flame inputs) is re-dumped in the tuning session.
- `tools/look_shots.toml` is not rendered in this patch: U changes no render code, so the before/after look comparison belongs to the render follow-up (L2-7), whose "before" is after-U.

### 7.6 Systems addendum

**(a) Existing canonical systems the patch uses:**

- **Fixed-point kits** (`fields-gas.md`): one `quantize` of one host double. No new rounding, shift or reciprocal.
- **PhysicsRunner / PhysicsEngine** (`sim-core.md`): `eos.dx` is already bound from `gmap.tile_size_m` (`physics_runner.py:1063, 1571`). No new dial, no Python glue.
- **Trace transport** (`fields-gas.md`): smoke keeps riding the bulk face flux. No second coefficient.
- **Face-flux energy step** and **Energy closure identity** (`energy-books.md`): step 6 and `price_face` are untouched, and energy rides the corrected dq inside `e_transport_net_sum`. No new counter.
- **Batched resident engine** (`device-residency.md`): the fold table stays the one per-config table, already keyed on dx. No new launch, no new sync.
- **Class P / Class T + the twin ledger**, **CUDA harness**, **A/B lockstep harness** and **GOLDEN_AGGREGATE** (`determinism-gates.md`): §7.2-7.3.
- **Tamed wind (render)** (`render-effects.md`): the law stays; its measured calibration is re-run in the follow-up.
- **Pinned match plays**: re-recorded.
- **Benches**: this bench, `bench_resident_scaling.py` and `wind_drift_probe.py`. Never a new probe.

**(b) New reusable systems:** none. `eos_transport_step_q` is a helper of an
existing system. **Its rule line goes where it fires** (L2-9): `fields-gas.md`,
beside Trace transport, as its own subsection *Transport step*: *step 1's
displacement and face coefficient come only from `eos_transport_step_q`
(dt_s/dx, quantized once), on every backend; wind is m/s everywhere (Erik,
2026-11-20).* `fields-gas.md`'s `paths:` gain `cpp/src/eos_solver.h`,
`cpp/src/cuda_eos_step.cu`, `cpp/src/cuda_eos_resident.cu` and
`cpp/src/cuda_sl_advection.cu`, the files holding the device folds, so the
rule fires on every fold site. No CLAUDE.md index line (a helper, not a
system).

**(c) Stubs** (L2-11): `stubs/breach_physics.pyi` is regenerated from the
CUDA build (`docs/dev_setup.md`'s stubgen line), so `cuda_eos_sl_advect` and
`eos_sl_advect_ref` show the required `dx`.

### 7.7 Open questions for Erik (settled for this patch by the brief)

1. **Re-tune in patch U, or after it?** Settled: after. Patch U lands with every dial of §7.3 unchanged, as a pure units fix with green gates. Then one fire + smoke tuning session at the screen (`fire-smoke-tuning` skill) over §7.3's list. Until that session, smoke at 0.333 m travels about 3× further per second.
2. **Is gas diffusion m²/s too?** Settled for this patch: it stays per tile² (identical at 1.0 m either way; one look change at a time). If it should be m²/s, that is its own small patch: `d·dt/dx²` in `trace_diffusion_dd_q` and in the gases stability door, with the shipped values rescaled by 0.111 to keep today's 0.333 m look.
3. **The render re-fit** of `_WIND_VREF` / `_WIND_VSENS`: its own render-only patch right after U, with the look shots before and after.

### 7.8 Critique (one critic at a time, verdicts recorded before the next)

| lens | model | verdict | findings (to resolve in v2 of §7) |
|---|---|---|---|
| 1. Completeness and correctness of the units change | Opus | SOUND WITH FIXES | (1) PROVED: twin debt is 12 twins, not `[twins.eos]` alone: `eos_solver.h`, `cuda_eos_step.h`, `cuda_resident_tick.h` and `bulk_transport.cpp` are `files` of eos, water, trace_tail, div_source, energy_flux, radiation_sweep, temperature, fire, stamp, water_tail, swarm_larva (P) and stamp_rows (T); mark all level, or drop the cosmetic rename and comment reword to shrink it. (2) PROVED: §7.5's test pins the SL leg through `eos_sl_advect_ref`, not `step()`'s backtrace (`eos_solver.cpp:831`); dropping the fold there passes it. Measure the SL leg through the live step (e.g. a wind step located in metres via `dbg_mg_inputs` div_u at 1.0 and 0.5 m); state the mass leg as Σ(N1−N0)·dx²/(L_m·dt). (3) PROVED config, INFERRED effect: §7.3's dial list misses `[blast] expansion_m3_per_strength` (calibrated at 0.333 m; 3× the mass at the same u after U), `heat_mj_per_strength`, `[physics.combustion] burn_rate` and `draw_r` (P-O2b widened the draw while resupply ran at 1/3 speed). (4) INFERRED, low: the range guard must sit inside `ensure_fold_table`'s poisoning try (`cuda_eos_resident.cu:919-932`), the `k_flux_q` pattern, and the guard is a behaviour change at 1.0 m when dt_s > 1 s (unreachable today). (5) PROVED, low: the bench twin must use `np.float32(dx)` and n_sub; `dx` on `eos_sl_advect_ref` / `cuda_eos_sl_advect` is required with no default; synthetic callers (`test_thermal_mass_axis.py:764/900/1006`, `cuda_p62_check.py:117`) need a chosen dx. (6) PROVED, info: the float-ratchet argument is moot (the TUs are not in `SIM_TUS`), and `step()` keeps `dt_s_d` for `k_flux_q`. Checked sound: no missed transport site; bit identity at 1.0 m on all four paths; the range and n_sub claims; the outflow limiter at any Courant; trace diffusion per tile² breaks nothing. Noted, not breakage: above 64 m/s at 0.333 m (blasts) mass stays capped at 1 tile/substep, so the solve/mass mismatch survives in that regime. |
| 2. Gates, regression, rollout, systems reuse | Opus | SOUND WITH FIXES | (1) PROVED: eight ship_demo (0.333 m) match pins move for the right reason, not two: `test_bots_v1.py:503`, `test_zombie_control.py:225` (also `tools/match_pin_check.py`'s), `test_exposure.py:588`, `test_grenade_kinds.py:667`, `test_horde_behaviours.py:52`, `test_turning_law.py:563`, `test_turn_rate.py:317`, `test_unit_impulse.py:883`; each gets a dated "#217 patch U" re-pin comment. The box pins (`_PRE_122_BOX`, `_STAGE1_BOX`, `_RE_PIN_1_BOX`) are 1.0 m oracles that must NOT move (1.0 m inferred for three). (2) PROVED: mark 14 twins level (adds combustion and edit_queue to lens 1's twelve). (3) PROVED: this PC has an RTX 3060 + nvcc 12.3; order is code → re-pins → `--mark-level` ×14 last → `cpp/build_cuda.bat` in the patch worktree → `tools/cuda_green.py` → full suite → merge from that checkout; plus `tools/match_pin_check.py --backend cuda` and `--backend resident` against the new pin. (4) Tests at 1/3 m with measured margins need a category each: re-measure and restate with rationale (`test_drag2_venting_gate.py` legs 4/5, leg 2's strict xfail XPASS goes to Erik, add it to the `k_drag2` dial note; `test_air_boundary.py` reflection/refill gates; `test_eos_checkerboard.py` XPASS goes to Erik); Erik-ruled properties where red is a real finding, never re-pinned (`test_wind_doorway_e2e.py`, `test_wind_studio.py:101`, `test_gen_ship_vents_air.py` `PRESSURE_BAND`); transport-independent, must stay green (`test_trace_resolution`, `test_sky_exchange`, `test_explosion`, `test_fire_vfx_ref`). (5) PROVED: the branch is behind origin/main (#221 test_arena, #97 water_studio); bring it up to main and re-grep. (6) Rule: the E2E reproduction is missing; `tools/wind_drift_probe.py` (plume drift ≈ 0.35-0.5·|u|) on smoke_light_studio and fire_studio, before and after, is the reproduction and the acceptance. (7) `_WIND_VREF`/`_WIND_VSENS` are measured fits, not Erik dials: a named render-only follow-up patch right after U; render `tools/look_shots.toml` before and after; `tests/data/fire_smoke_studio_flame_inputs.npz` goes stale (re-dump in the tuning session). (8) Replays recorded before U at 0.333 m hit `ReplayDivergence` by design (`FORMAT_VERSION` stays 8); RL runs, saved soldiers and imitation records made before U are on old physics (resume needs `--allow-code-change`; tell the game-ai-lead role); tournament pins `docs/rl_agent_rung1_1d_pins/*.json` move. (9) PROVED: the rule line must fire where the device folds live: `fields-gas.md`'s `paths:` miss `eos_solver.h`, `cuda_eos_step.cu`, `cuda_eos_resident.cu`, `cuda_sl_advection.cu`. (10) §7.2 slips: the float-ratchet claim is vacuous; `test_b1`/`test_b2` stay green because no physics runs; `bp.EOSSolver()`'s default dx = 0.333 so its direct users change too (books-only asserts). (11) `stubs/breach_physics.pyi:1272,1348` regenerated; `dx` required on the SL entries. Scope: one patch is right. |
