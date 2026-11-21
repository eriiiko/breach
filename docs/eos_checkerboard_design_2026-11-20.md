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
as the first bullet of its Trace transport section (as built: a section of its
own is refused by `test_claude_rules_paths.py`, which wants every rule section
to name a CLAUDE.md index row, and this helper has none): *step 1's
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

### 7.9 As-built (2026-11-20)

**What was built** (branch `217-eos-checkerboard`, after section 7 v2 `f91aac41`):

- `8b56b151` the code: `eos_transport_step_q` (`eos_solver.cpp`, out of line, range-guarded to [0, FP_ONE]) at all six fold sites; `dt_s_q` renamed `dtdx_s_q` in `EOSHostPrestage` and `EnvSub`; `dx` required on both SL test entries; `flux_to_dq`'s header states the coefficient's units and the int32 bound; the bench twin transcribes the fold. One deviation from v2: `cuda_p62_check.py`'s synthetic case `(40, 40, dt 2.0, wind 4)` became `(dt 1.0, wind 8)` (the same 8-tile march), because the new guard refuses dt_s/dx = 2 at dx = 1.0: the L1-4 behaviour change, met in a synthetic test, never in the game.
- `be1ac33f` `tests/test_transport_in_metres.py` (both legs through the live step).
- `db612102` the rule line, as the first bullet of `fields-gas.md`'s Trace transport section (a section of its own is refused by `test_claude_rules_paths.py`), with `eos_solver.h` and the three device fold TUs on its `paths:`.
- `0d2884d7` the eight ship_demo re-pins (category A); `84fc8d37` drag2 leg 4 restated (category B); `35b1e479` the stubs regenerated from the CUDA build (the regen also picks up bindings added since #63's last one).

**The oracles held.** `GOLDEN_AGGREGATE` unchanged on the CPU build
(`d5d03433...487b`, matches golden). The 1.0 m box pins `_PRE_122_BOX`,
`_STAGE1_BOX`, `_RE_PIN_1_BOX` (x2) pass unchanged. Before the twin marking,
on this worktree's CUDA build: `test_cuda_p62_sl_advection` (with the new 1/3 m
cases), `test_cuda_prestage_parity`, `test_cuda_s8a_residency`,
`test_cuda_batch_invariance` and `test_cuda_eos_step` green. The energy closure
is exact: `_quiet_books_bench.py` 1440 ticks EXACT (gas + solid),
`_sealedbox_bisect_bench.py` 432 ticks EXACT in every variant.

**The property test catches the fold being dropped** (verified, not assumed):
on the pre-U build the mass leg reads 0.4998 x N*U and the front moves 0.128 m
at 0.5 m tiles (expected 1.0 and 0.254 m; 1.0 m reads 1.0001 and 0.256 m on
both builds); a mutant that drops the fold only at `step()`'s own SL backtrace
fails the SL leg (0.128 m) and passes the mass leg (1.0001).

**The E2E reproduction, before and after** (`tools/wind_drift_probe.py`;
median over source-free windows of centroid drift / plume-weighted mean u):

| level | before | after |
|---|---|---|
| smoke_light_studio (168 ticks) | 0.451 (IQR 0.321-0.972, 125 windows) | **0.924** (IQR 0.699-1.241, 131 windows) |
| fire_studio, frag at (24,16) tick 24 (120 ticks) | 0.339 (IQR 0.333-0.344, 90) | **0.965** (IQR 0.890-1.029, 90) |

The plume now drifts at about the wind speed, as the ruling says.

**The checkerboard bench `--table`** (information; cb case, amp 0.1 %, 20 s; before / after):

| tile | gap_rms % | corr | cbN % | cbT | Tmin | Tmax | u max | cbT / cbN |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 0.577 / 0.577 | -0.76 / -0.76 | -0.346 / -0.346 | -0.39 / -0.39 | -7.97 / -7.97 | 5.99 / 5.99 | 0.228 / 0.228 | 1.13 / 1.13 |
| 0.500 | 1.804 / 1.962 | -0.91 / -0.76 | -0.975 / -1.193 | -2.10 / -1.31 | -4.98 / -4.82 | 4.67 / 8.83 | 0.488 / 0.369 | 2.15 / 1.10 |
| 0.333 | 4.024 / 3.985 | -0.91 / -0.73 | -1.753 / -2.347 | -5.99 / -2.81 | -16.39 / -13.25 | 27.93 / 25.60 | 0.736 / 2.114 | 3.42 / 1.20 |
| 0.250 | 7.159 / 7.188 | -0.92 / -0.66 | -2.727 / -4.019 | -12.01 / -4.72 | -24.97 / -19.14 | 25.23 / 31.94 | 0.811 / 2.399 | 4.40 / 1.17 |
| 0.333, gamma 1.0001 | 3.875 / 3.994 | -0.94 / -0.72 | -3.737 / -3.259 | -0.00 / -0.00 | | | 0.440 / 2.876 | |

The 1.0 m row is identical (bit identity). The temperature image of the
density pattern, cbT/cbN, grew like 1/dx before (2.15, 3.42, 4.40) and is now
the same at every tile size (1.10-1.20), as section 2.6 inferred. The
checkerboard itself is not cured (gap_rms the same size; cbN grew at 0.333 and
0.25 m): patch U was never a remedy for it (section 4 (f)).

**Cost** (CPU, ship_demo, 24 + 96 ticks, information): tick 19.23 -> 18.68 ms,
"7 EOS run_substeps" 15.59 -> 15.04 ms. The flow differs, so it is not a
like-for-like substep count; within noise of "no cost".

**Reds left red** (the categories of section 7.5; none re-pinned):

- C (Erik-ruled): `test_wind_doorway_e2e.py` x4 (its setup no longer reaches the asserted over-pressure: 1.146 atm vs > 1.3), `test_gen_ship_vents_air.py::test_no_vented_room_leaves_its_pressure_band` (mess room 2.35 % vs the 2 % band).
- D: `test_drag2_venting_gate.py` leg 2 at k2 = 0.25 XPASSes its strict xfail (ratio 1.2222 < 1.5); k2 = 0.5 / 1.0 stay xfail. `test_eos_checkerboard.py` stays xfail.
- Not classifiable by section 7.5, left and reported: `test_air_boundary.py::test_ambient_gate2_rail_counts_the_acoustic_overshoot` (at k_drag2 = 0 the T_MAX_PHYS rail fires ~600 times a tick for ~40 ticks, 24668 in all, where the property is a one-time transient; at the shipped k_drag2 it is 48 and the sibling gate passes); `test_grenade_kinds.py::test_e2e_four_cycles_w_throws_each_kind_after_the_wind_up` (door_test: the thrower is KNOCKED_DOWN by his own frag's blast for ~100 ticks and the later throws never happen); `test_fire_feedback.py::test_plume_raises_own_atmosphere_wind_points_outward` (the sealed-room plume's neighbour wind turns inward from about tick 48); `test_wave_push.py::test_a_blast_pushes_the_units_its_wind_reaches_and_nothing_else` (the marine behind the wall is now pushed); `test_unit_schema.py::...[bots]` and `test_unit_store.py::test_identity_ship_demo_every_tick` (the new bots match has no corpse conversion, so their non-vacuity guards fail); `test_cuda_conductor.py::test_the_conductor_replays_the_pinned_matches_to_their_end` (the device conductor, class T, refuses the new 3321-tick bots match at tick 806: an ApplierFault in slot 9e(d) vents, a vent ledger past its proved range; bots_v1 replays equal to its end).
- On this PC, failing on main too: `test_fire_tuning_level`, `test_larva_studio` (PNG bytes).

**Findings** (for Erik; nothing changed for them in this patch):

1. **Blasts at 0.333 m go numerically wild.** smoke_light_studio's first charge (t 25): before U, density stays 0.18-3.98 x ambient and the sealed room's pressure rises to ~4.9 atm (heat). After U, within 5 ticks one cell holds 49-113 x ambient while others empty to 0, the mean gas temperature goes negative (-94 game degrees), the room's mean pressure falls to 0.19 atm with no wall lost and no mass lost (exact books), and the plume-weighted wind stays near 30 m/s to the end of the run. Raising `N_SUB_MAX` to 24 or 48 tames the first blast but not the second. The same scene at `tile_size_m = 1.0` (the code path patch U leaves bit-identical) shows it too at the third charge (mean T -436, N 10 x), milder. Inference: the blast regime runs the donor-cell transport above Courant 1 at the substep cap against a once-per-tick pressure solve; before U the 1/dx under-transport hid it at 0.333 m. This is the regime section 7.4 called "not new", and it is worse than that; with L1-3's blast dials it belongs to the next session with Erik, before or with the tuning.
2. The quiet test_arena (`_sealedbox_bisect_bench.py` nofire) now carries winds up to 40.5 m/s (8.6 before), and the 60 s quiet run drifts -4.5 game degrees over the accountable set (-0.41 before), every joule counted.
3. The ship_demo matches run two to three times longer (G8p 272 -> 730 ticks, the controller pin 1349 -> 3321; the winning team flipped in four of eight pins) and lost their corpse conversion.

### 7.8 Critique (one critic at a time, verdicts recorded before the next)

| lens | model | verdict | findings (to resolve in v2 of §7) |
|---|---|---|---|
| 1. Completeness and correctness of the units change | Opus | SOUND WITH FIXES | (1) PROVED: twin debt is 12 twins, not `[twins.eos]` alone: `eos_solver.h`, `cuda_eos_step.h`, `cuda_resident_tick.h` and `bulk_transport.cpp` are `files` of eos, water, trace_tail, div_source, energy_flux, radiation_sweep, temperature, fire, stamp, water_tail, swarm_larva (P) and stamp_rows (T); mark all level, or drop the cosmetic rename and comment reword to shrink it. (2) PROVED: §7.5's test pins the SL leg through `eos_sl_advect_ref`, not `step()`'s backtrace (`eos_solver.cpp:831`); dropping the fold there passes it. Measure the SL leg through the live step (e.g. a wind step located in metres via `dbg_mg_inputs` div_u at 1.0 and 0.5 m); state the mass leg as Σ(N1−N0)·dx²/(L_m·dt). (3) PROVED config, INFERRED effect: §7.3's dial list misses `[blast] expansion_m3_per_strength` (calibrated at 0.333 m; 3× the mass at the same u after U), `heat_mj_per_strength`, `[physics.combustion] burn_rate` and `draw_r` (P-O2b widened the draw while resupply ran at 1/3 speed). (4) INFERRED, low: the range guard must sit inside `ensure_fold_table`'s poisoning try (`cuda_eos_resident.cu:919-932`), the `k_flux_q` pattern, and the guard is a behaviour change at 1.0 m when dt_s > 1 s (unreachable today). (5) PROVED, low: the bench twin must use `np.float32(dx)` and n_sub; `dx` on `eos_sl_advect_ref` / `cuda_eos_sl_advect` is required with no default; synthetic callers (`test_thermal_mass_axis.py:764/900/1006`, `cuda_p62_check.py:117`) need a chosen dx. (6) PROVED, info: the float-ratchet argument is moot (the TUs are not in `SIM_TUS`), and `step()` keeps `dt_s_d` for `k_flux_q`. Checked sound: no missed transport site; bit identity at 1.0 m on all four paths; the range and n_sub claims; the outflow limiter at any Courant; trace diffusion per tile² breaks nothing. Noted, not breakage: above 64 m/s at 0.333 m (blasts) mass stays capped at 1 tile/substep, so the solve/mass mismatch survives in that regime. |
| 2. Gates, regression, rollout, systems reuse | Opus | SOUND WITH FIXES | (1) PROVED: eight ship_demo (0.333 m) match pins move for the right reason, not two: `test_bots_v1.py:503`, `test_zombie_control.py:225` (also `tools/match_pin_check.py`'s), `test_exposure.py:588`, `test_grenade_kinds.py:667`, `test_horde_behaviours.py:52`, `test_turning_law.py:563`, `test_turn_rate.py:317`, `test_unit_impulse.py:883`; each gets a dated "#217 patch U" re-pin comment. The box pins (`_PRE_122_BOX`, `_STAGE1_BOX`, `_RE_PIN_1_BOX`) are 1.0 m oracles that must NOT move (1.0 m inferred for three). (2) PROVED: mark 14 twins level (adds combustion and edit_queue to lens 1's twelve). (3) PROVED: this PC has an RTX 3060 + nvcc 12.3; order is code → re-pins → `--mark-level` ×14 last → `cpp/build_cuda.bat` in the patch worktree → `tools/cuda_green.py` → full suite → merge from that checkout; plus `tools/match_pin_check.py --backend cuda` and `--backend resident` against the new pin. (4) Tests at 1/3 m with measured margins need a category each: re-measure and restate with rationale (`test_drag2_venting_gate.py` legs 4/5, leg 2's strict xfail XPASS goes to Erik, add it to the `k_drag2` dial note; `test_air_boundary.py` reflection/refill gates; `test_eos_checkerboard.py` XPASS goes to Erik); Erik-ruled properties where red is a real finding, never re-pinned (`test_wind_doorway_e2e.py`, `test_wind_studio.py:101`, `test_gen_ship_vents_air.py` `PRESSURE_BAND`); transport-independent, must stay green (`test_trace_resolution`, `test_sky_exchange`, `test_explosion`, `test_fire_vfx_ref`). (5) PROVED: the branch is behind origin/main (#221 test_arena, #97 water_studio); bring it up to main and re-grep. (6) Rule: the E2E reproduction is missing; `tools/wind_drift_probe.py` (plume drift ≈ 0.35-0.5·|u|) on smoke_light_studio and fire_studio, before and after, is the reproduction and the acceptance. (7) `_WIND_VREF`/`_WIND_VSENS` are measured fits, not Erik dials: a named render-only follow-up patch right after U; render `tools/look_shots.toml` before and after; `tests/data/fire_smoke_studio_flame_inputs.npz` goes stale (re-dump in the tuning session). (8) Replays recorded before U at 0.333 m hit `ReplayDivergence` by design (`FORMAT_VERSION` stays 8); RL runs, saved soldiers and imitation records made before U are on old physics (resume needs `--allow-code-change`; tell the game-ai-lead role); tournament pins `docs/rl_agent_rung1_1d_pins/*.json` move. (9) PROVED: the rule line must fire where the device folds live: `fields-gas.md`'s `paths:` miss `eos_solver.h`, `cuda_eos_step.cu`, `cuda_eos_resident.cu`, `cuda_sl_advection.cu`. (10) §7.2 slips: the float-ratchet claim is vacuous; `test_b1`/`test_b2` stay green because no physics runs; `bp.EOSSolver()`'s default dx = 0.333 so its direct users change too (books-only asserts). (11) `stubs/breach_physics.pyi:1272,1348` regenerated; `dx` required on the SL entries. Scope: one patch is right. |

## 8. Blast-regime stability (v2: cause and design, 2026-11-20)

This is the stack's third patch (Erik, 2026-11-20). Phase 1 measures the
cause and designs the fix; no sim code changed.

**v2** folds the two critiques of §8.15 into the text, with §7's convention:
every change that answers a finding carries its id (L1-n = lens 1, L2-n =
lens 2), and §8.15 ends with a disposition table, one row per finding. v2
also does the merge of step 0 (§8.0), so it describes the merged base
`3ac5cfe9`. `eos_solver.cpp`, `cuda_eos_resident.cu`, `cuda_eos_step.cu` and
`cuda_kick_compression.cu` did not change in that merge, so v1's line numbers
in them hold.

v1's numbers come from the live path at `2425bc6a` (patch U built). v2's new
numbers come from the merged base, on this worktree's CPU build
(`cpp/build/Release`) and CUDA build (`cpp/build_cuda`, RTX 3060), on the Work
Desktop. The instruments:

- the `--stability` mode of `tests/_eos_checkerboard_bench.py` (commit `6dd0ac78`);
- `tools/bench_conductor.py` (from main, P7e.5);
- two scratch readers over the bench's own loop (the k estimate per tick, §8.6.2, and the batch model, §8.7). They read; they probe nothing new. The k columns they print become columns of `--stability` in the patch (§8.9).

Per tick, `--stability` reports:

- `n_sub`, the substep count.
- `Csub` = max|u_entry|·dt_s/dx and `Ctick` = max|u_entry|·dt/dx.
- N's extremes over N0.
- The mass-weighted T, `Tmw` = ΣE/ΣN − T_amb.
- The stored P against the state pressure p* = C·E.
- `eps` = γ·dt·max|div u*|, the compression the solve linearizes. Its right-hand side is p*(1 − eps).
- The kick loop's kinetic-energy budget, in % of the room's tick-0 thermal energy E0.

`--tps` runs the whole game at another tick rate. That EMULATES updating the
pressure k = tps/24 times per 1/24 s. The other slots then run more often too;
this is noted where it matters.

PROVED means measured, or read in the code. INFERENCE says so.

### 8.0 Step 0: the merged base (L2-4)

**Done in this doc patch** (v2):

- **The merge.** `git fetch origin`, then `main` (`dde25e03`, 36 commits ahead, P7e.5 among them) merged into the branch: `3ac5cfe9`.
  - One conflict, `tests/twin_manifest.toml`: 11 twins carry a `level_at` mark on both sides (`fd514e9e` for patch U here, `1aebec6c` for P7e.5 on main). Resolved to this branch's marks. Every other hunk of main (its new comments and rows, its generated `files` lists) was taken as merged; the result differs from main's manifest in those 11 `level_at` lines only.
  - The code merged without a conflict. It changed `cuda_resident_tick.*`, `cuda_unit_books.*`, `bindings.cpp`, `physics_engine.h`, `swarm.*`, `conductor.py`, `physics_runner.py` and `swarm.py`, and brought `tests/cuda_conductor_batch_check.py` and `tools/bench_conductor.py`.
- **The twin ledger** on the merged base (`tools/twin_debt.py`, read only): 11 twins in debt (water, eos, trace_tail, div_source, radiation_sweep, temperature, fire, stamp, water_tail, stamp_rows, swarm_larva). Neither side's mark covers the merged tree; they go level only through the patch's own step 0, below.
- **Rebuilt** in the worktree: CPU (`cmake --build cpp/build --config Release`) and CUDA (`cpp/build_cuda.bat`), both exit 0.
- **§8.1's before tables, re-run on the merged CPU build.** The studio rows of ticks 27-35 reproduce §8.1 to every printed digit; the merge did not touch the CPU path. The arena, read every tick instead of every 24th, peaks higher than v1's sampled table: over 3-30 s, Ctick 7.8, eps 5.4, Nmin 0.050, wind up to 62.6 m/s. It is the same run, sampled more finely, and §8.1 now carries those numbers.
- **The conductor gates on the merged base:** `pytest tests/test_cuda_conductor_batch.py tests/test_cuda_conductor.py` on the merged CUDA build: 9 passed, 2 failed (20 min). Both reds are patch U's vent-ledger fault, which lens 2 predicted (L2-4):
  - `test_cuda_conductor_batch` (its hand scenes, swarm, swap, syncs and fault parts pass). Its MATCHES part fails: every ship_demo match is refused by the device conductor with `ApplierFault` in slot 9e(d) vents. bots@11 is refused at tick 627, bots_v1@12 at 831, bots@13 at 694, bots_v1@14 at 728. Each env ALONE is refused too, so the cause is not the batch.
  - `test_cuda_conductor` pinned matches: bots refused at tick 806 of 3321, after 806 equal ticks; bots_v1 equal to its end (730 ticks). Identical to §7.9's red.
  - Category: U's red, not §8's and not P7e.5's. The new matches run longer under U, and a vent ledger passes its proved range (§8.4: vents drain rooms at a constant mass rate). It is recorded here so that §8's own conductor leg (T7) is told apart from it.

**Left to the implementation patch, in this order, before its first code commit:**

1. `tools/twin_debt.py --mark-level` for the 11 twins, then `tools/cuda_green.py` on the merged build. The ledger is then level before §8 code exists.
2. The full suite on the merged base. Its reds are categorised with §7.5's categories and §8.11, so that U's reds, P7e.5's and §8's own stay apart.
3. The A/B baselines of T6 (§8.9), captured with the merged base's build.

### 8.1 Measurements

**smoke_light_studio, 0.333 m, 24 tps, first charge.** This is a legacy row:
it mints 1 N of air at the centre, and its `heat_amount` of 5000 rails at
T_MAX_PHYS.

| tick | nsub | umax | Csub | Ctick | eps | Nmin | Nmax | Tmw | P mean / p* mean | kick % | drag % | clamp % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 27 | 4 | 7.5 | 0.23 | 0.94 | 0.88 | 0.56 | 1.16 | 72 | 1.03 / 1.25 | 0.00 | 0.00 | 0 |
| 28 | 8 | 20.2 | 0.32 | 2.53 | 2.63 | 0.185 | 1.54 | 60 | 1.06 / 1.21 | 0.01 | 0.00 | 0 |
| 29 | 8 | 41.0 | 0.64 | 5.14 | 2.91 | 0.004 | 2.21 | 29 | 0.64 / 1.10 | 15.3 | 6.1 | 8.5 |
| 30 | 8 | 193.5 | 3.03 | 24.2 | 18.0 | 0.004 | 49.3 | 20 | 0.55 / 1.07 | 3.9 | 2.1 | 2.1 |
| 33 | 8 | 96.4 | 1.51 | 12.1 | 18.7 | 0.000 | 113.6 | 1 | 0.30 / 1.01 | 1.2 | 0.7 | 0.6 |
| 35 | 8 | 89.8 | 1.40 | 11.2 | 13.1 | 0.000 | 81.1 | −3 | 0.19 / 1.00 | 1.0 | 0.5 | 0.5 |

Over ticks 25-36, in % of E0:

- The kick debited 36.2 % from the gas as kinetic energy.
- Drag returned 16.0 %.
- The velocity cap destroyed 19.2 % (counted).
- The T_MIN rail minted 3.2 %.
- Step 1 changed KE by −0.7 %, unbooked.

The largest pile, at tick 33, held N 113.6 with p* 130 atm, while its stored P
was 0.91 atm. At tick 34 a pile of N 54 had p* 71 and stored P 0.11.

**Variants of the same scene:**

| variant | what it changes | result |
|---|---|---|
| `N_SUB_MAX` 48 | transport substeps only; Csub ≤ 0.25 (0.78 peak) | the first charge survives (Nmin 0.016). The second goes wild the same way: Nmax 126, Nmin 0, stored P ≤ 1.0 atm against a p* mean of 3.9 |
| 48 tps | the pressure updated 2× per 1/24 s | wild (Nmax up to 200) |
| 96 tps | 4× | each charge recovers within ~0.5 s; Nmax reaches 35 transiently |
| 192 / 384 tps | 8× / 16× | tame: Nmax ≤ 3.6, no empty cell, P = p*, kick and clamp ≈ 0. The wind is 190 m/s at detonation, under 9 m/s by 0.4 s, and 3-5 m/s by 1.5 s |
| `tile_size_m` 1.0 | — | first charge: Ctick ≤ 3.3, eps ≤ 3.2, no empty cell. Second and third: 4-11 empty cells, kick up to 11.8 % of E0 per tick; it recovers by ~3.3 s |

**The quiet test_arena, 30 s.** The scene is a sealed glass box with no vents
and no fire. The seal pushes 0.5 atm into the wall-layer cells.

| run | max wind | Ctick | eps | Nmin | Tmw at end |
|---|---|---|---|---|---|
| 24 tps (shipped) | 4 m/s at tick 1; up to 31 m/s by 3 s; 11.7-62.6 m/s from 3 s on (rms 2.5-3.2) | ≤ 7.8 | ≤ 5.4 | 0.050-0.07 | −3.0 K |
| 48 tps | 2-8 m/s | ≤ 0.5 | ≤ 0.17 | 0.75-0.9 | −0.1 K (20 s) |
| 96 tps | 0.4-0.6 m/s | ≤ 0.02 | ≤ 0.01 | 0.89 | 0.0 K (15 s) |
| 24 tps, `k_drag2` 0 | 446 m/s at 2 s; 999.6 (U_MAX) at 4 s | 56-125 | 7-9 | 0.005 | −142 K; 720 T_MAX_PHYS rail hits |

(v2: the shipped row is read every tick on the merged base; v1 sampled every 24th tick and printed 35-54 m/s, Ctick 2-6.7, eps ≤ 4.9, Nmin 0.09-0.13. Same run.)

At 24 tps the hot spot sits at one convex wall corner, (47, 77), in a cell
thinned to N 0.17-0.27. 14-16 of the 20 fastest cells touch a solid. The wind
has no odd-even share (cb share of u_x 0.00); N has −0.2.

**Two baselines that set the cost picture:**

- **ship_demo, idle (no units).** n_sub = 8 on every tick from about tick 24.
  - The return vent's aperture cell (51, 47) is empty from tick 24.
  - Its 246-tile room drains: mean N 0.95 → 0.27 in 20 s.
  - Meanwhile the supply aperture tile (16, 46) climbs to 3.6 N0.
  - The estimator's K·dt·|∇P|/N̂ term at the empty cell (N̂ floored at 1e-3) pins n_sub at the cap.
  - Winds there are only 1-3.5 m/s until the room is nearly empty (20 m/s at 20 s).
- **The GOLDEN_AGGREGATE scenario** (1.0 m, hull breach to vacuum): n_sub = 8 on 119 of 120 ticks, max wind 78.7 m/s on every tick, 4 cap hits.

### 8.1.1 The design's regime, emulated before the build (v2: L1-1, L1-4)

Two emulations on the merged CPU build, with no code change.

- **k = 1 with the ceiling kept.** 24 tps with `U_MAX` 64 m/s. 64 m/s is below c_amb, so the cap² plane is 64² on every cell (`K_pre_scan`'s `ratio >= ratio_umax` branch), which is (B) with no sub-ticks. This is the mutation tests T2 and T3 must fail under (L1-1).
- **k = 8 with the ceiling.** 192 tps with `U_MAX` 64 and `N_SUB_MAX` 2: one tile per pressure update and half a tile per substep. This is lens 1's proxy (L1-4). It also runs every other slot 8×, and its cap still destroys (no (C)).

| run | smoke_light_studio (legacy rows), charges at about ticks 24 / 36 / 48 | fire_studio frag at (24,16), tick 24 |
|---|---|---|
| before (merged base) | Nmax 113.6 / 131.8, Nmin 0, stored P / p* down to 0.02 and staying | Nmax 1.76, Nmin 0.036; wind median 8.9 m/s over ticks 30-48 and 5.6 over 48-120 (the slosh) |
| k = 1 + ceiling 64 | Nmax 35.5 (ticks 24-40) and 75.8 (40-72), Nmin 0, P / p* down to 0.03: **still wild** | Nmax 2.10, Nmin 0.035; wind median 8.8 and 5.6: the slosh unchanged |
| k = 8 + ceiling (proxy) | per charge, Nmax 2.3 / 4.1 / 11.5 and Nmin 0.068 / 0.148 / 0.140. P / p* falls on each detonation tick (to 0.74, 0.18, 0.08) and is back to ≥ 0.99 within 4 ticks @24; it rises monotonically (0.18 → 0.87 in 0.75 of a tick @24) | Nmax 1.19, **Nmin 0.086**; wind median 2.1 and **0.9**: no slosh |

What it says:

- **The mutation breaks the tests (L1-1, PROVED).** The ceiling alone does not tame the studio. So "k forced to 1, ceiling kept" is a real mutation of T2 and T3.
- **L1-4's sink, at k = 8 (PROVED in the emulation).** The frag's source tile does not empty: no cell falls below 0.086 N0. The criterion stays as a test leg (T2), because the emulation is not the build.
- **The studio's P / p* dip is a lag, not a sink (PROVED).** On a detonation tick the legacy rows dump their heat into p* before the solve. The solve catches up within about four ticks @24, at the ceiling's pace. T3 is restated for it (§8.9).
- **N 11.5 beside N 0.14 is not a pile (INFERENCE).** At equal pressure it is the hot/cold mixture of the legacy rows' heat, which rails at T_MAX_PHYS (N ∝ 1/T). So an N band is not the pile test where heat varies 10×. T2 tests pressure instead (§8.9).

### 8.2 The cause

1. **PROVED: the limit is the single pressure update per tick, not the transport's Courant number.**
   - **What the code does.** The substep loop (`eos_solver.cpp:811`) wraps only 1a (SL of u) and 1d (mass and energy flux). p* (:951), the solve (:998), the kick (:1009), step 6 (:1347) and the recovery (:1665) run once per tick. So u is frozen across the substeps, except for its own SL.
   - **The experiment.** With Csub ≤ 0.25 (N_SUB_MAX 48), the second charge still goes wild. With the pressure updated 8× (192 tps), all three charges are tame.
   - **The onset.** Over ticks 27-29, Ctick rises 0.94 → 2.53 → 5.14 and eps 0.88 → 2.63 → 2.91, while Csub never exceeds 0.64. Mass moves several tiles per pressure update, and the solve linearizes a compression above 1: at tick 28 the rhs p*(1 − eps) is negative on 8 cells.
   - **The published condition.** Kwatra et al. 2009 take dt from |u| for the whole step; the code substeps the transport alone. At 0.333 m and 24 tps, one tile per update is 8 m/s; at 1.0 m it is 24 m/s.
   - **Why U exposed it.** Patch U made the transport honest, which tripled Ctick at 0.333 m for the same wind. That is also why 1.0 m needs a bigger blast before it fails.
2. **PROVED: emptied cells are sinks the solve cannot fill within a tick, so piles are never pushed apart.**
   - The row mass m_i = 1/aK_i (`:1831`) is huge where p* is tiny. That pins those cells' P near 0, and a pile next to them solves to a fraction of an atm (N 113.6, p* 130, P 0.91).
   - The stored P mean falls to 0.19 atm while the state pressure mean stays at 1.00. The "pressure loss" is in the stored P, not in the air: the air's energy is all still present.
3. **PROVED: near-empty cells are kicked hard and then steer their neighbours' mass.**
   - The kick divides by the cell's own N̂, floored at 1e-3 (`:1088`). The solve's face mobility uses the arithmetic face mean (`:1869`).
   - The face velocity of steps 1d and 6 is the unweighted mean (u_i+u_j)/2.
   - In the wild phase, 18-207 cells a tick hit the cap. The cells at ≥ 90 m/s have median N 0.15-0.39 N0, while the 50-110× piles move at 5-20 m/s.
   - Piles form against walls: the pile's neighbour list holds solids.
4. **PROVED: the ~90 m/s plateau is the drag stage's ceiling, not a wind.**
   - Quadratic drag maps any speed u to u/(1 + kd2·u), with kd2 = k_drag2·dt = 0.009375 (`:1262`). Its limit is 1/kd2 = 106.7 m/s.
   - A cell capped at 1000 m/s (U_MAX) leaves at 96.4 m/s; one capped at c_amb = 300 leaves at 78.7. Both values recur in the traces: 96.4 in the studio, 78.7 in the golden scenario.
   - With k_drag2 = 0 the quiet arena runs to U_MAX in 4 s. Today the drag dial is the hidden limiter.
5. **PROVED: why the temperature falls while the books close.**
   - The kick debits ΔKE from `gas_energy` at the post-kick speed (`:1104`, e_kick_ke_sum).
   - The cap then scales u down and books the removed KE to `e_clamp_destroyed_sum` (`:1213`). That is an export, and it never returns to the gas. Over ticks 25-36 it was 19.2 % of E0, about 56 K of mass-weighted temperature.
   - Step 1 changes KE outside every counter: −0.7 % in the studio window, −10.8 % in the k_drag2 = 0 arena.
   - The mass-weighted T fell from +103 to −3 K. The "−94" was the CELL-UNWEIGHTED mean, which is dominated by emptied cells sitting at T_MIN.
   - For scale: a cell moving at 567 m/s carries kinetic energy equal to its whole thermal energy (k_ke = 9.1e-4 K/(m/s)²).
6. **PROVED: the quiet-arena winds are the same failure at low amplitude.**
   - At 24 tps, Ctick reaches 7.8 and eps 5.4 (v2, read every tick); at 48 tps they stay ≤ 0.5 and ≤ 0.17. At 96 tps the wind is 0.4-0.6 m/s.
   - **INFERENCE:** the 2-8 m/s left at 48 tps, where the per-update Courant number is ≤ 0.5, does not come from the update cadence. It is the collocated coupling (§3; F4 and F5 below) at a convex wall corner, and curing it belongs to (d)+(a).
   - Of the −3.0 K arena drift, kick − drag accounts for ~0.35 K and the rails for ~0.5 K. The rest lies outside the EOS. It was not split further, and it vanishes at 96 tps.
7. **PROVED: idle ship_demo pays for 8 substeps on every tick because of one emptied vent aperture** (§8.1). It is a cost of the same 1/N̂ term, paid today.

The previous session inferred the cause to be "donor-cell above Courant 1 at
the substep cap". **That inference is refuted:** the onset happens at
Csub ≤ 0.64, and the N_SUB_MAX 48 run fails with Csub ≤ 0.25. The limiter does
truncate above Courant 1 (F6), but only after the failure has begun.

### 8.3 Numerics review (independent, code-only)

A reviewer read the code only, with no runs, and tested 1D float toys that
transcribe steps 1-7. Its findings, against what the live 2D integer path
shows:

| # | finding (reviewer) | live path, 0.333 m | label |
|---|---|---|---|
| F1 | the advective CFL is enforced on transport only; p*, solve, kick and step 6 run once per tick | N_SUB_MAX 48 fails and 192 tps is tame; the onset is at Ctick 2.5-5 with Csub ≤ 0.64 (§8.2.1) | CONFIRMED |
| F2 | the cap destroys heat: bracket 1 debits at the uncapped speed, and bracket 4's removal is exported | the cap destroyed 19.2 % of E0 over the first blast's 12 ticks, and 32 % in the k_drag2 = 0 arena. That is most of the net kick debit (36.2 − 16.0) | CONFIRMED |
| F3 | near-empty cells pin P near 0, so a pile solves to ~0.25 atm | N 113.6 / p* 130 → P 0.91; N 54 / p* 71 → P 0.11; P mean 0.19 against a p* mean of 1.00 | CONFIRMED |
| F4 | the odd-even null mode (wide kick against compact solve) | the §2-§3 measurements; the quiet-arena wind has no odd-even share, N has −0.2 | CONFIRMED for N (already in §3); it does not drive the arena's fast wind |
| F5 | the 1e-3 kick floor plus the unweighted face mean let massless cells move their neighbours' mass | fast cells have median N 0.15-0.39 N0 while piles move at 5-20 m/s; the arena's hot spot is a thinned cell at a wall corner | CONFIRMED (structure and correlation); INFERENCE for how much of the pile-building it causes |
| F6 | the donor-cell limiter silently truncates above substep Courant 1 | true from tick 30 (Csub 1.3-4.7), but cells empty at Csub 0.32-0.64, and the N_SUB_MAX 48 run fails with no substep above 0.78 | REFUTED as the trigger; CONFIRMED as code behaviour in the wild phase |
| F7 | momentum is not conserved (SL u against flux N), so KE changes outside the books | the unbooked KE column: −0.7 % of E0 over the first blast, −10.8 % in the k_drag2 = 0 arena | CONFIRMED in sign (a loss) and order of size; INFERENCE for its share of the arena drift |
| F8 | the blast source is S·dt ≈ 144 tile volumes on one tile, which needs ~288 m/s face speed | fire_studio frag: eps on the detonation tick is 202 = γ·144.3; the source tile's N falls to 0.09 by tick +3 | CONFIRMED |

Where I disagree with its recommendation:

- **"A cap on n that never binds."** The cost would be unbounded: by its own criterion S·dt/n ≤ 0.3, the frag alone asks n ≈ 400. I keep a cap and bound the regime past it instead (§8.6: the budget ceiling, and a cap that returns the kick's energy as heat).
- **"Spread the blast source so S·dt/n ≤ 0.3."** That changes how a calibrated blast delivers its volume. It is a source-data decision for the blast session with Erik, not for this patch (§8.4).
- **Face projection (Rhie-Chow / MAC) as the second patch.** I agree in substance: it is the stack's (d)+(a). Its design must take F5 with it: a density-weighted face velocity, and a kick whose mobility is the solve's face mobility.

I agree with substepping the whole tick (F1) and with the energy-neutral cap
(F2); both are in §8.6.

### 8.4 The wind sources (survey; measurement only, no source changed)

The verdicts:

- (i): the input is unrealistic, a source-data or dial problem for a later session with Erik.
- (ii): the input is fine and the solver amplifies it.
- (iii): both.

| source | live, 0.333 m, after U | back-of-envelope (assumptions) | verdict |
|---|---|---|---|
| Vents (ship_demo, `q_circ` 16) | 1-3.5 m/s at the return grille at first. Its aperture cell is empty from tick 24 and n_sub sits at 8. The room drains to 0.27 N0 in 20 s, with 20 m/s by then; the supply tile reaches 3.6 N0 | 16 tile volumes/s = 4.4 m³/s through one 0.333 × 2.5 m grille (0.83 m²), i.e. 5.3 m/s; a real grille gives 1-3 m/s. A fan stalls at a few hundred Pa and never draws a room to vacuum. The rate is in tile volumes, so it scales with dx² | (iii) |
| Generic frag (`frag_standard`, fire_studio) | 22 m/s at +3 ticks, then a 3-9 m/s slosh with a ~15-tick period for > 4 s. At 192 tps it decays 10 → 1 m/s in 3 s | 0.184 kg of Comp B ≈ 0.9 MJ. Isobaric expansion bound (γ−1)/γ·E/p0 ≈ 2.6 m³, against the row's 40 m³. Heat 10 MJ, against 0.9. A real 0.18 kg charge gives ~50-100 m/s at 2.5 m for 2-3 ms, displacing ~0.1-0.3 m of air | (iii) |
| Studio charges (`studio_small/medium/large`, legacy rows) | 85-96 m/s (the drag ceiling) for the rest of the run; the first one raises the room's mean T by 103 K | `heat_amount` 5000 heat units = 156 MJ at 0.333 m (31 kJ per unit). It rails at T_MAX_PHYS, so ~50 MJ lands, against ~0.9 MJ for its 0.184 kg fill | (iii) |
| Fire plumes (fire_smoke_studio, six fires) | 0.5-1.7 m/s within 4 tiles of the fire at 24 tps (0.7-3.7 at 96 tps); the room median is 0.13-0.17 | a top-down plane has no vertical plume (1-5 m/s). In-plane, expansion outflow Q/(ρc_pT) over the fire's perimeter is cm/s, and entrainment inflow ≲ 1 m/s | neither: plausible, not amplified |
| Pumps / airlocks | not reached cheaply: airlock_demo's pump idles with no unit on the plate | 8 N/s through one port tile, the same class as the vents | INFERENCE: (iii), like vents |

### 8.5 Options and the choice

| option | cures | cost | verdict |
|---|---|---|---|
| raise `N_SUB_MAX` | nothing structural | n_sub × transport | rejected: measured to fail (§8.1) |
| a positivity flux limiter at any Courant | nothing: N is already ≥ 0 (the outflow limiter), and the failure happens at Csub < 1 | — | rejected (PROVED: positive and still wild) |
| semi-Lagrangian or flux-form SL mass transport | transport accuracy only; mass still moves with no pressure response, and non-conservative SL breaks the exact books | — | rejected |
| solve/kick consistency (face mobility in the kick, density-weighted face velocity) | F4/F5: the amplifier and the quiet residual | moves every 1.0 m result | right, and it is (d)+(a), later in the stack |
| **limit the wind** (Erik's idea) with the drag dial | lowers the plateau of a numerically made wind; the kick keeps feeding it, and drag turns it into heat | free | treats the symptom. The dial is a physical behaviour and Erik's call. Today it is already the hidden limiter (§8.2.4) |
| limit the wind: lower U_MAX or a speed cap | lowers the plateau, but the cap destroys KE (F2), so a lower cap drains heat faster | free | treats the symptom and worsens F2 |
| limit the wind: a Courant cap tied to the tick, u ≤ dx/dt (8 m/s at 0.333 m) | stability, by construction | free | changes physics: every wind, blast push and breach outflow is capped at 8 m/s, and the cap depends on dx and tps. Rejected as the cure |
| **substep the whole tick** (F1's remedy), with a bounded budget, a ceiling tied to that budget, and a cap that returns the kick's energy | F1; F3 (no empties in the sealed rooms emulated, §8.1.1; a one-tile source at the cap stays a gap, L1-4); F2; bounds F5/F6 | k solves, only on ticks with fast air | **chosen** |

A wind limit cures the cause only where the limit IS the stability condition,
the per-update Courant number. The choice below includes exactly that: a
ceiling tied to the sub-tick budget. It is not the cure. It binds only past the
budget, where the rest of the design cannot reach.

### 8.6 The design (v2): whole-step sub-ticks (k), a budget ceiling, a cap that returns the kick's energy

#### 8.6.1 (A) Sub-ticks, and where the loop lives (L2-3, L1-7)

**One call, one host read a tick (L2-3).** The loop lives in three places, one per backend:

- the CPU: inside `EOSSolver::step(dt)`;
- the batched resident engine (every N): inside `eos_step_resident`;
- the per-call device path: inside the C++ entry of `cuda_eos_step`.

`physics_runner._resident_chain` is unchanged and calls the EOS stage once, as
today (`device-residency.md:40`). No loop in Python glue, and no extra host sync.

**What runs once at entry.** Lens 1 (L1-7) found these left out; the device pre-stage already runs them once.

- The counter resets (`eos_solver.cpp:354-401`) and the device books' one memset.
- Step 0, `p_prev := atmosphere`: **`gmap.wave_p` is written once, at entry.** The water splash (`water_solver.cpp:337-345`) reads |P − P_prev| as the tick's transient, and the unseal and applier rows read tick semantics. `wave_p` is also a digested field.
- The pre-stage scan: the cap² / c_LOCAL plane, max|u|, max_du, the absorb plane, `cmask`, the k-set and k's reductions (§8.6.2), and n_sub (§8.6.3).
- The coeffE / coeffS caches, folded at (dt_k, n_sub). They are constant across the tick's sub-ticks, because k and n_sub are.

**What runs per sub-tick, k times at dt_k:**

- steps 1a/1d (n_sub substeps of dt_k/n_sub);
- div u*, with the div source S at dt_k (k·S·dt_k = S·dt);
- Dalton N and p*;
- the MG build and solve;
- the kick with its brackets, the cap ((B), (C)) and drag;
- step 5, step 6 (K3), and step 7, the recovery.

**The warm start.** For sub-tick ≥ 2 it is the previous sub-tick's stored P, read from `atmosphere` itself. Step 5 is that array's only writer inside the step, and it runs after the solve, so no buffer is needed and the caller's `p_prev` is never touched after entry (L1-7). `debug_pstar_from_prev` reads the same source.

**The recovery runs per sub-tick (L1-7).** That is necessary: sub-tick j's p* reads the mirror sub-tick j−1 wrote.

- Rails still write back only when one binds.
- `t_max_phys_hits`, `energy_floor_hits` and `e_rail_sum` now accumulate over the tick's sub-ticks.
- `energy-books.md`'s "once per tick" is amended (§8.13).

**Once per tick, after the EOS (unchanged, L1-7):** the trace tail (diffusion, decay, stranded; `physics_engine.cpp:791`) and every slot outside 7.

**k = 1 with the legacy switches is today's code path** (§8.6.6).

**The inventory of per-tick quantities** (L1-7; the implementer completes it in the as-built):

| kind | quantities | when k > 1 |
|---|---|---|
| rates folded with dt | `dt_q`, `Kdt_raw`, aK's dt²/dx² (`Kdt2dx2_raw`), `absorb_dt_q`, `kd_q`, `kd2_q` with `rad_dead_q32`, `k_flux_q`, the dt of `eos_transport_step_q`, the div source's S·dt | re-folded at `dt_k = eos_subtick_dt(dt, k)`, each a host double quantized once (the `k_flux_q` idiom) |
| per-tick factor | `sponge_udamp` (B3c, ambient maps) | default: applied on the last sub-tick only (the factor stays exact, with no integer root) |
| per-tick term not scaled by dt | `sponge_sigma`, the σ-mass added to m while m ∝ 1/aK ∝ k² (L1-7) | default: σ·k², an exact shift because k is a power of two, so its weight against m stays the per-tick one |
| entry only | `p_prev` / `wave_p`, the cap² plane and c_LOCAL, the absorb plane, `cmask`, coeffE/S, k, n_sub | once |
| state carried across sub-ticks | `atmosphere` (the warm start), wind, the temperature mirror, `gas_energy`, gas | carried |
| books | every counter | accumulated over the sub-ticks, reset once at entry |
| after the EOS | the trace tail, slot 9 | unchanged |

Both sponge defaults are decided against the B3c sponge's own gate,
`test_air_boundary.py`. The implementer may choose the other form if that gate
says so, and records it.

**A note for ruling 4 (drag).** Quadratic drag's per-step map u/(1 + k_drag2·dt·u)
is the exact solution of du/dt = −k_drag2·u² over dt. So k sub-ticks at dt_k
compose to the same map, and the law per second does not change with k.
Linear drag and the absorb factor change only at second order.

#### 8.6.2 The choice of k (L1-3, L1-5, L1-9, L2-2)

k is one integer per env, chosen once per tick at entry from integer reductions (the n_sub idiom).

**The k-set** is the cells k is measured over:

- gas cells: not solid, not vacuum, not the ambient ring;
- with Dalton N ≥ `N_K_MIN`;
- with no vacuum or ambient-ring cell within `R_K` = 2 tiles (4-neighbour distance).

`N_K_MIN` = 3277 raw = quantize(0.05), i.e. 5 % of the 1-atm density at
T_AMB in the Q16 N currency. It is an integer constant and valid on space
maps, which have no n_amb (L1-5). `R_K` is an integer constant too.

**The terms:**

- `k_u = ceil(max_kset |u_entry| · dt / (C_P·dx))`, with C_P = 1.
- `k_div = ceil(γ·dt·max_kset |div u_entry| / ε*)`, with **ε* = 1**: the solve's right-hand side p*(1 − γ·dt_k·div u) stays positive in every sub-tick. v1 had 0.5. 1 is the positivity condition itself, and it halves a breach's k (below).
- `k_src = ceil(max S·dt / (4·C_P))` on a tick with a div source. S is reduced on the device in the pre-stage, which lens 1 checked sound.
- **`k = min(K_MAX, pow2ceil(max(k_u, k_div, k_src)))`, so k ∈ {1, 2, 4, 8}** (L1-9). dt_k = dt/k is then exact in float. It is still computed by ONE helper, `eos_subtick_dt(dt, k)` (`eos_solver.h`, out of line in `eos_solver.cpp`), which the CPU, the per-call host stage and `ensure_fold_table` all call.

**The kick term du is NOT in k. This refutes L1-3's sum, on measurement (table below).**

- du is the per-tick kick increment K·dt·|∇P|/N̂, not a displacement.
- Where a flow is steady (a breach outflow, a vent's draw, a slosh), drag and the cap remove it again inside the same kick loop.
- Summed into k, it holds k up on quiet and breach ticks:
  - idle ship_demo runs at k = 2-4 on 53 % of ticks;
  - the 1.0 m golden breach sits at k = 8 on 119 of 120 ticks;
  - the fire_studio frag's slosh holds k = 2 for 4 s.
- du stays in n_sub verbatim, which is HEAD's n_sub, so k = 1 keeps HEAD (§8.6.3).

**Why L1-3's in-tick acceleration is still bounded without the sum.**

- Inside a tick, the mass of sub-tick j moves at the speed sub-tick j−1's kick left. (B) holds that speed at or below U_bud = K_MAX·dx/dt.
- So mass moves at most max(C_P, K_MAX/k) tiles between solves: 1 tile when k = K_MAX, and 4 tiles in the worst case (k = 2, a tick whose own kick reaches the ceiling).
- The next tick's k is computed from that speed.
- T1 asserts exactly this bound, per sub-tick (§8.9).

**Measured** on the merged CPU build, from the entry state of every tick of
today's trajectories (a scratch reader; float estimates of the integer
estimator). The cells give the mean k per tick:

| scene | ticks | unfiltered ceil(Ctick) (lens 2's measure) | L1-3's sum (u + du), N filter only, ε* 0.5 | **v2** (the k-set, no du, ε* 1) |
|---|---|---|---|---|
| ship_demo, idle | 480 | 1.52 | 1.66 (k 2-4 on 53 % of ticks) | **1.00** |
| range_hall, idle | 240 | 1.00 | 1.00 | **1.00** |
| range_hall, frag at the centre | 240 | 1.05 | 1.11 | **1.06** |
| range_hall, C4 at the centre | 360 | 1.11 | 1.16 | **1.12** |
| range_hall, C4 against the hull (a breach) | 360 | 5.92 | 3.84 | **2.54** |
| fire_studio, frag | 120 | 1.23 | 1.96 | **1.27** |
| GOLDEN_AGGREGATE scenario (1.0 m, a breach) | 120 | 2.97 | 7.94 | **1.59** |
| smoke_light_studio (today's wild trajectory) | 168 | 6.79 | 6.85 | **6.82** |

**The breach in detail (v2 k, mean, by window; detonation at tick 24):**

| ticks | 24-72 | 72-144 | 144-240 | 240-360 |
|---|---|---|---|---|
| range_hall C4 against the hull | 5.90 | 3.15 | 2.70 | **1.00** |

- The C4's own blast ends by about tick 72.
- From then on, k_u is 1. k_div holds k at 2-8 while the room drains: the expansion fan toward the mouth has |div| ~ u/r, and at 0.333 m it reaches past 2 tiles.
- k is back to 1 once the room has drained, about 9 s after the C4. So **a breach does not pin k for the rest of the episode** (L2-2), and the cost table carries it (§8.7).
- These are today's trajectories, where a C4 at 0.333 m is still in the wild regime. INFERENCE: after the fix, blast tails shorten and the means fall. The patch's `--stability` columns re-measure them.

#### 8.6.3 n_sub per sub-tick (L1-3)

- **n_adv** is HEAD's estimate, **uncapped**: `ceil(dt·u_est / (CFL_ADV·dx))`, with u_est = max|u| + max_du + 1, clipped at max(c_LOCAL, U_MAX) as today (`:715-728`).
- **The rule:** `n_sub = min(N_SUB_MAX, max(ceil(n_adv/k), F))`, where F = ceil(K_MAX/(k·CFL_ADV)) when k > 1 and the ceiling switch is on, and F = 1 otherwise.
- **At k = 1** this is min(N_SUB_MAX, n_adv): HEAD's n_sub exactly.
- **At k > 1** every |u| ≤ U_bud after any kick (B), and the SL sample and the face mean are convex combinations. So a substep moves at most U_bud·dt_k/(n_sub·dx) = K_MAX/(k·n_sub) ≤ CFL_ADV = 0.5 tiles. Without this floor, sub-ticks 2..k of a detonation tick (entry u ≈ 0, so n_sub = 1) run at Csub 1.0, the F6 regime (L1-3).
- **The cost:** k·n_sub ≥ 16 transport substeps on such a tick (8 today at the cap), paid by that env alone (masked, §8.6.7).
- **n_sub is fixed for the tick:** one estimate, one read. A per-sub-tick device re-estimate (each env's own n_sub from its own sub-tick, with the loop run to the bound) is a later lever, not this patch.

#### 8.6.4 (B) The budget ceiling (L1-8, L1-6)

- The cap stage's per-cell ceiling becomes `min(cap2_plane_[i], U_bud²)`, with U_bud = K_MAX·C_P·dx/dt: 64 m/s at 0.333 m, 192 m/s at 1.0 m, 48 m/s at 0.25 m.
- Past the budget, the velocity the next update transports therefore never exceeds one tile per update.
- It moves no mass slower than today. The donor-cell limiter already caps real mass motion at one tile per substep × 8 (§7.4).
- It is a value of an existing stage, as the Interior drag rule requires (extend its stages, never a parallel damping site).

**It holds breaches too (L1-8).** A hull breach's mouth wind is sonic, about 300 m/s; the golden scenario rails at c_amb. At 0.333 m it is now held at 64 m/s.

- The mass flux hardly changes, because the limiter caps it at one tile per substep today.
- Every wind reader changes:
  - the wind load on units, which goes as u²: about (300/64)² ≈ 22× less push at a breach mouth;
  - the renderer's tamed wind;
  - the recorder.
- This is ruling 2 (§8.14), and those tests are category A (§8.10).

**Dead at shipped resolutions (L1-6).** After (B), cap² = U_bud² wherever the cap binds, at every shipped dx.

- U_bud < c_amb whenever dx < c_amb·dt/K_MAX ≈ 1.56 m, and the cap² plane floors at c_amb².
- So the per-cell c_LOCAL plane (D2v2) and U_MAX never bind at shipped resolutions.
- They stay: they bind at dx ≥ 1.56 m, or with the ceiling switched off. Noted in §8.13, not removed here.

#### 8.6.5 (C) The cap returns only what the kick supplied (L1-6)

**The quantities.** Per accountable cell, in the kick loop, with KE(·) the existing `ke_energy(n_bulk_ke, u²)`:

- KE0: at the kick's entry;
- KE1: after bracket 1, the kick;
- KEpre: before the cap, after the absorb and the sponge;
- KEc: after the cap.

**The rule:**

- removed = KEpre − KEc. It is ≥ 0, and 0 when the cap does not bind.
- **heat = min(removed, max(0, KE1 − max(KE0, KEc)))**: the part of this sub-tick's kick above both the entry speed and the cap.
- `gas_energy += heat`, booked to a new counter, `e_cap_heat_sum`. It sits in the identity's EOS group, beside `e_drag_heat_sum`.
- **removed − heat stays DESTROYED and counted in `e_clamp_destroyed_sum`, which stays LIVE as a tripwire.**
  - It is structurally about 0 once (B) binds, because u0 ≤ U_bud at the kick's entry.
  - It is non-zero only for a speed that arrived from outside the kick: a FieldEdit or level-load wind, a K_MAX hot reload, or F7's unbooked SL KE moving onto a denser cell.

**Why it cannot mint.** heat ≤ removed, and heat ≤ KE1 − KE0, the kick's own debit from that same cell's gas. So no energy is minted, and no hot spot forms elsewhere.

**Gated (L1-6).** On the fire_studio frag (no legacy rows), per tick: no tick may show a T_MAX_PHYS hit, or an `e_rail_sum` larger than the same run with `cap_heat` off. On the studio's legacy rows, whose `heat_amount` 5000 rails by design, these are reported, not gated.

**Files.**

- The counters' readers: `bindings.cpp` and the stubs.
- The device: `cuda_kick_compression.cu` and its reference `eos_kick_compression_reference`; `cuda_resident_tick.cu`'s `kc(...)` fold (`:1332-1345`) gains one slot, with `KICK_CNT_SLOTS` (L1-10).
- Every Python transcription of the EOS group gains the term. These are the readers of `e_drag_heat_sum`:
  - tests: `test_e1_hot_rail`, `test_thermostat_books`, `test_p_e3_drag`, `test_drag2_venting_gate`, `test_bulk_gas_payload`, `test_heat_source`;
  - benches: `_sealedbox_bisect_bench`, `_quiet_books_bench`, `_blast_bench`, `_fire_bench`, `_vent_bench`, `_drag2_sweep_bench`, `_eos_checkerboard_bench`;
  - CUDA checks: `cuda_kick_check`, `cuda_s8a_check`, `cuda_combustion_burn_check`;
  - the tool `tools/storm_ledger.py`;
  - `docs/ray_engine_v2_scheme_study_2026-09-13/currency_audit_t2.py:104` (L1-10).

**It reverses rule D6 for the cap stage** (`fields-gas.md` Interior drag; code `:1206-1209`). That is a modelling ruling, ruling 3 (§8.14), and the patch amends the rule text (§8.13).

#### 8.6.6 Separate switches, and what "k = 1 is HEAD" means (L1-1, L2-6, L2-8)

Three `EOSSolver` fields, default ON:

- `subtick_loop`: k from the estimate. Off: k ≡ 1, and n_sub is HEAD's.
- `subtick_ceiling`: U_bud. Off: the cap is `cap2_plane` alone, HEAD.
- `cap_heat`: §8.6.5. Off: bracket 4 destroys everything, HEAD.

They are test seams, not dials: not in `config.toml`, set by tests through the solver (as `debug_pstar_from_prev` is), and every one is a key of the device fold table. Never off in play.

- **Legacy, all three off, is HEAD bit for bit on every path.** The standing guard (L2-6): with the legacy switches, the GOLDEN_AGGREGATE scenario on the new build hashes to the old `d5d03433…487b`. This is `test_b6_logic_golden.py`'s "sole cause" standard: the old setting reproduces the previous value on the same build.
- **The identity claim, exact (L1-1).** With the defaults, a tick is bit-identical to HEAD if and only if k = 1, no cell's speed exceeds U_bud, and the cap does not bind.
- **K_MAX is a PHYSICS dial (L2-8).** `[physics.eos] K_MAX`, default 8, is bound in PhysicsRunner only. The ceiling is K_MAX·dx/dt, so K_MAX moves blast and breach physics. Training and play share it; a run records it through its config hash. It is not a perf dial: "lower it to train faster" would train on another game.

#### 8.6.7 Per-env k in the batched resident engine (L1-2, L2-1: a requirement)

**Read at the merged base:**

- `cuda_eos_resident.cu`: `eos_step_resident` (:1023-1255), `device_prestage` (:953-988), `K_pre_fold` (:737-770), `ensure_fold_table` (:909-946), `mg_build_device` (:627-676);
- `cuda_batch.cuh`: `sub_off`, `env_off`, `bptr`;
- `cuda_resident_tick.h`: `EnvSub` (:79-86);
- `cuda_kick_compression.cu`: `kick_compression_launch_resident`;
- main's P7e.5 seam: `run_resident_conductor`, `resident_tail_defer` / `_finish`.

**Today**, only the transport rows are per env (`EnvSub`: n_sub, `dtdx_s_q`, `k_flux_q`, `flux_pu_cap`, `c_local_q`). The kick's and the MG's dt folds are batch-uniform, passed by value. The substep loop runs to the batch's max n_sub, with `sub_off` masks.

**The design:**

1. **The fold table is keyed (k, n_sub).**
   - 32 rows: k ∈ {1, 2, 4, 8} × n_sub ∈ [1, N_SUB_MAX], built on the host once per config. The key gains K_MAX and the three switches.
   - Each row's folds come from the ONE function per fold, at dt_k = `eos_subtick_dt(dt, k)`.
   - `EnvSub` gains `k`, `dt_q`, `Kdt_raw`, `Kdt2dx2_raw`, `absorb_dt_q`, `kd_q`, `kd2_q` and `rad_dead_q32`, beside `dtdx_s_q`, `k_flux_q` and `flux_pu_cap`.
   - A refused row poisons its env, as today (inside the `try`).
2. **The pre-stage picks each env's (k, n_sub) row.**
   - `K_pre_fold` picks it from the env's own reductions.
   - The k-set maxima (|u|², |div u|, S) are three more `atomicMax` slots of the pre arena, filled in `K_pre_scan` and a k-set div scan beside `K_max_du`.
   - `K_pre_fold` writes the batch's max k and max n_sub as two words. ONE 16-byte D2H, ONE `tick_sync_point("max_n_sub")`. The name is kept, so the capture check's two named syncs stay two.
3. **The loop:** for t < k_max: [substeps s < n_sub_max] → [mid-stage] → [MG build + V-cycles] → [kick] → [`K_store_atm`] → [K3 + recovery].
   - **The mask** `tick_off(es, t)` = t ≥ `es[env].k`, beside `sub_off`, block-uniform as today. Every EOS kernel past the pre-stage takes it:
     - the SL and the bulk flux (`tick_off || sub_off`);
     - `K_div_u`, `K_div_source`, `K_ntot`, `K_pstar`;
     - every MG build kernel at every level (`K_L0_*`, `K_C_*`, `K_recip`);
     - every V-cycle kernel (`eos_mg_vcycle_resident` gains the `es` pointer; nullptr means unmasked, the per-call path);
     - `kick_kernel`, `K_store_atm`, the energy-flux passes and the recovery kernel.
   - **Copies.** The loop's device-to-device copies of scratch (`svx`, `svy`, `st`) are harmless for a masked env, since they are scratch only. No state is copied inside the loop: `p_prev` is entry-only.
4. **Per-env folds.** The kernels read the dt folds from `es[env]` in place of the by-value `KickScalarFolds` / `MGScalarFolds` fields; the dt-free fields stay by value. N = 1 and the per-call path use row 0 of the same kernels: one code path.
5. **Books.** The layout is unchanged; the counters accumulate per env across sub-ticks (`atomicAdd` into the arena). The kick slots gain `e_cap_heat_sum` (§8.6.5).
6. **Batch invariance** (env i of N == env i alone) holds by construction: each env's k, n_sub and folds are its own, and a masked env does nothing. It is gated with a quiet and a blasting env in one batch (§8.9 T7).
7. **P9 (graph capture).** The loop has two nested data-dependent counts (k_max, n_sub_max), so the WHILE-node plan needs a nested conditional node (L1-10). Recorded for P9; not built here.

**Feasible in one patch: yes.**

- Every EOS kernel already has the (N, h, w) launch and the block-uniform early return.
- Per kernel the change is one predicate. For the kick and the MG level-0 build it is also reading 1-3 folds from `es`.
- The V-cycle is the largest edit: one pointer through its launch helpers.
- No staged fallback is proposed, and batch-max k is not used.

### 8.7 Cost (Erik: training speed; L2-1)

**Measured** on the merged build, this PC (RTX 3060), with `tools/bench_conductor.py` (10 warm-up, 60 timed and 20 profiled ticks; one recorded match latched into every env; the rooms quiet, so k = 1 everywhere today):

| level | N | ms / tick | ms / env-tick | EOS segment, GPU ms | EOS segment, host ms | radiation sweep, GPU ms |
|---|---|---|---|---|---|---|
| range_hall | 1 | 7.26 | 7.26 | 1.69 | 0.58 | 0.27 |
| range_hall | 16 | 9.04 | 0.565 | 1.64 | 0.56 | 1.71 |
| range_hall | 64 | 16.61 | 0.260 | 2.51 | 0.61 | 6.50 |
| ship_demo | 1 | 9.65 | 9.66 | 3.09 | 1.98 | 0.63 |
| ship_demo | 16 | 20.35 | 1.27 | 7.26 | 1.98 | 5.10 |
| ship_demo | 64 | 58.52 | 0.914 | 24.13 | 2.21 | 19.34 |

**What an extra sub-tick costs.**

- **The floor. This refutes L2-1's "+2-8 % once GPU-bound", PROVED from the table.**
  - On range_hall the EOS stage is latency-bound on the GPU, not launch-bound on the host.
  - At N = 1 its GPU segment (1.69 ms) is three times its host launch time (0.58 ms). From N = 1 to N = 64 it grows by only 0.82 ms.
  - So a masked sub-tick, in which most envs return at the top of each kernel, still costs the chain's kernel latency on the GPU: a floor F per extra sub-tick, paid at the batch's MAX k at any N.
  - F is not measured directly, because no masked loop exists yet. It lies between the stage's host time and its GPU time at N = 1, which includes the once-per-tick pre-stage and the sync gap: **0.6-1.2 ms on range_hall, 1.2-2.5 ms on ship_demo.**
- **The bandwidth part** is what per-env masking removes: b = (2.51 − 1.69)/63 ≈ 0.013 ms per env-sub-tick on range_hall, and 0.33 ms on ship_demo.
- **The model.** Per-env k costs (E[max k] − 1)·F + N·(E[mean k] − 1)·b a tick. Batch-uniform k costs (E[max k] − 1)·T_EOS(N).
- **The k statistics** come from a Monte Carlo over 45 s episodes (1080 ticks), with blasts at random ticks, lens 2's loadouts (`[loadouts.rifleman]`: 2 frags and 1 C4), and §8.6.2's v2 k sequences measured on range_hall:
  - a frag: 8, 8, 4, 4, 2;
  - a C4: 8 ×5, 4, 4, 2 ×9;
  - a hull breach: 213 ticks, mean 3.6;
  - the detonation tick: k_src = 8.

**range_hall (the training room)**, extra cost a tick:

| episodes | N | E[max k] | E[mean k] | P(batch k > 1) | **per-env k** | batch-uniform k |
|---|---|---|---|---|---|---|
| the learner armed (2 frags + C4) | 1 | 1.08 | 1.085 | 0.02 | +2 % | +2 % |
| | 16 | 2.22 | 1.085 | 0.32 | **+8-16 %** | +22 % |
| | 64 | 4.6 | 1.085 | 0.79 | **+14-26 %** | +55 % |
| both teams armed (4 frags + 2 C4) | 1 | 1.17 | 1.17 | 0.05 | +4 % | +4 % |
| | 16 | 3.20 | 1.17 | 0.54 | **+15-30 %** | +40 % |
| | 64 | 6.23 | 1.17 | 0.95 | **+20-39 %** | +79 % |
| both teams, and 10 % of episodes breach the hull | 1 | 1.22 | 1.22 | 0.06 | +5 % | +5 % |
| | 16 | 3.7 | 1.22 | 0.63 | **+19-37 %** | +49 % |
| | 64 | 6.8 | 1.22 | 0.98 | **+22-43 %** | +88 % |

The per-env ranges span F at its two bounds.

**ship_demo at N = 64**, the same model: the learner armed, per-env +11-18 % against uniform +150 %; both teams, +17-29 % against +216 %. The uniform figure counts a full idle EOS stage (n_sub 8) for every extra sub-tick, so it is an upper bound.

**Reading:**

- **Per-env k is required.** It buys most where the EOS is bandwidth-bound: on ship_demo at N = 64, +11-29 % instead of +150-240 %.
- **On the small training room the floor dominates:** +14-43 % at N = 64 instead of +55-88 %. A "+2-8 %" is not reachable by masking alone.
- **Levers on the floor.** Each is measured in the patch, and none is decided here:
  1. fewer V-cycles on sub-ticks ≥ 2, which are warm-started and have σ/k smaller. This re-opens `mg_cycles = 8`, with numbers.
  2. P9's graph capture of the loop, which removes the host launch gaps, part of F.
  3. fusing the chain's small kernels, a later perf patch.
  
  INFERENCE: (1) is the largest, since the V-cycles are most of the chain's kernels.
- **Caveat.** The k sequences are today's CPU trajectories, in which a C4 at 0.333 m is still in the wild regime. INFERENCE: after the fix, blasts are calmer, their tails shorter, and E[max k] falls.

**N = 1, the played game.** On the GPU, as v1: a k = 8 blast tick costs about 13-15 ms (3.2 today). On the CPU backend, about 40-60 ms: 8 solves plus k·n_sub ≥ 16 transport substeps at about 1.5 ms each. That is a hitch of 3-6 ticks per blast on the CPU backend.

**v1's idle saving is withdrawn (L1-5).** n_sub stays HEAD's, with its unfiltered du, so idle ship_demo still pays n_sub 8 for its emptied vent aperture. **ACCEPTED GAP:** that is the vents' source data (§8.4), for the blast and vent session.

**The cost gate (L2-1).**

- `tools/bench_conductor.py`, before and after, at N = 1 / 16 / 64, on ship_demo and range_hall.
- The blasts must be STAGGERED across envs. The bench today latches one recorded match into every env, so the patch gives it a per-env shift of the latched intents (env e runs the match e·Δ ticks later). That is a mode of the existing instrument, not a new probe.
- It reports ms per env-tick and P(batch k > 1).
- The acceptance is the measured number, recorded in the as-built against this table.

### 8.8 What it changes

**At 0.333 m, blasts stay physical in the solver's sense:** no empty cells and no piles; P tracks p* within about four ticks of a detonation (§8.1.1); no heat drain.

- Blast winds past 64 m/s are held at the ceiling.
- The frag's one-tile source asks for about 288 m/s, so part of its 40 m³ is not delivered in its tick. The source carries no state, so that part is lost, not deferred. INFERENCE: its knockdown range shrinks. Restoring it is the blast retune's job, with Erik; the source data is 15× too large anyway (§8.4).

**Breaches (L1-8).** The mouth wind is held at 64 m/s at 0.333 m and at 192 m/s at 1.0 m. The wind load on units at a breach falls about 20×; the mass flux hardly changes.

**At 1.0 m:** see §8.6.6. The ceiling is 192 m/s, and the golden breach runs k = 2 on most ticks of its first 3 s (§8.6.2).

**The quiet arena** (INFERENCE, from the 48-tps emulation): 2-8 m/s instead of 35-63, and a drift of −0.1 K instead of −3.0 K. The residual goes to (d)+(a).

**Goldens and pins:** §8.10. The field digest needs no version bump: no field, dtype or currency changes.

### 8.9 Gates, oracles, property tests, the reproduction

**The E2E reproduction, before and after,** with `--stability` on the live path. The before numbers are the merged base's (§8.0, §8.1.1).

- **smoke_light_studio, 72 ticks.** After, all of these must hold:
  - T2's pressure band and T3's lag;
  - Tmw never below its pre-blast value minus the thermal solver's losses;
  - the plume-weighted wind back under 10 m/s within 1 s of each charge.
- **fire_studio frag** (`--detonate 24,16,24`): median wind under 3 m/s from 1 s after the detonation (5.6 today; 0.9 in the proxy).
- **`--arena`, 30 s:** max wind and drift, against today's 62.6 m/s and −3.0 K.
- **ship_demo idle:** n_sub, k and the EOS ms per tick, before and after.
- **`tools/wind_drift_probe.py`** on smoke_light_studio and the fire_studio frag: the median of drift ÷ wind stays ≈ 1 (patch U's result).

**Property tests,** in a new `tests/test_eos_substep_whole_tick.py`. Each names the property it protects and the change that must break it, and none pins a count. They read three solver test seams the patch adds:

- `dbg_last_k`;
- `dbg_solves_last_step`, the MG solve calls in the last `step()`;
- `dbg_subtick_umax`, the k-set's max|u| at each sub-tick's transport entry. It is recorded only when a test turns it on.

**T1. Each pressure update sees bounded mass motion.** This replaces v1's test 1, which held by construction (L2-5).

- The checks:
  - solves per tick == k;
  - per sub-tick, `dbg_subtick_umax`·dt_k/dx ≤ max(C_P, K_MAX/k) + one quantum;
  - per substep, `dbg_subtick_umax`·dt_k/(n_sub·dx) ≤ CFL_ADV + one quantum.
- These are read from snapshots of what each sub-tick transported, not from k's own estimate.
- Two flows:
  - the fire_studio frag, whose detonation tick has entry u ≈ 0: L1-3's case;
  - a uniform 60 m/s wind in a sealed 0.333 m room.
- It must break:
  - (a) if the loop wraps 1a/1d only (solves 1 ≠ k);
  - (b) if the n_sub floor is dropped (Csub reaches 1.0 on sub-ticks ≥ 2 of the detonation tick);
  - (c) if the ceiling is dropped (sub-tick Courant above K_MAX/k).

**T2. No pile and no sink (L1-4, L2-2).** Three legs:

- **The studio's first charge, 2 s.**
  - The checks: from 0.25 s after the detonation, every accountable cell's p* stays within [0.25, 4]× the room's mean p*; Nmin ≥ 0.02 N0 throughout.
  - It tests pressure, not N, because the legacy rows' heat makes an isobaric N vary 10× (§8.1.1). The band is confirmed from the patch's before/after numbers and recorded.
- **The fire_studio frag.**
  - The checks: the source tile and its 4-ring keep N ≥ 0.05·N0 on every tick (0.086 in the proxy; 0.035 with k = 1 and the ceiling kept); Nmax ≤ 4.
- **A breach: range_hall, a C4 against the hull, 10 s.**
  - The check: from 0.25 s after the detonation, every k-set cell of the vented room keeps p* within [0.25, 4]× the k-set's mean p* (the band near the mouth is not in the k-set).
  - Its k tail is a measured number of the cost gate, not a property here.
- It must break, in every leg, with `subtick_loop` off and the ceiling kept (L1-1). Measured for the studio and the frag (§8.1.1).

**T3. The stored pressure tracks the air.**

- The check, in the studio run: |mean P − mean p*| / mean p* < 5 % on every tick outside the 0.25 s after each detonation. Inside that window the ratio rises monotonically and reaches 5 % by its end (proxy: 0.18 → 0.99 within 4 ticks @24; before: 0.02-0.19, and staying).
- It must break with `subtick_loop` off and the ceiling kept (measured: 0.03, staying).

**T4. The cap returns only what the kick supplied (L1-6).**

- The checks:
  - for a cell kicked past its cap, Δgas_energy over the kick loop == −kick bracket + drag + cap heat, exactly;
  - heat == min(removed, max(0, KE1 − max(KE0, KEc)));
  - `e_clamp_destroyed_sum` == removed − heat;
  - a cell whose ENTRY speed is above U_bud (a FieldEdit wind) shows destroyed > 0: the tripwire is live.
- It must break if bracket 4 reverts to an export (heat 0), or if it heats the whole removal (the entry-speed leg).

**T5. The energy closure identity still closes,** exactly in int64 across whole ticks, with k > 1 (the frag run).

- The check: `test_e1_hot_rail`'s transcription of the identity, with `e_cap_heat_sum`.
- It must break if a counter resets per sub-tick instead of per tick.

**T6. k = 1 / legacy is HEAD (L2-6).**

- (a) A committed property: with the legacy switches, the GOLDEN_AGGREGATE scenario on the new build hashes to `d5d03433…487b`.
- (b) The A/B lockstep harness, every field and every tick: the legacy switches on the new build against the merged base's captured trajectory (`save_trajectory`, step 0), on two scenes:
  - the golden scenario;
  - a named 0.333 m scene, a new `make_sim` in `tests/field_ab_harness.py`: `test_arena` with its vents, 240 ticks.
- (c) The identity claim with the DEFAULTS: on range_hall idle (k = 1 on every tick, wind far below U_bud), the defaults equal the legacy switches bit for bit, every tick.
- It must break on any edit to the k = 1 path, or if a switch fails to restore HEAD.

**T7. CPU == CUDA, and the batch (L2-7c).**

- `cuda_s8a_check` (N = 1 == CPU, tol 0), with a frag leg so that k > 1.
- `cuda_batch_invariance_check`, with envs at different k.
- `cuda_conductor_batch_check` gains a leg: a quiet env and a blasting env (the same level, one with a frag), each equal to itself alone, every tick.
- `cuda_conductor_check` and `test_cuda_conductor`.
- `cuda_prestage_parity_check`: the row is now (k, n_sub).
- `cuda_tick_capture_check`: still two named syncs.

**The gate order (L2-9).**

0. Step 0 (§8.0).
1. The code: the CPU reference first, then the device twins, in one patch.
2. Goldens and pins:
   - GOLDEN_AGGREGATE re-baselined once, with this section as its rationale;
   - `test_b6_logic_golden.LOOP_GOLDEN_TRAJ_DIGEST` in the same commit, with the same rationale;
   - the 11 `cuda_*_check` readers follow the constant;
   - the ship_demo pins (category A);
   - the 1.0 m inventory per §8.10.
3. `tools/twin_debt.py --mark-level`, in the LAST commit. Trust the tool's list; it will include `eos`, `div_source`, `energy_flux` (`cuda_kick_compression.cu`) and every twin whose files include `physics_runner.py` or `eos_solver.h`.
4. `cpp/build_cuda.bat` in the worktree.
5. `tools/cuda_green.py`.
6. The full suite, `pytest tests -q`.
7. `tools/match_pin_check.py --backend cuda` and `--backend resident`, against the new pins.
8. The cost gate (§8.7) and the E2E reproduction, recorded in the as-built.

A commit after step 5 voids the stamp and restarts at step 3.

**Informational, not gates (L1-11).** These depend on vents, blast data or (d)+(a):

- the idle ship_demo cost and its vent aperture;
- the quiet arena's residual wind below about 8 m/s (F5, for (d)+(a); its gate expects it);
- the frag's knockdown range (the blast's source data);
- the studio's legacy rows' rail counts.

### 8.10 The regression map (L2-7)

**Goldens.**

- **GOLDEN_AGGREGATE** is re-baselined once. Its breach runs at k = 2 on most ticks of the first 3 s (§8.6.2), and cells past 192 m/s now cap there.
- **`test_b6_logic_golden.LOOP_GOLDEN_TRAJ_DIGEST`** (`:149`; 1.0 m, live EOS, a door venting a chamber to vacuum, `:181-194`) moves with it, in the same commit, with the same rationale. Every past re-baseline moved it.
- **The CUDA checks that read GOLDEN_AGGREGATE follow the constant:** `cuda_combustion_check`, `cuda_conduction_check`, `cuda_eos_step_check`, `cuda_fire_check`, `cuda_kick_check`, `cuda_mg_solve_check`, `cuda_p62_check`, `cuda_s3_check`, `cuda_swarm_check`, `cuda_trace_saturation_check`, `cuda_trace_smoke_check`.

**The 1.0 m inventory (L2-7b), classified at step 0 against the merged base's suite run:**

- **The golden scenario and its swarm fixtures** (the breached 16×16 room of `field_ab_harness`): `_xarch_perfield_digest`, `test_swarm_digest`, `test_swarm_tick`, `test_field_ab_harness`, `cuda_swarm_check`, `cuda_swarm_batch_check`, and the other `field_ab_harness` readers. Category A: the golden moves and they follow it.
- **The 1.0 m box match pins** (`_PRE_122_BOX`, `_STAGE1_BOX`, `_RE_PIN_1_BOX`): a sealed box. Expected k = 1 and no binding cap, so they hold (INFERENCE). If one moves, it is category A with this rationale.
- **`test_b1_signal_bus`, `test_b2_nodes`:** no physics; unchanged.

**0.333 m.**

- The eight ship_demo pins of §7.3: category A, re-pinned again after U's re-pin.
- `test_wave_push`, and any test that measures a push or a wind at a breach mouth: category A, with L1-8's rationale (the ceiling holds breach winds at 64 m/s), classified at step 0.
- The rest per §7.5's categories.

**Replays and RL artefacts (L2-7d)**, §7.5's paragraph widened to 1.0 m:

- **Replays** recorded before §8 hit `ReplayDivergence` wherever k > 1 or the cap binds, now at 1.0 m too (any breach, vent to vacuum or blast). `FORMAT_VERSION` stays 8: the re-simulation differs, which is what the check is for.
- **Saved soldiers** (`agents/saved.py`), **the imitation records** (`agents/records.py`; re-made, because the observation's wind columns change) and **the rung-1 tournament pins** (`docs/rl_agent_rung1_1d_pins/*.json`) are on the old physics.
- **A run in progress** keeps its tagged checkout (`rl-agent.md`, Training snapshot). A resume refuses the code change by name and needs `--allow-code-change`.
- **For the game-ai-lead role** (a note to carry into its next session):
  - §8 changes blast and breach physics for every agent;
  - K_MAX is part of the game a run trains on;
  - a training batch pays the cost of §8.7.

### 8.11 The reds

Expected to turn green (INFERENCE unless marked):

- finding 1, the studio blast (the proxy of §8.1.1 shows it);
- `test_air_boundary.py` ambient gate 2. It is the same runaway at k_drag2 = 0; the signature is PROVED in the arena (U_MAX and the T_MAX_PHYS rail), but that test's own scene is still to be run;
- `test_grenade_kinds`, the self-knockdown, which comes from the frag's sustained winds (the proxy: no slosh). The knockdown calibration moves with (B);
- `test_fire_feedback`, where the plume turns inward from about tick 48. It looks like the same limit cycle as the slosh measured after the frag.

Other causes, one line each:

- the quiet arena's winds below about 8 m/s: the collocated coupling, for (d)+(a);
- the CUDA conductor's vent-ledger fault (patch U): vents drain rooms at a constant mass rate (§8.4, source data). On the merged base it also reds `cuda_conductor_batch_check`'s match part, alone and batched, at ticks 627-831 (§8.0);
- `test_wind_doorway_e2e` and `test_gen_ship_vents_air` (Erik-ruled): patch U's faster transport equalizes pressure. Category C, not instability;
- `test_wave_push`: category A (§8.10, L1-8);
- the non-vacuity guards of `test_unit_schema` / `test_unit_store`: match dynamics. They are classified at step 0 and again after the patch.

### 8.12 Place in the stack, and the merge unit (L2-11)

**Before (e).** The two are independent in quiet rooms, where k = 1. Three reasons to go first:

- (e)'s own gates, the full suite and the pressure arc's convergence re-run, are confounded today by blasts that go wild.
- (e) changes the multigrid, which this patch calls k times per tick, so its residue measurement belongs on the final cadence.
- The reds of §8.11 belong to this patch, not to (e).

**The merge unit (L2-11): U + §8 merge to main together, as the first unit, once §8's gates are green.**

- U alone leaves the blast regime wild (its finding 1), so U should not merge without §8.
- After §8, U's reds of §7.9 are either green or classified.
- Then (e) and (d)+(a) merge separately, each on its own green.
- Reasons:
  - main moves weekly on the conductor seam this patch edits;
  - every later patch re-pins the same eight pins and the golden again;
  - training stays on the old physics until the unit lands.
- Diffusion in m²/s comes last.

The order: U + stability (one merge), (e), (d)+(a), diffusion in m²/s.

### 8.13 Systems and rule amendments (L2-10)

**(a) Existing canonical systems the fix must use:**

- **PhysicsRunner / PhysicsEngine:** the sub-tick loop lives in `EOSSolver::step`, `eos_step_resident` and the per-call `cuda_eos_step` entry, never in Python glue (L2-3); `K_MAX` is bound in PhysicsRunner only.
- **Interior drag:** the ceiling and the cap's heat belong to the existing cap stage, not a new site. Drag stays as it is.
- **Energy closure identity / Gas energy seam / Gas energy recovery:** one new counter in the EOS group (`e_cap_heat_sum`). The recovery and its rails run per sub-tick and write back only when a rail binds.
- **Face-flux energy step:** runs per sub-tick at dt_k, with `k_flux_q` re-folded.
- **Trace transport:** unchanged. Trace rides each sub-tick's face flux, and `eos_transport_step_q(dt_k, n_sub, dx)` stays THE transport scalar.
- **Fixed-point kits:** no new rounding; `eos_subtick_dt` is a helper of the EOS step, not a kit function.
- **Batched resident engine:** one chain, one sync (max k and max n_sub in one read), per-env (k, n_sub) rows, masked kernels, books in the one arena.
- **Class P / Class T + the twin ledger, CUDA harness, A/B lockstep harness, GOLDEN_AGGREGATE, Pinned match plays:** §8.9, §8.10.
- **Benches:** `_eos_checkerboard_bench.py --stability` (gaining the k columns) and `tools/bench_conductor.py` (gaining the per-env stagger); never a new probe.
- **Not built here:** at shipped resolutions (B) makes the D2v2 per-cell c_LOCAL cap plane and U_MAX dead (§8.6.4). They stay; removing them is a separate cleanup.

**(b) New reusable systems:** none. The sub-tick count is a property of the EOS step.

**(c) Rule changes, written in the patch:**

- **New** (`energy-books.md` and `fields-gas.md`, where the loop lives): *The EOS updates pressure at least once per C_P tiles of air motion: steps 0-7 run k times per tick (k ∈ {1, 2, 4, 8}, dt_k from `eos_subtick_dt` alone), with k from the one entry estimate over the k-set (wind, divergence, source; never the kick increment), capped at K_MAX; a transport-only substep loop is never the stability mechanism. Past K_MAX the cap stage's budget ceiling K_MAX·dx/dt holds. K_MAX is a physics dial shared by training and play.*
- **Amended, `energy-books.md:40`** (Gas energy recovery), "once per tick" → *once per EOS sub-tick, over the whole accountable set; rails write back only when one binds (its counters accumulate over the tick)*.
- **Amended, `fields-gas.md` Interior drag, D6:** *the absorb and sponge stages EXPORT theirs, counted; the cap stage returns as heat the part of its removal that the same sub-tick's kick supplied (`e_cap_heat_sum`) and DESTROYS the rest, counted (`e_clamp_destroyed_sum`, a tripwire)*, citing ruling 3.
- **Amended, `device-residency.md:36`:** *Substep loops run to the batch's max n_sub, and the EOS sub-tick loop to the batch's max k; an env past its own is masked (`sub_off`, `tick_off`); the fold table's rows are (k, n_sub).*
- **Amended, `device-residency.md:41`:** a new resident core is masked inside the sub-tick loop as well as the substep loop, and the gates add `cuda_conductor_batch_check` (a quiet and a blasting env in one batch).
- **The bench rule** (§6(b), draft): `--stability` and its k columns join it.

### 8.14 For Erik (rulings, each with the default the build uses)

1. **K_MAX = 8 is a physics dial, not a speed dial.** It sets how fast air may move past the budget: 8 tiles a tick, 64 m/s at 0.333 m. Training and play must use the same value. Raising it costs speed on blast ticks; lowering it trains on a different game. *Default: 8, shared by training and play.*
2. **The 64 m/s ceiling also holds breaches, not only blasts.** At 0.333 m a hull breach's mouth wind goes from about 300 to 64 m/s, so its push on a unit is about 20× smaller. The amount of air leaving barely changes. A blast's knockdown also moves; its source data (the frag's 40 m³ and 10 MJ against about 2.6 m³ and 0.9 MJ for a real charge; the studio charges' 156 MJ; vents as a constant mass flow) is for the blast and vent session. *Default: the ceiling applies everywhere; blast and breach pushes are re-tuned in that session.*
3. **Energy the speed cap removes becomes heat again, but only what that same step's push put in.** Today the cap destroys it (rule D6), which is why blast rooms cooled by about 56 K. Anything above that, which no push in the step supplied, stays destroyed and counted as an alarm. *Default: return it as heat; rule D6 is amended for the cap.*
4. **The drag dial `k_drag2` stays as it is.** It was the hidden speed limit; the ceiling now does that job. Its law per second does not change with sub-ticks. Whether real air should be slowed more is your call. *Default: unchanged at 0.225.*
5. **Training cost.** A training batch pays for its loudest env: at 64 envs in the range room, about +14-43 % per tick with each env sub-ticking only as often as it needs, against +55-88 % if all envs did. Cutting that further means fewer solver cycles on the extra sub-ticks, measured in the patch. *Default: build per-env sub-ticks; report the measured cost before training runs on it.*

**Ruled (Erik, 2026-11-21): "5 decisions: yes to all".** All five as
defaulted: (1) K_MAX = 8, a physics dial shared by training and play; (2) the
64 m/s ceiling holds for blasts and breaches; (3) the cap's removed energy
returns as heat, only what that step's push supplied, and rule D6 is amended
for the cap; (4) `k_drag2` unchanged in this patch; (5) per-env sub-ticks are
built, and the measured training cost is reported before any training runs on
it. The build follows §8.6 with these values.

### 8.15 Critique (one critic at a time, verdicts recorded before the next)

| lens | model | verdict | findings (resolved in v2 of §8; the disposition table follows) |
|---|---|---|---|
| 1. Correctness and completeness of §8.6 | Opus | SOUND WITH FIXES | The core is right, and the int64 identity across sub-ticks is exact by construction. Items 1-4 must be fixed in the doc before the build. (1) PROVED: K_MAX = 1 does not reproduce HEAD, because U_bud = K_MAX·dx/dt is then 8 m/s; it needs separate switches for the sub-tick loop, the ceiling and the cap heat. Mutate tests 2-3 with k = 1 and the ceiling kept. Restate identity as k = 1 AND no cell above U_bud AND no binding cap. (2) PROVED structure: the batched path needs per-env k. That means per-env fold columns (Kick/MG folds are batch-uniform by value), a (k, n_sub) table, and env masks on every EOS kernel. The batch loops to max k: P(some env k > 1) is 0.81 at N = 16 and 0.9988 at N = 64. Measure eos_step_resident at forced k = 8 for N = 16 / 64. (3) PROVED: k and n_sub are frozen at entry. A detonation tick has entry u ≈ 0, so sub-ticks 2-8 run at Csub = 1.0 (the F6 regime). k = max(k_u, k_kick) undercounts the sum. Fix: take k from ceil((max|u| + filtered du)·dt/dx); for k > 1 set n_sub ≥ ceil(K_MAX/(k·CFL_ADV)); leave n_adv uncapped before dividing; state test 1 per sub-tick. (4) INFERENCE: at the cap, the frag's one-tile div source still empties and becomes an F3 sink. Add a frag leg to tests 2-3, plus a pre-build proxy `--stability --tps 192` with U_MAX 64 and N_SUB_MAX 2. (5) PROVED: the idle-saving claim contradicts k = 1 identity. Filter k_div and du too, compute the k histogram offline, and make the 5 % threshold an integer constant (space maps). (6) PROVED: (C) reverses rule D6 (fields-gas.md Interior drag, code :1206-1209). This is a modelling ruling missing from §8.13. Heat only the part the kick supplied, keep e_clamp_destroyed_sum live as a tripwire, and gate e_rail_sum / t_max_phys_hits. (7) PROVED: the per-tick items left out. Recovery per sub-tick amends energy-books.md "once per tick". p_prev (`gmap.wave_p`) must be written at sub-tick 1 only, with an internal warm-start buffer, or the water ripple splash (`water_solver.cpp:337-345`) and unseal / applier break. The entry pre-stage runs once. Add sponge_sigma. (8) INFERENCE: (B) caps breach winds at 64 m/s (sonic ~300 today), so breach knockdown drops ~20×; ruling 2 must name breaches. (9) PROVED: dt_k is not canonical. Use one `eos_subtick_dt(dt, k)` or k in {1, 2, 4, 8}. (10) Lists to extend: cuda_resident_tick.cu:1251-1256 KICK_CNT_SLOTS, bindings, stubs, currency_audit_t2.py, and P9's graph-capture nesting. (11) F5 and F7 are rightly out of scope; say which gates are informational because they depend on vents and blast data. |
| 2. Gates, regression, rollout, systems reuse, training cost | Opus | SOUND WITH FIXES | (1) PROVED, measured: §8.7 costs only N = 1. The batch loops to the max k across envs (`cuda_eos_resident.cu:1101-1127`, `device-residency.md:36`). range_hall with 2 frags + 1 C4 per rifleman in 45 s episodes: mean batch-max k 2.3-3.2 at N = 16, 4.6-6.2 at N = 64, 7.4-7.9 at N = 256, against a mean per-env k of 1.09-1.18. That is about +50-90 % tick cost at N = 64 against the P7e.5 profile. Per-env k with masked kernels brings it to about +2-8 % once GPU-bound, with a launch floor of about 1.3 ms per extra sub-tick. Make per-env k a stated requirement, rewrite §8.7, and gate with `tools/bench_conductor.py` at N = 1/16/64 with blasts staggered across envs. (2) PROVED, measured: a C4 breach of range_hall's hull holds k_u ≥ 6 on every tick to episode end (wind 47.6 m/s at the breach). Filter k_u like k_kick (N ≥ 5 %) or exclude vacuum-adjacent faces; add a breach leg to tests 2-3 and the cost table. (3) PROVED: §8.6 puts the loop in Python `_resident_chain`, which means k host syncs a tick (against `device-residency.md:38`) and orchestration in Python glue. Put the loop inside `eos_step_resident` / `EOSSolver::step` / the per-call path, with one call and one read a tick. (4) PROVED: the branch is 36 behind main, and P7e.5 rewrote this seam (`_resident_chain`, `run_resident_conductor` for N envs, the deferred tail fold, `cuda_resident_tick.*`, bindings, physics_runner). merge-tree conflicts only in `tests/twin_manifest.toml`. Step 0: merge main, rebuild CPU + CUDA, `--mark-level` + `cuda_green`, re-run §8.1's before tables on the merged base, and categorise the reds (the conductor batch check will likely show U's vent-ledger fault). (5) PROVED: test 1 holds by construction. Assert instead that solves per tick == k and that mass moved between solves is ≤ 1 tile. (6) The k = 1 path needs a lasting guard: with lens 1's legacy switches on, the default scenario still hashes to `d5d03433…487b`. Capture the A/B baseline on the merged base before any code commit, and name a 0.333 m scene. (7) Regression gaps: `test_b6_logic_golden` loop digest; the list of 1.0 m breach/vent tests (swarm scenarios share the breached level); the conductor gates (`cuda_conductor_check`, `cuda_conductor_batch_check` with a quiet + blasting env leg, `test_cuda_conductor`); replays diverge at 1.0 m where k > 1; saved soldiers, imitation records, rung-1 tournament pins and runs in progress (`--allow-code-change`). Tell the game-ai-lead role. (8) K_MAX is NOT a perf dial: the ceiling is K_MAX·dx/dt, so training and play must share it. (9) Gate order: merged base, code, GOLDEN/B6/re-pins, `--mark-level` last commit, `build_cuda`, `cuda_green`, full suite, `match_pin_check --backend cuda/resident`. (10) Rule amendments named: `energy-books.md:40`, fields-gas D6, `device-residency.md:36,41`. (11) INFERENCE: the merge unit is U + §8, with (e) and (d)+(a) after. Sound: canonical reuse, test 5, the bench exercises k > 1, training rooms at rest are k = 1, placement before (e). |

**How v2 answers each finding** (where, or ACCEPTED GAP, or REFUTED with its evidence):

| id | severity | finding (short) | v2 |
|---|---|---|---|
| L1-1 | high | K_MAX = 1 does not reproduce HEAD (U_bud is then 8 m/s); the tests' mutation is wrong | Three switches (§8.6.6). Legacy = HEAD, and the identity claim is stated exactly. The mutation of T2/T3 is "loop off, ceiling kept", measured to stay wild (§8.1.1) |
| L1-2 | high | batched path needs per-env dt_k: folds, a (k, n_sub) table, masks on every kernel; cost of max k | §8.6.7: 32-row table, folds in `EnvSub`, `tick_off` on every EOS kernel incl. MG and V-cycle; costed at N = 1/16/64 (§8.7) |
| L1-3 | high | bounds frozen at entry: Csub 1.0 on sub-ticks ≥ 2; k = max undercounts the sum; test 1 per sub-tick | n_sub floor ceil(K_MAX/(k·CFL_ADV)) with n_adv uncapped (§8.6.3); T1 per sub-tick with the honest bound max(C_P, K_MAX/k) (§8.9). **The sum is REFUTED** on measurement (§8.6.2): it holds idle ship_demo at k 2-4 on 53 % of ticks and the golden breach at 8; the in-tick speed is bounded by (B) instead |
| L1-4 | high | at the cap the frag's source tile empties into an F3 sink | Emulated (192 tps, U_MAX 64, N_SUB_MAX 2): no cell below 0.086 N0, no slosh (§8.1.1). T2's frag leg keeps N ≥ 0.05 N0 at the source and its ring. ACCEPTED GAP: at the cap the source tile's rhs can still be negative (eps/8 ≈ 25); its cure is the source data (§8.4) |
| L1-5 | medium | idle saving contradicts k = 1 identity; k_div pins idle; histogram; 5 % as an integer | n_sub stays HEAD's (unfiltered); the saving is withdrawn, ACCEPTED GAP (vents' source data). k_div filtered on the k-set; histogram measured (§8.6.2: idle ship_demo k = 1 on 480/480 ticks); `N_K_MIN` = 3277 raw, a constant valid on space maps |
| L1-6 | medium | (C) reverses D6; heat only the kick's part; keep the tripwire; gate rails; D2v2 dead | §8.6.5 formula; `e_clamp_destroyed_sum` live; rails gated on the frag; ruling 3; rule amended (§8.13); D2v2 / U_MAX dead noted (§8.6.4, §8.13) |
| L1-7 | medium | per-tick items: recovery cadence, p_prev, entry pre-stage, sponge_sigma, trace tail | §8.6.1 (wave_p once; warm start from `atmosphere`; recovery per sub-tick with the rule amended; sponge σ·k² and udamp last sub-tick; trace tail once) |
| L1-8 | medium | (B) caps breach winds; ruling 2 must name breaches | §8.6.4, §8.8, ruling 2; breach-push tests category A (§8.10) |
| L1-9 | low | dt_k not canonical | k ∈ {1, 2, 4, 8} AND one helper `eos_subtick_dt` (§8.6.2) |
| L1-10 | low | list gaps: KICK_CNT_SLOTS, bindings, stubs, currency_audit_t2, P9 nesting | §8.6.5 (slots, bindings, stubs, currency_audit_t2.py:104), §8.6.7 item 7 (P9) |
| L1-11 | info | F5, F7 out of scope; say which gates are informational | §8.9 "Informational" |
| L2-1 | high | §8.7 costs N = 1 only; per-env k a requirement; gate with bench_conductor staggered | §8.6.7 (requirement, built); §8.7 rewritten with measured profile and model at N = 1/16/64; the staggered cost gate. **Partly REFUTED:** "+2-8 % once GPU-bound" does not hold on the training room, the EOS chain being latency-bound on the GPU (1.69 ms GPU vs 0.58 ms host at N = 1); measured model +14-43 % at N = 64 vs +55-88 % batch-max; the floor levers named |
| L2-2 | high | a hull breach pins k ≥ 6 to episode end | The k-set (N ≥ 5 %, > 2 tiles from vacuum) and ε* = 1: the breach's k returns to 1 when the room has drained (~9 s; §8.6.2), mean 2.54 over 15 s; breach leg in T2; in the cost table |
| L2-3 | high | the loop must live in C++ (one call, one read), not `_resident_chain` | §8.6.1; `_resident_chain` unchanged |
| L2-4 | high | merge main first (step 0) | Done here (§8.0): `3ac5cfe9`, one manifest conflict, rebuilds, before tables reproduced; mark-level, cuda_green, reds and A/B capture left to the patch's step 0 |
| L2-5 | medium | test 1 tautological | T1 rewritten on independent observables (solves per tick == k; per-sub-tick Courant from snapshots) with three named mutations |
| L2-6 | medium | lasting k = 1 guard; A/B baseline at the merged base; a 0.333 m scene | T6 (a)-(c); baseline captured at step 0; `test_arena` + vents named as the 0.333 m `make_sim` |
| L2-7 | medium | regression map: B6 loop golden, 1.0 m inventory, conductor gates, replays and RL artefacts | §8.10; T7 conductor legs; game-ai-lead note |
| L2-8 | medium | K_MAX is not a perf dial | §8.6.6, ruling 1 |
| L2-9 | low | gate order missing | §8.9 "The gate order" |
| L2-10 | low | rule amendments not planned | §8.13 (c) |
| L2-11 | low | the merge unit | §8.12: U + §8 first, then (e), (d)+(a) |

### 8.16 As-built (2026-11-21)

Built on branch `217-eos-checkerboard` after the rulings (§8.14, Erik
2026-11-21: "5 decisions: yes to all"). PROVED means measured on this
build or read in its code; INFERENCE says so.

**What was built.**

- `34c1050a` the rulings recorded; `cbfa0afd` step 0: the named 0.333 m A/B
  scene (`field_ab_harness.arena_vents_sim`) and `bench_conductor --stagger`;
  the T6(b) baselines captured on the merged base `3ac5cfe9` before any code
  commit (untracked dev pickles, golden 120 + arena 240 ticks).
- `ae24fc24` the code, CPU and CUDA in one commit:
  - `EOSSolver::step`: the entry part (resets, step 0, the cap² scan with the
    ceiling, max|u| / max_du, `eos_kset_scan`, the k-rule, n_sub, dt_k, the
    dt-folded rates at dt_k, cmask, coeffE/S) once; steps 1-7 in a loop of
    `k_ticks`; the warm start of sub-tick ≥ 2 is `atmosphere`; σ << 2·log2 k
    (`mg_build_levels(..., sigma_shift)`); `sponge_udamp` on the last
    sub-tick; the counters reset once and accumulate.
  - One FP_HD k-rule `eos_subtick_k` and n_sub rule `eos_subtick_n_sub`
    (`eos_solver.h`), host helpers `eos_subtick_dt`, `eos_subtick_n_floor`,
    `eos_budget_cap2_q32`, `eos_check_k_max` (out of line, `/fp:strict`),
    one host k-set scan `eos_kset_scan` (step() and the per-call pre-stage),
    its device twin `K_kset_scan` + `K_src_max`.
  - The cap stage in all three kick loops (step(), the reference, the device
    kernel): heat = min(removed, max(0, KE1 − max(KE0, KEc))) with
    removed = −(bracket 4's dE), so heat + destroyed == HEAD's destroyed
    exactly; `e_cap_heat_sum` (KICK slot 15, `KICK_CNT_SLOTS` 15 → 16).
  - The per-call CUDA path loops the sub-ticks around its device substep
    chain, host mid-stage, MG and the per-call kick / flux entries (counters
    zeroed once, accumulated).
  - The batched resident engine: the source staged before the pre-stage;
    the (k, n_sub) fold table (4 × (N_SUB_MAX + 1) rows, every fold the one
    function at dt_k, the key holding K_MAX and the switches); `K_pre_fold`
    picks each env's row; `K_absorb` builds the absorb plane per env after
    the fold; ONE 16-byte D2H (max n_sub, max k) at the one `max_n_sub`
    sync; the loop `t < k_max` inside `eos_step_resident`; `tick_off(es, t)`
    on every EOS kernel (mid-stage, the MG build at every level, every
    V-cycle kernel through `MGLevelDev.{es, t}`, the kick, `K_store_atm`,
    K3, the recovery); the substep kernels' `sub_off` reads the sub-tick from
    the high bits of `sub_tick(t, s)` (one predicate for both loops).
  - `K_MAX = 8` in `[physics.eos]`, bound in PhysicsRunner.
- `da04717c` every Python transcription of the EOS group gains
  `e_cap_heat_sum`; `--stability` gains k / Cupd / epsk / caph; the CUDA
  checks follow (below). `bca7bc8e` `tests/test_eos_substep_whole_tick.py`
  and the `dbg_last_kset` seam. `8141439c` the golden re-baselines.
  `88c46b99` the rule amendments and the T7 legs.

**Deviations from §8, each with its evidence.**

1. **The ceiling is folded into the cap² plane at its one build (the scan),
   not min'd in the kick loop.** The same values reach the kick (the plane is
   trusted verbatim, D5), one site per backend instead of three kick loops.
   `u_max_hits` counts only where U_MAX itself binds.
2. **The k-set's max |S| covers the cells the source is applied to** (not
   solid, vacuum or ring) and takes |S| (a negative source would also need
   updates). On every shipped source S ≥ 0, so this is the design's term.
3. **The substep kernels carry the sub-tick in the high bits of their
   substep index** (`sub_tick(t, s) = t << 8 | s`), so the existing `sub_off`
   predicate masks both loops; kernels without a substep index take
   `(es, t)`. Equivalent to §8.6.7 item 3's masks, fewer parameters.
4. **cuda_prestage_parity_check's coverage rule** is now "every n_sub AND
   every k reached" with a fast-wind sweep for the k = 8 rows, since at
   k = 2 the floor pins n_sub at 8 (F₂ = 8).
5. **cuda_kick_check PART 2 replays one kick at k = 1** (`subtick_loop` off,
   the ceiling and the cap heat on, the ceiling folded into its cap² rebuild):
   a single-kick replay cannot reconstruct k kicks. The k > 1 CPU == GPU is
   `cuda_eos_step_check` (k compared per tick, max k == K_MAX required) and
   `cuda_s8a_check` PART 1h.
6. **cuda_eos_step_check's hardness bar** reads the LAST sub-tick's p*
   (9.2 atm on this scenario after k − 1 updates, ≥ 20 before): restated to
   ≥ 5 atm plus max k == K_MAX, dated.

**The §8.6.1 inventory, as built.** Re-folded at dt_k: `dt_q` (MG rhs),
`Kdt_raw` (kick), `Kdt2dx2_raw` (MG operator), `absorb_dt_q`, `kd_q`, `kd2_q`
with `rad_dead_q32`, `k_flux_q` / `flux_pu_cap`, the transport step
`eos_transport_step_q(dt_k, n_sub, dx)`. At dt (entry): `dt_q` of the k-rule
and n_adv, `Kdt_raw` of the max_du scan, `gdt_q = quantize(γ·dt)`,
`cp_dx_q = quantize(dx)`, U_bud. Per-tick factor: `sponge_udamp`, last
sub-tick. Per-tick term: σ << 2·log2 k. Entry only: p_prev / wave_p, the cap²
plane and c_LOCAL, cmask, coeffE/S, k, n_sub, the k-set. Carried:
`atmosphere` (the warm start), wind, temperature, gas_energy, gas. Books:
reset once, accumulated. After the EOS: the trace tail, unchanged. The div
source is subtracted every sub-tick at its rate S (k·S·dt_k = S·dt).

**Findings (measured on this build).**

1. **The k-rule lags the legacy studio charges by two ticks.** The studio's
   legacy rows dump heat into p* with no divergence source, so k is chosen
   from a calm entry state: k = 1, 1, then 2, 8 (ticks 25-28). Nmin falls to
   0.042 there (T2's 0.02 floor holds). INFERENCE: a p* spike is the kick
   increment the design refused as a k term (§8.6.2); a payload row with
   `strength > 0` (the generic explosion) has a source and gets k = 8 on its
   tick (the frag, the C4).
2. **A 1 / 2 limit cycle in k.** After the frag (and in the quiet arena) k
   alternates 1, 2, 1, 2: a k = 2 tick ends calm (u ≈ 4 m/s) so the next runs
   k = 1, whose single kick lifts u to ≈ 12 m/s. T1 holds (the bound per
   sub-tick is max(C_P, K_MAX/k)); the per-update Courant number of a k = 1
   tick after such a kick is ≤ 2. INFERENCE: k from the entry state lags the
   in-tick kick by one tick, which the design accepted (§8.6.2).
3. **The quiet arena's fastest cell sits beside the ambient ring, outside the
   k-set** (within R_K of the ring, by design): 14-18 m/s at (54, 3) while the
   k-set max is 7-9 m/s. Over 30 s: winds 6-24 m/s (median of the per-tick
   max 11.0), against 11.7-62.6 before and the 48-tps emulation's 2-8: the
   design's inference (§8.8) is NOT met in full. Drift −0.39 K over 30 s
   (−3.0 before). Residual for (d)+(a) and the ring's coupling.
4. **The hull breach's k tail outlasts the drain.** range_hall's C4 breach
   runs k = 8 for ≈ 1.5 s, then alternates 1 / 4 / 8 on residual pockets
   (≥ 5 % N, > 2 tiles from the vacuum) for the rest of the 10 s run
   (§8.6.2 measured on the pre-§8 trajectory: back to 1 after ≈ 9 s).
5. **The cap heat is large where the legacy rows rail.** smoke_light_studio
   over 3 s: the cap returned 123 % of E0 as heat (it destroyed it before),
   the T_MAX_PHYS rail removed 708 % (the rows' `heat_amount`, by design);
   the mass-weighted T ends +1145 K (it fell to −3 K before). On the frag (no
   legacy rows) the cap returned 0.89 % of E0, destroyed ≈ 1e-7 of that
   (the tripwire stays ~0), no T_MAX_PHYS hit, no rail.

#### Drag after §8 (the units audit; report only, nothing changed)

**The law, read term by term** (`eos_solver.cpp` kick loop, `cuda_kick_
compression.cu` `kick_kernel`, the reference; the folds in `step()` and
`kick_scalar_folds`):

- Stage L, `u ← u·(1 − kd_q)` with `kd_q = quantize(k_drag·dt_k)`. `k_drag`
  is 1/s, `dt_k` s: a per-second law, no tile length anywhere. PROVED.
- Stage Q, `u ← u / (1 + kd2_q·|u|)` with `kd2_q = quantize(k_drag2·dt_k)`
  and |u| in m/s (Q16): `k_drag2` is 1/m (a length scale L = 1/k_drag2 =
  4.4 m at the shipped 0.225), `k_drag2·dt·|u|` dimensionless. No dx in the
  fold, the product or the dead zone (`rad_dead_q32 = ceil(2^16/kd2_q)²`, a
  speed below which the product truncates to 0). PROVED.
- The heat (bracket 5) is `ke_energy(N, Δ|u|²)` at the derived k_ke: m/s in,
  the gas_energy currency out, no dx. PROVED.
- **So no drag term carries the tile-versus-metre bug patch U fixed in
  transport**: the law is in metres and seconds at every dx, before and after
  U. PROVED. The bug sat in what the drag was TUNED against: at 0.333 m the
  transport moved air a third of the metres its wind said, so "how far a
  puff coasts" (Erik's by-eye target, 2026-10-06, `smoke_light_studio`) was
  judged on a plume moving at 1/3 speed. INFERENCE: for the same coast
  distance in tiles, the honest value is about 3× larger in the linear
  regime (x = ln(1 + k₂u₀t)/k₂, so less than 3× where the log saturates);
  that is a look ruling for the drag session, not a units correction.
- **Per-tick versus per-second with sub-ticks.** Stage Q's map is the exact
  solution of du/dt = −k₂u² over dt_k, so k sub-ticks compose to the same
  per-second law (§8.6.1 note); stage L differs at second order
  ((1 − k₁dt/k)^k against 1 − k₁dt: 4e-8 relative at the shipped k₁dt).
  PROVED (algebra). The one honest per-sub-tick effect is rounding:
  `kd2_q` = 614.4 → 614 at k = 1 (−0.07 %) but 76.8 → 77 at k = 8 (+0.26 %),
  and the dead zone rises from 0.0016 to 0.013 m/s. INFERENCE: negligible.
- **The old "ceiling" was a step-size artefact, not a speed.** The implicit
  map's supremum is 1/(k₂·dt): 106.7 m/s per 24-tps tick, the "~90 m/s
  plateau" of §8.2.4. Per sub-tick it is 1/(k₂·dt_k), k× higher; the budget
  ceiling (64 m/s at 0.333 m) binds long before it. PROVED (algebra) and
  measured below.

**Measured on this build** (`--stability`, sums of the per-tick columns, % of
the scene's tick-0 thermal energy E0):

| scene | k_drag2 | max wind (m/s) | drag heat | cap removal (all returned as heat) | k mean | runaway? |
|---|---|---|---|---|---|---|
| quiet arena, 30 s | 0.225 | 24.1 | 0.028 % | 0 (no clamp) | 1.48 | no |
| quiet arena, 30 s | **0** | 24.7 | 0 | 0 (no clamp) | 2.00 | **no** (before §8: U_MAX 999.6 m/s in 4 s, §8.1) |
| fire_studio frag, 5 s | 0.225 | 59.5 | 0.071 % | 0.89 % | 1.31 | no |
| fire_studio frag, 5 s | 0 | 63.9 | 0 | 0.97 % | 1.61 | no |
| smoke_light_studio, 3 s | 0.225 | 63.9 | 2.8 % | 123 % | 3.82 | no |
| smoke_light_studio, 3 s | 0 | 63.9 | 0.003 % | 100 % | 4.29 | no |

(Before §8, studio ticks 25-36: drag returned 16.0 % of E0 and the cap
destroyed 19.2 %, §8.1.)

**Reading.**

- **Drag is no longer the limiter.** With `k_drag2 = 0` the quiet arena stays
  at ≤ 24.7 m/s for 30 s and no scene runs away; the blast winds stop at the
  budget ceiling (63.9 m/s), not at drag's step artefact. PROVED (the table).
- **What drag still does** is physical-scale damping: 0.03 % of E0 in 30 s
  of quiet arena, 0.07 % over a frag, 2.8 % over the studio's three legacy
  charges (where the cap's removal is 40× larger). Turning it off raises the
  quiet arena's median max wind from 11.0 to 12.3 m/s and k's mean from
  1.48 to 2.00 (the cost of a calmer-looking room). INFERENCE: its value is
  now a look and a cost question, Erik's drag session.
- **No term is in the wrong units**; the honest-value question is the
  tuning target (above), not the law.
