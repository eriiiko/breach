# Ray engine v2 — adversarial critique 3: determinism and CUDA (2026-09-15)

> **Status:** critique pass, remit = **DETERMINISM AND CUDA ONLY**. Physics
> (critique 1) and engine integration (critique 2) are settled in v3 and are not
> re-litigated; where a finding below touches physics it is because a
> determinism *gate* the design names cannot pass without a physics decision.
> **Reviews:** `docs/ray_engine_v2_design_v3_2026-09-15.md` at `fire-12` `968f62a`.
> **Against:** `CLAUDE.md` (iron rules + canonical table) ·
> `docs/architecture/engine/14_determinism_and_number_ingress.md` ·
> `cpp/src/fixed_point.h`, `cpp/src/cuda_fixedpoint_device.cuh` · the existing
> CUDA twins (`cuda_temperature.cu`, `cuda_raycaster.cu`, `cuda_bulk_transport.cu`,
> `cuda_resident.h`) · `physics_engine.cpp`, `bindings.cpp` · `tests/field_digest.py`,
> `field_ab_harness.py`, `cuda_harness.py`, `test_no_float_in_sim_tu.py`,
> `test_ingress_lint.py`, `_xarch_perfield_digest.py`, `cpp/CMakeLists.txt` ·
> `docs/rl_env_arc_proposal_2026-08-27.md` §A · the scheme study's `sweep_ref.py`
> and `stability_study.py` · `temperature_solver.cpp` Pass 1 + `gas_energy.h`.
>
> Every finding carries a file:line citation into the tree at `968f62a`, or the
> quoted output of a probe I wrote (`probe_sweep_q.py`, an exact-integer
> transcription of §2.3 in gather form plus a float port of `sweep_ref.py`; run
> in the session scratch directory, nothing written into the repo). Nothing in
> the tree was modified except this file. Line numbers verified 2026-09-15.
>
> **Terms used below, defined once.** *Door 1/2/3/4* — the four audited ways a
> number may enter synced state (engine/14 §3: integer arithmetic; a constant
> quantized once at load; an audited float chain of `+ − × ÷ √` quantized at the
> write; the seeded RNG). *Twin* — the CUDA kernel that must produce the CPU
> solver's bytes exactly. *Gather form* — each cell reads its upwind neighbours'
> stored outputs (no cell writes into another). *Wavefront* — the set of cells
> that can be computed in parallel because all their upwind cells are already
> done. *Atomic* — a read-modify-write that several GPU threads can issue at the
> same address without corrupting it. *Forcecast* — pybind11's default that
> silently converts a numpy array to the parameter's dtype by *copying* it.
> *Ratchet* — `test_no_float_in_sim_tu.py`, which counts lines containing the
> words `float`/`double` per sim TU and fails if a count rises.

---

## BOTTOM LINE

**P0 is cuttable as scoped, with two additions to its gate list (4b and 4f
below). P1 is NOT cuttable as scoped. P3 is blocked twice over.** The sweep's
integer core is right — I could not break conservation, the ambient fixed
point, the virtual ring, the Fleck integer form or the `E°⁻¹` inverse ("What
I could not break", at the end) — but the
design's account of *what in the tree the widening and the clamp touch* is
wrong in ways that make the CUDA gates red on the day P1 lands and again on
the day P3 flips:

1. **The int64 widening list is incomplete, and the survivors read wrapped
   values silently (BLOCKING, P1).** §3 names two pybind signatures and one
   CUDA header, "both die with the old cast at P3". But `rad_net` is
   `const int32_t*` in the temperature solver's own signature
   (`cpp/src/temperature_solver.h:646`, read at `temperature_solver.cpp:248`),
   in `PhysicsEngine::step_tail` (`physics_engine.cpp:77`), in the
   `TemperatureSolver.step` direct binding that gate 5 is written against
   (`bindings.cpp:2062-2081`), in the `step_tail` binding (`:3176-3281`), and in
   the CUDA temperature twin (`cuda_temperature.h:110`, `.cu:202`, `:220`, `:487`,
   `:568`, `:583-585` — which H2D-copies `nb = n·4` bytes of the plane). None
   dies at P3. pybind's default `forcecast` means an int64 numpy array handed
   to a `py::array_t<int32_t>` parameter is converted into a truncated *copy*
   with no error, so every un-widened reader folds wrapped garbage. The CUDA
   temperature gates (`tests/cuda_thermal_mass_check.py` et al., run by the
   suite whenever `cpp/build_cuda` exists) go red at P1. §3 also contradicts
   itself: "at P1 only the new binding carries int64" cannot coexist with
   `gmap.rad_net` widening at P1 while the old int32 cast "still feeds the fold".
2. **The clamp lives in the temperature twin, which no patch extends
   (BLOCKING, P3 and P5).** §2.8 puts the maximum-principle clamp in Pass 1;
   the Pass-1 fold has a GPU twin, `cuda_temperature.cu::temp_convert_unified`
   (`:194-240`), gated at tol 0. P4 ports only `cuda_radiation_sweep`. So at P3
   the CPU fold clamps and the GPU fold does not: the temperature backend's
   existing tol-0 gates fail at P3, before P4 exists, and again at P5 for gas.
   `rad_clamp_hits` / `e_rad_clamp_drop_sum` also need two new pinned slots in
   `TEMPERATURE_ENERGY_SLOTS` (`cuda_temperature.h:147`, 13 today).
3. **Gate 4 cannot pass on any scene with a unit beside a solid (BLOCKING,
   P1's gate and P3's live behaviour).** The body share absorbs at ambient but
   never emits (§2.3), so a wall in a body's shadow gathers less than `amb_m`
   and books `rad_net < 0` every tick. Probe [3]: a marine on air beside an
   ambient wall gives the adjacent wall cell `rad_net = −49 638` per tick,
   `−775` counts of ΔT at `thermal_mass = 64`, on a solid whose floor is 0 —
   which trips `t_low_rail_hits`, the counter gate 4 requires to be zero and
   `temperature_solver.cpp:283-288` calls "a RED". The same mechanism falsifies
   §6.3's "radiation alone cannot drive a gas cell below ambient" for shadowed
   smoke at P5. This needs a physics ruling (OPEN QUESTIONS, Q1; the finding
   is 4f).

Beyond those: the E-table bake, moved "verbatim" into a header, would compile
under `bindings.cpp`'s global `/fp:fast` (a `k4·scale + 0.5` chain MSVC may
contract to an FMA) — §5; the per-ordinate constants `s_m` are libm `cos`/`sin`
quantized at load, the exact class engine/14 §8 case 3 exempts only with a
written rationale — §5; the gas Fleck chain's first product overflows int64
(probe [8]) — §1; the CUDA twin's wavefront launch count (3 072–6 128 launches
per tick) is absent from §10 and §3's scratch shape contradicts §8.2's
concurrent ordinates — §2; gate 2's "`f < 1` forced on" is vacuous in the
excess form — §4; and "a pure function of digested state is not digested" is
contradicted by `obstacles` — §6.

Counts: **3 BLOCKING · 12 REQUIRED · 10 NOTE** (the blockers are 6a, 7b and
4f; the REQUIRED CHANGES list has 16 entries because its last entry bundles
four NOTE-level edits). Fourteen of the design's claims about the tree were
checked and hold, four do not, two are overstated (the claims table before the
REQUIRED CHANGES list).

---

## 1. The Fleck factor in integers

### 1a. The integer form is door 1 and backend-identical — **checked, TRUE**

`f_q = floordiv_q((int64)T_abs_q << 16, D)`, `D = max(T_abs_q + 2L_q, 4L_q)`.
`floordiv_q` (`cpp/src/fixed_point.h:562-566`) is `FP_HD`, already used on the
device by `cuda_bulk_transport.cu:227`, and its correction branch fires only
when the operands disagree in sign. Here both are positive (`T_abs_q >= 65536`
because `T_MIN = -292` at `config.toml:722` keeps `T_abs >= 1`; `L_q >= 0`
because `E[T] >= E[0]`), so the call reduces to a plain int64 `/`, which C++
and CUDA both define as truncation toward zero (`fixed_point.h:550-556` says so
and is the reason the helper exists). Identical on every backend by the
standard.

The algebra: `1 + alpha*g = max(1 + g/2, g)` with `alpha = max(1/2, 1 - 1/g)`
— `max` is monotone in alpha and `g > 0`, so `1 + max(1/2, 1 - 1/g)*g =
max(1 + g/2, 1 + g - 1)`. At `g = 2` both branches give 2 (continuous); as
`g -> 0` the max is 1 and `f -> 1`. Substituting `g = 4L/T_abs` gives
`f = T_abs / max(T_abs + 2L, 4L)`. Correct.

Probe [6] against the float `1/(1 + alpha*g)` over T in {0 .. 15996} x three
`(a, heat_inv_shift)` pairs:

```
   T=    0 a=0.5 his=3: L_q=2^  0.0 g=    0.000 f_q=65536 (1.00000) float=1.00000  (T_abs<<16)=2^40.2  D=2^24.2
   T= 1800 a=0.5 his=3: L_q=2^ 25.9 g=    1.805 f_q=34448 (0.52563) float=0.52564  (T_abs<<16)=2^43.0  D=2^28.0
   T=15996 a=0.5 his=3: L_q=2^ 37.7 g=  848.297 f_q=   77 (0.00117) float=0.00118  (T_abs<<16)=2^46.0  D=2^39.7
   worst |f_q/ONE - f_float| over the grid = 1.41e-05  (<= 1/65536 = 1.53e-05)
```

`0 < f_q <= ONE`, and `f_q == ONE` exactly iff `L_q == 0` — both asserted in
the probe. Overflow: `(T_abs_q << 16) <= 2^46`, `D <= 2^40`,
`a*(E[T] - E[0]) <= 65536 * 3.62e12 < 2^58`. Nothing approaches 2^63.

### 1b. The solid `L_q` chain needs the int64 `shr_round0` the design already owes — **NOTE**

`shr_round0` takes a `q16` (`fixed_point.h:410`); its device port is also
`q16` (`cuda_fixedpoint_device.cuh:224`). The design says the kit gains an
int64 twin (section 3). The operand is non-negative here, so a plain `>>`
would do — but the fold at `temperature_solver.cpp:255` and its device copy
(`cuda_temperature.cu:224`) apply `shr_round0` to the *signed* `rad_net`, so
the int64 twin is genuinely needed there and must be one `FP_HD` function in
`fixed_point.h`, never re-derived in the `.cu` (CLAUDE.md "Fixed-point kits").

### 1c. The gas `L_q` chain overflows int64 as specified — **REQUIRED**

Section 2.8 / 6.3: the gas `L_q` is "the excess through the deposit chain
(`recip_N`, `recip_cv`)" with "the int32-first-operand limit lifted;
`mul128_shr` is the primitive". But `deposit_dT_wide_q16`'s MSVC and device
bodies (`fixed_point.h:314-330`, `cuda_fixedpoint_device.cuh:112-117`) form
`stage1 = deposit * recip_n` as a **plain int64 product** before the 128-bit
step — sound for an int32 deposit (`|a*b| < 2^62`), unsound for a wide one.
Probe [8]:

```
   dep*recip_n with recip_N at N=0.01 (n_floor_heat): 2^64.4  OVERFLOWS int64
   dep*recip_n with recip_N at the self-floor denom=3: 2^72.1  OVERFLOWS int64
```

(`dep = E[3999] = 3.62e12`; `recip_N` at `n_floor_heat = 0.01` is 2^22.6, at
`reciprocal_q16`'s self-floor `denom = 3` it is 2^30.4.) The same chain is
the P5 radiative gas deposit itself (`rad_net -> dT`), so this is not a
Fleck-only problem. **Required:** specify the staged form — e.g.
`mul128_shr(mul128_shr(dep64, recip_n, FP_SHIFT), recip_cv, RECIP_SHIFT)`, two
narrows instead of the existing chain's one — state its rounding (it differs
from the `heat` deposit's one-narrow chain by <= 1 LSB, acceptable, but it is
a *different* rounding and must be declared), and confirm it is the FP_HD
`mul128_shr` on both backends (`fixed_point.h:186-212`; the `__CUDA_ARCH__`
branch is the verbatim device body). Magnitude-then-sign for the signed
deposit, as section 6.3 says.

## 2. The gather-form sweep and CPU<->GPU bit-identity

### 2a. Race-freedom inside an ordinate — **checked, TRUE, with one condition**

In gather form every thread writes only its own cell's `store[i]`,
`rad_net[i]`, `rad_flux[i]`, `rad_amb[i]`, `rad_fluence[i]` and reads two
`store[]` entries of the previous wavefront. For shear, x-major, cell `(y, x)`
reads `(y, x-sx)` and `(y-sy, x-sx)` — both in column `x-sx`, so column `x` is
a wavefront and no two of its cells touch the same address. For step, `(y, x)`
reads `(y, x-sx)` and `(y-sy, x)` — both have `sx*x + sy*y` one less, so an
anti-diagonal is a wavefront. Probe [1]/[2] asserts a valid topological order
at every read (`assert v is not None`) and reproduces the same integers for
any valid order — which is exactly what makes the twin structural. The
**condition**: wavefronts must be separated by a launch boundary or a grid
sync, and each ordinate's `store` must be its own plane if ordinates overlap
in time. Which brings:

### 2b. Section 3's scratch shape contradicts section 8.2's concurrent ordinates — **REQUIRED**

Section 3: "sweep scratch (per-ordinate stored outflow) int64 (h, w) ... keyed
`(N, h, w)`". Section 8.2: "Ordinates may run sequentially ... or concurrently
with int64 `atomicAdd`". If ordinates run concurrently, a single `(N, h, w)`
store is written by 16 ordinates at once and the gather reads the wrong
ordinate's outflow — a race the tol-0 gate would catch, but the design should
not ship it. **Required:** either (i) ordinates sequential, scratch
`(N, h, w)`, **no atomics anywhere** (each cell's four sums are written by one
thread per wavefront) — or (ii) ordinates concurrent, scratch `(N, 16, h, w)`
(4 MB at 128x256 — fine), per-cell sums by int64 atomics. State which. 2c
forces (ii) or a grid sync.

### 2c. The launch count is missing from section 10 — **REQUIRED**

"The CUDA twin launches one wavefront at a time" (section 8.2). At 128x256,
S16 half-offset: 8 x-major ordinates x 256 column wavefronts + 8 y-major x
128 = **3 072 launches per tick** for shear; step's anti-diagonals give
16 x (128 + 256 - 1) = **6 128** (P6 light). At ~3-10 us per launch on Windows
WDDM that is 10-60 ms — the whole 41.67 ms budget (R-H) — and none of it is in
section 10's cost table, which counts cell-updates only. The remedies keep
bit-identity by the gather argument: one launch per wavefront *index* covering
all 16 ordinates (`blockIdx.y = m`; needs 2b(ii)); a cooperative persistent
kernel with `grid.sync()` per wavefront; or a captured CUDA graph (the S8
residency direction). **Required:** count the launches in section 10 and
choose.

### 2d. int64 `atomicAdd` on the device — **NOTE**

CUDA has no signed 64-bit `atomicAdd`; the idiom in this tree is
`atomicAdd((unsigned long long*)p, (unsigned long long)v)`
(`cuda_temperature.cu:120`, `cuda_bulk_transport.cu:313`,
`cuda_eos_resident.cu:940` "int64 atomicAdd on two's complement is
order-free"). Unsigned wrap *is* the two's-complement signed add, so the
reinterpreted int64 is exact as long as the true sum stays in range — which
the headroom argument guarantees. Say so in section 8.2 rather than "int64
`atomicAdd`".

### 2e. Headroom — **checked, TRUE**

`E[3999] = 3 622 281 227 645 ~ 2^41.7` (probe header, the exact bake). Along a
ray, `i_out = stream - floor(stream*a) - floor(stream*b) + floor(src*a)`:
when `src <= stream` the floors are monotone so `i_out <= stream`; when
`src > stream`, `i_out <= src + 1`. So the stream never exceeds the largest
source upstream by more than one count per cell, i.e. `< 2^41.7 + 256`; the
leak stage `i_in*(1-k) + amb_m*k <= max(i_in, amb_m)` cannot raise it. Per
cell, 16 ordinates: `|rad_net|, rad_fluence < 2^46`. Probe [1] measured
`max|rad_net| = 2^41.7` on a scene seeded to the table top. Intermediate
products: `stream*a < 2^58`, `(E[T] - E[0])*w_m < 2^54`, `ex_m*f < 2^54`. The
design's "< 2^50" is conservative and correct; nothing relies on wrap.

## 3. The virtual ambient ring — **checked, TRUE, by algebra and probe**

For ordinate `m` with major share `s_m`, an out-of-grid read returns
`i_out = amb_m`, giving `fa_v = floor(amb_m*s_m)`, `fb_v = amb_m - fa_v`.

*Step* (`sweep_ref.py:57-67`): the float seeds are `bx = E_amb*w*fx_w` on the
x-inlet column and `by = E_amb*w*(1 - fx_w)` on the y-inlet row. Gather: an
x-inlet cell reads its x-upwind virtually -> `fa_v` (`bx` truncated); a
y-inlet cell reads its y-upwind virtually -> `fb_v` (`by` under the remainder
convention, the same convention every interior split uses). The corner gets
`fa_v + fb_v = amb_m` exactly; the float corner gets `bx + by = E_amb*w`.

*Shear*, x-major (`sweep_ref.py:132-138`): the push form seeds
`(1-f)*bnd + f*bnd` on the inlet column, `f*bnd` on the inlet row, and
subtracts `f*bnd` at the corner "counted twice". Gather: a column cell
`(y, col)` reads both upwind cells virtually -> `fa_v + fb_v = amb_m` (float:
`bnd`); a row cell `(row, x > col)` reads `(row, x-sx)` in-grid (at the fixed
point its `i_out = amb_m`, so `fa = fa_v`) and `(row-sy, x-sx)` virtually ->
`fb_v`; the corner reads both virtually -> `amb_m`. The double-count
correction exists only because the push form seeds by whole rows and columns;
the gather form has nothing to correct. Probe [2] confirms every cell of the
three planes is exactly zero on a uniform-ambient field, both transports, S16
and S12, with `k = 0.10` on and `f = 0.3` forced.

Probe [5], integer gather vs the float push port on a random scene:

```
  shear: max|int - float| = 51443.2 counts,  max|rad_net| = 34714610685,  ratio = 1.48e-06
  step : max|int - float| = 11676.1 counts,  max|rad_net| = 37801519434,  ratio = 3.09e-07
```

— the shift truncation, as P0 will state it.

## 4. The excess-form Fleck factor and the clamp

### 4a. The excess form's exact fixed point — **checked, TRUE; the v2 form fails**

Probe [2] (above) is the excess form. Probe [2b] is the v2 whole-emission
form on one ambient cell, `a = 0.91`, `f = 0.995`: `rad_net = 111` counts per
ordinate. Row 22 is right, and it is right *for any `f`*: at ambient the
excess is zero, so `src = amb_m` regardless of `f`, and `emitted =
floor(amb_m*a) = abs_mat` by the same integers. Conservation (probe [1]) holds
with `f` randomised per cell, both transports, `k` on and off, S16 and S12,
bodies present:

```
  shear k=    0 S16: sum_net=  -4994928061640 sum_flux= 3275759417489 sum_amb=   1719168644151  identity=0  OK
  step  k= 6554 S12: sum_net=  -6166063775942 sum_flux= 3552096666173 sum_amb=   2613967109769  identity=0  OK
```

(all eight configurations `identity=0`; each of the three sums individually
non-zero, as gate 1 demands).

### 4b. Gate 2's "`f < 1` forced on" is vacuous in the excess form — **REQUIRED**

CLAUDE.md 2026-09-08 / section 2.9 preamble: "no gate may pass vacuously". In
the excess form `f` multiplies `E[T] - E[0]`, which is zero on the uniform
ambient field gate 2 specifies, so forcing `f < 1` exercises nothing. The
non-vacuous property is the **enclosed isothermal box**: opaque walls two
cells thick at `T0`, transparent interior — the inner wall layer must be an
exact per-cell zero for any `f`, because every gather it receives is another
`T0` cell's `src` (`a = ONE` makes `emitted = src` exactly) split and re-summed
under the same shift. Probe [4], `T0 = 1263`, `f = 0.3`:

```
  shear: inner wall layer max|rad_net| = 0   outer layer (sees sky) min = -54809448   interior air = 0
  step : inner wall layer max|rad_net| = 0   outer layer (sees sky) min = -45044888   interior air = 0
```

The outer layer legitimately cools to the ambient sky. **Required:** restate
gate 2 as (a) uniform ambient, per-cell zero, any `a`, `w`, and `f` *by
construction — say so*; (b) the enclosed isothermal box's inner layer,
per-cell zero, `f < 1` forced, S16 and S12 — the successor of
`test_pf1a_radiation_books.py:399-443`'s isothermal-lattice property.

### 4c. Clamp idempotence and saturation — **checked, TRUE; one undefined case — REQUIRED**

`E_inv(phi)` = largest `b` with `E[b] <= phi`, mapped to `T = 4b` game.
`e_bucket_of((4b) << 16) = (4b << 16) >> 18 = b` (`raycaster.h:210-214`), so
`E[E_inv(phi)] = E[b] <= phi`. Probe [7]: `True` over eight boundary probes
including `E[3999]` and `3*E[3999]`. Two things the design leaves out:

- **`phi < E[0]` has no bucket.** Probe [7]: `E_inv(E[0]-1) = None`. It is
  reachable — a cell in a body's shadow (section 6 below) has `phi` below
  `16*amb_m`. The binary search must define the case (`T_cap = 0`, i.e. "no
  radiative warming above the floor", is the consistent choice since such a
  cell's `rad_net <= 0` anyway). **Required.**
- **Saturation is at 15 996, not 16 000.** `E_inv(E[3999]*3) = 15996 game`.
  So the clamp binds *below* `T_MAX_PHYS` and "the existing `T_MAX_PHYS` rail
  then owns" it (section 2.6) is backwards on the radiative sub-step: the rail
  can never engage there. Gate 4's "`t_max_phys_hits` ... non-zero in a
  deliberately over-driven one" can only be satisfied by the `heat` deposit
  branch (`temperature_solver.cpp:311-313`), not by radiation. Restate gate 4
  per counter. **Required.**

### 4d. The clamp on gas is a deposit reduction — **consistent with the seam, NOTE**

`gas_energy::deposit_railed` (`gas_energy.h:98-112`) applies the ceiling to
the stored `E` and books `e_rail`; a reduced `dE` with its own counted drop is
the same shape, and the mirror-only-write trap the "Gas temperature is a
mirror" row records is avoided by construction. One subtlety: `T_after` for
the gas clamp is `mirror_q(E + dE, N)`, a floor-division; reducing `dE` so the
mirror "lands at" `min(T_after, max(T_before, T_cap))` needs
`dE_new = N*T_target_abs - E` (exact), not a scaled `dE` — say which.

### 4e. "Compiled out (a test-only flag on the binding)" — **NOTE**

A flag on a pybind binding is a runtime argument, not a compile-time switch.
Gate 5's non-vacuity needs a `clamp_enabled=True` keyword on the
`TemperatureSolver.step` binding (the `rad_net = py::none()` precedent at
`bindings.cpp:2081`), never a global static — a global a test flips is exactly
the class of hidden state that forks a digest.

### 4f. Gate 4 cannot pass with a body beside a solid — **BLOCKING (bottom line 3)**

Section 2.3: the body share `b_i = d_i - a_i` "absorbs into `rad_flux`, never
emits". Section 6.2: "A marine on air: the marine takes it all, and the stream
behind it is zero — a real shadow." Both are exactly what the scheme does,
and the consequence is that a body is a radiative *sink at every temperature,
ambient included*: a cell downwind of it gathers less than `amb_m` while it
still emits `floor(amb_m*a)`, so its `rad_net` is negative every tick. On a
thermal solid the fold's floor is 0 = ambient (`temperature_solver.cpp:290-294`)
and every negative landing is a `t_low_rail_hits` — which section 2.9 gate 4
requires to be *zero* in the normal scenarios and the solver's own comment
(`:283-288`) calls "a RED, not a shrug". Probe [3], a 7x7 ambient room, a wall
column `a = 0.9` at `x = 6`, one marine (`d = ONE`, `a = 0`) on air at `(3, 5)`:

```
   rad_net at wall cells x=6, rows 0..6: [-4893, -9182, -37992, -49638, -37992, -9182, -4893]
   rad_flux at the marine: 389472   identity: 0
   -> with heat_inv_shift=6 (thermal_mass 64) the wall's dT this tick = [-76, -143, -593, -775, -593, -143, -76] counts
```

Every wall cell in the marine's shadow cools, every tick, from the moment P3
makes units absorb — so gate 4 is red on any scene with a unit near a wall,
and in the live game the low rail fires continuously. At P5 the same stream
deficit lands on smoke (`a_gas > 0`) and cools it *below ambient*, which
section 6.3 asserts cannot happen.

Conservation is untouched (`identity: 0` above), so this is not a books
problem; it is a gate the design cannot pass as written, and it needs a
physics ruling (OPEN QUESTIONS, Q1). The exact fix, if Erik takes it: the
body re-emits at ambient with the material's own arithmetic —
`emitted_body = (amb_m * b_i) >> 16`, `i_out += emitted_body`,
`rad_flux[i] += abs_body - emitted_body`. At ambient `abs_body ==
emitted_body` by the same integers, so probe [2]'s per-cell zero holds with
bodies present; the identity holds unchanged with `rad_flux` now signed
(negative only in another body's shadow), and `exchange.py:299-340`'s
`max(heat, rad_flux)` floors it at 0. P0's integer reference should carry this
variant so the ruling is measured, not argued.

## 5. The number doors

### 5a. The E-table bake, moved into a header, compiles off the /fp:strict floor — **REQUIRED**

The bake (`cpp/src/raycaster.cpp:62-97`) is not "one double multiply": it uses
`std::floor`, `llround`, and the chain `v = (double)k4 * scale; ... (int64)(v
+ 0.5)`. That last chain is `k4*scale + 0.5` — a multiply feeding an add,
which MSVC under `/fp:fast` is allowed to contract into a fused multiply-add
(one rounding instead of two), and a one-ULP change on `v` flips `+0.5`
rounding for any bucket that sits near a half — a different table on one
build, i.e. a moved digest. Today the chain lives in `raycaster.cpp`, which is
on the `/fp:strict` list (`cpp/CMakeLists.txt:177-189`). Section 2.6 moves it
"verbatim" into `cpp/src/emissive_table.h`, header-inline, "owned by
`PhysicsEngine`". A header-inline function is compiled by every TU that
instantiates it; `bindings.cpp` is compiled under the global `/fp:fast`
(`CMakeLists.txt:23-25`, "the pure-glue bindings.cpp ... keep the global
/fp:fast"), and the bake is triggered from a binding today
(`physics_runner.py:404` -> `bake_emissive_table`). **Required:** the bake's
*body* lives out-of-line in a strict TU (`radiation_sweep.cpp`, or an
`emissive_table.cpp` added to the strict list); the header carries only the
integer lookups (`e_bucket_of`, `E_inv`) and the table pointer. Also say that
the ratchet does not scan headers (`test_no_float_in_sim_tu.py:49-56` lists
`.cpp` only), so "0/0/0 on the ratchet" is a statement about
`radiation_sweep.cpp` alone, not about the bake — and note the ratchet counts
*lines containing the word* `float`/`double`, comments included
(`:28-33`), so a 0/0/0 baseline forbids the word in comments too.

### 5b. The per-ordinate constants come from libm `cos`/`sin` — **REQUIRED**

Section 2.3: `s_m` = shear `ONE - f_m`, `f_m = abs(minor/major)`; step
`abs(mu)/(abs(mu)+abs(eta))`, "quantized ONCE at load, door 2". The source of
`mu`, `eta` is `cos((m + 1/2) * 2*pi/16)` (`sweep_ref.py:24-27`). Door 2 is a
*config constant* snapped once; a value computed at load by a libm
transcendental is engine/14's case 3 (`materials.py:640`, `:659` — the
`math.log2` shift tables, exempted only with a written rationale and an
"integer-`bit_length` TODO"). The cross-machine risk is real but tiny (a
last-ULP difference flips `quantize` only when the exact value sits within
~1e-16 of a half-count); the rule is not tiny, and `<cmath>` `cos` in a sim TU
is the banned list in spirit even though the ratchet cannot see it.
**Required:** pin the 16 (and 12, for gate 2's S12 run) `s_m` values as
integer literals in the TU with a test that recomputes them within one count
(the "door-2 constants checked in" route engine/14 §3 door 4 names), or
compute them with the kit's `sin_q16`/`cos_q16` plus `floordiv_q`. Never
`std::cos` at load in `radiation_sweep.cpp`.

### 5c. The remaining door claims — **checked**

- `E[0]`, `w_m`, `amb_m = (E[0]*w_m) >> 16`: integer, door 1. TRUE.
- `heat_atten_q` through `optics_fixed.py`: door 2; the ingress lint scans
  `src/simulation/**/*.py` (`test_ingress_lint.py:134`), so the new module is
  covered automatically. TRUE.
- Per-unit `heat_atten` quantized "at the unit boundary": the unit attribute
  is a Python float today (L2 int-backed attrs are owed, engine/14 §7); the
  row array built in `_stamp_units_cpp` (`gamemap.py:1785-1805`) is the
  boundary. Fine, but name the quantize call (`optics_fixed.quantize_scalar`)
  and note the Python reference path `_stamp_units_python` (`gamemap.py:1839`)
  must apply the same MAX with the same quantized value, because
  `tests/test_stamp_units_cpp_ab.py` compares the two. **NOTE.**
- `[gases.*] heat_absorb` in the `beam_absorb_q16` idiom (`gases.py:148-165`):
  that idiom is one float divide then `quantize_scalar` — door 3 then 2. TRUE.
- `a_gas = min(ONE, sum_g (heat_absorb_q16[g]*N_g,raw) >> 16)`: int32 x Q16
  < 2^47, door 1. TRUE.
- The Fleck divide by `floordiv_q`, `E_inv` by binary search: door 1. TRUE.
- **P6/P7 `L°` table from `renderer/blackbody.py`**: that module computes its
  ramp with `np.power` and `np.log` (`blackbody.py:123`, `:129-130`, `:140`) —
  libm transcendentals, SIMD-dispatched per CPU (engine/14 §2 table). A table
  computed that way at load and quantized is *not* door 2; it is fine while
  light is render-only, and it becomes a determinism hole the day P7 puts
  `light_q` in the digest. **Required (P6):** bake `L°` by an algebraic chain
  (the `E°` pattern: integer `K^4`, one strict double multiply, round) or check
  the 4000 integers in as constants with a regeneration script under `tools/`.

### 5d. Guards coverage for the new TU and module — **checked, TRUE with one gap**

`/fp:strict` list: per-source, `CMakeLists.txt:177-189` — the design adds the
TU there. Ratchet: `SIM_TUS` at `test_no_float_in_sim_tu.py:49-56`, baseline
dict at `:340-346` — the design adds both. Lint: automatic. The pre-existing
gap the design should not inherit silently: `eos_solver.cpp`, `combustion.cpp`,
`bulk_transport.cpp`, `sky_exchange.cpp` are on the strict list but **not** on
`SIM_TUS`, so the ratchet's "every sim TU" is already false; not v3's fault,
worth one line so the new TU's 0/0/0 is not mistaken for suite-wide coverage.
**NOTE.**

## 6. Digest, goldens, and the widening

### 6a. The widening list — **BLOCKING (bottom line 1), the full inventory**

Every site that types `rad_net`/`rad_amb`/`rad_flux` as 32-bit, from
`grep -rn` over `src tests tools renderer cpp/src`:

| site | dies at | in the design's list? |
|---|---|---|
| `gamemap.py:479`, `:499`, `:514` allocations | — | yes |
| `temperature_solver.h:646` `const int32_t* rad_net` (the solver's signature) | never | **no** |
| `temperature_solver.cpp:248` `const int32_t rn = rad_net[i]`, `:255` `shr_round0` | never | `:255` only |
| `physics_engine.cpp:77` `step_tail(... const int32_t* rad_net ...)` | never | **no** |
| `bindings.cpp:2062-2081` `TemperatureSolver.step` direct binding (`rad_net_obj.cast<py::array_t<int32_t>>`) — gate 5's entry | never | **no** |
| `bindings.cpp:3176-3281` `PhysicsEngine.step_tail` binding (`rad_net.cast<py::array_t<int32_t>>`) | never | **no** |
| `cuda_temperature.h:110`, `.cu:202`, `:220`, `:487`, `:568` (`cudaMalloc nb`), `:583-585` (`cudaMemcpy nb` = `n*4` bytes) | never | **no** |
| `bindings.cpp:609-611` (CUDA cast binding), `:2276-2278` (CPU cast binding) | P3 | yes |
| `cuda_raycaster.h:115-121`, `cuda_raycaster.cu:193`, `:376`, `:402-404`, `:420-424` | P4 | header only |
| `tests/test_pf1a_radiation_books.py:100-102`, `test_pr1_fire_plane_cast.py:146-148`, `test_fire_heat_source.py:85-89`, `cuda_pr1_fire_plane_check.py:76` | P3/P4 | yes |
| `tests/cuda_s2b_raycaster_live_check.py:307-308` (`zeros_like`, dtype-agnostic) | P4 | — (harmless) |

The **forcecast** hazard makes the omissions silent: pybind11's
`py::array_t<T>` default flags include `forcecast`, so a `.cast<py::array_t<int32_t>>()`
of an int64 array (`bindings.cpp:2068`, `:3241`) or a by-value
`py::array_t<int32_t>` parameter (`:609`, `:2276`) converts into a temporary
int32 copy — values truncated modulo 2^32, no exception, and for the *output*
planes the writes land in the temporary and are discarded. The design's own
P1 gate "goldens unmoved (asserted)" would pass on the CPU build (the fold
reads a copy of an int32-valued plane, still exact) while the CUDA temperature
gates fail (the twin copies half the plane). **Required at P1:** widen every
row above that does not die at P3, in one commit, and make the two surviving
bindings `py::array_t<int64_t, py::array::c_style>` without forcecast so a
stale caller fails loudly.

### 6b. "A pure function of digested state is not digested" is not the spec's rule — **REQUIRED**

`DIGEST_FIELDS` (`tests/field_digest.py:79-100`) contains `obstacles` — a
per-tick output of `stamp_units` (`physics_engine.h:465`), a pure function of
`permeability` (material) and unit positions, both digested elsewhere. The
file's own rationale is "Topology fields (wall_hp/material/obstacles/
is_vacuum) are included: a wall destruction mutates them and that IS synced
state" (`:75-78`). So the tree's precedent for a stamp output is *digested*,
and the design's rule would exclude `dyn_heat_atten_q` on a principle the
digest does not follow. The other precedent (`heat_inv_shift`, `face_shift`,
`thermal_solid`, `cool_shift`, `heat_atten` — static material projections,
none digested) supports `heat_atten_q`'s exclusion. Engine/14 §5 L3's purpose
is to *name the field* of first divergence; a `dyn_heat_atten_q` desync would
surface a tick later as `temperature`. **Required:** decide `dyn_heat_atten_q`
explicitly, and if it enters, do it at P3 in the same commit as the P3
re-baseline — a `DIGEST_SPEC_VERSION` v6 bump plus every golden regenerated is
what P3 already does, so the marginal cost is zero then and non-zero at any
other time. `rad_*` exclusion by the wipe (`field_digest.py:50-52`) and
`rad_fluence` by the same argument: TRUE, consistent.

### 6c. "Goldens unmoved at P1" — **checked, TRUE under conditions the design does not state — REQUIRED**

The claim holds if: (i) the old cast's arithmetic is unchanged — widening its
accumulator from int32 to int64 changes any scene that wrapped, and
`tests/test_pf1a_radiation_books.py:322-340` documents a firestorm scene at
"96% of INT32_MAX" whose bound was lowered to avoid wrap, with `rad_net`'s
"documented out-of-band [wrap] contract (`raycaster.h rad_signed_add`)". That
contract and that test change meaning at P1, not P3 — re-disposition it at
P1 (an int64 accumulator makes the scene exact; the assertion should say so).
The A/B default scenario (`field_ab_harness.py:109-139`, one 0.8 fire on 16x16)
is far from wrap, so the committed goldens hold — assert `max|rad_net| <
2^31` on it once, so the claim is checked rather than believed. (ii) The
fourth `stamp_units` output leaves the three existing outputs' bytes unchanged
— true if the C++ and Python paths grow identically (5c). (iii) The Pass-1
clamp is dormant with `rad_fluence == nullptr` and the int64 `shr_round0`
returns the same value as the q16 one on int32-range input — true by
construction, assert it in the kit's test. (iv) Nothing in step 2b writes a
digested field — true, it writes the four rad planes (wiped) only.

### 6d. Section 3's P1 shadow planes are undeclared — **REQUIRED**

Section 11 P1: "the sweep writes shadow planes"; section 3 lists `rad_net`,
`rad_amb`, `rad_flux`, `rad_fluence` as *the* planes and widens them at P1
while the old cast still fills the first three at P1 (`physics_runner.py:819`,
`:1207` -> `bindings.cpp:2276-2278`). Two writers into one plane in one tick
would make gates 1-2 unassertable. **Required:** name the P1 shadow planes
(e.g. `rad_net_sweep`, `rad_flux_sweep`, `rad_amb_sweep`, `rad_fluence`, all
int64, wiped beside the others, deleted or renamed at P3 when the old writer
dies), or state that P1 widens the live planes and the shadow sweep writes a
second set. Either way, the "one `fill(0)` line" in the conductor becomes four
at P1 and three fewer at P3.

## 7. The A/B lockstep harness and the CUDA harness

### 7a. What the A/B harness can and cannot see — **NOTE**

`capture_trajectory` snapshots after `sim.step()` (`field_ab_harness.py:292`),
i.e. after the conductor's wipe (`simulation.py:1603-1616`): the four per-tick
planes are always zero in an A/B snapshot and in the digest. So gates 1-3 and
7 cannot run through the A/B harness or the digest; they need a direct call
(the sweep binding, or `PhysicsEngine.step_tail`) with the planes read *before*
the wipe. The design's P4 gate "tol-0 lockstep on every plane" through
`tests/cuda_harness.py` is right in kind — `run_cuda_script` (`:105-112`) is a
subprocess that can call anything — but the check script must call the engine
directly, as `cuda_pr1_fire_plane_check.py` does today. Say so. And at P3
"the A/B lockstep harness" (gate 8) proves run-to-run determinism of one
build and *localises* the expected before/after difference; it cannot be a
0-ULP gate across a law change — state which of the two P3 uses it for.

### 7b. The clamp lives in the temperature twin, which no patch extends — **BLOCKING (bottom line 2)**

`rad_clamp_hits` and `e_rad_clamp_drop_sum` live on `TemperatureSolver`
(section 2.8). The GPU temperature twin returns its counters through a
pinned block, `TEMPERATURE_ENERGY_SLOTS = 13` (`cuda_temperature.h:147`,
enum `cuda_temperature.cu:101-116`, folded at `physics_engine.cpp:349-362`), and
its two hit counters through `hits`/`low_hits` (`cuda_temperature.cu:226`,
`:231`). The clamp on the device needs: `rad_fluence` (int64 plane, H2D),
the `E°` table pointer (`e_table` H2D, as `cuda_raycaster.cu` does today), a
device `E_inv` (the same 12-compare binary search, `FP_HD` in
`emissive_table.h`), a third hit counter, and slot 13 for the gas drop at P5.
**Required:** list these in P3 (solids) and P5 (gas), or move P4 ahead of P3
and put the temperature-twin clamp in P4 — either sequencing is fine; the
current one is not.

### 7c. The `E°⁻¹` device twin — **NOTE**

A binary search is branch-divergent per thread but data-independent in
*result*; the integer compares are exact on both backends. Keep the loop a
fixed 12 iterations (the `sqrt_q16_dev` precedent, `cuda_fixedpoint_device.cuh:242-259`:
"FIXED 32-trip loop ... data-INDEPENDENT") so the device and host bodies are
one `FP_HD` function.

## 8. The residency path (RL-batch habits section A)

### 8a. The existing rad planes are not on the residency path today — **REQUIRED**

`_RESIDENT_SYNCED` / `_RESIDENT_MASKS` (`gamemap.py:124-160`) carry
`gas_energy` (int64) but none of `rad_net`, `rad_amb`, `rad_flux`; the old
cast fills them on the host mirror at `_step_resident` step 1
(`physics_runner.py:1207`) and `step_tail` reads them from the mirror
(`:1356-1372`). "`rad_fluence` allocated through the residency path from day
one" therefore means: **all four** planes join `_RESIDENT_SYNCED` at P1 (the
int64 precedent is `gas_energy`, `:130`), which also makes them part of
`to_host()`'s *default* set (`gamemap.py:1741-1747`) — harmless only because
the resident tick forbids a defaulted `to_host()` (`physics_runner.py:1340-1345`).
Say all of that; today the design implies the three existing planes are
already resident.

### 8b. What "the twin" is at P4, concretely — **NOTE**

The precedent (`cuda_resident.h:6-24`): a `*_launch_resident(...)` core that
takes device pointers and does launches only, wrapped by a per-call `*_step`
that mallocs, H2D-copies, launches, syncs, D2H-copies, frees — which is what
`step_tail` dispatches today for temperature (`cuda_temperature.cu:535-595`).
At P4 the sweep is dispatched the same way from step 2b, on the host mirror,
on both the normal and the resident path — so per tick it pays H2D of
`temperature`, `heat_atten_q`, `dyn_heat_atten_q`, the gas planes (P5) and D2H
of four int64 planes (~2 MB at 128x256 in total). That is the same tax the
temperature twin pays and it is absent from section 10. The `(N, h, w)` shape
for the *core* is section A rule 1 and is right; the wrapper is `N = 1`.

### 8c. The dormancy flag contradicts section A rule 4 — **NOTE**

"A *device-side* all-ambient flag may skip the launch" (section 8.2): a device
flag can only skip a *host* launch after a D2H readback and a sync — which is
host-side gating with a stall, the thing rule 4 forbids. The honest form is no
skip at all: a uniform-ambient cell costs the same as any other and computes
exact zeros (probe [2]), and the sweep's whole thesis is that cost is set by
the grid. Delete the flag.

### 8d. Consistent with section A otherwise — **checked, TRUE**

Rule 2 (no new host tick logic): the sweep is step 2b of
`PhysicsEngine::step_tail`, which the resident path already brackets on the
mirror (`physics_runner.py:1356`). Rule 5 (scratch keyed `(N, h, w)`): see
2b for the ordinate axis. Rule 6 (golden discipline): P1 asserts goldens
unmoved.

---

## 9. Claims about the current code

### Checked and TRUE

| claim (v3) | where | evidence |
|---|---|---|
| `floordiv_q` at `fixed_point.h:562`, exact, backend-identical | §2.8 | `fixed_point.h:562-566`; device use at `cuda_bulk_transport.cu:227` |
| `reciprocal_q16` (`:458`) is truncated Newton, not used | §2.8 | `fixed_point.h:445-508` |
| `E°` already int64, 4000 x 4-unit buckets, `e_bucket_of` saturates | §2.3, §2.6 | `raycaster.cpp:55-97`, `raycaster.h:203-214` |
| `E°[3999] ~ 3.6e12` | §3 | probe: `3 622 281 227 645` |
| `T_MAX_PHYS = 16000` at `config.toml:186`, `temperature_solver.h:394` | §2.6 | verified |
| `T_MIN = -292`, `kelvin_ambient = 293`, slope 1 | §2.8 | `config.toml:722`, `:813-814` |
| The Pass-1 fold at `:247-299`, `shr_round0` at `:255`, applied-dT booking at `:297-299`; gas branch `deposit_railed` at `:373-388` | §2.8, §6.3 | `temperature_solver.cpp` |
| Six closure groups | §6.3 | `tests/test_thermostat_books.py:67-92` |
| Wipes at `simulation.py:1603`, `:1610`, `:1616`; slots 6/7 at `:1392`/`:1405` | §3, §6.2 | verified |
| `/fp:strict` list at `CMakeLists.txt:177-189`; ratchet `SIM_TUS` `:49-56`, stale comment `:47-48` | §8.1 | verified |
| Nothing outside `raycaster.cpp` calls `march_ray*`; the render march is `lighting.py:293` | §0 row 1 | `grep`: only `renderer/lighting.py:293` (`cast_source_directional`) |
| 46 old-law tests in four files (11 + 11 + 14 + 10) | §11.1 | `grep -c "def test_"` |
| `gamemap.py:487-494` order-freedom argument; `:137` always-upload set; `:1356`/`:1578` `heat_atten` seam; `:1749-1826` stamp wrapper | §2.7, §3 | verified |
| `physics_runner.py:819`, `:1207` cast sites; `:1356` resident bracket; `:1557-1560` dormancy early-out | §0 row 16, §8.2 | verified |
| Branch green: 2453 passed | App. B | this session: `2453 passed, 4 skipped, 4 xfailed in 198.06s` |

### Checked and FALSE

| claim | why |
|---|---|
| "the pybind signatures that take `rad_net`/`rad_amb`/`rad_flux` (`bindings.cpp:609-611`, `:2276-2278` — both die with the old cast at P3, so at P1 only the *new* binding carries int64)" (§3) | four more 32-bit sites survive P3: `temperature_solver.h:646`, `physics_engine.cpp:77`, `bindings.cpp:2062-2081`, `:3176-3281`, plus the CUDA temperature twin (`cuda_temperature.h:110`, `.cu:202/220/487/568/583`). §6a |
| "the CUDA header (`cpp/src/cuda_raycaster.h:108ff`, dies at P4)" is the CUDA side of the widening (§3) | `cuda_temperature.{h,cu}` reads `rad_net` and never dies. §6a |
| "for `Φ >= E°[3999]` the inverse returns the table top, which the existing `T_MAX_PHYS` rail then owns" (§2.6) | the table top is 15 996 < 16 000; the clamp binds first and the rail never engages on the radiative sub-step. §4c |
| "radiation alone cannot drive a gas cell below ambient because its excess emission is zero there" (§6.3) | true only when `stream >= amb_m`; behind a body share the stream is below `amb_m` and `rad_net = -emitted < 0`. Probe [3]. Bottom line 3 |

### Checked and OVERSTATED

| claim | what is true |
|---|---|
| "one formulation, two backends, no atomics inside an ordinate" makes bit-identity *structural* (§2.3, §8.2) | true inside an ordinate; across ordinates §3's `(N, h, w)` scratch and §8.2's "may run concurrently" are incompatible, and the per-wavefront launch count is 3 072-6 128 per tick. §2b-2c |
| "`E°`, `L°`, `w_m`, `s_m`, `amb_m` ... quantized once at load (door 2)" (§8.1) | `s_m` derives from libm `cos`/`sin`; `L°` from `np.power`/`np.log`. Door 2 names *config constants*; these are load-time transcendentals (engine/14 case 3). §5b-5c |

---

## REQUIRED CHANGES, in priority order

1. **(BLOCKING, P1)** Complete the int64 widening inventory (§6a table): the
   temperature solver's signature and read, `step_tail`'s parameter, the two
   surviving bindings (int64, no forcecast), the CUDA temperature twin's
   pointer, malloc size and memcpy size, the int64 `shr_round0` twin. One
   commit. Resolve §3's "at P1 only the new binding carries int64".
2. **(BLOCKING, P3/P5)** Put the clamp into `cuda_temperature.cu` in the
   patch that makes it live (P3 solids, P5 gas): `rad_fluence` H2D, the `E°`
   table H2D, an `FP_HD` `E_inv`, a third hit counter, slots 13-14 of
   `TEMPERATURE_ENERGY_SLOTS` with the pinned enum — or reorder P4 before P3.
3. **(BLOCKING, gate 4 / P3)** Rule the body share's ambient emission (§12
   Q1). If the body re-emits at ambient (`emitted_body = (amb_m*b) >> 16`,
   `rad_flux += abs_body - emitted_body`), the ambient fixed point is exact
   with bodies present, the identity is unchanged with `rad_flux` signed, and
   `exchange.py:299-340`'s consumer floors it at 0. If the sink stays, gate 4
   must exempt shadowed cells and the P-F1a low rail changes meaning.
4. Name the P1 shadow planes and their lifetime (§6d).
5. Fix the gas `L_q` / radiative-deposit chain: staged `mul128_shr`, declared
   rounding, both backends (§1c).
6. Bake `E°` out-of-line in a strict TU; header = lookups only; say the
   ratchet does not see headers or comments (§5a).
7. Pin `s_m` (S16 and S12) as integer literals with a recompute test; no
   `std::cos` in the TU (§5b).
8. Choose the CUDA wavefront mechanics; make the scratch shape and the
   atomics story consistent; count the launches in §10 (§2b-2c, §2d).
9. Define `E_inv` for `phi < E[0]`; restate gate 4 per counter
   (`t_max_phys_hits` from the `heat` branch only) (§4c).
10. Restate gate 2: (a) uniform ambient by construction, (b) enclosed
    isothermal box, inner layer, `f < 1` (§4b).
11. Decide `dyn_heat_atten_q`'s digest membership; if in, at P3 with the
    re-baseline (§6b).
12. Re-disposition `test_pf1a_radiation_books.py`'s wrap-contract scene at
    P1; assert the A/B scenario is wrap-free (§6c).
13. Add all four rad planes to `_RESIDENT_SYNCED` at P1 and say the three
    existing ones are not resident today (§8a).
14. P4's check script calls the engine directly, before the wipe; state what
    the A/B harness proves at P3 (§7a).
15. `L°` (P6) by an algebraic bake or checked-in constants, before P7 (§5c).
16. The gas clamp's `dE_new` is `N*T_target_abs - E`, exact (§4d); the gate-5
    switch is a binding keyword, not "compiled out" (§4e); `_stamp_units_python`
    grows with the C++ stamp (§5c); delete the dormancy flag (§8c).

## OPEN QUESTIONS — only Erik can settle these

1. **Does a body radiate?** The body share is a pure sink at any temperature.
   A marine at room temperature standing by a wall cools the wall every tick
   (probe [3]) and trips the low rail the design calls a RED. The exact fix is
   one line (the body emits at ambient, the same integer arithmetic as the
   material share) but it is a physics choice — it says a unit is a grey body
   at ambient, and it makes `rad_flux` signed (negative in another body's
   shadow). The alternative is to accept shadow-cooling and re-scope the rail.
2. **Does `dyn_heat_atten_q` enter the digest at P3?** Zero marginal cost
   then; the digest names the diverging field one tick earlier; the precedent
   cuts both ways (`obstacles` in, `heat_inv_shift` out).
3. **P4 before P3, or the temperature-twin clamp inside P3?** Either keeps
   the CUDA gates green; the first keeps P3 smaller, the second keeps the
   sequence.

## Suite state

`C:/Users/steen/anaconda3/python.exe -m pytest tests -q` at `968f62a`:
`2453 passed, 4 skipped, 4 xfailed, 3 warnings in 198.06s`. Matches Appendix
B. Nothing in the tree was changed.

## What I could not break

- **Conservation.** `sum(rad_net) + sum(rad_flux) + sum(rad_amb) == 0` exactly,
  int64, randomised `a`, `d >= a`, `k`, `T`, `f`, both transports, leak on and
  off, S16 and S12, bodies present, streams to `2^41.7` (probe [1], eight
  configurations, all `identity=0`).
- **The uniform-ambient fixed point in the excess form.** Every cell of all
  three planes exactly zero, any `a`, `w`, `f`, both transports, S16 and S12,
  leak on (probe [2]); and the enclosed isothermal box's inner layer exactly
  zero at `f = 0.3` (probe [4]).
- **The virtual ring.** Reproduces `sweep_ref.py`'s seeding for both
  transports including the shear corner, by algebra and by the fixed point.
- **The Fleck integer form.** Door 1, one plain int64 `/` on both backends,
  within one count of the float over the whole table (probe [6]); the
  `max(1 + g/2, g)` algebra including `g = 2` and `g -> 0`.
- **`E_inv` idempotence** (`E[E_inv(phi)] <= phi`, probe [7]).
- **Headroom.** The stream is bounded by the largest upstream source plus one
  count per cell; per-cell sums stay below `2^46`; no product above `2^58`.
- **Race-freedom inside an ordinate**, given one plane per ordinate and
  wavefronts separated by a launch or a grid sync.
- **The seam story on gas** (a counted deposit reduction through
  `deposit_railed`; no mirror write).
