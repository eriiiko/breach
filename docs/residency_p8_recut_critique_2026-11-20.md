# Critique of the P8 re-cut (2026-11-20)

Adversarial review of the paragraph "P8 RE-CUT (2026-11-20" in
`docs/residency_batching_design_2026-11-07.md` (§4.5, before §4.6), read against the code at
`a7f7b924` and the inventory `docs/residency_p8_observation_inventory_2026-11-20.md`. Nothing built
or changed. One-liners ran with `C:/Users/steen/anaconda3/envs/data/python.exe` (numpy 2.5.3,
Python 3.12.14, CPU without AVX-512; `cpu_features`: AVX2 True, FMA3 True, AVX512F False); the two
timing runs used the main checkout's CPU build (`cpp/build/Release`, main `dde25e03`), since this
worktree has no build.

Severity: BLOCKER = the plan as written cannot reach its gate or targets the wrong thing; MAJOR = a
patch will fail, overrun or pass vacuously unless the plan changes; MINOR = a trap to name in the
brief.

---------------------------------------------------------------------------------------------

## Lens 1: decision 1 (floats exact by one shared function)

**1. MAJOR: the list of "four places that call libm" is incomplete. `onto_reach_q` calls `pow`, not
`sqrt`, and its result is SYNCED SIM STATE.**
Evidence: `src/simulation/action_spec_v1.py::onto_reach_q` (line 242) is
`quantize_scalar(float(dx * dx + dy * dy) ** 0.5)`. CPython's float `**` calls the C runtime's
`pow`, which is not correctly rounded. Measured: `float(k) ** 0.5 != math.sqrt(k)` for 8011 of the
integers k in [0, 4e6), the first at k = 343, 525, 550, 971, 1372 (14 below 7200, so inside throw
range). After `quantize_scalar`, 0 of the 4e6 differ. The inventory (F step 3, I table) calls this
"sqrt, IEEE, exact-class", and the plan copies that. The value goes into the throw latch, so it
reaches the digest. The same pattern sits in `control_bots.py:1296` (`_lob`), which is host-only.
Change: add it to decision 1's list. In P8.0b, change the host to `math.sqrt` (or `quantize_scalar`
of the exact integer square root). Show "no quantized reach moves" with an exhaustive sweep over
every d2 the largest batch map allows, so the change needs no re-pin. The device decoder (P8.6) then
uses IEEE `sqrt`.

**2. MAJOR: the decision is bit-reproducible as described, but it buys almost nothing that is ever
served. A counted check in the SERVED dtype is the better default for `log` / `log1p`. Keep only the
cheap half of decision 1 (`hypot` written as `sqrt`).**
Evidence:
- Every log result is used only as a value and then cast to float32 or float16. No flag, mask,
  argmax or sort reads a squashed value: the far argmax reads `load`, which is integer-valued, and
  the masks and sorts are integer. So a 1-ulp float64 difference never propagates. It shows only
  if it moves the cast. Measured: 5e6 random values in (1, 4] (the squash's output range), each
  moved by 1 ulp with `nextafter`. The float16 cast changed 0 times, the float32 cast 0 times. The
  odds per differing value are about 2^-29 for float32 and about 2^-42 for float16.
- On this machine numpy's `np.log` and `math.log` agree on 3e6 inputs, `np.log1p` and
  `math.log1p` on 1e6, and `math.hypot`, `np.hypot` and `sqrt(dx*dx+dy*dy)` on 1e6 integer pairs.
  The real cross-machine hazard is elsewhere. numpy dispatches float64 `log` to its own AVX-512
  (SVML-derived) loop on AVX-512 CPUs. On such a host, `squash` (`math.log`, line 351) and
  `squash_array` (`np.log`, line 363) can disagree with EACH OTHER inside one host observation.
  This does not happen on this CPU. It can on a Linux node.
- A hand-written log costs a reference move (P8.0b, a patch) and three implementations that must
  match op for op: Python scalar (`squash`, `d_c`), numpy vector (`squash_array`) and CUDA. It
  also slows the host builder, which is both the gate's reference and the production farm until
  P8.10.

Proposed form:
- (a) `hypot` becomes `sqrt(dx*dx + dy*dy)` on both sides (`_hit_from` line 1167, `d_c` callers
  1367 / 1398, the wind's `np.hypot` line 938). This is IEEE and correctly rounded. It equals
  CPython 3.12's correctly rounded `math.hypot` for integer inputs below 2^26 (measured: 0
  mismatches), so the host does not move on the `_hit_from` / `d_c` corpus.
- (b) `log` / `log1p`: the card uses CUDA's double `log` / `log1p`. Every gate compares in the
  served dtype, bit for bit, prints any flip and fails on a flip. The flip count is reported, so a
  failure is a finding, never a tolerance.
- (c) Write the shared module only if (b) ever goes red on a corpus, or if Erik wants
  cross-machine identity of the host reference for its own sake.

If decision 1 is kept anyway: state P8.0b's bound as the measured maximum ulp error of the chosen
implementation over a dense sweep, reported in the served dtype. "At most the last bit" is not
true for a fixed-sequence polynomial log, which is typically 1 to 2 ulp from true and up to about
3 ulp from libm. `d_c`'s `log1p(d) / log1p(L)` compounds two such errors, so P8.0b's gate as
worded would go red on its own design.

**3. MAJOR: numpy's summation order is NOT sequential for one of the two reward folds. The plan
lists it as an open item "checked in P8.7"; it is answered here.**
Evidence: `src/rl_env.py::EnvRunner._reward`.
- Line 900 (named profiles): `(counters[:, :, None] * G[None]).sum(axis=1)` reduces over a
  non-last axis. Measured equal to a sequential loop: 0 mismatches over 64 x 24 x 40 random
  float64 values.
- Lines 897-899 (`"counters"`): `(old * w[:k0]).sum(axis=1)` and `(c32[:, k0:] * w[k0:]).sum(axis=1)`
  sum a 2-D float32 array along its LAST, contiguous axis. That is numpy's unrolled pairwise sum
  (8 partial accumulators from 8 terms up). k0 = 7 + `len(CAUSE_NAMES)` = 7 + 11 = 18. Measured on
  float32 against a sequential loop: k = 7 gives 0 mismatches; k = 12, 16 and 20 give 5284, 10830
  and 8678 mismatches of 20000.

Change: in P8.7, do one of two things. Either write both host folds as an explicit loop in a named
order (a host change that moves the `"counters"` reward by float32 ulps; the named profile does not
move), or refuse the `"counters"` profile on the device backend by name. Do not reproduce numpy's
unrolled order on the card.

**4. MINOR: signed zero and NaN must be part of "value for value".**
Evidence: `_hit_from` returns `-dx / n` with `dx = float(hit[1])`, so a hit along y gives -0.0.
`raw["wind_x"] = j * dx` gives -0.0 when j = 0 and dx < 0 (`FieldPlanes.__init__` lines 942-943).
`np.array_equal` treats -0.0 as equal to +0.0, which is what the existing obs tests use
(`tests/test_obs_spec_v1.py`).
Change: the P8 gates compare the served arrays as raw bits (`view(np.uint16 / np.uint32)`), and
each gate counts NaNs (expected 0) separately.

**5. MINOR: double to half on the card.**
Evidence: numpy converts float64 to float16 directly. Going through float32 differs on 275 of 5e6
values in [-4, 4] (measured). Whether `__double2half` is a single correctly rounded step depends on
the architecture and toolkit: native `cvt.rn.f16.f64` on sm_90, a software path on the other four
of `CMAKE_CUDA_ARCHITECTURES` (75 to 89).
Change: one integer-bit-operation `double -> half` function in the observation module, held by an
exhaustive test against numpy over every float16 rounding boundary ±1 ulp of float64. It lands in
P8.0, because P8.0 already writes float16 (`far_f[0]`, exact k/16, so it does not exercise the
problem).

**6. MINOR: the CUDA flags are already right; only the brief needs to say where the new code goes.**
Evidence: `cpp/CMakeLists.txt` (lines 263-266) sets `--fmad=false -prec-div=true -prec-sqrt=true
-ftz=false` on the whole `breach_physics` target, so every new obs TU inherits "contraction off"
with no work. numpy cannot contract: each ufunc rounds on its own.
Change: put the obs double arithmetic in its own `.cu` / `.cuh` files, not in
`cuda_unit_lib.cuh`, so the unit kernels' integer library carries no double code.
