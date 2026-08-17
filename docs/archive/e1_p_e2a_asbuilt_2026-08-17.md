# P-E2a as-built — conduction in energy form, with a per-face limiter (energy-books arc, 2026-08-17)

**Status: as-built record for P-E2a (design
`energy_transport_design_2026-08-16.md` v2.2, §2.3 + §6 row 4). Branch
`storm-damping`, base `3a15103` (the P-E1 as-built docs commit). CPU **and**
CUDA in the SAME commit — the tree is never parity-broken at a commit
boundary. No dial moved; `config.toml` untouched. Code commits: `5d7e692`
(the patch) and `38ed00b` (two gate call-sites the patch's own message got
wrong — see §8.1).**

---

## 0. Headlines

1. **Conduction's energy books close to an IDENTITY, not a bound.** On every
   tick, on both backends, `Σ_cells ΔT_i·C_real_i == e_cond_trunc_sum +
   e_cond_cap_sum`, because `Σ_cells ΔE_i == 0` **exactly** in int64. Gated on
   150 ticks of a heterogeneous field, on 400 ticks of the sealed-room E2E, and
   on 121 synthetic CUDA configs.
2. **Face antisymmetry is exact and independently verified.** A separate Python
   transcription of §2.3 reproduces the C++ field **bit-for-bit**, and its
   per-face ΔE array sums to **exactly 0** over every shared face — worst
   residual `0` over 218,396 live-face samples.
3. **The maximum principle survived the change of currency, measured.** Worst
   aggregate excursion beyond the frozen field's `[min, max]`: **0 raw counts**
   over 5 seeds × 60 ticks of a 14×14 mixed-material field. Worst per-face
   fraction `|ΔE| / (g·C_min)` = **0.250000** against the limiter's 0.5 ceiling
   — a 2× margin, so `cond_limit_hits` is **0** everywhere and the limiter is
   the structural backstop the design says it is, not an operating mechanism.
   In the 11,760-pair single-face sweep (640× capacity ratios) there were
   **zero** gap inversions and **zero** gap growths.
4. **The biggest silent energy channel in the sealed room is closed.** The old
   ΔT law moved equal ΔT to both ends of a solid↔air face whose capacities
   differ by ~32×, so it moved (or destroyed) 32× more energy on one side than
   the other. That is now exactly conservative. Measured consequence on the
   window row: the `tail` pass's gas-thermal contribution flipped from
   **−20.67 to +0.81** over 4800 ticks — the wall no longer drains the gas at
   32× the honest rate.
5. **Kirchhoff re-gate GREEN on both backends** (the P-R4 headline): equal-T
   pairs net `rad_net` exactly 0 (CPU: `test_pf1a_radiation_books` 11/11,
   including gate (iii)'s `max|rad_net| == 0` on isothermal lattices at 3
   temperatures; GPU: `rad_net` byte-identical to the CPU on 5 multi-source
   casts + 20 evolving ticks + a 30-tick all-backends-on trajectory across all
   26 synced fields). The conduction rework did not disturb it.
6. **CPU↔CUDA tol 0 across the lockstep set**, including the six new energy
   counters, per-call **and** resident.
7. **The cold rail is gone on the window row.** `t_min_gas` over 4800 ticks:
   **−0.1908 → 0.0000**; ticks below −250: 0 both sides (the −288.65 floor of
   P-E0 was already closed before this rung). §6 reports the full re-measured
   row — **no fix was chased; §2.9 owns the engine.**
8. **Suite: 48 failed / 2173 passed / 5 skipped / 1 xfailed** against a
   baseline of 48 / 2169 / 5 / 1. **Set-diff empty in both directions**; the
   +4 passes are this patch's four new conduction gates.

---

## 1. THE OLD LAW, TRANSCRIBED BEFORE THE REWRITE (design §2.3 requires this)

`temperature_solver.cpp` Pass 2, as it stood at `3a15103`:

```cpp
scratch_.resize(n);
int32_t* temp_new = scratch_.data();
const int NO_FACE = no_face;

for (int y = 0; y < h; ++y) {
  for (int x = 0; x < w; ++x) {
    const int i = y * w + x;
    const int32_t* fs = &face_shift[i * 4];   // [N,S,E,W] for this tile
    const int32_t ti = temperature[i];
    int64_t acc = 0;
    for (int d = 0; d < 4; ++d) {
      const int s = fs[d];
      if (s == NO_FACE) continue;             // grid edge or kappa==0 -> no face
      const int ny = y + DY[d], nx = x + DX[d];
      if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
      const int32_t tn = temperature[ny * w + nx];
      acc += (int64_t)(tn - ti) >> s;         // arithmetic shift == floor
    }
    temp_new[i] = (int32_t)((int64_t)ti + acc);
  }
}
for (int i = 0; i < n; ++i) temperature[i] = temp_new[i];
```

**The exact per-face rate.** `(T_j − T_i) >> face_shift[i][d]`, i.e. a fixed
fraction `2^-s` of the temperature gap, where `s` comes from the load-time
harmonic-mean face table (`materials.py::_build_conduction_tables`), clamped to
`[SHIFT_MIN=2, NO_FACE=63]`. Shipped values: hull-hull 2, wood-wood 8,
hull-air 10, wood-air 11.

**The skip sets.** A face is skipped iff **this cell's own** `face_shift` entry
is `NO_FACE` (grid edge, or κ==0 on either side of the pair — the table writes
`NO_FACE` for both orders, so it is symmetric in practice); plus a
belt-and-braces bounds guard. **There is no medium test and no vacuum/ring/
solid skip in this pass**: every cell participates, including open vacuum cells
and ambient-ring cells. Conduction is κ-keyed only — it is deliberately NOT one
of the six `MEDIUM-TEST SITE n/6` marks (design §2.2: furniture, conductivity 0,
therefore has COOL_SHIFT as its one loss channel).

**The solid↔air branch: there isn't one.** That is the point. Air was given a
small nonzero `conductivity` (0.024) at EOS-P2, which made hull↔air a live face
in the *same* whole-grid loop, with no branch and no capacity term. Decisions
item 7 called this "the primary sealed-room energy sink" — and it was a sink
precisely because it was not conservative.

**The convex bound that gave the maximum principle for free.** Writing
`r_d = 2^-s_d`, the update is `T_i' = T_i + Σ_d r_d (T_j − T_i)`, i.e. a
weighted combination of `T_i` and its neighbours with weights `r_d ≥ 0` and
`1 − Σ r_d`. With `SHIFT_MIN == 2` every `r_d ≤ 1/4` and there are 4 faces, so
`Σ r_d ≤ 1` and the update is **convex** — no new extremum can ever be created,
unconditionally stable for all time (proposal §2.6).

**Why it had to go, stated in its own terms.**

- It relaxes TEMPERATURES. Across hull↔air the endpoints' capacities differ by
  ~32× (thermal_mass 32 vs N·c_v ≈ 1), so "i loses ΔT and j gains the same ΔT"
  moves 32× more ENERGY into the light side than it takes out of the heavy one
  — a mint or a destruction of the same class the arc exists to close, invisible
  to every gate because the flat `Σ T` looked conserved.
- It is **not even antisymmetric in ΔT**. The arithmetic right shift rounds
  toward −∞: for a gap `g > 0` the cold side gains `floor(g/2^s)` while the hot
  side loses `ceil(g/2^s)`. Every conducting pair therefore lost 0 or exactly 1
  count per tick, uncounted. (`test_eos_p2_sealed_room_energy`'s old epsilon
  bound was built on precisely this identity, and dies with it — §7.)

---

## 2. THE NEW LAW (design §2.3, four constraints)

### 2.1 Capacity — one transcription, shared by both backends

`conduction::cell_capacity_q` (`temperature_solver.h`, FP_HD so the `.cu`
calls the same function) returns a cell's heat capacity in Q16.16, such that
raw energy `E = C·T`:

| medium | `cap_used` (the divisor of record) | `cap_real` (what the counters price with) |
|---|---|---|
| object (`thermal_solid`) | `1 << (min(heat_inv_shift, CAP_SHIFT_MAX) + 16)` | `1 << (min(heat_inv_shift, 30) + 16)` |
| gas | `(max(N,n_floor_q)·c_v_q) >> 16`, clamped to `[1, 2^28]` | `(max(N,0)·c_v_q) >> 16`, clamped to `[0, 2^28]` |

- The object capacity is **exactly the divisor Pass 1's `heat >> heat_inv_shift`
  deposit uses**, so a deposit and a conduction gain of equal energy raise T by
  equal amounts. No new dial.
- The gas capacity uses **the same `n_floor_heat` dial the deposits use**
  (design §2.3's explicit instruction). Its VALUE was not touched — P-E2b owns
  that. `n_floor_q` was simply hoisted out of the Pass-1 block to the top of
  `step()` so both readers share one quantization; `c_v_q = quantize(c_v)` is
  new (Pass 1 needs `1/c_v` via `make_recip`, the capacity needs `c_v` itself).
- `CAP_SHIFT_MAX = 12` (C ≤ 4096) is the **int64-overflow guard** on the face
  product `g·C_min` (|g| ≤ 2^31 raw ⇒ product ≤ 2^59, with room for the 4-face
  sum and the `ΔT·C` counter products). The shipped table's largest
  `thermal_mass` is **32** — seven doublings below the ceiling — and gas `N·c_v`
  never approaches 4096 either, so it is inert. It is a clamp rather than an
  assert because a clamp is deterministic on both backends, and when it binds
  its energy is **counted** (`cap_used ≠ cap_real` ⇒ `e_cond_cap_sum`).
- `cap_used ≥ 1` is a divide-by-zero guard for direct-binding callers that set
  `n_floor_heat` or `c_v` to 0. One raw count is 2^-16 of a unit; it never binds
  in the sim.

The two planes (`cap_used_`, `cap_real_`) are built **once per `step()`, ahead
of every pass**, because they depend only on frozen inputs (the medium mask, N,
the dials) and never on T. Three passes read them: Pass 0a prices its wipes,
Pass 2 IS the law, Pass 3 prices its cooling. They are transient
`mutable std::vector<int64_t>` scratch — never synced, never digested (R4).

### 2.2 The face quantum — constraints 1 and 4

`conduction::face_energy_q`:

```cpp
const int64_t d    = t_j - t_i;
const int64_t g    = (d < 0) ? -d : d;
const int64_t cmin = (cap_i < cap_j) ? cap_i : cap_j;
const int64_t full = g * cmin;            // energy to close the gap through C_min
int64_t q          = full >> s;           // the face's baked rate
const int64_t lim  = full >> LIM_SHIFT;   // constraint 4, LIM_SHIFT == 1
if (q > lim) { q = lim; ++*limit_hits; }
return (d < 0) ? -q : q;
```

**Constraint 1 (antisymmetry) lives in the magnitude-first shape.** The
magnitude is computed from `|ΔT|` and the sign re-applied afterwards, and every
other input is symmetric in the endpoint pair (`min`, `g`, `s`). So evaluating
the same face from the other cell returns EXACTLY the negation — no rounding
mode, shift, or clamp inside can break it, because they all act on the
magnitude. This is the precise repair of the old law's `floor` asymmetry.

`s` is `max(s_ij, s_ji)`: the pass reads **the neighbour's facing `face_shift`
entry** as well as its own, and skips the face if EITHER is `NO_FACE`. The
shipped harmonic-mean table is symmetric, so this is bit-identical to reading
one side only; it is done so antisymmetry is **structural** rather than a trust
in the bake.

**Constraint 4 (the limiter).** Moving `E` across a face changes the gap by
`E/C_i + E/C_j ≥ 2E/C_min`. Capping `|ΔE|` at `(g·C_min)/2` therefore
guarantees the gap can never invert — no endpoint passes the donor. This is the
P-R4 `LIM_SHIFT`/A1.6 shift idiom (`cuda_raycaster.cu:263-264` precedent). It
restores per face what the convex bound used to give for free; the AGGREGATE
bound is still `SHIFT_MIN`'s (§2.3 below).

### 2.3 The pass — constraints 2 and 3

Gather form, double-buffered, single-writer, no atomics for the physics:

```cpp
int64_t de = 0;
for each of the 4 faces:  de += face_energy_q(ti, T[j], cap_i, cap_used_[j], s, &cond_limit_hits);
if (de == 0) { temp_new[i] = ti; continue; }              // exact rest
const int64_t dT = fixedpoint::floordiv_q(de, cap_i);     // R2 + R3
e_cond_trunc_sum += dT * cap_i - de;                      // <= 0, always
e_cond_cap_sum   += dT * (cap_real_[i] - cap_i);
temp_new[i] = (int32_t)(ti + dT);
```

**Constraint 2 (endpoint-local conversion, R2).** Each end divides the energy
it received by ITS OWN capacity. A hull tile taking gas energy warms 32× less
than the gas cooled, because it is 32× heavier.

**Constraint 3 (one-way counted guards).** The endpoint divide is the shared
`floordiv_q` (toward −∞) — the same helper §2.1.5's recovery and §2.7's
expansion branch use. Truncation toward zero would round a LOSING cell's ΔT
up, i.e. MINT, and both backends' `/` agree on that mint so **only the ledger
could see it**. Toward −∞ the residual destroys in both signs. Both residual
terms are counted in ENERGY, not as hit counts.

**The aggregate maximum principle.** With `s ≥ SHIFT_MIN == 2` and
`C_min ≤ C_i`, each face contributes at most `ΔT = g/2^s ≤ g/4`, so the update
is still a convex combination over the four neighbours and no new extremum can
appear — **up to the ≤1-raw-count slack `floordiv_q` can add on a cell that is
losing energy**, which is the price of the one-way rounding R3 demands. Measured
slack in practice: **0** (§5.3).

**The skip set is UNCHANGED.** Vacuum, ring and solid cells all still conduct
exactly where they did. This patch changed the currency, not who participates.

**Cost.** One int64 divide per active cell per tick (cells with `ΔE == 0` take
the early-out), plus four extra int32 loads per cell for the facing shifts.

### 2.4 Pass 3 / sky / ambient-ring — named as SIGNED channels (L3-6)

**No law changed here. Instrumentation only**, so §7's "every creator named and
counted" is checkable:

- **Pass 3 ambient cooling / sky** (`e_cool_sum`). It relaxes T toward 0 from
  BOTH sides, so on a sub-ambient tile it **creates** energy. An
  "ambient-cooling" counter that only ever went one way would hide the creator
  half of the same line of code — which is exactly finding L3-6.
- **Pass 0a open-vacuum wipe** (`e_vac_wipe_sum`) — destroys what a breach
  vents, creates if it pins a sub-ambient cell up to 0.
- **Pass 0a ambient-ring pin** (`e_ring_pin_sum`) — the §5 boundary channel,
  bidirectional. A cell that is both vacuum and ring is attributed to vacuum
  (the test order matches the condition's own; pinned so both backends and the
  ledger agree).

All three are priced at `cap_real` — the same currency as the ledger's
`Σ N·T_abs` estimator.

**Explicitly NOT in scope here, and why:** the Pass-1 deposit drop counter and
the Pass-1 T_MAX_PHYS / LOW rail energy twins are **P-E2b's row** (§6:
"Pass-1 drop counter"), and the raycaster's rule-4 sky charge is created in
`cuda_raycaster.cu`, upstream of this TU. Named here so the boundary is a
decision, not an omission.

---

## 3. The counters, and where they surface

Six new `mutable int64_t` fields on `TemperatureSolver`, all `def_readonly` in
`bindings.cpp`, all in `tools/storm_ledger.py::counters()` under a new
`temp.` prefix (the function grew a third holder and a `(holder, prefix,
names)` shape):

| counter | meaning | sign |
|---|---|---|
| `e_cond_trunc_sum` | endpoint `floordiv_q` residual | ≤ 0 (one-way) |
| `e_cond_cap_sum` | the capacity floor (thin gas) / ceiling term | signed |
| `cond_limit_hits` | constraint-4 per-face engagements | count |
| `e_cool_sum` | Pass 3 ambient cooling / sky | **signed** |
| `e_vac_wipe_sum` | Pass 0a open-vacuum wipe | **signed** |
| `e_ring_pin_sum` | Pass 0a ambient-ring pin | **signed** |

**They ACCUMULATE across `step()` calls** — the `t_max_phys_hits` idiom of this
class — and are therefore deliberately NOT in `PER_TICK_COUNTERS`: the ledger
diffs them. This is a considered departure from P-E1's reset-at-entry choice:
the CUDA path is a separate free function, and a reset-at-entry contract would
have needed a second, independently-maintained reset site. Accumulating means
the CUDA path only ever folds `+=` into the same fields, exactly as
`t_max_phys_hits` / `t_low_rail_hits` already do.

`breach_cuda::temperature_step` gained a nullable `int64_t*
energy_counters_out` (6 slots, order pinned by the `C_*` enum in the `.cu` and
by `TEMPERATURE_ENERGY_SLOTS` in the `.h`); `physics_engine.cpp` folds them
into the solver's fields right after the call.

**CUDA structure:** one new kernel `temp_cap_build` (the CPU pre-pass loop
verbatim) plus the rewritten `temp_conduct`; `temp_zero_vacuum` and `temp_cool`
gained a `cap_real` plane and the counter block. Device scratch: two int64
planes + a 6-slot counter block. Both backends call the **same** `conduction::`
helpers out of `temperature_solver.h`, which the `.cu` now includes — the law
exists in one place.

---

## 4. Gates 1–2 — baseline and builds

- **Pristine baseline captured FIRST**, name-sorted: **48 failed, 2169 passed,
  5 skipped, 1 xfailed** — exactly the brief's expectation.
  `test_air_boundary::test_ambient_gate3_udamp_band_absorbs_reflection` is in
  it as the carried red awaiting Erik's ruling (P-E1 as-built §8); it was left
  alone.
- `cpp\build_cpu_data.bat` and `cpp\build_cuda_lenovo.bat` both `BUILD_EXIT=0`.

---

## 5. Gates 3 and 5 — antisymmetry, the books, and the limiter, with numbers

### 5.1 Gate 3 — FACE ANTISYMMETRY, EXACT, both backends

**CPU** (`test_conduction_face_energy_is_exactly_antisymmetric`): an
independent Python transcription of §2.3 is run alongside the solver on a
10×10 mixed-material field with random N, for 12 ticks. Two assertions:

| assertion | result |
|---|---|
| every shared face's two directed ΔE entries sum to 0 | **worst residual 0**, over 218,396 live-face samples across the wider 5-seed sweep |
| the mirror's field == the C++ field | **bit-identical, every tick** |

The second is what makes the first a statement about the shipped law rather
than about the mirror.

**Both backends, as an identity:** because `Σ ΔE == 0` exactly, the whole-grid
books close. Asserted per tick on the CPU
(`test_conduction_energy_books_close`, 150 ticks on a 14×14 field:
`Σ ΔT·C_real == e_cond_trunc_sum + e_cond_cap_sum`, with `d_trunc ≤ 0` every
tick) and on the GPU by counter parity — `cuda_conduction_check` PART 1
compares all six counters bit-for-bit over 121 configs and PART 2 over 120
ticks, so the GPU's `Σ ΔE` is 0 for the same reason the CPU's is.

### 5.2 Gate 4 — KIRCHHOFF RE-GATE (hard), both backends

| backend | gate | result |
|---|---|---|
| CPU | `test_pf1a_radiation_books.py` (11 tests) | **11 passed** — incl. gate (ii) `Σ rad_net + Σ rad_amb == 0` exactly on a real firestorm, and gate (iii) `max\|rad_net\| == 0` on isothermal sealed lattices at three temperatures, with the non-vacuity control (breaking the isotherm at one tile does move energy) |
| GPU | `cuda_s2b_raycaster_live_check` PART 1 | 5 multi-source live casts, `rad_net` + `rad_flux` **byte-for-byte == CPU**; `\|rad_net\|peak = 0` on every equal-T cast |
| GPU | PART 1b | 20 ticks of an evolving fire → radiation cast, byte-identical every tick (peak 18,846,800 counts — non-vacuous) |
| GPU | PART 2 | all-backends-on 30-tick trajectory: **all 26 synced fields bit-identical, incl. `heat` and `temperature`** |

The conduction rework did not disturb it. (The file's only failing leg is the
pre-existing shared-canonical-golden comparison — §7.)

### 5.3 Gate 5 — the MAXIMUM-PRINCIPLE REPLACEMENT, measured

Measured on 5 seeds × 60 ticks of 14×14 mixed AIR/HULL/WOOD fields with random
N spanning `[0, 3·ambient]`, live face shifts {2, 7, 8, 10, 11}:

| observable | measured | bound |
|---|---:|---|
| **worst per-face fraction** `\|ΔE\| / (g·C_min)` | **0.250000** | limiter ceiling **0.5** — a 2× margin |
| `cond_limit_hits` (C++ counter) | **0** | the limiter is a structural backstop, never an operating mechanism at shipped shifts |
| **worst aggregate excursion** beyond the frozen field's `[min, max]` | **0 raw counts** | the stated ≤1-count `floordiv_q` slack |
| single-face sweep: worst gap INVERSION | **0** | 11,760 pairs, capacity ratios to 640× (hull C=32 vs floored gas C=0.05) |
| single-face sweep: worst gap GROWTH | **0** | a face only ever relaxes |

**Reading:** 0.250000 is exactly `2^-SHIFT_MIN`, i.e. the hull-hull face at the
fastest legal shift — the aggregate convex bound is what actually binds, and
the limiter sits a factor of 2 above it. That is the design's intent ("pinned
≤ 1/2, the safe side of the f=2 line"): it exists so a mis-baked shift of 0 or
1, or a future faster material, cannot let a floored thin endpoint close 4× the
gap in a tick. **Worst-case overshoot fraction is therefore 0 in the field and
0.25 per face against a 0.5 allowance.**

### 5.4 Gate 7 — CPU↔CUDA lockstep, tol 0

| gate | result |
|---|---|
| `cuda_conduction_check` **PART 1** | **121 configs** bit-identical on `temperature` + rail hits **+ all six P-E2a energy counters**. Coverage asserted non-vacuous per counter (a counter that is 0 on every config proves nothing when compared): `e_cond_trunc_sum` 1.65e9, `e_cond_cap_sum` 3.17e13, `e_cool_sum` 2.36e15, `e_vac_wipe_sum` 2.25e14 (all as Σ\|per-config\|), `cond_limit_hits` 0 (asserted 0) |
| `cuda_conduction_check` **PART 2** | 120 ticks bit-identical, `temperature` + rail hits + the six counters, peak \|T\| 11,622 K-rel |
| `cuda_conduction_check` **PART 3** | engine dispatch A/B (`set_temperature_backend`): bit-identical over 30 ticks, peak \|temperature\| 4,587,380 counts |
| `test_cuda_thermal_mass` (axis lockstep, `thermal_solid != solid`) | **PASS**, now comparing all seven values |
| `test_cuda_cool_shift` (per-tile decay grid) | **PASS**, now comparing all seven values |
| `test_cuda_eos_step` PART 1 (per-call chained, 120 ticks) | bit-identical across all EOS fields, six digests, five rail counters |
| `test_cuda_s8a_residency` (**resident** path) | **PASS** |
| `test_cuda_thermal_mass_eos` PART 2 (step) / PART 3 (**resident**) | **PASS** |
| `test_cuda_bulk_flux` PARTs 1/2/3 | **PASS** (incl. P-E1's energy-books closure oracle) |
| `test_cuda_p66_conduction`, `p62`, `p63`, `p68`, `p69`, `s3`, `s4a`, `trace_smoke`, `s2b` | every parity leg bit-identical; each file's ONLY failing leg is the pre-existing shared canonical golden (§7) |

---

## 6. Gate 6 — the LEDGER, and the cold-rail window row RE-MEASURED

### 6.1 Conduction's global drift IS the counted floor terms

Pump-off row, `--ticks 4800 --damp 0.0 --pf1b` (the audit command P-T0 and
P-E1 both ran):

| `temp.` counter | run total (raw Q16.16²) |
|---|---:|
| `e_cond_trunc_sum` | **−480,884,298,383** |
| `e_cond_cap_sum` | **0** |
| `cond_limit_hits` | **0** |
| `e_cool_sum` | −3,024,042,777,903,104 |
| `e_vac_wipe_sum` | 0 (sealed map, no open vacuum) |
| `e_ring_pin_sum` | 0 (sealed map, no ambient ring) |
| `t_max_phys_hits` / `t_low_rail_hits` | 0 / 0 |

`e_cond_cap_sum == 0` says the `n_floor_heat` floor never bound on a real bench
run (bench gas sits at ambient N), so **conduction's entire global drift over
4800 ticks is the one counted, one-way endpoint-truncation term** — the §7
property, measured rather than assumed. `cond_limit_hits == 0` confirms §5.3 on
a real trajectory too.

Per-pass row deltas vs P-E1's measurement on the same command:

| pass | quantity | P-E1 | P-E2a |
|---|---|---:|---:|
| eos | `eth_gas` | +275.1 | **+291.9** |
| eos | `n_bulk` / `n_o2` | 0 / 0 | **0 / 0** (unchanged — P-T0's headline survives) |
| combustion | `eth_gas` | −403.3 | −403.1 |
| tail | `eth_gas` | +56.71 | **+71.86** |
| tail | `t_obj` | — | +119.5 |

The `tail` row (conduction + cooling) now delivers *more* thermal energy to the
gas and visibly warms objects (`t_obj +119.5`), which is the direct signature of
§0.4: the wall no longer takes 32× what the gas gives up.

### 6.2 THE COLD-RAIL WINDOW ROW — re-measured PRE vs POST, across exactly this patch

`--ticks 4800 --damp 0.005 --pf1b --set k_wind_strip=0.5` (audit §7 command).
**PRE was measured by reverting the tree to the parent commit and rebuilding**,
so this is a clean one-patch delta, not a comparison across two law changes:

| observable | PRE (P-E1, `3a15103`) | POST (P-E2a) |
|---|---:|---:|
| `t_min_gas` minimum over the run | **−0.1908** (tick 4789) | **0.0000** |
| ticks with `t_min < −250` | 0 | 0 |
| ticks with `t_min < −100` | 0 | 0 |
| **`tail` pass `eth_gas`** | **−20.67** | **+0.8128** |
| `tail` pass `t_obj` | −189.1 | −188.9 |
| `tail` pass `t_solid` | −0.0584 | 0 |
| `eos` pass `eth_gas` | +58.63 | +61.85 |
| `eos` pass `ke` | 118.4 | **33.75** |
| `eth_transport_delta` run total | −6.120e10 | −1.365e10 |
| `eth_compression_delta` run total | +3.358e10 | **+9.612e9** |
| `e_ts_residual` run total | −1.045e10 | +2.612e8 |
| `work_clamp_hits` / `energy_floor_hits` / `t_max_phys_hits` | 0 / 0 / 0 | 0 / 0 / 0 |
| amplifier max gain | 1.1× | 1.1× |

**Does energy-form conduction + the limiter change the reservoir back-feed?
Yes, and in the direction the design predicted.**

1. **The back-feed's sign flipped.** The reservoir loop's conduction leg — the
   pinned hull conducting into the supercooled pocket "for free" (§2.9) — was
   the `tail` row removing 20.67 units of gas thermal energy per run. It now
   *adds* 0.81. The free leg is gone because the wall's ΔT is now priced by the
   wall's own 32× capacity.
2. **The compression pump on this row fell 3.5×** (+3.358e10 → +9.612e9), and
   the EOS pass's kinetic energy fell by the same factor (118.4 → 33.75). The
   loop had less thermal energy to convert.
3. **`t_min_gas` reaches exactly 0**: on this row the gas never goes sub-ambient
   at all, so §2.9's engine (compression on a negative game-T) has nothing to
   act on here. Note the −288.65 rail P-E0 measured on HEAD was **already**
   closed before this rung (PRE reads −0.19); P-E2a takes the residual to 0.

**No fix was chased.** §2.9 owns the engine and it lands as its own designed
patch after P-E5; this row is reported as a measurement for §7's window-row
expectations, which P-E4 sets.

---

## 7. Gate 8 — suite accounting, line by line

**Post-patch: 48 failed, 2173 passed, 5 skipped, 1 xfailed** (baseline
48 / 2169 / 5 / 1).

**Set-diff against the 48-red baseline: EMPTY IN BOTH DIRECTIONS.** Nothing
became green, nothing new went red, name for name.

**Pass arithmetic (2169 → 2173, +4):** the four new gates in
`test_temperature_conduction.py` —
`test_limiter_bounded_no_endpoint_passes_the_donor` (the rewrite of
`test_discrete_maximum_principle`, so net 0 there),
`test_single_face_gap_never_inverts_at_any_capacity_ratio`,
`test_conduction_face_energy_is_exactly_antisymmetric`,
`test_conduction_energy_books_close`, and
`test_conduction_limiter_is_inert_at_shipped_face_shifts`. (Five added, one
replaced in place: +4.)

**Authorized rewrites executed (Appendix A P-E2a):**

| file | what died | what replaced it |
|---|---|---|
| `test_temperature_conduction.py` | `test_discrete_maximum_principle` (the convex-update premise) | the limiter-bounded property, split correctly into the AGGREGATE assertion (no new extremum beyond the ≤1-count floordiv slack, on heterogeneous fields) and the PER-FACE one (a single face's gap never inverts, in isolation, swept across 640× capacity ratios) |
| `test_temperature_conduction.py` | `test_air_conducts_with_solids`'s `temp.sum() <= before` | the capacity-weighted sum `Σ C·T` — the flat sum now legitimately RISES across a hull↔air face (the wall shedding one degree warms the light gas by ~32), which is the physics, not a defect |
| `test_eos_p2_sealed_room_energy.py` | the plain-ΣT metric AND its `pairs × n_ticks` epsilon bound (both derived from the old law's `floor(x)+floor(−x) ∈ {0,−1}` identity) | `Σ C·T`, and — strictly stronger — the counter IDENTITY `Δ(Σ C·T) == Σ of the five named channels`, asserted **every tick** on both scenarios. Scenario (b) additionally asserts the drain is ATTRIBUTED: the two space-facing channels must dominate conduction's counted truncation by 10× |
| `cuda_conduction_check.py` | `cuda_temperature_step`'s int64 return | a 7-tuple `(t_max_phys_hits, *six energy counters)`, all compared |

`test_pf1a_radiation_books.py` was **not touched** — its floor counters needed
no change (11/11 green throughout), so the Appendix-A allowance went unused.

**Reds already in the 48-red baseline, still red for the same reason (the
shared canonical golden `28678e9d…` vs the value P-T0 left at
`8203584350ae69a5…`):** `test_cuda_eos_step`, `test_cuda_mg_solve`,
`test_cuda_p62_sl_advection`, `test_cuda_p64_kick_compression`,
`test_cuda_p66_conduction`, `test_cuda_p68_fire`, `test_cuda_p69_combustion`,
`test_cuda_s2b_raycaster_live`, `test_cuda_s3_water`, `test_cuda_s4a_smoke`,
`test_cuda_trace_smoke`, `test_w6_armory`, plus the non-CUDA baseline reds.

**The canonical golden did not move again**, for P-E1's measured reason: that
A/B scenario carries `temperature` identically 0 in every cell on every tick,
and this law is an exact no-op on a uniform-ambient T field (`g == 0 ⇒ ΔE == 0`
on every face). The post-patch digest is byte-for-byte the same
`8203584350ae69a5…` P-T0 left. No re-baseline recorded at this rung.

**Pins, all green:** `test_no_transport_mint`,
`test_transport_delta_is_one_way_negative`,
`test_hot_scenario_prefix_is_deterministic`,
`test_pocket_variant_is_deterministic`,
`test_window_scenario_prefix_is_deterministic`,
`test_hot_scenario_reaches_the_audit_anatomy`.
**`test_no_rail_hits` stays `xfail` with owner P-E4** — the cross-rung red
idiom, unchanged.
`test_air_boundary::test_ambient_gate3_udamp_band_absorbs_reflection` remains
the one carried red, untouched, still awaiting Erik's gate-semantics ruling
(P-E1 as-built §8).

---

## 8. Deviations, and things the brief or the design got wrong

### 8.1 `cuda_temperature_step` had THREE callers, not one

The P-E2a commit message asserts `cuda_conduction_check.py` "is its only
caller". **It is not** — `cuda_thermal_mass_check.py` and
`cuda_cool_shift_check.py` call it too, and both went red with
`TypeError: int() argument … not 'tuple'` the moment the CUDA half was built.
This was a signature-surface red, not a parity break (every field and parity
leg in both files was green before and after). Fixed in `38ed00b`, and since
both are TEMPERATURE-SOLVER lockstep gates they now compare all seven values
rather than just unpacking `[0]`. **Reported because the claim was made in a
commit message and was wrong**; the sweep that should have caught it is
`grep -rn cuda_temperature_step --include=*.py`, which is now in this record.

### 8.2 The design's per-face property needed splitting to be true

Design §2.3 constraint 4 says the limiter "restores the discrete maximum
principle the ΔT form had for free". Read literally as "no endpoint may pass
the donor across any face", that is **false for the 4-face aggregate and always
was** — a cell with three hot neighbours and one barely-hotter fourth will end
up past the fourth, under the old law as much as the new one. That is ordinary
diffusion. The first draft of the rewritten test asserted the aggregate form
and correctly failed.

The property actually splits in two, and both are now asserted separately:
**per face, in isolation**, the limiter guarantees the gap never inverts (this
is what constraint 4 buys, and it is what stops a floored thin endpoint closing
4× the gap); **in aggregate**, `SHIFT_MIN` still gives the convex bound, exactly
as before. Recorded because a future reader will otherwise re-derive the same
confusion from the design's one-sentence phrasing.

### 8.3 `test_cuda_p64_kick_compression` PART 2 was ALREADY diverging

While walking the lockstep set I found `cuda_kick_check` PART 2 reporting
`advection replay != solver digest_advect` from tick 0 and field divergence
from tick 1. **This is a pre-existing P-E1 debt, not this patch's**, and it was
verified by MEASUREMENT rather than argument: the tree was reverted to the
parent commit, CUDA rebuilt, and the check re-run — the output is
**byte-for-byte identical, down to every hex digest**
(`ref=0x060eec62fe736c00 solver=0x9a77899bddafdb83` at tick 0, and so on).

The cause is P-E1's own declared contract change: `eos_sl_advect_reference`
became u-only and "is no longer comparable to `digest_advect`", which also
moved across the flux call. **P-E1's as-built §6 lists
`test_cuda_p64_kick_compression` PART 1/2 as "bit-identical", which
over-claims** — PART 1 is, PART 2 is not. Flagged for the arc rather than
fixed here: repairing that replay premise is P-E1's authorized-rewrite
territory, and doing it inside a conduction patch would be exactly the silent
scope creep the project forbids.

### 8.4 Counters accumulate rather than reset per tick

P-E1's five EOS counters reset at `step()` entry and are listed in
`PER_TICK_COUNTERS`. P-E2a's six do the opposite — they accumulate, like
`t_max_phys_hits` / `t_low_rail_hits` already do in this class, and the ledger
diffs them. Rationale in §3: the CUDA path is a separate free function, so a
reset-at-entry contract would need a second reset site to maintain in lockstep.
Reported because it is a deliberate departure from the sibling patch's idiom.

### 8.5 `CAP_SHIFT_MAX` is a clamp on a quantity the design does not discuss

Nothing in §2.3 bounds the endpoint capacity, but `g·C_min` must fit int64 with
room for a 4-face sum. `CAP_SHIFT_MAX = 12` is the guard, seven doublings above
the shipped maximum `thermal_mass` of 32. Chosen as a deterministic clamp
(identical on both backends) rather than an assert, and any energy it would
imply is counted through `e_cond_cap_sum` — so it cannot become a silent
channel. Named here because it is a constant this patch invented.

### 8.6 The plain-ΣT death is larger than "a metric premise"

Appendix A calls `test_eos_p2_sealed_room_energy`'s change a metric premise.
It is worth stating what the old metric was hiding: that module's docstring
proved `Σ T` monotonically non-increasing under conduction, and used it as the
sealed room's conservation gate. The proof was correct **about the old law** and
the law was wrong about the physics — the gate was measuring a quantity blind to
the room's largest energy channel. The replacement asserts an identity against
named counters, which is a strictly stronger statement and would have caught
the original defect.

---

## 9. Files touched

**Code (CPU):**
- `cpp/src/temperature_solver.h` — the `conduction::` energy kit
  (`cell_capacity_q`, `face_energy_q`, `opposite_dir`, `CAP_SHIFT_MAX`,
  `LIM_SHIFT`), the six counters, the two capacity scratch planes, the old
  law's transcription + the new law's contract in the header block.
- `cpp/src/temperature_solver.cpp` — the capacity pre-pass; Pass 2 rewritten;
  Pass 0a and Pass 3 instrumented; `n_floor_q` hoisted and shared.
- `cpp/src/bindings.cpp` — the six counters exported; the isolated GPU entry's
  return became a 7-tuple.
- `cpp/src/physics_engine.cpp` — the GPU dispatch's counter fold.

**Code (CUDA):**
- `cpp/src/cuda_temperature.h` — `energy_counters_out` + `TEMPERATURE_ENERGY_SLOTS`.
- `cpp/src/cuda_temperature.cu` — includes `temperature_solver.h` for the shared
  kit; new `temp_cap_build`; `temp_conduct` rewritten; `temp_zero_vacuum` /
  `temp_cool` instrumented; the `C_*` slot enum + `cadd`.

**Tests / tools:**
- `tests/test_temperature_conduction.py` — the independent Python transcription
  of §2.3 + four new gates + the two authorized rewrites.
- `tests/test_eos_p2_sealed_room_energy.py` — metric + drift-bound re-derivation.
- `tests/cuda_conduction_check.py` — 7-tuple contract, counter parity, per-counter
  coverage assertions.
- `tests/cuda_thermal_mass_check.py`, `tests/cuda_cool_shift_check.py` — the two
  call sites of §8.1, now comparing all seven values.
- `tools/storm_ledger.py` — the `temp.` holder in `counters()`.
- `docs/e1_p_e2a_asbuilt_2026-08-17.md` — this record.

**Untracked, regenerable:** `_fire_tuning_artifacts/ledger_pe2a.npz`,
`ledger_window_pe2a.npz`, `ledger_window_PRE.npz`.
