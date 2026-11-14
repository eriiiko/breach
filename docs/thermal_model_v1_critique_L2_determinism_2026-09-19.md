# Thermal model v1 — critique L2: determinism, integers, CUDA (2026-09-19)

> **Lens**: determinism, integer arithmetic, `/fp:strict` + the float ratchet,
> the number-ingress doors, the digest/golden gates, CPU↔GPU bit-identity, int64
> headroom, and the #54 closure identity.
> **Document under critique**: `docs/thermal_model_v1_design_2026-09-19.md`.
> **Out of remit** (by instruction): §2's rulings R1–R9, the ACCEPTED GAPs, the
> physics/thermodynamics/calibration values, and §8's L1 row (not read, so these
> conclusions are independent).

---

## Verdict summary

**2 BLOCKING · 11 REQUIRED · 5 NOTE.**

| # | Severity | One line |
|---|---|---|
| B1 | **BLOCKING** | `t_amb_q` is the wrong scalar. It is the Kelvin offset of the temperature scale, not the ambient the sweep radiates at (`amb_m` is). Promoting it to a plane delivers none of R3 — **and all four T1 gates still pass.** |
| B2 | **BLOCKING** | T4a's `dyn_heat_atten_q`-into-the-digest is a spec v5→v6 membership bump. `DIGEST_SPEC_VERSION` is salted into every per-field hash, so **every golden in 19 files moves at T4a, by construction.** "The one golden re-baseline of the arc" at T4b is not achievable as the patch plan is cut. |
| R1 | REQUIRED | `t_amb_q` is shared with the arc #54 gas-energy seam; the design must state it stays scalar. |
| R2 | REQUIRED | Name the ingress door per plane. `optics_fixed` **raises** above 1.0, so it cannot take the ambient plane, and no `temperature_fixed.py` exists. |
| R3 | REQUIRED | Say whether the planes are mutable in-tick — that is what decides `DIGEST_FIELDS` vs `SIM_FIELDS` vs the residency masks. The design names none of the four lists. |
| R4 | REQUIRED | `k_leak_q` is per-tile authored data that is not a material property; it needs a `level_lib.py` writer and a `level.toml` home. §7.2 gives it no ingress at all. |
| R5 | REQUIRED | §6 item 4's fixed-point property only holds on a *uniform-ambient neighbourhood* once the ambient is really per-cell. |
| R6 | REQUIRED | The Q16 leak has a truncation dead-band: at ambient the per-ordinate stream is **10 counts**, so §6 item 2 compares `0 == 0` and passes vacuously below ~17 counts. |
| R7 | REQUIRED | §6 item 8's "`rad_amb`'s grown" is the wrong ledger term. `rad_amb` is not in the P-G5 identity; the term that grows is `e_solid_deposit_sum`. |
| R8 | REQUIRED | The `cool_shift` deletion surface is much larger than §7.3: a CUDA Pass 3, **two pinned positional counter slots**, `e_cool_sum` (written from the same line), a required binding arg, and a resident mask. |
| R9 | REQUIRED | The cool-shift and #54 tests die at **T4b**, not T5. `test_thermostat_books.py`'s assertion (b) asserts on the deleted counter. |
| R10 | REQUIRED | T4a's "clamp on the CUDA temperature twin" is new device work not costed: a 4000-entry int64 E° upload and a `rad_fluence` H2D that the twin has no parameter for today. |
| R11 | REQUIRED | The pins the future P4 sweep twin needs (whose ambient the virtual ring reads; `ret` per cell; plane-vs-scalar `t_amb` on device) must be written **at T1**, in the reference. |
| N1 | NOTE | `radiation_sweep.cpp` is a hard 0/0/0 ratchet — the two type names are forbidden even in prose there. |
| N2 | NOTE | Two new `.noconvert()` required plane kwargs break every direct `step_tail` / `RadiationSweep.run` caller at once; "goldens unmoved" ≠ "suite green". |
| N3 | NOTE | The int64 headroom argument survives, but restate it as a **bound on `E°[3999]`**, not as the measurement it currently is. |
| N4 | NOTE | `tools/storm_ledger.py` reads counters inside a bare `try/except`, so a deleted counter vanishes silently instead of failing. |
| N5 | NOTE | The `T_abs_q << 24 < 2^54` bound is exactly tight today and stops holding if an unbounded per-cell offset ever lands. |

**The single most important finding is B1.**

---

## 1. The two new planes vs the number-ingress doors (§7.2)

### B1 (BLOCKING) — `t_amb_q` is the wrong scalar. Promoting it to a plane delivers none of R3, and every T1 gate still passes.

This is not a critique of R3 (a ruling, out of remit); it is that the design
names a symbol which does not mean what §7.2 says it means.

**What `t_amb_q` actually is in the sweep.** It appears in exactly one place:

```
radiation_sweep.cpp:179   const int64_t t_amb = t_amb_q;
radiation_sweep.cpp:211   int64_t T_abs = (int64_t)temperature[i] + t_amb;
radiation_sweep.cpp:215   f_q24_[i] = fleck_enabled ? fleck_f_q24(T_abs, L) : F_ONE;
```

It is the **absolute-zero offset of the temperature scale** — what turns a game
temperature (a ΔT above ambient) into Kelvin for the Fleck denominator
`g = 4L/T_abs`. Design v3 §2.3 says so outright: *"the denominator is on the
absolute temperature because β = 4aT³/c_v is defined on Kelvin; a
delta-above-ambient denominator diverges at ambient."* Its value is bound at
`physics_runner.py:1053` as `self._eos_t_amb_raw()`, i.e.
`max(1, gas_fixed.quantize_scalar(self.eos.T_AMB_K))` = `293 << 16` — **the very
integer arc #54's gas-energy seam runs on** (`physics_runner.py:1859-1868`;
`bindings.cpp:3556  py::arg("t_amb_q") = 0  // arc #54 T_AMB_K raw`;
`cuda_temperature.h:139  int32_t t_amb_q = 0`, commented *"T_AMB_K raw (only
read with gas_energy)"*). `PhysicsEngine::step_tail` passes **one** `t_amb_q` to
both `radiation.run(...)` (`physics_engine.cpp:241`) and the temperature pass.

**What the ambient radiation stream actually is.** Not `t_amb_q`:

```
radiation_sweep.cpp:173   const int64_t e0    = e_table[0];
radiation_sweep.cpp:175   const int64_t amb_m = (e0 * w_m) >> FP_SHIFT;
radiation_sweep.cpp:178   const int64_t ret   = (amb_m * k) >> FP_SHIFT;
```

`amb_m` comes from `E°[0]` — the *bottom bucket of the emissive table*, covering
`T_game ∈ [0, 4)`, i.e. 293–297 K (`emissive_table.h:25-35`,
`emissive_table.cpp:51-63`). Ambient is `T_game = 0` **by construction of the
table's domain**; `e_bucket_of` returns bucket 0 for every `T_q <= 0`
(`emissive_table.h:57-61`, whose own comment says *"a tile below ambient does not
emit less than the ambient floor in this model"*). `amb_m` is the ambient stream,
`ret` the ceiling's ambient return (`:178`, `:291`), and
`emit_body = (amb_m * b) >> 16` the body's ambient re-emission (`:286`).
**None of the three reads `t_amb_q`.**

**Therefore**: promoting `t_amb_q` to a plane gives per-tile *absolute-zero
offsets*. It moves only the Fleck damping factor. It cannot express "room temp in
ship, 0 K outside", because the ambient the sweep radiates at is `E°[0]`, and
`amb_m`, `ret`, `emit_body` and the Pass-1 clamp ceiling `E°⁻¹(Φ)` would all
stay pinned at 293 K.

**Why the T1 gates cannot catch it.** I walked §6 items 1–4 against the literal
implementation (`t_amb_q` becomes an `int32_t*`, filled uniformly with
`293 << 16`, read at `:211`):

| §6 gate | Result on the literal implementation | Why |
|---|---|---|
| 1 uniform case reproduces the scalar era | **PASSES** | identical arithmetic, identical integers |
| 2 a per-tile leak leaks per tile | **PASSES** | `k_leak_q` genuinely is per tile |
| 3 conservation with non-uniform leak *and* ambient | **PASSES** | `f` is applied *before* the sweep (design v3 §2.8: "conservation is untouched and the books need no counter"), so a per-cell `t_amb` is conservation-neutral by construction |
| 4 an ambient cell is an exact fixed point at its *own* ambient | **PASSES** | with a uniform ambient it is, and `amb_m` is unchanged |

Four gates, zero detection. §6 item 4's *"Breaks if `amb_m` is baked from a
global"* shows the author knows `amb_m` is the load-bearing quantity — but
`amb_m` is **not** `t_amb_q`, and the patch row (T1) and the draft rule (§7.2)
both name `t_amb_q`.

**What the design must state instead** (facts to settle before T1 spawns; no
implementation proposed):

1. The per-tile ambient plane is a **new** quantity, not a promotion of
   `t_amb_q`, and must carry a different name — `t_amb_q` is a shared
   unit-system constant that arc #54 requires to stay scalar (R1).
2. State the plane's **denomination**. If it is a game temperature (ΔT above
   293 K, the frame `GameMap.temperature` uses), a uniform fill of **0**
   reproduces `E°[0]` exactly through `e_bucket_of`. A uniform fill of
   `293 << 16` does not: `e_bucket_of(293<<16) = 73` and `E°[73] = 2524` against
   `E°[0] = 161` (computed below from `rad_scale_derived = 2.125632e-08`) — a
   15.7× ambient. §6 item 1 *would* catch that, which is the only reason the
   denomination error is recoverable at all.
3. State which of `amb_m`, `ret`, `emit_body` and the Pass-1 clamp ceiling become
   per-cell. All four are ambient-derived; §3.1 mentions only `ret`.
4. State what the **virtual ring** reads when the ambient is per-cell (`:267-268`,
   `io_a = in_a ? store[...] : amb_m`). The ring is outside the grid and has no
   cell of its own. Conservation survives any choice (whatever integer arrives is
   booked as `-fa`/`-fb` into the *reading* cell's `rad_amb`, `:274-275`), so this
   is not a correctness accident waiting to happen — but it **is** a pin the CUDA
   twin needs (R11), and §6 item 3 already asserts against "the wrong cell's
   ambient" without saying which cell is right.

### R1 (REQUIRED) — `t_amb_q` is not free to become a plane: it is shared with the arc #54 seam.

`_eos_t_amb_raw()` is the single source for the sweep's `t_amb_q`, the
temperature solver's `t_amb_q` (the gas-energy seam's born-at-ambient offset),
`GameMap._gas_energy_t_amb_raw`, and `EOSSolver::T_AMB_K`'s A7 floor. The #54
identity is denominated against it: `bindings.cpp:2782` reads a room's relative
energy as `Σ (E − N·T_AMB_raw)`, and `bulk_transport.cpp:351-366` mints gas at
`q * t_amb64` ("MINTED at ambient"). A per-tile `t_amb_q` would make "born at
ambient" a per-tile quantity and that readback wrong. The design must state
explicitly that the #54 `t_amb_q` stays scalar and is untouched by T1.

### R2 (REQUIRED) — name the ingress door per plane; `optics_fixed` will *refuse* the ambient one.

- **`k_leak_q`** is a coefficient in [0, 1], so `optics_fixed.quantize` is the
  right door and its *raise-never-clamp* contract (`optics_fixed.py:34-46`) gives
  exactly the refusal the remit asks about. The scalar path already refuses
  twice: an explicit `0.0 <= k <= 1.0` check plus `quantize_scalar`
  (`physics_runner.py:458-465`), and a re-check in the engine
  (`radiation_sweep.cpp:160-163`). **The design must state that a per-cell
  re-check replaces `:160-163`** — an `int32_t*` plane silently skips that scalar
  guard, and the pre-pass's existing per-cell ingress loop (`:185-192`, which
  already enforces `0 <= a <= d <= ONE` and throws) is where it belongs.
- **The ambient plane is a temperature, not a coefficient.**
  `optics_fixed.quantize` **raises** above 1.0, so §7.2's "Filled at level load
  through the Q16 boundary module" cannot mean `optics_fixed`. And there is **no
  `temperature_fixed.py`** in the tree: `src/simulation/` has atmosphere / fire /
  gas / optics / unit / wall / water / wave `_fixed.py` and nothing else. Today's
  temperature scalars go through `gas_fixed.quantize_scalar`
  (`physics_runner.py:1867`, `tests/test_thermostat_books.py:55`), whose own
  docstring scopes it to "the smoke + 5 gas planes". The design must make the
  reuse-or-new call: which door, and what range it refuses. Without a stated
  range the failure mode is **silent, not loud** — `e_bucket_of` clamps
  `T_q <= 0` to bucket 0 and saturates above 15996 game, so a level author
  writing "0 K outside" gets 293 K with no error whatsoever.

### R3 (REQUIRED) — say whether the planes are mutable in-tick; that is what decides digest membership.

Neither plane needs `DIGEST_FIELDS` **if** it is authored once at load and never
written during a tick — the `floor_height` / `conductivity` / `heat_atten_q`
precedent. `heat_atten_q` is not digested because it is a pure projection of
`material`, which *is* digested, and it is patched through the one seam
(`gamemap.py:1675`, `on_tile_changed`).

`k_leak_q` is explicitly **not** material-derived — R2's reason for per-tile is
"what is above and below it", which no `[materials.*]` row knows. So it is
independent authored state, and the open question is whether anything mutates it.
If a breach that removes a deck can raise a tile's leak, it is synced mutable
state and belongs in `DIGEST_FIELDS` (spec bump, goldens regenerated in the same
commit); if it is immutable after load it belongs in `SIM_FIELDS` and the
residency masks only, like `heat_atten_q` (`field_ab_harness.py:83-90`). **The
design says neither.** The same question applies to the ambient plane, which R8's
Dirichlet extension point would eventually want to write.

Whichever way it falls, four lists need naming and the design names none:
`GameMap._RESIDENT_MASKS` (the static-mask precedent, `gamemap.py:148-202`, with
the `__setattr__` stale-pointer guard and the one upload at `enable_residency`);
`field_ab_harness.SIM_FIELDS`; and — if mutable — `DIGEST_FIELDS` **and**
`tests/field_digest_spec.toml`, together.

**The Recorder needs nothing**, and the design is right to be silent there:
`Recorder.DEFAULT_FIELDS` (`recorder.py:81-83`) records dequantized state for
blowup forensics, and design v3 row 18 already settled that the radiation planes
are never recorded. Neither new plane is a per-tick evolving field, so neither is
a `.npz`-contract or DTYPE-class extension. I checked; nothing is owed.

### R4 (REQUIRED) — the level-data door is unnamed.

CLAUDE.md's level rule is one writer ever: `level_lib.py` (write) /
`level_loader.py` (read), "never hand-write level.toml". `k_leak_q` is per-tile
authored data that is *not* a material property, so it needs a home in that
format and a writer. §7.2 gives it no ingress sentence at all — the one
half-sentence ("Filled at level load through the Q16 boundary module") is about
the ambient plane.

### N1 (NOTE) — `radiation_sweep.cpp` is pinned at a **hard 0/0/0** float ratchet, comments included.

`tests/test_no_float_in_sim_tu.py:350-352`: the ratchet "counts LINES CONTAINING
THE WORD, comments included, so 0/0/0 forbids the two type names even in prose
there." T1 adds two planes to that TU; the natural explanatory comment ("a
per-tile ambient temperature as a floating-point value…") trips a hard gate.
Cheap to avoid, cheap to state. (`radiation_sweep.h` is not in `SIM_TUS` — only
the `.cpp` — so the header's prose is unconstrained.)

### N2 (NOTE) — the sweep/step-tail plane args are `.noconvert()` and required, not defaulted.

Design v3 row 36 made every plane argument `.noconvert()` so a stale caller
raises `TypeError` rather than silently forcecasting into a discarded temporary.
Two new required plane kwargs therefore break every direct `step_tail` /
`RadiationSweep.run` caller in the test tree at once
(`tests/test_radiation_sweep_shadow_wiring.py`,
`tests/test_radiation_sweep_reference.py`, the gate scripts). That is designed
behaviour, not a defect — but "goldens unmoved" is not the same claim as "the
suite is green", and T1's scope should say so.

---

## 2. T1's central claim: does the uniform case reproduce the scalar era bit for bit?

**Yes — achievable, and for `k_leak_q` nearly free. The hazard is not
reproducibility; it is that reproducibility is guaranteed for the wrong reason
(B1).**

### The leak plane: bit-for-bit is structural. `ret` is the only hoist that must move.

Today `k` and `ret` are hoisted once per run (`:177-178`), then used per cell at
`:277` (`leaked = (i_in * k) >> FP_SHIFT`), `:278` and `:291`. A per-cell `k_i`
turns `leaked` into `(i_in * k_i) >> 16` — the same single multiply, same single
arithmetic shift, same operand order, no re-association. `ret` must move
**inside** `visit` as `ret_i = (amb_m_i * k_i) >> 16`. Under a uniform fill both
produce the identical integer, because the hoist was pure CSE over
loop-invariant data and nothing about the truncation order changes. Integer `*`
and `>>` are exact; there is no accumulation whose order could shift.

I checked that the exact fixed point survives a per-cell `ret`, since that is
what the hoist was implicitly protecting. At a cell whose whole upwind
neighbourhood sits at its own ambient `A`:

```
fa + fb  =  (A·s >> 16) + (A − (A·s >> 16))  =  A       exactly   (:271-273)
leaked   =  (A·k_i) >> 16   ==   ret_i = (A·k_i) >> 16  identical integers
stream   =  A − leaked + ret_i  =  A
abs_mat  =  (A·a) >> 16     ==   emitted = (src·a) >> 16   (src = A when ex = 0)
i_out    =  A                        →  rad_net = rad_flux = rad_amb = 0
```

`fb` being the *remainder* rather than a second shift (`:272`, "never a second
shift") is what makes `fa + fb == A` exact, and per-cell `k` does not touch it.
The fixed point is per-cell and stays per-cell. **Sound.**

### The ambient: `amb_m` is a genuine hoist that must move per cell — §6 item 4 already says so.

`amb_m` (`:175`) is loop-invariant **and run-invariant**: one integer
`(E°[0] · w_m) >> 16` for the whole grid, read at four sites — `:267` and `:268`
(the virtual ring), `:284` (`src = amb_m + mul128_shr(ex_m, f, 24)`), and `:286`
(`emit_body`). Moving it per cell is a real change of shape, not a CSE undo.
Under a uniform fill it is bit-identical **provided the plane's denomination maps
to bucket 0** — §1 B1 point 2.

One arithmetic detail the reference must pin: `w_m = FP_ONE / n_ordinates` is
integer division (`:174` — 4096 at S16, **5461** at S12, not 5461.33), so
`amb_m = (E°[b] * w_m) >> 16` is not exactly `E°[b]/16`. That is already true
today and must be preserved verbatim per cell. The temptation when moving it
inside the loop is to fold the two steps (`(E°[b] * w_m) >> 16` into
`E°[b] >> 4`), which is **not** the same integer at S12 and not the same integer
at S16 either once `E°[b]` is large.

### R6 (REQUIRED) — the ambient quantum is **ten counts**, and it makes §6 item 2 vacuous.

At the sweep's own calibration (`config.toml:646`,
`rad_scale_derived = 2.125632e-08`), reproducing the exact bake chain of
`emissive_table.cpp:51-63`:

```
E°[0]     = 161                 amb_m at S16 = (161 · 4096) >> 16 = 10
E°[3999]  = 1 497 197 365       amb_m at top = 93 574 835        (2^26.5)
```

The per-ordinate ambient stream is **10 counts**. With the derived `k_leak = 0.10`
(`k_leak_q = 6554`), `ret = (10 · 6554) >> 16 = 1` and `leaked = 1` at ambient —
the fixed point holds, with one count of resolution. At `k_leak = 0.06`
(`k_leak_q = 3932`) both `ret` and `leaked` are **0** at ambient: the out-of-plane
channel is identically inert until a cell's incoming stream reaches
`65536 / k_leak_q` ≈ 17 counts.

That is a legitimate Q16 truncation dead-band; it is *symmetric* (`leaked` and
`ret` truncate the same way, so it cannot manufacture energy and conservation
stays structural) and it does not threaten bit-identity. **But it means §6 item 2
— "two cells with different `k_leak_q` book different `rad_amb`" — is only
testable above that threshold.** A gate written with two ambient cells and two
small `k_leak_q` values compares `0 == 0` and passes vacuously. Given this
project's own 2026-09 incident record (five test fixtures silently measuring
nothing — CLAUDE.md's igniter row), the gate statement must name the stream level
at which it measures.

### R5 (REQUIRED) — §6 item 4's wording is not achievable as literally stated once the ambient is per-cell.

"An ambient cell is still an exact fixed point — per cell, with bodies, at its
*own* ambient temperature." Once `amb_m` is per-cell, a cell sitting at its own
ambient **next to a cell at a different ambient** is *not* a fixed point: its
upwind neighbours' stored outflow carries *their* `amb_m`, so `i_in ≠ amb_m_i`,
hence `stream ≠ amb_m_i` and `abs_mat − emitted ≠ 0`. That is correct physics
(radiation transports across an ambient gradient) and is exactly the point of R3
— but the property only holds on a **uniform-ambient neighbourhood**, and the
gate must say so, or it will be written to fail and then "fixed" by softening the
wrong thing.

---

## 3. CPU↔GPU bit-identity for a per-cell leak and a per-cell ambient

### The sweep's future twin (P4): structurally fine, but three things must be pinned now.

The gather form already reads four per-cell planes inside `visit` (`temperature`,
`heat_atten_q`, `dyn_heat_atten_q`, `heat_inv_shift`) and its correctness
argument — *"each cell reads its two upwind neighbours' stored outflow and
recomputes the split from the same integers"* (`radiation_sweep.h`) — does **not**
involve the ambient at all: `fa = (io_a·s_m) >> 16` and
`fb = io_b − ((io_b·s_m) >> 16)` (`:271-272`) touch only the stored outflow and a
checked-in constant. Two more per-cell reads add no atomics, no ordering, no
divergence. Both `fleck_L_solid_q` and `fleck_f_q24` are already `FP_HD`
(`radiation_sweep.h`), and `e_bucket_of` / `e_inv_q` are `FP_HD` *"so the CUDA
twins (the P3 clamp in cuda_temperature.cu, the P4 sweep) share ONE definition
with the host"* (`emissive_table.h:37-38`). **The structure is right; I checked
and found nothing that breaks.**

### R11 (REQUIRED) — but three pins must be written at T1, in the reference, not discovered at P4.

1. **Whose ambient the virtual ring returns.** `:267-268` reads `amb_m` when the
   upwind cell is out of grid. The only atomics-free device choice is *the
   reading cell's own* plane entry; any other choice (a separate global
   "space ambient", the nearest in-grid cell's) is a different kernel and a
   different integer. §6 item 3 asserts against "the wrong cell's ambient"
   without ever saying which is right.
2. **`ret` is per cell, computed once per `visit`.** Today it is a run constant
   (`:178`). On device it becomes a register computed from two plane reads. Same
   integer either way — but it must be in `sweep_ref_q.py` first (CLAUDE.md's
   integer-reference rule: *"Any change is made THERE FIRST … and the C++ written
   against it — never the other way round"*), because the reference is what gate
   0 and the P4 twin are both written against.
3. **`t_amb` stays a scalar parameter on the device.** If B1 is resolved by
   adding a *new* ambient plane, then `RadiationSweep::run`'s `t_amb_q` stays an
   `int32_t` and `cuda_temperature`'s stays an `int32_t` (`cuda_temperature.h:139`,
   the gas-energy seam's offset). If instead `t_amb_q` is promoted, two kernels
   in the same tree take the same-named quantity as a pointer and as a scalar —
   a bit-identity trap with a name collision on top.

### R10 (REQUIRED) — T4a's "clamp on the CUDA temperature twin" is new device work the row does not cost.

I checked `cuda_temperature.cu` and `cuda_temperature.h` for the clamp's inputs:
**`e_inv_q`, `rad_fluence`, `e_table` and `rad_clamp_hits` do not appear at all.**
The twin has the widened `const int64_t* rad_net` (P1 landed that,
`cuda_temperature.h:112`) but no fluence parameter and no emissive table. So
T4a's one clause implies, on the device: a new `const int64_t* rad_fluence`
parameter with its malloc/H2D, a 4000-entry × 8-byte (32 KB) E° table upload or
`__constant__` residency, the `e_inv_q` call between the saturating add and the
rails in the pinned order (`temperature_solver.cpp:273-296` says *"ORDER IS
PINNED … and the CUDA twin pins the identical order"*), and a `rad_clamp_hits`
return path. The pieces exist and are `FP_HD`; the work is real and uncosted, and
T4a's gate is "CUDA tol-0", which is precisely what will fail if any of it is
missed.

### R8 (REQUIRED) — deleting `cool_shift` breaks the CUDA twin and two **pinned positional counter slots**.

§7.3 lists "the `COOL_SHIFT`/`COOL_SHIFT_VACUUM` globals, the per-row column, and
the Pass-3 relax-to-ambient block" — singular. The real surface:

| Site | What is there |
|---|---|
| `cpp/src/temperature_solver.cpp:634-688` | the CPU Pass 3 |
| `cpp/src/cuda_temperature.cu:418-470` | the **device** Pass 3 (`cool_shift_grid`, `vac_offset`, `cool_shift_floor`) |
| `cpp/src/cuda_temperature.cu:565-588` | its `d_csg` malloc + H2D |
| `cpp/src/cuda_temperature.cu:101-118` | `C_COOL = 3` and `C_THERMOSTAT = 12` in a **pinned positional enum**, `C_SLOTS = 13` |
| `cpp/src/cuda_temperature.h:118-131` | the mirrored slot-order comment |
| `cpp/src/physics_engine.cpp:377-388` | the positional fold `cond_counters[3]` … `[12]` |
| `cpp/src/bindings.cpp:1983, 1940-1952, 3540` | `.def_readonly("e_cool_sum")`, the three dial properties, and `py::arg("cool_shift_grid")` — **required, not defaulted** |
| `src/simulation/gamemap.py:202, 1489, 1697` | `cool_shift` in `_RESIDENT_MASKS`, built in `_update_caches`, patched in `on_tile_changed` |
| `src/simulation/physics_runner.py` (both paths) | `gmap.cool_shift` passed on the normal **and** resident tick |

Two slots vanishing from a positionally-pinned block is the sharp edge: either
the slots are **tombstoned** (keep `C_SLOTS = 13`, always 0) or every index
after them renumbers, and `cuda_temperature.cu`, `cuda_temperature.h`'s comment
and `physics_engine.cpp`'s fold must move in **one** commit. The design says
nothing about either. Deleting the CPU Pass 3 only would leave the device
relaxing to ambient and the CPU not — a CPU/GPU divergence, which the CUDA tol-0
gate exists to catch but which the design should not be relying on it to discover.

**And `e_cool_sum` is written from the same line as `e_thermostat_sum`**
(`temperature_solver.cpp:682` and `:686`). §7.3 names only the latter.
`e_cool_sum` has independent consumers that will break:
`tests/test_eos_p2_sealed_room_energy.py:74,154,282,323,346` (it is a **term in
that test's own identity**, and asserted `== 0` twice),
`tests/test_temperature_conduction.py:503` (`== 0`), and the `E_COUNTERS` name
tuples in `tests/cuda_conduction_check.py:107-109`,
`tests/cuda_cool_shift_check.py:84`, `tests/cuda_thermal_mass_check.py:202`
(hard `getattr` → `AttributeError` once the pybind member is gone).

### N4 (NOTE) — `tools/storm_ledger.py` swallows a deleted counter silently.

`storm_ledger.py:217-225` reads the same names inside a bare
`try: … except Exception: pass`. After the deletion the key simply disappears
from the ledger dict rather than failing — the quiet-failure mode this project's
own rules are written against. Worth a line in T4b's disposition.

---

## 4. int64 headroom with a per-cell ambient

**Unchanged. The bound survives; the measurement behind it does not, and the
design should say which one it is relying on.**

### The bound, re-derived with a per-cell ambient.

Let `M = max_i max(amb_m_i, src_i)`, where `src_i = amb_m_i + mul128_shr(ex_m_i,
f_i, 24) ≤ amb_m_i + ex_m_i`. Induct over the wavefront, assuming every stored
outflow so far is `≤ M`:

- `fa = (io_a·s) >> 16 ≤ (M·s) >> 16`; `fb = io_b − ((io_b·s) >> 16)` is
  non-decreasing in `io_b`, so `fb ≤ M − ((M·s) >> 16)`. Hence
  **`i_in ≤ M`** exactly (and an out-of-grid read is `amb_m_i ≤ M` by
  definition). `:271-273`.
- `stream = i_in − ((i_in·k_i) >> 16) + ((amb_m_i·k_i) >> 16)` is a convex
  combination of `i_in` and `amb_m_i` up to two floors, so
  **`stream ≤ M + 1`**. `:277-278`.
- `i_out = stream − abs_mat − abs_body + emitted + emit_body`, with
  `abs_mat + abs_body ≥ stream·(a+b)/2¹⁶ − 2` and
  `emitted + emit_body ≤ M·(a+b)/2¹⁶` since `a + b = d ≤ ONE`. So
  **`i_out ≤ M + 3`**. `:281-287`.

That is the same "a stream never exceeds the largest upstream source by more than
one count per cell" shape the header comment (`radiation_sweep.cpp:37-41`) already
asserts, with a small constant instead of one. Over a 128×256 grid the
accumulated slop is a few thousand counts against accumulators of order 2³⁰ —
irrelevant.

**The ceiling on `M` is unchanged** because a per-cell ambient is still an `E°`
lookup and `e_bucket_of` saturates at bucket 3999: `amb_m_i ≤ (E°[3999]·w_m) >> 16`,
which is exactly the ceiling `src_i` already had. So
`rad_fluence[i] = Σ₁₆ stream ≤ E°[3999]`, and the widest product,
`stream · ONE`, is `≤ E°[3999] · 2¹⁶ / 16`. Numerically:

| scale | `E°[3999]` | `rad_fluence` ceiling | widest `stream·ONE` |
|---|---|---|---|
| sweep's derived `2.125632e-08` | 2³⁰·⁵ | 2³⁰·⁵ | 2⁴⁶·⁵ |
| old fitted `5.1427e-5` (the header's "~2⁴¹·⁷") | 2⁴¹·⁷ | 2⁴¹·⁷ | 2⁵⁷·⁷ |

Both inside critique 3's `< 2⁴⁶` per cell / `< 2⁵⁸` per product, and the widest
product of all (`ex_m · f_q24`) goes through the kit's 128-bit
`mul128_shr` anyway (`:284`), so no headroom argument is load-bearing on it.
**Conclusion: a per-cell ambient does not change the overflow argument.**

### N3 (NOTE) — but restate it as a bound, not as a measurement.

Today `amb_m = 10` and the realized magnitudes are eight orders below the bound.
A per-cell ambient makes `amb_m_i` up to `E°[3999]·w_m >> 16 = 93 574 835`
*reachable in a scene a level author can write* — so the P0b measurements
(`2^62.13 at S12, 2^61.72 at S16`) that currently back the claim no longer
describe the operating range. The bound above still holds; the design should say
it is relying on the bound. One sentence.

### N5 (NOTE) — `T_abs_q << 24 < 2^54` is exactly tight today.

Design v3 §2.3 asserts `T_abs_q << 24 < 2⁵⁴`. At `T_MAX_PHYS = 16000` game,
`T_abs_q = (16000 << 16) + (293 << 16) ≈ 2³⁰·⁰`, so `<< 24` is `≈ 2⁵⁴·⁰` — the
bound is met with no margin. Nothing overflows even if it were exceeded (`:284`'s
operand is already `int64`, and `floordiv_q(T_abs_q << 24, D)` with `D ≥ T_abs_q`
still returns `≤ 2²⁴`), but the *stated* bound stops being true the moment an
unbounded per-cell offset lands — which is another reason the ambient plane needs
a range at its door (R2).

---

## 5. The #54 closure identity with a channel deleted (T4b)

### Sound: "close with the term removed, not zeroed" is right, and "six groups" is the correct count.

I verified this against `tests/test_thermostat_books.py:67-92`. `_terms()` returns
a **six**-entry tuple — EOS / thermal-solver-gas / combustion-gas / Python seams /
water-evac / **solid side** — and the solid side is itself the sum of four
counters:

```
int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
+ int(tsolver.e_thermostat_sum) + int(comb.e_comb_solid_heat_sum)
```

So deleting the thermostat removes **a term inside group 6**, not a group. §6
item 8's "still close over six groups … with the ambient thermostat's term gone"
is exactly right, and §7.3's "the identity must close with the term removed, not
zeroed" is the correct instruction. I went looking for a miscount here and did
not find one.

### R7 (REQUIRED) — but "and `rad_amb`'s grown" is the wrong ledger term.

`rad_amb` is not in the P-G5 identity at all. It is a plane in the sweep's **own**
conservation law, `Σ rad_net + Σ rad_flux + Σ rad_amb ≡ 0` (design v3 §2.3, gate
1) — a statement about the radiation field, which starts at ambient and ends at
ambient within a tick and never enters `solid_energy_books_sum`.

The term that actually grows in the P-G5 ledger is **`e_solid_deposit_sum`**. The
radiative fold books its *applied* ΔT there, post both rails:

```
temperature_solver.cpp:328-329
    e_solid_deposit_sum += ((int64_t)temperature[i] - t_before_rad) * cap_real_[i];
```

which is why the fold needs no new counter — the header block says so, and the
Pass-1 clamp deliberately sits *before* that booking (`:271-292`) so the clipped
landing is what gets priced. **That is genuinely elegant and I checked it holds.**
But an implementer told to "grow `rad_amb`" in the books will either hunt for a
counter that does not exist or add one and double-count. One sentence fixes it:
*the thermostat's job moves into `e_solid_deposit_sum`, which already books it.*

### R9 (REQUIRED) — the cool-shift and #54 tests die at **T4b**, not T5, and one of them asserts on the deleted counter.

T5's row disposes of "the 46 old-law tests". None of the following is an old-law
test; they are #54 / cool-shift-axis gates that stop being satisfiable the moment
Pass 3 is deleted, which happens at **T4b**:

- `tests/test_thermostat_books.py` — assertion (b) is
  `assert int(tsolver.e_thermostat_sum) < 0` (`:143`), on a counter that no longer
  exists. Its companion monotone-decay assertion survives *numerically* but its
  stated **reason** ("the thermostat is the only channel with anywhere to put net
  energy in this closed, fireless scenario", `:138-142`) becomes false — with
  `k_leak` live, the out-of-plane channel is that channel. A property test whose
  reason has silently changed is exactly what CLAUDE.md's "tests assert
  properties, and name the change that must break it" rule is for.
- `tests/test_cool_shift_axis.py`, `tests/cool_shift_axis_gate_a_capture.py`
- `tests/test_cuda_cool_shift.py` + `tests/cuda_cool_shift_check.py` (a CUDA
  harness pair; the wrapper will fail on a missing kernel arg)
- `tests/test_temperature_cooling.py`
- `tests/test_eos_p2_sealed_room_energy.py` — `e_cool_sum` is a **term in its own
  identity** at `:74` and `:154`, and asserted `== 0` at `:282` and `:346`
- `tests/test_temperature_conduction.py:503` — `assert int(solver.e_cool_sum) == 0`

**Sound, and worth recording:** the float ratchet is *not* a problem here.
`temperature_solver.cpp` is a `MIGRATED_FLOOR_TUS` entry at `float 4 / double 6`,
and Pass 3 owns at least the `float o2_vacuum_thresh` chain (`:612`), so the
counts drop. `test_baseline_is_not_stale_low` (`:466-487`) is explicitly "a nudge,
not a gate" and always passes. Nothing to do beyond tightening the baseline.

---

## 6. The single golden re-baseline — is one enough?

### B2 (BLOCKING) — no. T4a's spec bump moves every golden, by construction, in T4a's own commit.

T4a's row includes: *"`dyn_heat_atten_q` into the digest, spec v6."* That is a
`DIGEST_FIELDS` **membership** change. Two checked-in rules bite:

1. `tests/field_digest.py:121`:
   ```
   h.update(f"FIELD_DIGEST_V{DIGEST_SPEC_VERSION}\n".encode("ascii"))
   ```
   The version is **salted into every per-field hash**. `_xarch_perfield_digest.py`
   states the consequence verbatim from the last time it happened (P-G0, v4→v5):
   *"DIGEST_SPEC_VERSION is itself hashed into every per-field digest — so EVERY
   GOLDEN this suite carries moves on a spec bump, by construction, even when the
   underlying arithmetic is untouched."*
2. CLAUDE.md's own iron rule: *"Field digest | Membership/dtype change = version
   bump + regenerate all goldens **same commit**."* The regeneration cannot be
   deferred to T4b.

The blast radius is **19 source files** carrying `GOLDEN_AGGREGATE`
(`tests/_xarch_perfield_digest.py`, 11 `cuda_*_check.py`, and
`test_b6_logic_golden.py`, `test_radiation_sweep_shadow_wiring.py`,
`test_s3b_fire_determinism.py`, `test_thermostat_books.py`,
`test_vent_determinism_and_serialize.py`, `test_vent_dormancy.py`,
`test_w6_armory.py`).

And T4a is **also a physics move** independent of the schema: it flips the fold
onto the sweep's planes (a different emission scale — `rad_scale_derived` is
2419× smaller than the fitted `rad_scale`, per R9's own numbers), turns the
Pass-1 clamp live, and makes units absorb the body share. `temperature`, `heat`
and everything downstream of them move. T4a's stated gate is "full suite", which
cannot be green without a re-baseline.

So the arc as cut needs **two** golden movements, not one. Either the design says
T4a and T4b land as a single commit with a single re-baseline (in which case
T4a's separate gate row is fiction and should say so), or it admits two.

### B2 (continued) — and the P-G0 procedure for telling a schema move from a physics move is unavailable when both land together.

The P-G0 precedent is not just a rationale format; it is a *method*. Its rebase
note records: *"Verified NOT a physics move directly (not just inferred from the
spec-bump argument): captured every SIM_FIELDS array (raw, not hashed — immune to
the version salt)…"* That check works precisely because P-G0 was schema-only. A
combined T4a re-baseline cannot use it: the raw arrays move too, so a genuine
accidental regression inside the spec bump is indistinguishable from the
intended fold flip. The design must state how that distinction is preserved. (I
am naming the property that is lost, not proposing the patch cut that would
preserve it — that is L3's territory.)

### Sound: T1, T2 and T3 really do leave the goldens alone.

I checked rather than assumed. T1's planes feed only `RadiationSweep::run`, whose
four outputs are the shadow planes `rad_net_sweep` / `rad_flux_sweep` /
`rad_amb_sweep` / `rad_fluence` — none of which is in `DIGEST_FIELDS` or
`SIM_FIELDS` (`radiation_sweep.cpp:226` says so, and the spec toml's field list
confirms it). `k_leak` stays at `0.0` until T4b (`config.toml:597`), so even the
shadow numbers barely move. T2 and T3 are reports. **Provided** B1 is resolved
such that the uniform ambient fill lands on bucket 0 (§1 B1 point 2), T1's
"goldens unmoved" claim is correct.

One consequence the design should note anyway: if R3 resolves toward `k_leak_q`
being **mutable** synced state, then `DIGEST_FIELDS` gains a field **at T1**, and
T1 becomes a third golden-moving patch. That is the real reason R3 must be
settled on paper first.

---

## 7. What I checked and found sound

Stated explicitly, because independence is only worth something if the negatives
are reported too.

1. **The gather form's exact fixed point survives per-cell coefficients.**
   `fb` as the remainder rather than a second shift (`:272`) makes
   `fa + fb == amb_m` exact, and nothing about a per-cell `k` or a per-cell
   `ret` touches that. Verified by hand, term by term (§2).
2. **Conservation is genuinely structural and survives both planes.**
   Every integer leaving the stream is added to a ledger as the same integer
   (`:274-275`, `:289-292`, `:300-301`); the ring's two books close the boundary
   whatever value the ring carries. §6 item 3 will pass for the right reason.
3. **The Fleck factor is conservation-neutral.** It is applied before the sweep
   (`:196-215`), so a per-cell `t_amb` cannot break gate 1 — which is also
   exactly why it cannot deliver R3 (B1).
4. **int64 headroom is unaffected by a per-cell ambient** (§4, induction plus the
   `E°[3999]` ceiling).
5. **The #54 identity's "six groups" count is correct**, and the thermostat is
   correctly identified as a *term*, not a group (§5).
6. **The radiative fold needs no new counter.** `e_solid_deposit_sum` books the
   applied ΔT after the clamp and both rails (`temperature_solver.cpp:328-329`),
   and the clamp is deliberately placed before that booking. This is the right
   design and I found no hole in it.
7. **The Recorder owes nothing** (`recorder.py:81-94`, design v3 row 18).
8. **The float ratchet does not obstruct T4b.**
   `test_baseline_is_not_stale_low` is a nudge, not a gate
   (`test_no_float_in_sim_tu.py:466-487`).
9. **`e_inv_q` / `e_bucket_of` / `fleck_*` are already `FP_HD`**, so T4a's CUDA
   clamp and P4's sweep twin share one definition with the host by construction
   (`emissive_table.h:37-38`, `radiation_sweep.h`). The structure the design
   depends on is in place; only the plumbing is uncosted (R10).
10. **T1 is correctly routed reference-first** through `sweep_ref_q.py`, per
    CLAUDE.md's integer-reference rule. That is the right ordering and the one
    thing that makes B1 cheap to fix if it is fixed on paper now.
