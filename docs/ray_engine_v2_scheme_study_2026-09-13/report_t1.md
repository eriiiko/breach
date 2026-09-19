# T1 — per-tile ambient (`amb_m`) + uniform `k_leak` live in the sweep

> Issue #12, ray-engine-v2. Worktree `breach-t1`, branch
> `12-t1-ambient-plane-leak-live` off `fire-12`.
> Design: `docs/thermal_model_v2_design_2026-09-19.md` R2/R3, §3.3, §5 T1, §6 2-4.
> **Written incrementally** — every section below is filled as its work lands.

## 0. Status

| step | state |
|---|---|
| D1 reference (`sweep_ref_q.py`) | **done** |
| D1 reference gates G1-G12 (+2 new blocks) | **done** — ALL GATES PASS, re-run at the tail (§5.1) |
| D2 engine (`radiation_sweep.{h,cpp}` + binding + runner) | **done** |
| D3 gate 0 bit-for-bit | **done** — 162 radiation tests pass (§5.2) |
| D3 goldens unmoved | **done** — nothing that could move one was touched (§5.5) |
| D4 item 2 (uniform == scalar era) | **done** — pinned on the ENGINE, against the pre-patch reference (§5.3) |
| D4 item 3 (non-uniform genuinely varies) | **done** — and unpassable by a hoist; validated by breaking it (§5.4) |
| D4 item 4 (conservation with per-cell ambient AND k_leak > 0) | **done** — identity exactly 0, 12 configurations (§5.3) |
| full suite | **done** — 2622 passed, 0 failed (§5.6) |

> **The patch is complete.** §5 is the evidence; §6.4–§6.7 are what the tail
> found that §1–§4 could not have known.

## 1. The scalar that was named wrong, and the one that is right

`t_amb_q` is the **absolute-zero offset of the temperature scale** — the integer
that turns a game temperature (a delta above ambient) into Kelvin for the Fleck
denominator `g = 4L/T_abs` (`radiation_sweep.cpp:211`). It is bound from the same
`293 << 16` the arc #54 gas-energy seam runs on, and it **stays a global
scalar**: untouched by this patch (thermal v2 §3.3, L2-B1, L2-R1).

The ambient the sweep *radiates* at is `amb_m = (E°[0] · w_m) >> 16` — hoisted
once per run at `radiation_sweep.cpp:175` before this patch. **That** is what
becomes per-cell.

### 1.1 The ambient is an emissive LEVEL, and could not have been a temperature

Forced, not a taste. `e_bucket_of` returns bucket 0 for every `T_q <= 0`
(`emissive_table.h:57-61`), so `E°[T] >= E°[0]` for **every** temperature the
table can be asked about. A temperature-denominated ambient therefore cannot
express an ambient *below* `E°[0]` — and "0 K outside the hull", the thing R3
exists to make expressible, is exactly such an ambient. The per-cell quantity is
an **emissive level in the E-table's own units**, with the invariant
`0 <= amb[i] <= E°[0]`.

That upper bound is load-bearing: it is what makes the per-cell excess
`E°[T_i] - amb[i]` non-negative for every temperature, which the excess form of
the Fleck factor (design §2.8 row 22) and the sweep's positivity both rest on. A
hotter-than-room ambient (a furnace deck above) is refused at the door rather
than clamped downstream; it is design §7.4 territory and owes its own argument.

## 2. Decisions — which cell's ambient, at each of the four sites

All four are **cell i's own ambient**. One reason, four times: the per-cell fixed
point of design §2.3 ("at exact ambient a body disturbs nothing") survives
per-cell ambients only if every ambient-derived term at cell i is the *same
integer*.

| site | which cell | why |
|---|---|---|
| the cell's own emission floor, `src = amb_m_i + f*ex_m` | i | the excess is measured over i's own ambient, so the two halves sum back to `E°[T_i]*w_m`. Makes `ex_cell` per-cell-ref, and the Fleck pre-pass's `L` with it |
| `ret`, the ceiling's ambient return | i | the ceiling above cell i is cell i's own out-of-plane boundary — the same one `leaked` left through. At `i_in == amb_m_i`, `leaked == ret` as the same integer, so a cell at its own ambient stays an exact fixed point |
| `emit_body`, the body standing on cell i | i | a grey body at the ambient of the cell it stands in (design row 25), so `abs_body == emit_body` by the same integers when the stream is at that cell's ambient |
| the **virtual ambient ring** (§2.7) | the **reading** cell, i | see below — the pin L2-B1 point 4 asked for |

### 2.1 The virtual ring — the pin, and its reasoning

The ring cell is outside the grid and has no cell of its own. Conservation
survives *any* choice (whatever integer arrives is booked as minus-fa / minus-fb
into the **reading** cell's `rad_amb`), so this is a pin, not a correctness
accident. It is pinned to **the reading cell's own `amb_m_i`** because:

1. it is the only choice that keeps a **boundary** cell at its own ambient an
   exact fixed point — a global sky constant would bathe a 0 K hull cell in
   room-temperature radiation;
2. it needs no new global and no second ingress door;
3. it reduces to the scalar era *exactly* when the ambient is uniform;
4. the CUDA twin (P4) reads a register it already holds instead of a neighbour
   outside the grid — no out-of-grid addressing, no second branch.

The consequence, stated so it is deliberate: **the sky beyond the grid is
whatever the boundary cell's own environment is.** A vacuum boundary sees a cold
sky; an interior boundary sees a room-temperature sky. Design §2.7's "the sky
beyond the grid is ambient, even in space" is preserved in the only sense that
survives a per-cell ambient.

### 2.2 Named gap — a body in a cold cell

`emit_body` using cell i's ambient means a marine standing on a vacuum tile is
modelled as radiating at the vacuum's ambient, not at ~310 K. Under thermal v2
**R4** (space is at room temperature in v1) no shipped scene has a cold ambient
anywhere, so this is a mechanism choice with **no live consequence today**. A
body with an emission temperature of its own is a T5-or-later question. The
alternative (a body on a global body-temperature reference) was rejected because
it breaks `abs_body == emit_body` at the per-cell fixed point, which is gate
G2a's property and the thing critique 3 §4f measured the cost of.

### 2.3 What is deliberately NOT made per-cell

- **`t_amb_q`** — the temperature-scale offset (§1). Shared with the arc #54
  seam (L2-R1); a per-tile version would make "born at ambient" per-tile and the
  relative-energy readback wrong.
- **The Pass-1 clamp ceiling `E-inv(Phi)`** (`temperature_solver.cpp`). It is
  ambient-derived and L2-B1 point 3 lists it, but it lives in the temperature
  solver, is dormant until P3, and D5 forbids touching that file. **T5 inherits
  the question**: when the fold flips, `E-inv` maps a fluence to a temperature
  through a table whose bottom bucket is the *interior* ambient, so a
  cold-ambient cell's clamp ceiling is still the room-temperature floor.
  Harmless while R4 holds (every ambient is `E°[0]`); it must be re-read the day
  a cold sky ships.

## 3. The reference change (D1)

`docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py`:

- **new `ambient_plane(e_ref, h, w, table)`** — the one door. Accepts `None`
  (`E°[0]` everywhere), an `int` (broadcast — the scalar era, and `0` is the
  scheme study's zero-sky configuration), or a plane. The scalar is broadcast
  **at the door**, so there is exactly one code path below it and no fork.
- **`validate_planes(a, d, k, amb=None, table=E)`** — gains the ambient
  invariant `0 <= amb[i] <= E°[0]`, raising like its three siblings. Called from
  `sweep_q` *after* the ambient plane is normalised.
- **`sweep_q`** — `amb_m` becomes `amb_m_cell[y][x]`, read **inside** the cell
  loop; `ret`, `emit_body`, `src` and both virtual-ring reads take it. `ex_cell`
  is now `E°[T] - amb_lvl[y][x]`. The old `E°[T] < e_ref` raise becomes provably
  dead on a validated scene and is kept as belt-and-braces for `validate=False`.
- **`fleck_prepass`** — takes the same scalar-or-plane `e_ref`, so a cell's `L`
  (its free excess emission) is measured over *its own* ambient. A cell facing a
  cold sky has more excess to shed and is damped accordingly.
- **Gates**: `G1` gains a **cold-sky-patch** block (non-uniform ambient *and*
  `k_leak` live, both transports, S16 and S12 — identity exactly 0, all three
  sums non-zero). `G2a` gains **non-vacuity (c)**: the same all-at-ambient scene
  with a cold-sky patch stops being a fixed point (113 non-zero cells, min
  `rad_net` = -103542), where a *hoisted* `amb_m` would give zeros everywhere.

**The uniform path did not move.** A sha256 over the four planes + the Fleck
plane + the stream telemetry, across gate 0's whole 32-configuration matrix, is
identical before and after:

    SCALAR_ERA_DIGEST = c965d26549fdc10d068cced722692d306c7be8c5bc49b55bdaeed9f6d707005b

`sweep_ref_q_gates.py` -> **ALL GATES PASS** (G1-G12).

## 4. The engine change (D2)

### 4.1 `cpp/src/radiation_sweep.{h,cpp}`

- `run()` takes **`const int64_t* amb_level`** (the per-cell ambient LEVEL,
  right after `e_table`), required and non-null. `amb_m` and `ret` leave the
  hoist block at the top of `run()`; `amb_m_` is computed per cell in the
  pre-pass (it needs `w_m`, so it cannot be computed earlier than the pre-pass
  anyway) and `ret` is formed inside the cell loop from it.
- The pre-pass gains the **ambient ingress re-check** `0 <= amb_level[i] <= E°[0]`
  beside the existing `0 <= a <= d <= ONE`, and `ex_cell_[i]` becomes
  `E°[T_i] - amb_level[i]`.
- **new `derive_ambient(is_vacuum, e_table, vac_level, n)`** — THE derivation
  (thermal v2 R3). A vacuum cell takes `vac_level`; every other cell takes
  `E°[0]` — interior air, solids, **and the ambient ring**, which *is*
  room-temperature air by definition, so only `is_vacuum` keys the select.
  `vac_level < 0` means `E°[0]` (R4, uniform, the scalar era). Above `E°[0]`
  raises. Fills the sweep's own scratch; nothing is authored and no new plane
  crosses the binding on the live path.

### 4.2 `physics_engine.{h,cpp}` — step 2b

`step_tail` gains **`int64_t rad_amb_vacuum_q = -1`** and calls
`derive_ambient` on the `is_vacuum` plane **already in its parameter list**,
then hands the result to `run()`. No new plane argument, no new allocation on
the caller's side, no level data.

### 4.3 `bindings.cpp`

- `RadiationSweep.run` gains `amb_level` as a keyword argument: an int64
  `(h, w)` array, or **`None`** — which broadcasts `E°[0]` at the door, exactly
  as the reference's `ambient_plane(None)` does, so the binding and the spec
  have the same door and the sweep below both has one path.
- `RadiationSweep.derive_ambient` is exposed so the derivation itself is
  gateable from Python (item 3 drives it directly).
- `PhysicsEngine.step_tail` gains `rad_amb_vacuum_q` (default `-1`).

### 4.4 `config.toml` + `physics_runner.py` — the two dials

- **`[physics.radiation] k_leak = 0.10`** — was `0.0`. Live at the derived
  value (thermal v2 R2): the fit of the exact slab law `S(r) = a/sqrt(a²+r²)`
  for a 2.5 m deck, max 8.2 % over r = 1..12 tiles. Quantizes to
  `k_leak_q = 6554` (0.1000061). The sweep is still in shadow, so this moves no
  golden; **T5's flip is where it starts cooling rooms.**
- **`[physics.radiation] vacuum_ambient_K = 295.0`** (new) — the temperature of
  space. The runner bakes it through the **same exact chain** the E° table uses
  (`K⁴` by repeated integer multiplication, one `rad_scale_derived` boundary
  multiply, round half up) into `self.rad_amb_vacuum_q`, and validates
  integer-valued Kelvin and `level <= E°[0]`. The default, `kelvin_ambient +
  2·k_temp_to_kelvin`, is bucket 0's own midpoint, so it bakes to **exactly
  `E°[0] = 161`** — verified live, not assumed — which is R4 and keeps the
  derived plane uniform. `vacuum_ambient_K = 0.0` is Erik's cold sky.

Measured on the default scenario after the change: `E°[0] = 161`,
`rad_amb_vacuum_q = 161` (equal), `k_leak_q = 6554`, the derived plane uniform
at 161, and one tick's shadow planes still `net = flux = amb = 0` — i.e. an
all-ambient scene is an exact fixed point **with the leak live**, which is
design §2.7's claim about `leaked == ret`, now on the live path.

### 4.5 Callers updated

`tests/_radiation_sweep_harness.py` (both drivers take `amb`, one shared
`amb_plane` door), `tests/test_radiation_sweep_reference.py`,
`tests/test_radiation_sweep_shadow_wiring.py` (it now reproduces the
conductor's *derivation*, not a `None`, so it cannot go vacuous when a cold sky
ships), `tools/bench_radiation_sweep.py`, `tools/fire_tuning_lab.py`.

## 5. Gate results (D3, D4)

Every number below was re-measured at the tail, not carried over.
Python: `C:/Users/steen/anaconda3/python.exe` (there is no conda `data` env on
this machine). CPU build only; no CUDA in this patch.

### 5.1 The reference's own gates

`docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q_gates.py` →
**ALL GATES PASS** (G1–G12). The two blocks §3 added are green and non-trivial:

    shear k= 6554 S16 COLD-SKY PATCH: net=  -9770926588038 flux=  3249619431691 amb=  6521307156347 identity=0  each-nonzero=True  OK
    shear k= 6554 S12 COLD-SKY PATCH: net= -10690861153394 flux=  3857242593855 amb=  6833618559539 identity=0  each-nonzero=True  OK
    step  k= 6554 S16 COLD-SKY PATCH: net= -11486377086464 flux=  6350301022008 amb=  5136076064456 identity=0  each-nonzero=True  OK
    step  k= 6554 S12 COLD-SKY PATCH: net=  -9004114471019 flux=  2453889927360 amb=  6550224543659 identity=0  each-nonzero=True  OK
    non-vacuity: a cold-sky patch (ambient level 0 on x < 3, every T still at
    ambient) -> 113 nonzero cells, min rad_net = -103542 (a 293 K wall facing
    0 K radiates; a HOISTED amb_m gives 0 everywhere)

### 5.2 Gate 0 — bit for bit, across the ambient axis

    pytest tests/test_radiation_sweep_reference.py tests/test_radiation_sweep_gates.py \
           tests/test_radiation_sweep_constants.py tests/test_radiation_sweep_shadow_wiring.py \
           tests/test_radiation_sweep_ambient_plane.py -q
    -> 162 passed in 4.23s

136 of those are the four pre-existing files (the count the orchestrator verified
before committing §4.5's work); **26 are the new ambient-plane file**. Gate 0 now
runs its 32-configuration matrix over three ambients — `uniform`, `cold-half`,
`random` — so the C++ is held to the reference on a *non-uniform* plane as well,
which is what makes it able to see a PARTIAL hoist (one of the four
ambient-derived sites left reading a global moves no integer under a uniform
ambient).

### 5.3 The three properties — `tests/test_radiation_sweep_ambient_plane.py`

The file the harness already named in §4.5. 26 tests.

**Item 2 — the uniform case is unchanged, integer for integer.** A sha256 over
gate 0's whole 32-configuration matrix (the four output planes + the Fleck plane
+ the stream telemetry, serialisation spelled out in `_scalar_era_digest`):

    SCALAR_ERA_DIGEST = 6a28a3fba102cf63fe51a0aab7880ae8e411ef5995a41c5bb49cd56798655f00

Three independent computations agree on it:

| what | why it is the scalar era |
|---|---|
| the **pre-patch reference**, `git show b3488fc:.../sweep_ref_q.py` | the spec the pre-patch engine was held to bit for bit by gate 0 — this is what the sweep produced BEFORE `amb_m` went per-cell |
| the **current reference**, `amb=None` | §3's claim, re-measured |
| the **current C++ engine**, `amb=None` | the claim design v2 item 2 actually makes, which §3's reference-side digest did not reach |

The constant is therefore a **historical** fact, not a snapshot of current
behaviour: it cannot be satisfied by "whatever the code does today", and it is
re-derivable at any time from the git object. It moves only when the uniform path
is deliberately changed, with written rationale — the golden discipline.
(§3's `c965d265…` was a reference-side number under an uncommitted
serialisation; `6a28a3fb…` is the same *fact* under a serialisation that is now
checked in and applied to the engine as well. No number moved.)

A second leg pins the **three doors**: `amb=None`, `amb=E°[0]` broadcast, and
`derive_ambient(is_vacuum, vac_level=-1)` give the same integers on every plane
over a *non-trivial* `is_vacuum` mask — which is why today's live path (R4) is
still bit-identical to the scalar era.

**Item 3 — the non-uniform half.** §5.4.

**Item 4 — conservation with both changes live at once.**
`Σ rad_net + Σ rad_flux + Σ rad_amb == 0` **exactly**, in int64, no tolerance,
with `k_leak = 0.10` live and a non-uniform ambient, over 12 configurations
(3 ambient shapes × 2 transports × S16/S12) — each of the three sums individually
non-zero in all twelve. The three shapes are a breach cold patch, **the R3
derivation driven off a real `is_vacuum` mask** (the shipped path), and a
per-cell random level over the whole legal range `[0, E°[0]]`. As the committed
test builds them:

    shear S16 cold-patch : net=     -127714529845 flux=     57140162244 amb=     70574367601 ident=0
    shear S16 derived    : net=     -127714807589 flux=     57139811512 amb=     70574996077 ident=0
    shear S16 random     : net=     -127714287435 flux=     57140320911 amb=     70573966524 ident=0
    step  S12 derived    : net=      -68029704458 flux=     32101950353 amb=     35927754105 ident=0

**Item 4 needed a non-vacuity guard, and it is worth saying why.** Look at the
first three lines: the three ambient shapes move the sums by about **5 parts in a
million**. The identity is *structural* — it closes for a uniform ambient just as
exactly — and this scene's totals are dominated by cells up to 32767 game, beside
which the ambient is a rounding detail. So `ident == 0` would still hold if the
ambient plane silently stopped reaching the sweep, and item 4 would have become a
conservation gate that had quietly lost its "**with a per-cell ambient**" half.
The test therefore also runs the same scene at `amb=None` and asserts the two
`rad_net` planes differ.

### 5.4 Item 3, and the proof that it cannot pass with a hoist

This is the test the patch exists for, so it is written as an argument rather
than as an assertion, and every step of the argument is **measured**.

**The scene.** Two identical compartments, mirror images about the vertical
midline: `out out | WALL | in in in in | WALL | out out`. Every plane the sweep
reads — `a`, `d`, `T`, `heat_inv_shift`, `thermal_solid` — is exactly
mirror-symmetric, and every cell sits at ambient. The **only** asymmetric input
in the whole scene is `is_vacuum`: the port side is open to space, the starboard
side is not. Two structurally identical hull walls, one facing vacuum and one
facing interior air.

**The argument.**

1. the **control** leg runs the same scene at `vac_level = -1` (R4) and measures
   that all four output planes come back *exactly* mirror-symmetric — the sweep
   commutes with the mirror — and that the scene is the exact G2a fixed point
   (zero non-zero cells across all three ledger planes);
2. the **hoist** leg re-measures that symmetry at every level the derived plane
   actually contains (0 and `E°[0]`);
3. therefore any implementation that collapses the ambient to one scalar —
   whatever value, from whichever cell — yields a mirror-**symmetric** output on
   this scene;
4. the **claim** leg asserts the output is **not** mirror-symmetric, and that the
   port wall loses strictly more than its twin at *every* row.

A v1-style patch (`t_amb_q` made per-tile, `amb_m` left hoisted) lands in (3) and
fails (4). Measured, shear/S16: port wall `rad_net = -115604`, starboard
`-2750` — **42×**, from one cold sky. `L < R < 0` holds at all 7 rows × 2
transports × S16/S12.

Two further legs: **localisation** (one vacuum cell; `rad_amb`'s *strict*
maximum is the cell the mask names) and **the derivation itself**
(`derive_ambient` keyed on `is_vacuum` alone, the R4 sentinel, the
`0 ≤ amb ≤ E°[0]` door raising).

**Validated by breaking it** (`feedback_review_the_first_instance`: a gate that
cannot fail is not a gate). Four perturbations were injected and each test re-run:

| injected bug | item 2 digest | item 3 hull | item 3 localisation | item 4 + its guard |
|---|---|---|---|---|
| `amb_m` hoisted at `E°[0]` — **the v1 bug** | pass | **FAIL** | **FAIL** | **FAIL** |
| the plane read once, at cell `[0][0]` | pass | **FAIL** | **FAIL** | pass |
| the plane read once, at the last cell | pass | **FAIL** | **FAIL** | **FAIL** |
| a mis-indexed per-cell read (roll by one column) | pass | pass | **FAIL** | pass |

**Item 3 is the only column that is `FAIL` on every hoist.** Item 4's two failures
are its *non-vacuity guard* firing, not conservation: both of those hoists happen
to collapse the ambient to `E°[0]`, which is the uniform run the guard compares
against. The `[0][0]` hoist collapses it to 0 instead — a different uniform
ambient — and item 4 sails through. The guard asks "did the plane reach the
arithmetic", never "is it read per cell"; the identity itself (§6.5) is blind to
all four.

Gate 0 was put through the same roll: `amb=uniform` passes it, `amb=cold-half`
and `amb=random` **fail** it (`rad_net differs at 44 cells`). See §6.4 for what
that table means.

### 5.5 Goldens unmoved

Three kinds of evidence, none of them "the suite was green".

1. **Nothing that could move one was touched.** Over the whole patch
   (`git diff --name-only b3488fc..HEAD`) no golden artifact, no digest spec, no
   baked-art PNG, and no file holding a golden constant appears — 16 files, all
   of them the sweep, its reference, its tests, its two tools, and
   `config.toml`. `tests/_xarch_perfield_digest.py` (GOLDEN_AGGREGATE) and
   `tests/field_digest_spec.toml` are untouched; `DIGEST_SPEC_VERSION` is
   unchanged.
2. **Both new dials terminate at the sweep.** `k_leak_q` and
   `rad_amb_vacuum_q` are read in `physics_runner.py` and handed to
   `step_tail` → `RadiationSweep::{derive_ambient,run}`; grep finds no other
   consumer. The sweep's four outputs are the shadow planes, and the only live
   reader of any of them — the Pass-1 clamp's `rad_fluence` in
   `temperature_solver.cpp:286`, behind `rad_fluence != nullptr && e_table !=
   nullptr` — is handed **`nullptr, nullptr`** by the conductor at
   `physics_engine.cpp:433` (read, not assumed; the comment above it says P3 is
   where they arrive). The sweep is still genuinely in shadow, so `k_leak` going
   live moves nothing until T5.
3. **The suite agrees** — §5.6, which collects every golden test there is.

### 5.6 Full suite

    C:/Users/steen/anaconda3/python.exe -m pytest tests -q
    2622 passed, 29 skipped, 4 xfailed, 3 warnings in 119.47s (0:01:59)

**0 failed.** No pre-existing failure to report. The 29 skips and 4 xfails are the
tree's standing ones (CUDA-gated and marked), unchanged by this patch.

## 6. Findings, surprises, and what T5 inherits

1. **The ambient could not have been a temperature** — §1.1. This is why the
   scalar-naming error (L2-B1) was not merely a rename: a per-tile `t_amb_q` is
   a *temperature* plane, and no temperature plane can express R3's "0 K
   outside".
2. **The Fleck factor is ambient-dependent too.** `L_q` is the cell's free
   *excess* emission, so the pre-pass reads the ambient as well — the change is
   not confined to the sweep loop. Under a uniform ambient it is bit-identical.
3. **T5 inherits the Pass-1 clamp ceiling** — §2.3.

*(4–7 added at the tail, from the gate work itself.)*

### 6.4 Design v2 §6 item 2's stated break condition is wrong

Item 2 reads *"the uniform case is unchanged … **Breaks if `amb_m` is
mis-indexed**."* It does not, and cannot. A uniform ambient plane holds the same
integer in every cell, so any read that lands **inside** the plane returns the
right value whatever index it uses: a uniform-ambient test is *structurally
blind* to an in-plane mis-index. Measured — a roll-by-one-column perturbation
passes the item 2 digest untouched (§5.4 table, last row). Item 2 catches only a
read that leaves the plane (a wrong stride, scratch read past its fill), and its
real content is the one it does deliver: **the uniform path did not move.**

The mis-index property is owned by **gate 0's non-uniform ambient axis**, which
§4.5's work added and which was measured to fail under exactly that roll. A
localisation leg was added to item 3 as well, so the property is also asserted
where a reader of the ambient-plane file will look for it. Three gates, three
distinct properties:

| property | owner |
|---|---|
| the uniform path did not move | item 2 (the scalar-era digest) |
| the ambient is read at the RIGHT cell | gate 0's `cold-half` / `random` axes; item 3's localisation leg |
| the ambient is read PER cell at all | item 3 |

### 6.5 As the design specifies them, items 2 and 4 are both blind to the hoist

Item 3 is the only column of §5.4's table that fails on every hoist. The v1 bug —
`t_amb_q` made per-tile, `amb_m` left hoisted — sails through the scalar-era
digest (a hoist *is* the scalar era), through gate 0's uniform axis, and through
**conservation**: the identity is structural and closes exactly for any ambient,
uniform included, so item 4 *as design v2 states it* cannot see a hoist at all.
The two `FAIL`s in its column are the non-vacuity guard §5.3 describes, and they
are opportunistic — they catch the hoists that happen to land on `E°[0]`, and
miss the one that lands on 0.

Stated plainly because it is the shape of the original failure: **a patch can be
green on every gate the design lists except one and still deliver nothing at
all.** Item 3 is not a nice-to-have leg of T1; it is the only thing standing
between R3 and a no-op. Anything that weakens it — a scene that stops being
mirror-symmetric, a move to the live calibration (§6.6) — takes the whole patch's
verification with it.

### 6.6 The live calibration resolves the ambient axis to a HANDFUL of levels — T5 inherits this

The gates run on `reference_table()`, baked at `[physics.fire] rad_scale =
5.1427e-5`, where `E°[0] = 389475`. The **shipped** sweep bakes at
`[physics.radiation] rad_scale_derived = 2.125632e-08` (P2b's derived
calibration), where **`E°[0] = 161`** — 2419× smaller. Since
`amb_m = (amb · w_m) >> 16` and `w_m = ONE/16`, the per-tile ambient the live
path can express spans `amb_m ∈ {0 … 10}`: **eleven distinct levels** between
"0 K outside the hull" and "room temperature outside".

That is not a defect — it is `T⁴` being honest. A 293 K surface radiates 161
counts a tick where a 1263-game fire radiates 124282 (772×, against the exact
`(1556/293)⁴ = 794`). But it has a consequence T5 must re-measure rather than
assume. The same hull scene at the live calibration:

| calibration | `E°[0]` | port wall `rad_net` | starboard wall `rad_net` |
|---|---|---|---|
| reference (`rad_scale`) | 389475 | −115604 | −2750 |
| **live (`rad_scale_derived`)** | **161** | **−50** | **0** |

The first-order effect survives quantisation; the **second-order** one — the
cold side chilling the rest of the compartment — quantises to **exactly zero**.
So at T5's flip: a hull wall facing a 0 K sky does cool, and nothing behind it
feels the sky at all until it is warmer than ambient. Whether that is acceptable
is a T5/T3 calibration question, not a T1 one. **Consequence for the gates,
written into the test file**: the item 3 scene must stay on `reference_table()` —
moved to the live table, half its assertions go vacuous while the file still
reports green.

**And it is about to tighten, not loosen.** `fire-12` gained **R13** while T1 was
in flight (`5d87b99`, Erik: the currency pin is 0.9 MJ/(m³·K)), which replaces
`rad_scale_derived = 2.125632e-08` with **1.6533e-08**. Measured through the same
bake:

| calibration | `E°[0]` | `amb_m` at S16 | distinct `amb_m` levels |
|---|---|---|---|
| reference (`[physics.fire] rad_scale`) | 389475 | 24342 | 24343 |
| live today (`rad_scale_derived`, the 0.7 provisional) | 161 | 10 | **11** |
| **after R13** (`rad_scale_derived`, 0.9) | **125** | **7** | **8** |

So whoever applies R13's number should expect the per-tile ambient axis to be
**eight** levels wide, not eleven. T1 deliberately does **not** apply it:
`config.toml` still carries 2.125632e-08, and §5 T3/T5 own `rad_scale_derived`.
A measurement handed forward, not a change.

### 6.7 The sweep commutes with the mirror, exactly

Measured, not assumed, and it is the lever §5.4's whole argument rests on: on a
mirror-symmetric scene with a uniform ambient, all four output planes come back
**exactly** mirror-symmetric — all four (transport × ordinate count)
combinations. Nothing in the remainder split, the gather offsets, the
per-ordinate constants or the traversal order introduces a handedness. Two uses
beyond T1:

* it is a free, strong, *snapshot-less* property gate for the sweep, available to
  any future patch that needs a non-vacuous invariant (P4's CUDA twin should
  preserve it, ordinate table and all);
* it is the reason a hoist is *provably* visible here rather than merely likely
  to be caught — §5.4 step (3) is a consequence of this symmetry, not a hope
  about how the numbers happen to fall.

## 7. Commits

Branch `12-t1-ambient-plane-leak-live`, off `fire-12` at `b3488fc`.

| hash | what |
|---|---|
| `f4a7775` | report skeleton, written incrementally from there |
| `99e0167` | **D1** — the reference takes a per-cell ambient LEVEL; ring, ceiling return and body all read the cell's own |
| `0031662` | **D2** — the engine's `amb_m` goes per-cell, derived from `is_vacuum`; `k_leak` goes live at 0.10 |
| `4fe6379` | gate 0 and the harness take the per-cell ambient *(committed by the orchestrator after the first implementer died mid-patch; its work, verified before committing)* |
| `588bbaf` | **D4** — the three properties (design v2 §6 items 2, 3, 4), 26 tests, validated by breaking them |
| `fd12caa` | the calibration trap named in the item 3 scene (§6.6) |
| `107a9d1` | **D3** — §0 closed, §5 gate results, findings §6.4–§6.7, §7 |
| `65f30b3` | item 4's non-vacuity guard on its ambient axis (§5.3) |
| `3f08ae3` | §5.3's item 4 numbers, re-measured through the committed test |
| `bba3392` | §7 completed; §6.7 wording |
| `22efec7` | §5.5 cites the conductor's `nullptr, nullptr` — read, not assumed |
| `42b834f` | §6.6 re-measured against **R13**, which landed on `fire-12` mid-patch |
| `21401db` | §5.4's perturbation table re-run after the guard; §6.5 rewritten |
| *(this commit)* | §7 final; §5.6 re-confirmed on the finished tree |

Three code/test commits (`99e0167`, `0031662`, `588bbaf` + `65f30b3`), one
inherited (`4fe6379`), the rest report. The branch is **13 commits** ahead of
`b3488fc`; `fire-12` itself moved one commit ahead in the same window (`5d87b99`,
R13 — docs only, no conflict with anything here).

**Not done, deliberately** (D5): `t_amb_q`'s meaning, `cool_shift`, the
temperature solver, conduction, any `[materials.*]` row, the fold flip, any
golden, `DIGEST_SPEC_VERSION`, and R13's `rad_scale_derived` value. Nothing was
pushed, merged, or rebased; the worktree is intact.
