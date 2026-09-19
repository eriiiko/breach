# T1 — per-tile ambient (`amb_m`) + uniform `k_leak` live in the sweep

> Issue #12, ray-engine-v2. Worktree `breach-t1`, branch
> `12-t1-ambient-plane-leak-live` off `fire-12`.
> Design: `docs/thermal_model_v2_design_2026-09-19.md` R2/R3, §3.3, §5 T1, §6 2-4.
> **Written incrementally** — every section below is filled as its work lands.

## 0. Status

| step | state |
|---|---|
| D1 reference (`sweep_ref_q.py`) | **done** |
| D1 reference gates G1-G12 (+2 new blocks) | **ALL GATES PASS** |
| D2 engine (`radiation_sweep.{h,cpp}` + binding + runner) | pending |
| D3 gate 0 bit-for-bit | pending |
| D3 goldens unmoved | pending |
| D4 item 2 (uniform == scalar era) | pending |
| D4 item 3 (non-uniform genuinely varies) | pending |
| D4 item 4 (conservation with per-cell ambient AND k_leak > 0) | pending |
| full suite | pending |

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

*(filled at D2)*

## 5. Gate results (D3, D4)

*(filled at the end)*

## 6. Findings, surprises, and what T5 inherits

1. **The ambient could not have been a temperature** — §1.1. This is why the
   scalar-naming error (L2-B1) was not merely a rename: a per-tile `t_amb_q` is
   a *temperature* plane, and no temperature plane can express R3's "0 K
   outside".
2. **The Fleck factor is ambient-dependent too.** `L_q` is the cell's free
   *excess* emission, so the pre-pass reads the ambient as well — the change is
   not confined to the sweep loop. Under a uniform ambient it is bit-identical.
3. **T5 inherits the Pass-1 clamp ceiling** — §2.3.

## 7. Commits

| hash | what |
|---|---|
| `f4a7775` | report skeleton |
