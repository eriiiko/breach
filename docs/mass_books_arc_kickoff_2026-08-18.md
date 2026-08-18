# Mass-books arc — kickoff + instrument spec (2026-08-18)

**Opened by the pressure arc's HUMAN-TEST.** Erik: *"fires dont blow up anymore,
but grenades still can, especially after i broke a wall with a high pressure
room."*

**AUDIT FIRST** (Erik's standing ruling). This document specifies the
*instrument* only. It deliberately does **not** propose a fix, and no dial may
be chosen before the instrument reports.

Seed + measurements: `docs/human_test_2026-08-18_mass_books.md`.
Evidence: `debug_blowup_20260818_040647.npz` — the first dump carrying
`wind_x`/`wind_y`/`inert_n2`.

---

## 1. The finding, in one paragraph

Total bulk N summed over the map grows **2.15×** across one play session
(2.90e8 → 6.26e8), monotonically. `playground` is `boundary = space,
ambient = None`: there is **no reservoir**, so no legitimate external source,
and the only sink — venting through the breach Erik blew — can only *remove*
mass. Locally one cell reaches **~710× ambient, doubling every tick for twelve
consecutive ticks**, while the *solved* pressure at that cell sits at 1.371: the
mass field and the pressure field have decoupled. Grenade bulk-N deposits are
real and by design, but are worth a few cells, not thousands.

## 2. Why no fix is proposed here

Three plausible mechanisms have already been proposed and **falsified by
measurement** in the space of two days:

| hypothesis | killed by |
|---|---|
| density-division amplifier (`u -= dt·K·∇P/N̂`, floored at 0.001) | the fastest cells are the **dense** ones — median N ≈ 10,954 in the top-1000 by \|u\|; low-N cells average \|u\| = 1.77 |
| semi-Lagrangian mass duplication | bulk mass does not use SL — it uses **donor-cell** flux |
| O₂ suffocation limiting the drive | O₂ only fell to 77.5% of initial; fires never starved |

And the donor-cell transport already carries a **per-cell outflow limiter**
bounding a cell's total outgoing flux to ≤ its own N, explicitly *"so the
non-negative clamp below never creates mass"* (`bulk_transport.cpp:146-165`).
So the obvious culprit is, on inspection, mass-exact by construction.

**The mint is unattributed.** That is precisely the condition under which the
energy-books arc's discipline paid off: instrument, measure, *then* design.

## 3. The instrument — a per-pass MASS LEDGER

Mirror the energy ledger exactly; it is proven and its idioms are already in
the file. Energy uses `eth_books_sum()` with per-pass brackets and named
channels (`eth_transport_delta`, `eth_compression_delta`, `e_floor_sum`,
`e_wipe_sum`, `e_ts_residual`, `e_drag_drop_sum`, …), and asserts the **counter
identity every tick**: `Δ(Σ C·T) == Σ of the named channels`.

### 3.1 The accountable sum

```
n_books_sum()  :=  Σ over the accountable set of  n_bulk[i]      (int64, raw Q16.16)
```

Accountable set = the same skip-set discipline the energy books use: exclude
`solid`; treat `is_vacuum` and the ambient ring as **named channels**, never as
silent sinks. Exactness matters more than elegance — this must be an exact
integer sum, not a float reduction, or the ledger cannot close to the LSB.

### 3.2 Named channels (one per writer of N)

Every pass that can change bulk N gets a bracket and a signed counter:

| counter | pass | expected sign |
|---|---|---|
| `n_transport_delta` | donor-cell bulk flux (per substep, summed) | ≈ 0 (conservative) |
| `n_combustion_delta` | P4 combustion: O₂ consumed → soot + inert-N₂ | ≈ 0 (design says N conserved) |
| `n_deposit_sum` | explosion / grenade bulk-N deposits | ≥ 0, **legitimate** |
| `n_vacuum_wipe_sum` | the `N := 0` wipe on vacuum cells | ≤ 0 |
| `n_ambient_clamp_sum` | ambient-ring reset to `N_amb` (dormant on space maps) | signed |
| `n_floor_sum` | any clamp that raises N (e.g. `N_FLOOR_SOLVER` paths) | ≥ 0 — **prime suspect** |
| `n_trunc_sum` | fixed-point truncation residual | small, signed |

### 3.3 The gate

```
assert  Δ(n_books_sum())  ==  Σ(all named channels)      every tick, both backends
```

A **property** gate, not a golden: it survives every legitimate retune and dial
change still queued, and goes red only on a real defect. This is the direct
analogue of `test_no_transport_mint`, which is what actually caught the energy
mint and stayed meaningful through an entire arc of changing behaviour.

Second gate, cheap and strong: on a **sealed** level with no deposits,
`Σ N` must be *bit-identical* tick over tick.

## 4. The missing fixture

Every bench we own is small and sealed — the same blindness that hid the
pressure bug for weeks. This arc needs a committed **blast + venting** fixture:
two rooms, one breached to vacuum, one grenade. That is the scenario Erik broke,
and it is *also* the scenario the one known lockstep divergence sits on (§5).

Build it from `tools/bench_two_room.py` + the parameterised generator sketched
during the pressure hunt (grid size and opening size as the swept axes).

## 5. Known pre-existing defect on this arc's own target scenario

`test_cuda_p64_kick_compression` **PART 2 (blast + venting trajectory) diverges
CPU↔GPU** — verified at both `mg_cycles` 2 and 8, so unrelated to the pressure
fix. `docs/archive/e1_p_e2a_asbuilt_2026-08-17.md` records P-E2a finding it and
handing it to P-E4; **P-E4's as-built claims it repaired, and it has not.**

Treat this as an arc gate, not a leftover: a CPU↔GPU divergence and a mass mint
on the same scenario may well be one bug seen from two sides.

## 6. Suggested ladder (to be critiqued, not executed blind)

- **P-M0** — the blast+venting fixture + a headless repro of the 2.15× growth.
- **P-M1** — the mass ledger (§3), CPU, instrument-only, no behaviour change;
  prove inertness by byte-identical digests.
- **P-M2** — CUDA twin + the p64 divergence repair (§5).
- **P-M3** — *measure*, then design. Design doc + adversarial critique round
  happens **here**, once the ledger names the minting pass — not before.
- **P-M4** — fix + HUMAN-TEST.

## 7. Standing constraints

- Determinism is a hard requirement: Q16.16 integer only in the sim path.
- Feel-adjacent changes never auto-merge; Erik plays before merge.
- The post-pressure **retune pass is blocked on this arc** — retuning against a
  substrate that mints mass bakes the mint into the dials.
