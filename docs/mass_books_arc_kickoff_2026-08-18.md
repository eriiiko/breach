# Mass-books arc — kickoff + instrument spec (2026-08-18)

**Opened by the pressure arc's HUMAN-TEST.** Erik: *"fires dont blow up anymore,
but grenades still can, especially after i broke a wall with a high pressure
room."*

**AUDIT FIRST** (Erik's standing ruling). This document specified the
*instrument* only and deliberately proposed no fix.

> **STATUS 2026-08-18 — the audit has reported.** §1.2 catalogues the events and
> §1.3 names the mint: `destroy_wall` seeds a newly-opened tile with the
> neighbour mean of its bulk gas and withdraws nothing, so **every destroyed
> wall creates one neighbour-mean cell of air out of nothing** — scaling
> linearly with how pressurized the room is. 87.7% of the session's 2.201×
> growth rides that path. Reproduced in isolation with no weapon, no explosion
> and no solver step. The remedy is **still** not chosen here: it is
> feel-adjacent and goes through P-M3's design + critique round (§6).

Seed + measurements: `docs/human_test_2026-08-18_mass_books.md`.
Evidence: `debug_blowup_20260818_040647.npz` — the first dump carrying
`wind_x`/`wind_y`/`inert_n2`.

**Numbers below were re-derived from the dump on 2026-08-18** (§1.1 records the
re-derivation and supersedes the seed doc's figures, which carried a unit
error). **Read the units rule before touching this dump:** the recorder
dequantizes a named list of planes and `gas_o2` is on it while `inert_n2` is
**not**, so

```
N_physical  =  gas_o2  +  inert_n2 / 65536          (ambient cell = 1.0)
```

`tools/analyze_blowup_dump.py:90-100` encodes this and is the canonical
converter — use it rather than summing planes by hand. Also: `atmosphere` in
this dump is the **solved pressure**, not N. Every figure below is in physical
cell-equivalents of ambient air.

---

## 1. The finding, in one paragraph

Total bulk N summed over the map grows **2.201×** across one play session
(5,591.9 → 12,306.8 cell-equivalents), on a map that starts with **6,257 gas
cells**: the engine minted **6,714.9 cells' worth of air — more than a second
atmosphere.** `playground` is `boundary = space, ambient = None`: there is **no
reservoir**, so no legitimate external source, and the only sink is venting
through the breach Erik blew. Locally one cell reaches **797.5× ambient** (snap
2239, y=65 x=95; the worst cell at the final snap sits at 709.6×) while the
*solved* pressure at that cell is 1.371 — the mass field and the pressure field
have decoupled. The mint is **not species-selective**: O₂ grows in lockstep,
1,172.5 → 2,522.2 (**2.151×**), so whatever adds mass adds *air*, not just N₂.

### 1.1 How the mint is delivered (measured, not inferred)

Re-deriving from the dump changes the arc's priors enough to record here:

| measurement | value |
|---|---|
| share of the mint arriving in discrete jumps (>1 cell-eq) | **100.3%**, in **62 events** |
| net contribution of all other snaps | **−0.3%** (the sink, see below) |
| one payload recurs *exactly* | **260.5 cell-equivalents**, ×3, at snaps 92 / 572 / 1292 |
| a second payload recurs | 72.4 cell-equivalents, ×2, at snaps 1975 / 2109 |
| footprint of the 260.5 event | 69 cells, 9×9 bbox, peak cell +10 cell-equivalents |
| minted **before any wall broke** (first destruction: snap 1509) | 868.2 cells = **12.9%** |
| snaps where total N falls | **2,337 of 2,399** — but never by more than 1 cell-eq |
| N resident inside solid cells, any snap | **exactly 0** |
| spatial spread at the final snap | top-1 cell = 10.6% of excess, top-50 = 48%, top-200 = 78%; 946 cells >2× ambient, 5 cells >100× |

Three consequences:

1. **This is not diffuse leakage.** 100.3% arrives in discrete events; every
   other snap nets −0.3%.
2. **The sink works; it is simply outpaced ~100:1.** Venting removes mass on
   2,337 of 2,399 snaps — but at under 1 cell-equivalent per snap (largest
   single fall: 0.02) against deposits of 72–260. Do not design against "the
   vacuum sink is dead"; design against a source three orders of magnitude
   larger than the drain.
3. **Stale mass in walls is ruled out.** Solid cells hold exactly zero N at
   every snap, so the `N := 0` clamp on solid
   ([`bulk_transport.cpp:212-229`](../cpp/src/bulk_transport.cpp#L212-L229)) is
   holding. A cell going solid→gas arrives empty — but it does not stay empty,
   which is §1.2.

### 1.2 The event catalogue — two populations, and only one is a weapon

Erik, on the session: *"i threw a grenade that didn't break any walls first, one
or a few, i think i also used the explosives."* Split the 62 events on whether
the obstacle grid changed in the same snap and that account matches exactly:

| | events | footprint | payloads | share of the mint |
|---|---|---|---|---|
| **no wall broken** | 4 | **69 cells**, tight 9×9 | **260.53 ×3, identical to 5 s.f.** | 12.3% |
| **wall broken same snap** | 58 | 150–500 cells, spread across most of the map | **52 distinct values, almost none repeating** | **87.7%** |

The three identical 260.53 events at snaps 92 / 572 / 1292 are the early
grenades: a fixed payload with a tight blast footprint, which is what a designed
deposit should look like. The other population is not that. **A weapon deposit
is a constant — it quantizes.** Fifty-two unique payloads in fifty-eight events
means the mass appearing at wall-break time is proportional to *local state*,
not to anything fired. It also tracks how many walls went at once (−4 walls →
251 cell-eq, −2 → 216, −1 → typically 40–130).

Reproduce with `tools/analyze_blowup_dump.py <dump> --mass-books`, which prints
the catalogue above and owns the units rule.

### 1.3 Root cause, confirmed by isolated repro: `destroy_wall` mints

The dump cannot settle it alone — in every one of those 58 events the explosive
both deposits *and* breaks the wall, so the two are confounded. Isolating the
destruction path settles it. `tools/repro_destroy_wall_mint.py` builds the
sealed two-room fixture, sums bulk N, calls `destroy_wall` on one tile, and sums
again — **no weapon, no explosion, not even a solver step**:

```
Sum N before               18,939,904 raw =    289.000 cell-eq
destroy_wall( 1,13)  dN =     +65,536 raw =     +1.000 cell-eq
destroy_wall( 2,13)  dN =     +65,536 raw =     +1.000 cell-eq
destroy_wall( 3,13)  dN =     +65,536 raw =     +1.000 cell-eq
destroy_wall( 4,13)  dN =     +65,536 raw =     +1.000 cell-eq
MINTED BY DESTRUCTION        +262,144 raw =     +4.000 cell-eq
```

**Every destroyed wall mints one neighbour-mean cell of air out of nothing**, and
the mint scales exactly linearly with how dense the neighbourhood is:

| room pressurized | mint for 4 walls |
|---|---|
| ×1 | +4.000 cell-eq |
| ×10 | +40.000 cell-eq |
| ×100 | +400.000 cell-eq |

The site is [`gamemap.py:1752-1754`](../src/simulation/gamemap.py#L1752-L1754),
at the end of `destroy_wall`:

```python
self.atmosphere[fy, fx] = self._neighbor_mean(self.atmosphere, fy, fx)
self._seed_bulk_gas_neighbor_mean(fy, fx)      # gas[O2], gas[INERT_N2]
```

`_neighbor_mean` **reads** the open 4-neighbours and **writes** their mean into
the newly-opened cell. Nothing is withdrawn from the donors. The intent is
documented and sound — "so we don't open with an artificial vacuum pulse" — but
the implementation is a source, not a transfer.

**The codebase already knows.** The A5 evacuation block twelve lines below spells
out the asymmetry ([`gamemap.py:1764-1767`](../src/simulation/gamemap.py#L1764-L1767)):

> *"The symmetric open half (`unseal_tiles`) withdraws its seed from the donors
> **instead of minting** (destroy_wall's neighbor-mean seed stays the rule for
> DESTRUCTION events only)."*

So a conservative seeding pattern was written, and `destroy_wall` was
deliberately left outside it. That parenthesis is the entire bug, and it
predates this arc.

This closes the audit. Everything it explains, it explains quantitatively:
Erik's *"especially after i broke a wall with a high pressure room"* is the
linear scaling above; the 52 non-repeating payloads are the neighbour mean
varying with local density; the payload growing with walls-broken is one mint
per tile; and the 12.9% minted before the first wall fell is the grenade
population, which is a separate question (§6, P-M3).

## 2. Why no fix is proposed here

Three plausible mechanisms were proposed and **falsified by measurement** before
the one in §1.3 was found — the record is kept because it is the argument for
audit-first:

| hypothesis | killed by |
|---|---|
| density-division amplifier (`u -= dt·K·∇P/N̂`, floored at 0.001) | the fastest cells are the **dense** ones — median N ≈ 10,954 in the top-1000 by \|u\|; low-N cells average \|u\| = 1.77 |
| semi-Lagrangian mass duplication | bulk mass does not use SL — it uses **donor-cell** flux |
| O₂ suffocation limiting the drive | O₂ only fell to 77.5% of initial; fires never starved |

> **Note on the seed doc.** `human_test_2026-08-18_mass_books.md` §4 is titled
> "Mechanism — semi-Lagrangian mass duplication at 27× CFL" and argues it as the
> finding. That section is **superseded** by the row above; a correction has been
> appended to that doc. Its *secondary* observation survives and is carried
> forward here: **|u| = 862 m/s exceeds `c_local` ≈ 640** on 976 cell-snaps even
> though the step-4 kick is supposed to clamp |u| to `c_local`, and `U_MAX =
> 1000` never bound either. An unbound velocity clamp is a live candidate for
> whatever the ledger ends up naming, and P-M1 should record `max|u|/c_local`
> alongside the mass channels so the two can be correlated in one pass.

And the donor-cell transport already carries a **per-cell outflow limiter**
bounding a cell's total outgoing flux to ≤ its own N, explicitly *"so the
non-negative clamp below never creates mass"* (`bulk_transport.cpp:146-165`).
So the obvious culprit is, on inspection, mass-exact by construction.

**The dominant mint is now attributed** (§1.3) and the audit that named it cost
two measurement passes, not a ledger. What is *not* settled: whether the 260
cell-equivalent grenade payload is the intended constant, and whether anything
else mints once destruction is made conservative. Those are §6/P-M3's, and the
ladder below still runs — the ledger's value was never only finding this one
bug, it is the standing property gate that keeps the books closed through every
retune still queued.

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

The set to mirror is `eth_books_sum()`'s, which skips
`solid || ts || is_vacuum || (ambient_mode && is_ambient)`
([`eos_solver.cpp:291-296`](../cpp/src/eos_solver.cpp#L291-L296)). Note what that
predicate is: **four dynamic flags**. Membership is not fixed for the session —
see §3.4.

### 3.2 Named channels (one per writer of N)

Every pass that can change bulk N gets a bracket and a signed counter:

| counter | pass | expected sign |
|---|---|---|
| `n_transport_delta` | donor-cell bulk flux (per substep, summed) | ≈ 0 (conservative) |
| `n_combustion_delta` | P4 combustion: O₂ consumed → soot + inert-N₂ | ≈ 0 (design says N conserved) |
| `n_deposit_sum` | explosion / grenade bulk-N deposits | ≥ 0, **legitimate** |
| `n_vacuum_wipe_sum` | the `N := 0` wipe on vacuum cells | ≤ 0 |
| `n_solid_wipe_sum` | the `N := 0` wipe on cells that **became solid** | ≤ 0 |
| `n_setentry_sum` | N in cells crossing **into** the accountable set (solid→gas, vacuum→gas) | signed — see §3.4 |
| `n_ambient_clamp_sum` | ambient-ring reset to `N_amb` (dormant on space maps) | signed |
| `n_floor_sum` | any clamp that raises N (e.g. `N_FLOOR_SOLVER` paths) | ≥ 0 — **prime suspect** |
| `n_trunc_sum` | fixed-point truncation residual | ≡ 0 for transport (see below), signed elsewhere |

Two of these are new relative to the energy ledger and were added after reading
the transport clamp:

- **`n_solid_wipe_sum` is not optional.** The vacuum wipe and the solid wipe are
  the *same clamp*, [`bulk_transport.cpp:212-229`](../cpp/src/bulk_transport.cpp#L212-L229)
  — `if (solid[i] || is_vacuum[i]) N = 0`, commented "mass in a cell that became
  solid must not linger". §3.2's original table named only the vacuum half. An
  unnamed *sink* breaks the tick identity exactly as loudly as an unnamed source,
  and this one fires on precisely this arc's scenario.
- **`n_trunc_sum` will read zero for the transport pass**, and that is a useful
  self-check rather than a wasted counter: the divergence apply is the
  conservative ± form where `dq_e[i]` is the same value removed from `i` and
  added to `i+1` ([`bulk_transport.cpp:196-212`](../cpp/src/bulk_transport.cpp#L196-L212)),
  so truncation cancels pairwise to the LSB. A nonzero reading there means the
  apply is no longer paired, which is itself a finding.

### 3.4 The set-membership seam (a hole inherited from the energy ledger)

Every energy bracket takes its pre/post sums **inside one tick, against one
snapshot of the flag arrays** — `eth_pre_transport` at
[`eos_solver.cpp:505`](../cpp/src/eos_solver.cpp#L505) and its close at
[`:567`](../cpp/src/eos_solver.cpp#L567) both see the same `solid`. Nothing
checks the identity *across* the tick seam. So a cell whose flags flip between
the end of tick *k* and the start of tick *k+1* silently changes what is in the
sum, and no channel names it.

This arc cannot inherit that blind spot: its seed event is Erik breaking a wall,
and the dump records **107 walls destroyed across 58 separate events**, the
first at snap 1509. The measurement in §1.1 partially exonerates the seam —
solid cells hold exactly 0, so nothing stale is carried in — but the *entry
event* is not benign:

```
@snap 1516:  2 cells went solid -> gas.
             N in them the snap before: 0
             N in them the snap after:  107 cell-equivalents
```

Two cells that were walls hold 107 cell-equivalents one snap later. **§1.3
identifies what puts it there** — `destroy_wall`'s neighbour-mean seed — which
is the seam behaving exactly as this section warned it could. Instrumenting the
seam is therefore no longer hypothetical: it is where the known mint enters, and
a ledger blind to it would close its books every tick while the map gained air.
P-M1 must either

- sample `n_books_sum()` on both sides of the destruction pass and book the
  difference to `n_setentry_sum`, or
- state explicitly that membership is frozen within the assert window, and gate
  the seam separately.

Silently choosing the first and not writing it down is how this hole got into
the energy books.

### 3.3 The gate

```
assert  Δ(n_books_sum())  ==  Σ(all named channels)      every tick, both backends
```

A **property** gate, not a golden: it survives every legitimate retune and dial
change still queued, and goes red only on a real defect. This is the direct
analogue of `test_no_transport_mint`, which is what actually caught the energy
mint and stayed meaningful through an entire arc of changing behaviour.

Second gate, cheap and strong: on a **sealed** level with no deposits,
`Σ N` must be *bit-identical* tick over tick. §3.2's `n_trunc_sum` reasoning says
this should hold exactly, so it is a real gate and not an aspiration.

Third gate, straight out of §1.1: **on a vented, deposit-free fixture, `Σ N`
must fall monotonically, and `n_vacuum_wipe_sum` must account for every unit of
the fall.** The dump shows the sink alive but weak (<1 cell-eq/snap against
72–260 per deposit), which is exactly the regime where a real leak hides inside
a plausible-looking drain. Asserting that the *named* channel equals the
observed fall — not merely that a fall occurs — is what separates the two.

## 4. The missing fixture

Every bench we own is small and sealed — the same blindness that hid the
pressure bug for weeks. This arc needs a committed **blast + venting** fixture:
two rooms, one breached to vacuum, one grenade. That is the scenario Erik broke,
and it is *also* the scenario the one known lockstep divergence sits on (§5).

Build it from `tools/bench_two_room.py` + the parameterised generator sketched
during the pressure hunt (grid size and opening size as the swept axes).

**It needs a pre-breach phase.** §1.1 measures 13% of the mint arriving before
the first wall fell, so a fixture that breaches immediately would scope the arc
to the wrong window and let the pre-breach source hide inside the blast. Run
grenades in the sealed configuration first, *then* breach, and keep the two
phases separately accounted. A third configuration — sealed, no deposits at all —
is the control that makes §3.3's gates meaningful.

## 5. Known pre-existing defect on this arc's own target scenario

`test_cuda_p64_kick_compression` **PART 2 (blast + venting trajectory) diverges
CPU↔GPU** — verified at both `mg_cycles` 2 and 8, so unrelated to the pressure
fix. `docs/archive/e1_p_e2a_asbuilt_2026-08-17.md` records P-E2a finding it and
handing it to P-E4; **P-E4's as-built claims it repaired, and it has not.**

Treat this as an arc gate, not a leftover: a CPU↔GPU divergence and a mass mint
on the same scenario may well be one bug seen from two sides.

## 6. Suggested ladder (to be critiqued, not executed blind)

**P-M0 is DONE** — §1.2's catalogue and §1.3's isolated repro, both committed as
tools (`analyze_blowup_dump.py --mass-books`, `repro_destroy_wall_mint.py`). It
cost two measurement passes and no engine change. The rest of the ladder stands,
reordered around what it found:

- **P-M1** — the mass ledger (§3), CPU, instrument-only, no behaviour change;
  prove inertness by byte-identical digests. Unchanged in scope: §1.3 names the
  *dominant* mint, not necessarily the only one, and the ledger is the standing
  gate that keeps the books closed through the retunes still queued.
- **P-M2** — CUDA twin + the p64 divergence repair (§5). Worth re-testing early
  now: the divergence sits on blast+venting, which is exactly the destruction
  path §1.3 implicates, and a Python-side mint that both backends inherit is
  *not* an explanation for a CPU↔GPU **divergence** — so if repairing
  destruction does not move it, they are genuinely two bugs.
- **P-M3** — design doc + adversarial critique for the fix. Three questions,
  none to be pre-judged:
  1. **How should `destroy_wall` seed without minting?** Erik's framing: *"we
     should either deposit 0 or perhaps 0.5 atm, or 1 atm… perhaps 1 atm would
     work pretty well, but i'm not sure if changing that value to something less
     would make any cool effects."* Three candidate shapes, and the distinction
     that matters is **fixed constant vs. conservative transfer**:

     | option | conserves? | pressure discontinuity at the seam | note |
     |---|---|---|---|
     | seed 0 | no — *deletes* | maximal (100 atm room opens onto a 0 atm cell) | the artificial vacuum pulse the seed was written to avoid |
     | seed a constant (0.5 / 1 atm) | no — mints or deletes ~1 cell-eq per wall | large in a pressurized room, none in an ambient one | removes the *scaling* (the 87.7%) but not the mint |
     | neighbour mean, **withdrawn from the donors** | **yes, exactly** | none | `unseal_tiles`' existing pattern; physically "gas expands into the new volume" |

     The third is the only one that closes the books, and it is also the only
     one with no discontinuity — but it is not free: withdrawing from donors at
     a *breach* pulls mass out of cells that are themselves venting, and that
     interaction needs its own look. Feel-adjacent, so it gets the full round.
     On "cool effects": a real breach into a pressurized room already produces a
     violent rush under conservative seeding, because the gradient across the
     opening is real. Buying that effect by seeding low instead means paying for
     it with a books defect in the opposite direction.
  2. **Is the 260-cell-equivalent grenade payload the intended constant?** That
     population is 12.3% of the mint and is untouched by any destruction fix.
  3. **Is the vent correctly weak?** <1 cell-eq/snap through an open breach to
     vacuum may be right or may be a second defect; it was never the mint, but
     it is why the mint accumulated instead of draining.
- **P-M4** — fix + HUMAN-TEST. Erik plays the same scenario: grenades in a
  sealed room, then breach a pressurized one.

**Gate to land with the fix, not before:** a test asserting `destroy_wall`
conserves Σ N. `repro_destroy_wall_mint.py` is that assertion already, minus the
`assert` — it deliberately exits 0 so no red test sits on main.

## 7. Standing constraints

- Determinism is a hard requirement: Q16.16 integer only in the sim path.
- Feel-adjacent changes never auto-merge; Erik plays before merge.
- The post-pressure **retune pass is blocked on this arc** — retuning against a
  substrate that mints mass bakes the mint into the dials.
