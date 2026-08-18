# P-M3 — `destroy_wall` seeding: design (2026-08-18, v1)

**Arc:** mass-books (`docs/mass_books_arc_kickoff_2026-08-18.md`).
**Status:** design, pre-critique. Feel-adjacent → **HUMAN-TEST before merge**.
**Decision owner:** Erik. The decision below was taken with him in session; this
document records it, its reasoning, and its gates — it does not re-open it.

---

## 1. The defect

`destroy_wall` ends by seeding the newly-opened tile with the neighbour mean of
its bulk gas ([`gamemap.py:1752-1754`](../src/simulation/gamemap.py#L1752-L1754)):

```python
self.atmosphere[fy, fx] = self._neighbor_mean(self.atmosphere, fy, fx)
self._seed_bulk_gas_neighbor_mean(fy, fx)      # gas[O2], gas[INERT_N2]
```

`_neighbor_mean` **reads** the open 4-neighbours and **writes** their mean into
the new cell. Nothing is withdrawn from the donors, so every destroyed tile
creates one neighbour-mean cell of air out of nothing.

Measured (`tools/repro_destroy_wall_mint.py`, no weapon, no explosion, not even
a solver step): 4 walls destroyed in the sealed two-room fixture mint exactly
+4.000 cell-equivalents, scaling linearly with local density — ×10 → +40.000,
×100 → +400.000.

**Why it blows up rather than drifts.** `find_burst_walls` fires on a pressure
*differential* above threshold, so a bursting wall is high-pressure **by
definition**, and the mint scales with exactly that pressure:

```
high pressure → wall bursts (the relief valve) → mint ∝ that pressure
              → higher pressure → more bursts → …
```

The emergent pressure-relief valve is a pressure amplifier. `_neighbor_mean`
also skips vacuum neighbours ([`gamemap.py:1571-1573`](../src/simulation/gamemap.py#L1571-L1573)),
so at a hull between a pressurised room and space the mean is taken over the
pressurised side only — peak seed at peak differential.

In one recorded session this path carried **87.7% of a 2.201× total mass
growth** (`docs/mass_books_arc_kickoff_2026-08-18.md` §1.2).

**The codebase already knew.** The A5 evacuation block twelve lines below
records the asymmetry as deliberate
([`gamemap.py:1764-1767`](../src/simulation/gamemap.py#L1764-L1767)): *"the
symmetric open half (`unseal_tiles`) withdraws its seed from the donors instead
of minting (destroy_wall's neighbor-mean seed stays the rule for DESTRUCTION
events only)."*

## 2. The decision, and why it is not "withdraw from the donors"

The conservative option — take the seed out of the donor cells, as
`unseal_tiles` does — closes the books exactly and produces no seam gradient.
It was considered and **rejected on physical grounds**, by Erik:

> *"An explosion does not eliminate matter, it just redistributes it."*

At a 1/3 m cell, destroying a tile is a drastic geometry change, and the wall's
own material does not vanish in reality — it becomes rubble that still occupies
volume. The sim carries no rubble, so withdrawing gas from the neighbours models
an expansion into void that does not physically happen. Withdrawal would be
right if the simulation were ground truth; it is not, and this is one of the
places where the missing model shows.

The cave case makes it concrete: a map 80% solid, blasted open to 40% solid,
would under withdrawal have its pressure fall by more than half — a cave that
suffocates you for digging it out. Real caves stay breathable because they are
connected to a reservoir. **Seeding ambient models the reservoir that is
actually there**, and on `boundary = ambient` maps it is not even a fiction.

**Decision: seed a CONSTANT at the map's ambient, and book it.**

The load-bearing property is *constant*, not the particular value. A constant
breaks the feedback loop with the burst valve dead — the seed no longer scales
with the pressure that triggered the burst. A neighbour-scaled seed of any size
keeps the loop closed.

Note that in the common case the constant **is** the neighbour mean: rooms sit
at ~1 atm, so the two agree. The constant only diverges where the room is
pressurised (the pathological case, where we want it capped) or depressurised
(a small bounded puff, accepted below).

## 3. Specification

### 3.1 The seed

| | value | note |
|---|---|---|
| `gas[O2]`, `gas[INERT_N2]` | the map's **ambient N**, split 0.21 / 0.79 | default sums to 1.0; read from the map's ambient, do not hard-code |
| `temperature` | **0** — explicitly written | `T_game` is a ΔT above ambient, so 0 *is* ambient |
| `atmosphere` | **not written** | `p*` is determined by the EOS from N and T; a third independent seed is how the current pair got out of sync |

Why "1 atm at ambient temperature" evaluates to `N = 1.0`: `T_game` is a ΔT and
`C = 1/eos_t_amb_k = 1/290`, so `p* = C·N·(T+290)` collapses to `p* = N` at
`T = 0`. The spec and "seed the ambient N constant" are the same number. The
ambient constant already exists (`[physics.eos]`, 0.21 + 0.79); this adds a
default, not a dial.

### 3.2 Breach tiles — skip the seed

A destroyed tile that joins the vacuum/ambient boundary (`on_edge_hull or
exposes`) is Dirichlet-pinned and has its N wiped by the bulk-transport clamp
next pass. Measured today: it is seeded +10 cell-eq and the next tick deletes
exactly −10 — a mint-then-delete round trip that nets zero through two broken
books.

**Do not seed a tile that becomes a breach cell.** It removes the round trip and
the seam is already handled by the Dirichlet pin.

### 3.3 Burning walls — clear `fire` explicitly

Walls burn. `destroy_wall` does not clear `fire`, so a burnt wall becomes an air
tile still carrying a fire value with no fuel beneath it (`on_tile_changed`
makes air non-flammable, `wall_hp → 0`). It is expected to decay on its own.

**Clear `fire` explicitly.** Erik's ruling: destroying a burning wall putting the
fire out is intended behaviour. Making it explicit means the behaviour is
deterministic rather than emergent from a decay rate that may be retuned.

### 3.4 Booking — `n_destruction_seed_sum`

A named, signed, `≥ 0` channel in the mass books, incremented by the seeded N per
tile. Trivially predictable — `ambient_N × tiles_seeded` — which makes it the
cheapest channel in the ledger to gate.

The ledger requires **attribution, not conservation**. A deliberate source is
legitimate; an invisible one is the defect. This is the same shape as the energy
books' deposit channels.

## 4. Why `T := 0` keeps the energy books closed

The energy books sum `n_bulk·T` over an accountable set that skips
`solid || ts || is_vacuum || is_ambient`
([`eos_solver.cpp:291-296`](../cpp/src/eos_solver.cpp#L291-L296)). A wall is
excluded and contributes 0. On destruction the cell **joins** the set:

- with `N = ambient_N, T = 0` it enters contributing `1.0 × 0 = 0` →
  **Δ(energy books) is exactly zero, no energy channel needed**;
- with `T` inherited from a hot neighbour or from the wall itself it enters
  contributing `N·T` with nothing naming it — an unbooked injection.

`destroy_wall` writes no temperature today, so a burning wall currently joins the
books hot. **This patch closes a pre-existing energy-seam hole as a side
effect**, and that is a claim the gate must actually check, not assume.

## 5. Scope

**In:** the four specification points in §3, and their gates.

**Out, deliberately:**

- **The grenade payload** (260.5 cell-equivalents per throw, 12.3% of the
  recorded mint). Erik's explicit ruling: not part of this scope. Untouched by
  this fix and still open.
- **The unbound velocity clamp.** |u| = 862 m/s against `c_local` ≈ 640 on 976
  cell-snaps, with `U_MAX = 1000` never binding. Separate defect, still open.
- **The full P-M1 mass ledger.** This patch ships the one channel it needs; the
  seven-channel ledger lands after, as the standing property gate.
- **Rubble porosity.** The principled way to model "the fragments still occupy
  volume" is reduced `dyn_permeability` on the destroyed tile, not gas from
  nowhere. Considered, deferred as a feature — the constant seed is the
  simplest honest expression of the same intent.
- **`unseal_tiles` / doors.** Already conservative, dormant, and Erik's
  reasoning distinguishes them ("I'm more willing to let the sliding door push
  on the atmosphere"). Left alone; the divergence in philosophy is deliberate
  and should be commented, not removed.

### ACCEPTED GAPs

- `ACCEPTED GAP:` **Solid thermal energy is not accounted anywhere.** Destroying
  a burning wall discards the heat stored in the wall material. That heat lives
  in the thermal-solid domain, which the energy books never tracked, so this is
  a pre-existing gap rather than one this patch opens. Not closing it.
- `ACCEPTED GAP:` **The seed mints on a depressurised map.** Blowing a wall in a
  vacuum-filled ship produces a small puff of air. Bounded at `ambient_N` per
  tile (~1.6% of map air across 107 destructions in the recorded session), not
  self-amplifying, and Erik has ruled it acceptable — *"i bet it looks quite
  cool"*. Named in the books, so it reads as a design channel, not a defect.
- `ACCEPTED GAP:` **Total minted mass can still be large on destruction-heavy
  maps.** The cave case mints ~2,400 cell-equivalents on a 6,000-cell map. Large
  but benign: spread at ambient, no concentration, no proportionality, so it
  cannot produce 800× cells or feed the burst valve. P-M1's gate therefore
  asserts `Δ(Σ N) == Σ named channels`, **not** `Δ(Σ N) == 0`.

## 6. Gates

1. **Conservation-with-attribution.** `tools/repro_destroy_wall_mint.py`
   promoted to an asserting test: after the change, `Δ(Σ N)` from a destruction
   equals `n_destruction_seed_sum` exactly, and is **independent of local
   density** — the ×1 / ×10 / ×100 sweep must give the *same* seeded total, which
   is the direct assertion that the feedback loop is broken.
2. **Breach path.** Destroying an edge-hull tile seeds nothing and the following
   tick shows no compensating wipe (today: +10 then −10; after: 0 then 0).
3. **Energy books.** `Δ(energy books)` across a destruction is exactly zero,
   including for a **burning** wall. This is §4's claim; it must be measured.
4. **Full suite green.** Meaningful only after P-M0b repairs the 37 fire tests
   that have been red on main since 547fb12 (2026-07-24) — until then "green"
   proves nothing about fire, and burning walls are in scope here.
5. **Determinism.** Digests unchanged where behaviour is unchanged; the seeded
   value is an exact Q16.16 constant, so no new quantisation path.
6. **CPU↔GPU.** Re-run `test_cuda_p64_kick_compression` PART 2 against the fix
   (P-M5). It sits on blast+venting, the same path. A Python-side mint that both
   backends inherit is *not* an explanation for a divergence, so if this does not
   move it, they are two bugs and we will have learned that cheaply.
7. **HUMAN-TEST.** Erik plays: grenades in a sealed room, then breach a
   pressurised one. Expected: no blowup, and peak room pressure in the low
   single-digit atmospheres (the design record says ~2.1 atm for a sealed room
   absorbing a grenade — a car tyre is ~3.3 atm absolute, and 100 atm was always
   pure bug).

## 7. Open questions for the critique round

Deliberately not resolved here — these are what the adversarial pass is for:

1. **Does the seam produce a velocity spike?** A 1 atm cell dropped into a
   genuinely over-pressured room is a real gradient, and we know the velocity
   clamp does not bind. At the intended 2–3 atm it should be negligible; that
   expectation is untested.
2. **Is `find_burst_walls` still correctly tuned once the amplifier is gone?**
   The valve was relieving against a source that was feeding it. With the source
   removed, `burst_threshold` may now be too eager or too lazy, and nobody has
   looked at it in that light.
3. **Ordering inside `destroy_wall`.** The breach decision (`breach_mask[fy,fx]
   = True`) currently happens *before* the seed. §3.2 depends on that order
   holding; if any caller path reaches the seed with the mask not yet set, the
   skip silently does not fire.
4. **Multi-tile destruction atomicity.** Several tiles can be destroyed in one
   tick (measured: up to 11). Each seeds independently against a state the
   previous seed already changed — but with a *constant* seed that coupling
   disappears. Worth confirming there is no remaining order dependence, since
   determinism is a hard requirement.
5. **Is `ambient_N` well-defined on every map?** `boundary = space, ambient =
   None` (the `playground` case) has no ambient reservoir. What does the
   constant read there — the config default, or something map-derived?
