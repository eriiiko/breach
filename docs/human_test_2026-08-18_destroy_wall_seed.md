# HUMAN-TEST 2026-08-18 — P-M3 destroy_wall seed: **PASS**
## …and the residue is the next arc, measured and isolated

Erik played `playground` on the P-M4a build (`worktree-agent-ae2f46162d52152a1`,
commit `ae7f0cb`), the same map and the same scenario that opened the mass-books
arc twelve hours earlier.

**Erik's verdict:** *"now it's much more stable, but not 100% actually. well, it
is stable but i still get individual pressure spikes. it is acceptable tho"* —
and, on the visuals, *"if i enable pressure visualization, i see some stuff…
some individual tiles that flashes yellow or white."*

Both correct, and the dump separates them cleanly: **the fix did what it claimed,
and the residue is a different defect.**

Evidence: `debug_manual_20260818_194038_velocity_clamp_seed.npz` (F8 manual dump,
775 snapshots ≈ 32 s, kept at the repo root as the seed for the velocity arc).

---

## 1. The destruction mint is fixed — measured in a real session

Wall-break events, from `analyze_blowup_dump.py --mass-books`:

| snap | deposited (cell-eq) | walls broken | per wall |
|---|---|---|---|
| 518 | 2.99 | 3 | **1.00** |
| 523 | 3.99 | 4 | **1.00** |
| 540 | 11.99 | 12 | **1.00** |
| 554 | 8.99 | 9 | **1.00** |
| 617 | 3.99 | 4 | **1.00** |
| 740 | 1.99 | 2 | **1.00** |

**Exactly one cell of ambient air per destroyed wall, every time, independent of
the room's pressure.** That is the design (§3.1) confirmed outside the fixtures.

| | before (2026-08-18 04:07) | after |
|---|---|---|
| per destroyed wall | 40–130 cell-eq, **52 distinct payloads in 58 events** | **1.00, every time** |
| share of deposits riding wall breaks | **87.7%** | **19.0%** |
| scaling with local pressure | linear | **none** |

The feedback loop with `find_burst_walls` is broken: the valve fires on a
pressure differential, and the seed no longer scales with it.

## 2. What is left is the grenade payload — scoped out, unchanged

Of 965.2 cell-equivalents deposited in the session, **81% is grenades**:
snap 263 = **260.53**, snap 503 = **521.06** (two in one snap). The same constant
this arc measured before touching anything, behaving identically.

Total N over the session: 5,593.0 → 6,554.3 = **1.172×** over 32 s, essentially
all of it grenades. Per Erik's ruling the grenade payload is out of this arc's
scope; it remains open.

Venting works: N falls on **762 of 774** snaps, netting −3.9 across the quiet
ones.

## 3. The flashing tiles are NOT mass creation — they are the velocity clamp

This is the finding that turns Erik's "not 100%" into a well-posed next arc.

```
worst cell                    433.5 x ambient
peak-pressure cell (23,95)    15.424 atm at 334.6 x ambient, T = 3.4
P_min                         -1.324 atm      <-- NEGATIVE, unphysical
peak |u|                      773.0 m/s       (local sound speed ~565 here)
T_max                         737.7           (ceiling 16000 — not thermal)
```

The discriminator is in the event table. **Snap 616 deposits 1.99
cell-equivalents in total — two walls, one each — yet a single cell in that event
gains 278.34.** Snap 617 deposits 3.99 and one cell gains 170.83.

Mass is not being *created* at those tiles. It is being **piled into them by
transport** far faster than it should, then draining back out — one or two ticks
of a 300×-ambient cell, which is exactly a tile flashing white.

Supersonic flow, a negative pressure, and transient 300× cells are three symptoms
of one cause: **`|u|` exceeding the local sound speed means advection is running
outside what the substep count can resolve.** This is the unbound velocity clamp
named in `mass_books_arc_kickoff_2026-08-18.md` §2 and deliberately scoped out of
P-M3 — |u| = 862 vs `c_local` ≈ 640 in the *previous* dump, 773 vs ~565 here,
with `U_MAX = 1000` never binding in either.

## 4. Verdict

**PASS.** The arc's defect is fixed and confirmed by a human test on the
scenario that produced it. The mass books now attribute destruction exactly;
what remains is one deliberately-deferred payload and one separately-named
defect.

Erik: *"it feels like the engine is finally starting to behave now!"*

## 5. What this hands the next arc

A velocity-clamp arc starts in a better position than this one did:

- a **recorded session** with the defect isolated and no mass mint confounding it
  (`debug_manual_20260818_194038_velocity_clamp_seed.npz`)
- a **discriminator that already works**: deposited-total vs peak-cell-delta
  separates "created" from "piled", and `--mass-books` prints both
- three **corroborating symptoms** to gate against — supersonic `|u|`, negative
  `P_min`, and transient ≫100× cells
- the knowledge that the step-4 kick's clamp to `c_local` is **not binding on
  this path**, which is where to start looking

Open, in priority order: the velocity clamp (this), the grenade payload (260
cell-eq/throw), and the post-pressure retune pass — which now also carries
`burst_threshold` (the relief valve is near-unreachable with the amplifier gone)
and fire being mistuned at both ends (`docs/TODO.md` item 3).
