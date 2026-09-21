# Ray engine v2 / thermal model — handoff

> **This file IS the prompt.** Point a session at it and it has everything.
>
> **UPDATED 2026-09-21 — q7 IS RULED. The arc is no longer blocked.**
> Supersedes the 2026-09-20 edition, which said "blocked on Erik (q7)".
> That is history; do not act on it.

---

## 0. Orient in ninety seconds

**q7 was ruled by Erik on 2026-09-21, and the arc turned.** The answer was not
the one the T5b report expected.

**THE DESIGN OF RECORD IS NOW `docs/thin_material_rows_design_2026-09-20.md`.**
Read it before anything else. It carries thirteen rulings *with their reasons*
— the thing that did not survive the previous handoff, which is why q7 had to be
re-derived from a summary that had lost the reasoning.

`docs/thermal_model_v2_design_2026-09-19.md` remains the underlying design; the
thin-rows doc **amends its R14** and answers **q7**, **q2** and **q5**.

Python on this box is **`C:/Users/steen/anaconda3/python.exe`** — there is no
conda `data` env here, whatever the project CLAUDE.md says. Tests:
`pytest tests -q`. Builds: `cpp\build_cpu_home.bat`, `cpp\build_cuda.bat`. Both,
always.

---

## 1. What q7 turned out to be

The flip exposed that combustion's fuel-bed deposit is **65 536× (exactly 2^16)
the physical heat of combustion**, balanced until now by an emission scale
**2 419× too strong**. Two wrongs had been making a playable right.

Three options were put to Erik: **A** keep the shipped dial (undoes the arc),
**B** take the derived 38.73 / 116.2 (physically exact — but a 17.7 kW fire
cannot heat a 153.9 kg tile, so fires go out), **C** B plus the two-node
skin/core solid (#68).

**Erik ruled B — and then rejected C.** The way out is neither another dial nor
another node:

> **Stop authoring flammable objects as 154 kg blocks.** A 154 kg block of wood
> genuinely does not ignite from a nearby fire — that is not a model defect, it
> is correct. Author the things that are *meant* to burn at the mass they really
> have, and the one-node model works.

The decisive argument: #68's skin layer (`C_bulk/32` = 4.8 kg) and a 1 cm wooden
panel (4.6 kg) **are the same object**. Two-node only adds the *core*, and Breach
has no use for it — its flammables are crates, kindling, props and foliage; its
structure is steel. **R6 is honoured, not overturned, and #68 stays unspent.**

A 0.5 cm panel ignites in **66 s** against a 30–70 s literature band, with
nothing tuned.

---

## 2. Erik's rulings — all of them, do not re-litigate

1. Derived `H_bed = 38.73`, `H_fuel = 116.2`. Never fit a dial over a model gap.
2. **No massive object ever burns.** Permanent, accepted as game design.
3. Author **dimensions** (`thickness_m`), never a `fill_fraction` — fill bakes a
   reference tile size into the row's physical meaning.
4. Derive at the **BASE** tile size (`tile_size_m_base` / `res_factor` doctrine).
5. `wood` (id 2) **repurposed** to a 0.5 cm panel, **name kept**.
6. `10_kg_bonfire` **dropped** — the id budget is full, and `kindling` already IS
   the declared 1–3 kg campfire row.
7. `furniture` 5 kg; `kindling` and `foliage` 2 kg.
8. Global level change (every wooden wall becomes burn-through-able): accepted.
9. **q5 → Option 1**: keep `wall_hp` as the fuel bar, **delete the timer**
   (`[fire] wall_damage` → 0). Option 2 (a separate `fuel_remaining` plane) is
   the recorded upgrade path, not a rejected alternative.
10. **The destroy decision moves to the chemistry channel**, same patch.
11. **The 3-minute fuel-out ruling is RETIRED** — replaced by derived physics.
    It had been set against a store that physically held 471 minutes.
12. **The measurement fixture changes** — measure on a THIN row, not the 154 kg
    crate that this design deletes.
13. `--res` is a **dev tool**, not a shipped setting. Cross-*project* correctness
    is what authoring by dimensions buys; `--res` invariance is a convenience.

---

## 3. State of the branches

- **`fire-12`** — green at `a704954`. The integration line. Nothing is authored
  here.
- **`12-t5b-the-flip`** — pushed, **unmerged**, **7 red on purpose** (the
  runaway). M3 is what makes them green. The design doc lives here.
- **M1 is MERGED** into the flip (`1feea98`); its branch and worktree are gone.

**Nothing merges to `fire-12` until the whole M-stack is green.**

---

## 4. The plan

| | patch | tier | state |
|---|---|---|---|
| **M1** | negative `thermal_mass` exponents (the sign lift) | Opus | **MERGED** `1feea98` |
| **M2** | dimensions + the four rows, derived at base resolution; re-anchor the bench on a thin row | Opus | next |
| **M3** | derived `H_bed`/`H_fuel`; delete the timer; move the destroy decision; **measure** the real burn durations | Opus | |
| **M4** | `tools/derive_material_row.py` + an `adding-a-material` skill + the CLAUDE.md rules | Sonnet 5 | |
| | **HUMAN TEST — Erik plays it** | | |
| **T6/T7/T8** | old-law deletion · CLAUDE.md walkthrough · ill-posed-test sweep | | |
| **P4–P7** | CUDA twin · smoke & gas (closes q6) · light (**P6b = 2nd human test**) · stealth + RL light | | |

Still open, unscheduled, **none blocking**: q1, q3 (M2 deliberately does NOT
half-solve it — see the recorded reversal in the design doc §4), q4.

---

## 5. The lesson this arc keeps teaching

**Five times** a check proved blind to the very thing it was named after: T1's
uniform-ambient test, T1's structural conservation identity, T2 (the #54 ledger
stayed balanced while 99.23 % of conducted energy vanished), T5a (T2's own
instrument would have gone tautological the moment T2's fix landed), and M1 (three
live consumers of a negative exponent, all invisible to the whole suite because
the shipped table reaches no negative exponent — the first one caught BEFORE it
shipped).

Every one was caught by **deliberately breaking the code**, never by running the
gate. **A green suite is not evidence.** Every patch validates its gates by
perturbation and names which gate caught what.

Two traps live on this stack specifically:

- **M1 did NOT re-baseline goldens, and that was right** — `heat_inv_shift` is
  not a digest field, and since M1 moves no authored value the goldens STAYING
  PUT is the neutrality proof. The general rule stands for M2, which *will*
  move values: a re-baselined golden records whatever the code does, so name
  the independent oracle.
- **A design that enumerates sites must say whether the list is exhaustive.**
  M1's brief named four consumers of `heat_inv_shift`; there are SEVEN, and
  breaking the three missed ones left the whole suite green.
- **M3 deletes the channel that currently ends fires.** "Fires still go out" and
  "burnt-out tiles are still destroyed" must be pinned by name *before* the timer
  is removed, or its removal is unfalsifiable. Note the asymmetry that makes this
  live: chemistry floors at `FUEL_FLOOR`, so nothing reaches the `<= 0` that the
  destroy decision tests.

---

## 6. How Erik works

Language first, code second · ask **one** question at a time, as plain text,
never a popup (he answers every question in a message and will silently drop one)
· precision over speed, explain *why*, explain jargon · **nothing is tuned until
the physics works** · tests assert properties, never snapshots · **look at the
picture before trusting a scalar summary** · he is happy with autonomous work, so
re-enter the loop only for rulings that are genuinely his.

His standard for this model, in his own words: *"the logic based off of first
principles — inspired by physics, and designed by us to find a good middle ground
between being able to compute everything in real time and have at least an OK
believable physics simulation."*
