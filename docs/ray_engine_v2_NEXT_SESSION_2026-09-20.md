# Ray engine v2 / thermal model — handoff (2026-09-20)

> **This file IS the prompt.** Point a session at it and it has everything: what
> is done, what is settled, what is blocked, and the one ruling that unblocks it.
>
> Supersedes `ray_engine_v2_NEXT_SESSION_2026-09-15.md`, which is now history.

---

## 0. Orient in ninety seconds

**Branch `fire-12`** is green — `2557 passed / 0 failed` with the CUDA gates, both
binaries current. **Branch `12-t5b-the-flip`** is pushed, **unmerged**, and has
**7 red tests on purpose**. Those seven are the finding, not a regression.

**The design of record is `docs/thermal_model_v2_design_2026-09-19.md`.** Read §2
(R1–R14, Erik's rulings, settled — do not re-litigate), the ACCEPTED GAPs
(decisions, not findings), §4 (the calibration ledger), §5 (the patch plan) and
§6 (the properties to pin).

Python on this box is **`C:/Users/steen/anaconda3/python.exe`** — there is no
conda `data` env here, whatever the project CLAUDE.md says. Builds:
`cpp\build_cpu_home.bat`, `cpp\build_cuda.bat`. Both, always.

---

## 1. THE ONE THING THAT MATTERS — q7, and it is Erik's

The flip exposed that **combustion's fuel-bed deposit is 65 536× the physical
heat of combustion — exactly 2¹⁶**, a Q16.16 transcription error between T3's
derivation and the engine's expression (`H_bed` wants **38.73**, carries
2.538e6; `H_fuel` wants **116.2**, carries 7.613e6). Verified independently: both
ratios are 2¹⁶ to four figures.

**The old raycaster's emission was 2 419× too strong and had been balancing it.**
The flip removes one half of that compensating pair, so the other half is now
visible. Two wrongs had been making a playable right.

Correcting the dial alone does **not** fix it, and that is the crux:

| option | what it means | consequence |
|---|---|---|
| **A** keep the shipped `H_bed` | a tile is a thin skin heated as if it were the bulk | today's runaway; the emission scale must go back to **fitted**, undoing the arc |
| **B** take the derived 38.73 / 116.2 | a tile is a 154 kg block; a 17.7 kW fire heats it 0.07 K/s | physically exact; **fires go out** |
| **C** B **plus issue #68** (two-node skin/core) | a fire heats a thin surface layer to flame temperature, the bulk lags | the real answer; **R6 deferred it** |

> **The arc's conclusion, stated plainly: you cannot have honest physics AND fire
> with a one-node tile.** The prep doc predicted this exactly — *"P3's HUMAN-TEST
> will not pass on the current material rows"* — and here it is, as a number.

**Nothing below T5b proceeds until Erik rules q7.** Report §12 has eight
questions; q7 is the blocking one. q1 (opaque rows emitting 0.85 while
extinguishing 1.0), q2 (are flammable rows objects or blocks — every one is
currently the same 154 kg), q3 (per-level conduction table), q4 (sub-dead-band
faces as a one-way sink), q5 (`wall_hp` still doing three jobs) are all real but
none blocks.

---

## 2. Done and merged (7 of 16)

| | patch | what it landed |
|---|---|---|
| ✓ | **T1** | `amb_m` per-cell from `is_vacuum`; `k_leak` live. Reference first. Its item-3 test **proves** hoist-impossibility rather than asserting it |
| ✓ | **T2** | the currency audit: `c_v` must be **0.0076849**, at the conversion seam, not in `gas_energy` |
| ✓ | **T3** | the real material table, every number cited. **D1 was a no-op** — the table already encoded R13's pin |
| ✓ | **T3b** | materials authored by **density**; `V_tile` cancels identically, so bit-identity across resolutions is structural |
| ✓ | **T4** | 103 dead tests retired **before** the flip could turn them red; refused one deletion that would have silently broken two arcs |
| ✓ | **T5a** | two provably-neutral widenings; caught that T2's expression **overflows int64 at 2 atm** with both backends agreeing on the UB |
| ⏸ | **T5b** | **THE FLIP** — all nine steps landed, 7 red pending q7 |

Remaining after q7: **T6** old-law code deletion · **T7** CLAUDE.md rules
walkthrough (Erik's ask) · **T8** ill-posed-test sweep · then ray-engine **P4**
CUDA twin · **P5a/b** smoke and gas · **P6a/b/c** light (**P6b is the second
human test**) · **P7** stealth + RL light.

---

## 3. The lesson this arc keeps teaching

**Four times** a check proved blind to the very thing it was named after:

1. T1's item 2 — a uniform-ambient test cannot see an in-plane mis-index
2. T1's item 4 — conservation is *structural*, so it holds for a sweep doing the
   wrong thing
3. T2 — the #54 closure identity stayed perfectly balanced while 99.23 % of
   conducted energy vanished
4. T5a — T2's own instrument would have become a tautology the moment T2's fix
   landed

Every one was caught by **deliberately breaking the code**, never by running the
gate. So: a green suite is not evidence. Every patch on this arc validates its
gates by perturbation and names which gate caught what. Keep doing that.

---

## 4. How Erik works

Language first, code second · ask **one** question at a time, as plain text, never
a popup (he tries to answer every question in a message and will drop one
silently — see the memory note) · precision over speed, explain *why*, explain
jargon · **nothing is tuned until the physics works** · tests assert properties,
never snapshots · **look at the picture before trusting a scalar summary** ·
he is happy with autonomous work, so re-enter the loop only for rulings that are
genuinely his.

His standard for this model, in his own words: *"the logic based off of first
principles — inspired by physics, and designed by us to find a good middle ground
between being able to compute everything in real time and have at least an OK
believable physics simulation."*
