# Bonfire tuning, headless half (#129) — 2026-10-29

Erik's wish (2026-10-22): a bonfire burns a couple of minutes, and once properly
lit it burns to the END — the fuel is consumed, the fire does not go out with fuel
left. Dial tuning only. Physics rows are Erik's: this report MEASURES and PROPOSES.

Instrument: `tools/fire_smoke_probe.py` on `levels/fire_smoke_studio`, 300 s, CPU
build of `2983f1f`. Two tiles: the furniture row's middle tile (25,22) and the lone
tile (37,27), each with vents on (the level as shipped) and `--no-vents`.
Every raw table is in this folder: `baseline_*.txt`, `proposal_wd054_*.txt` (10 s
steps), `runs/` (2 s steps, one file per grid point), `grid_single_dial.txt`,
`grid_pair.txt`.

Column meanings: **fade** = first time after the peak that the fire is below 10 %
of its peak; **fuel 0** = first time `wall_hp` reaches 0; **out** = first time the
fire is exactly 0 (the `I_min` snap); **fuel at out** = the fuel left then (the
"went out with fuel left" signature; negative = the fire burned on past an empty
tile, finding 3). The peak (0.42–0.61) is the heat-only igniter's transient at
6–8 s; the burning plateau is 0.10–0.19.

## 1. Baseline (shipped: `wall_damage` 0.36, `H_BED_M` 38.73 / `H_BED_SHIFT` 0)

| tile | vents | peak fire (t) | fade | fuel at fade | fuel 0 | out | fuel at out | peak tile K | hottest gas K | squad HP |
|---|---|---|---|---|---|---|---|---|---|---|
| row  | off | 0.44 (6 s) | 298 s | 9 % | never | never (0.04 at 300 s) | 9 % at 300 s | 1431 | 774 | — |
| row  | on  | 0.44 (6 s) | 190 s | 5 % | never | 256 s | 3 % | 1431 | 768 | 92 88 91 |
| lone | off | 0.59 (8 s) | 132 s | 6 % | never | 204 s | 3 % | 1419 | 486 | — |
| lone | on  | 0.61 (8 s) | 86 s  | 4 % | 130 s | 132 s | 0 % | 1420 | 481 | — |

The shipped dials miss both halves for the row: it smoulders 3–5 min and goes out
with 3–9 % char. (The squad's 8–12 HP loss happens in the first 10 s, at ignition,
and is identical in every run measured, proposal included.)

## 2. Grid

`grid_single_dial.txt` / `grid_pair.txt` hold all runs (11 + 7 single-dial values
and 7 pairs, x 4 cases). The row tile is the bonfire; the lone tile the minimum
fuel unit.

| wall_damage | H_BED_M | row, vents ON: fade / fuel 0 / out / fuel at out | row, vents OFF: fade / out / fuel at out | lone on / off: fade | peak tile K | gas K |
|---|---|---|---|---|---|---|
| 0.09 (x0.25) | 38.73 | — / — / — / 44 % at 300 s | — / — / 63 % at 300 s | — / — | 1431 | 760–769 |
| 0.18 (x0.5)  | 38.73 | — / — / — / 18 % at 300 s | — / — / 38 % at 300 s | 214 / — | 1431 | 761–793 |
| 0.27 | 38.73 | 300 / — / — / 7 % | — / — / 17 % at 300 s | 122 / 194 | 1431 | 762–769 |
| **0.36 (shipped)** | 38.73 | 190 / — / 256 / 3 % | 298 / — / 9 % | 86 / 132 | 1431 | 768–774 |
| 0.45 | 38.73 | 138 / — / 182 / 1 % | 190 / 258 / 4 % | 70 / 90 | 1431 | 758–762 |
| 0.50 | 38.73 | 122 / — / 160 / 1 % | 174 / 240 / 3 % | 62 / 74 | 1431 | 750–771 |
| **0.54 (x1.5)** | 38.73 | **110 / 132 / 140 / 0 %** | 146 / 214 / 3 % | 56 / 64 | 1431 | 755–782 |
| 0.63 | 38.73 | 90 / 100 / 118 / −1 % | 94 / 162 / 2 % | 50 / 54 | 1431 | 761–768 |
| 0.72 (x2) | 38.73 | 76 / 76 / 100 / −3 % | 82 / 120 / 1 % | 44 / 46 | 1431 | 760–770 |
| 0.90 | 38.73 | 54 / 48 / 78 / −7 % | 52 / 78 / −5 % | 38 / 44 | 1431 | 728–760 |
| 1.08 (x3) | 38.73 | 50 / 30 / 68 / −14 % | 50 / 66 / −10 % | 40 / 42 | 1431 | 741–766 |
| 1.44 (x4) | 38.73 | 44 / 18 / 62 / −33 % | 46 / 64 / −33 % | 38 / 38 | 1431 | 735–747 |
| 0.36 | 19.37 (x0.5) | 202 / — / 270 / 3 % | 292 / — / 9 % | 98 / 132 | 1429 | 755–773 |
| 0.36 | 30.98 (x0.8) | 194 / — / 258 / 3 % | 290 / — / 8 % | 94 / 134 | 1430 | 746–760 |
| 0.36 | 58.10 (x1.5) | 190 / — / 252 / 3 % | 264 / — / 7 % | 90 / 118 | 1432 | 755–776 |
| 0.36 | 61.97 (x1.6) | 192 / — / 254 / 2 % | 290 / — / 9 % | 88 / 122 | 1432 | 770–772 |
| 0.36 | 77.46 (x2) | 190 / — / 248 / 2 % | 294 / — / 8 % | 84 / 116 | 1434 | 775–793 |
| 0.36 | 154.92 (x4) | 164 / — / 210 / 2 % | 244 / — / 6 % | 76 / 108 | 1439 | 795–825 |
| 0.36 | 309.84 (x8) | 150 / — / 188 / 1 % | 210 / 274 / 3 % | 64 / 88 | 1450 | 853–881 |
| 0.45 | 77.46 | 140 / — / 180 / 1 % | 202 / 270 / 4 % | 66 / 86 | 1434 | 760–800 |
| 0.45 | 154.92 | 130 / — / 164 / 1 % | 174 / 242 / 3 % | 60 / 74 | 1439 | 811–816 |
| 0.50 | 77.46 | 116 / 144 / 148 / 0 % | 176 / 242 / 2 % | 58 / 70 | 1434 | 777–784 |
| 0.50 | 154.92 | 112 / 132 / 142 / 0 % | 160 / 228 / 3 % | 52 / 64 | 1439 | 812–814 |
| 0.54 | 77.46 | 106 / 122 / 136 / 0 % | 152 / 210 / 3 % | 54 / 62 | 1434 | 779–807 |
| 0.54 | 154.92 | 106 / 118 / 132 / −1 % | 120 / 182 / 2 % | 50 / 58 | 1439 | 795–816 |

(—: not reached within 300 s. H_bed range: the derivation's Drysdale 20–40 %
surface share is x0.8–x1.6 of 38.73; x0.5 and x2–x8 bracket it, x8 being the
Sept-6 lab's old factor.)

## 3. Findings

1. **`wall_damage` is the duration lever, and the only one.** Fade and out times
   scale roughly as 1/`wall_damage`, and the char left at the end falls with it,
   crossing 0 at 0.54 for the vented row and at about 0.9 for the unvented row.
   **H_bed is not the lever any more**: x0.5 .. x8 of the M3-derived 38.73 moves
   the row's duration by at most ~30 % and the char by 1–2 points, while raising
   the hottest gas by ~100 K at x8. The Sept-6 comment ("the fuel-vs-T-gate race
   needs H_bed x8") predates M3's currency fix and no longer describes the engine.
2. **Why the fire goes out with char left: the `I_min` snap, not cooling — #68
   is not the cause here.** With the capacity law `I_eq ≈ c·F·o2f·hot`, the fire
   follows the fuel down an exponential tail and snaps to 0 at `I_min` = 0.02,
   i.e. at F ≈ 0.02/(0.95·o2f·hot) ≈ 2–4 %, the char measured. The tile is still
   at 520–625 K when it snaps (the `hot` ramp spans 373–553 K): it has not cooled
   out. The char reaches 0 only when the drain outruns the fire's own relaxation,
   i.e. at a large enough `wall_damage`, which also shortens the fire. The
   unvented row is O2-starved (five tiles share one vitiated ring), so its `o2f`
   is lower, its tail longer and its char higher at every value. Duration and
   burn-to-the-end pull against each other: no value gives all four studio cases
   90–180 s AND fuel 0.
3. **A furniture tile is never destroyed when its fuel runs out.** Burn-through
   destroys a tile only `if (wall_hp[i] <= 0 && is_wall[i])`
   (`cpp/src/fire_simulation.cpp:398`), and `is_wall` is `gmap.solid`; furniture
   is permeable, not solid. At hp 0 nothing happens: the tile stays furniture,
   the fire (now F = 0) dies down over 10–45 s while `wall_hp` keeps going
   negative (to −47 % in the grid). Not a dial; recorded, not touched.

## 4. Proposal

**`wall_damage` 0.36 → 0.54; H_bed unchanged (38.73, the derived value).** 0.54
is the smallest value at which the studio's bonfire as shipped (the furniture
row, vents on) burns to the end: visible 110 s, fuel 0 at 132 s, out at 140 s
with 0 % left — inside the 90–180 s window, by moving only the duration dial.
H_bed stays because it buys little duration, nothing on the char, costs gas
temperature, and is a derived constant (Huggett + Drysdale), not a feel dial
(0.50 + H_bed x2 reaches the same result 6 s longer, +20–30 K gas). The
compromise, plainly: without vents the same row is visible 146 s but goes out at
214 s with 3 % char, and a lone crate is visible only 56–64 s (fuel 0 at
52–76 s, then burns a little past it, finding 3).

## 5. What moved in tests (at 0.54)

- `tests/test_w6_armory.py tests/test_light_field.py`: 31 passed, incl.
  `test_canonical_scenario_matches_sanctioned_golden` —
  **GOLDEN_AGGREGATE `1ff07369…` does not move** (the 30-tick canonical scenario
  burns no furniture).
- `pytest tests -q -k "fire or combust or ignite or bed"`: 155 passed, 2 skipped,
  2 xfailed.
- Full suite not run (not requested) — run it before merging.

## 6. For Erik to rule on

1. **Ratify `wall_damage` 0.54** (or pick another row of the grid) after seeing it
   on screen — HUMAN-TEST. "A couple of minutes" was read as 90–180 s visible.
2. **Which case is "the bonfire"?** 0.54 is chosen for the vented 5-tile row. If a
   sealed room's bonfire must burn to the end too, that needs ~0.9 (visible
   ~50 s) — or the char floor itself must move: it is `I_min` (0.02, global)
   through the capacity law, a dial outside this brief.
3. **Burnt-out furniture is not destroyed** (finding 3): should a non-solid
   flammable tile at hp <= 0 become burnt out, and its fire stop with its fuel?
   That is a code change, not a dial.
