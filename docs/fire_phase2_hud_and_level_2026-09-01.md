# Phase 2 — full tile-inspector HUD + fire-tuning level (fire session #12)

> **Status: SPEC — for the implementation agent, 2026-09-01.** Phase 1
> (G12) closed + blessed. Context: `docs/fire_mechanics_inventory_2026-08-31.md`
> §5 + Systems(b); issue #53 item 1; issue #12 session comments.

## A. Extend the tile-inspector HUD to the full field set

`renderer/hover_readout.py::pack_hover_readout` is **THE per-tile debug
probe** (rules-lifecycle: it becomes a CLAUDE.md canonical row in this
patch). Contract to write into its docstring: *all* tile-value display goes
through this one function; callers never read gmap fields directly for
display; when the resident tick's once-per-tick D2H sync goes away (Erik,
on record 2026-08-31), the swap to a device-side one-tile gather happens
INSIDE this seam and no caller changes.

Fields (currently: material, T game+K, fire I, five traces, O2). ADD:

| Field | Source | Display |
|---|---|---|
| Pressure | the atmosphere/pressure field | dequantized via its own `*_fixed` module — NEVER an inline /65536 (CLAUDE.md rule) |
| Bulk N | o2 + inert_n2 gas planes | dequantized sum, labeled `N` (plus keep the O2 line as-is) |
| Wind | `wind_x`/`wind_y` | m/s (they are TRUE velocity post-EOS — issue #51's caveat; label `m/s`), show vx, vy |
| Water | `water_depth` | dequantized depth |
| Fuel | `wall_hp` + material `hp` | raw hp AND `F = clamp01(wall_hp/hp_mat)` — the fire availability factor, THE fuel gauge (G3: fuel is what kills a fire) |
| Gas energy | `gas_energy` (int64) | the #54 ledger value for the tile, in a readable unit (pick one, document it) |

Keep the fixed-width one-corner table style (screenshot-proven); it may
grow a few rows but stay one block. Survey gmap/the `*_fixed` modules for
the exact names + scales — do not guess scales, read each field's own
boundary module. Headless-testable as today: extend
`tests/test_hover_readout.py` for every new field (pyray-free).

Also: add the CLAUDE.md canonical-systems row (Render/UI table):
`| Tile inspector | renderer/hover_readout.py::pack_hover_readout | THE per-tile debug readout (F6 hides) — tools/HUD read it, never roll a parallel field probe; future resident gather swaps inside this seam |`

## B. The fire-tuning level

`levels/fire_tuning/`, authored ONLY through `level_lib.py` by a new
client script `tools/make_fire_tuning_level.py` (one writer ever — never
hand-write level.toml; look at how existing levels/tools do it, e.g. the
playground + `tools/place_playground_vents.py` pattern). Content — each
station isolated enough that radiation (~2-tile ignition reach) doesn't
couple stations:

1. **Bonfire stage**: a 2×2 wood-crate cluster in a large ambient hall —
   the §1.1 reference fire (G1–G4 measurements happen here).
2. **Spread line**: a kindling line/ring around one wood crate — ignition
   propagation timing.
3. **Material row**: single isolated samples — wood, furniture, kindling —
   spaced ≥ 6 tiles apart.
4. **Sealed chamber**: small airtight room + one wood crate (O5/O2
   starvation: fire must die when O2 runs out). Verify with the airtight
   lint (`tools/level_airtight.py`).
5. **Door room**: same chamber shape but with a door to the hall (the
   flow/reignition case).
6. Ambient/space ring boundary like the playground (venting scenarios
   later); a small marine roster (1–2 units) so the level is playable and
   flashlight/HUD usable.

Gate: level loads + steps N=100 ticks headless without error, airtight
lint passes for the sealed chamber, level data files committed. No golden
or digest is touched by ANY of this patch (render/tools/tests/level data
only) — if a golden moves, that is a bug, stop.

## C. Rules

- Branch `fire-12`, this tree, no push, no branch switching. Python:
  `C:/Users/steen/anaconda3/python.exe`; tests `-m pytest tests -q`.
- Suite green except the 3 knowns (2× cool_shift parked, 1×
  test_bench_two_room scipy env). Stage explicit paths, never `git add -A`.
- One commit for A (+ CLAUDE.md row + tests), one for B — or one combined
  if the diff stays readable. Cite this spec in the message(s).
- Erik's look at the level + HUD in-game is the HUMAN feel check afterward
  — not a gate for the commit.
