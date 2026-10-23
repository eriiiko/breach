# P6a brief — light rides the sweep (2026-09-25)

Ray-engine-v2 arc, issue #12. This is the first of P6's three steps:
- **P6a (this one):** the sweep computes light. Invisible.
- **P6b:** the renderer consumes it, adds cone emitters and the sky boundary.
  HUMAN-TEST.
- **P6c:** the archive and the deletions.

Design: `docs/ray_engine_v2_design_v3_2026-09-15.md`, the P6a row of §11.5,
and **§4, §5, §6.3, §7, §8.1–8.3, §10, §11.2**. Read them. Erik ruled the clock
and the language on 2026-09-15 (§12, "R-P amended"): light runs on the C++/CUDA
sweep once per sim tick. **Oracle-gated, no HUMAN-TEST. Nothing on screen
changes. It merges on a green gate.**

## 1. What P6a builds

The light channels ride the SAME sweep as heat:
- three RGB channels, transported with **step** (Erik's choice on real-scene
  renders, §2.4); heat keeps **shear**;
- the net flux vector;
- the in-scatter glow at gas cells.

Inputs:
- a second baked table **L°[T]**;
- integer extinction planes **`light_atten_q`** (static) and
  **`dyn_light_atten_q`** (the `stamp_units` output), both (h,w,3) Q16;
- the smoke's RGB extinction as a per-cell integer term.

Outputs:
- **`light_q`** (h,w,3), integer RGB irradiance;
- **`light_flux_q`** (h,w,2);
- **`light_glow`** (h,w,3).

The only sources in P6a are thermal: every cell with a light extinction emits
`a_c · L°_c[T]`. Burning tiles, glowing hot solids and hot smoke are therefore
all lights, with no cap. Cone emitters, the sky boundary, the accessor,
`LightingPass` and every deletion are P6b/P6c, NOT here.

## 2. Decisions this brief makes (the deltas since design v3). Do not re-derive.

1. **One traversal, a transport step per channel.** This is the CLAUDE.md rule
   ("every payload is a channel on this sweep, never a second traversal; the
   transport step is a parameter"). Heat gathers from shear's upwind pair, and
   light from step's pair, in the same visit, each channel with its own stored
   outflow.
   - **CPU:** a row-major walk in the ordinate's direction is topological for
     both (the `radiation_sweep.cpp` comment already says so).
   - **CUDA:** when light is on, the twin's wavefront becomes step's
     anti-diagonal for all channels (h + w − 1 launches; that is also a
     topological order for shear). Heat-only keeps shear's column wavefront.
   - The gather form makes the order irrelevant to the integers, so **heat must
     come out BIT-IDENTICAL whether light is on or off, on both backends.**
     That is a gate.
2. **Light is requested, never always on** (§7.1: "headless training simply
   does not request the light channels"). The engine computes light only when
   asked. P6a adds the request, default OFF, so the live game and the golden
   do not move. P6b turns it on for the renderer.
3. **L° is a checked-in integer table** (3 × 4000 int64), generated OFFLINE by a
   `tools/` script FROM `renderer/blackbody.py` (the single ΔT→colour map:
   Helland/Bartlett chroma × Macklin intensity), with a test that re-runs the
   generator and matches within one count. Never `np.power`/`np.log` at load
   (§8.1). Its buckets are E°'s buckets. **L°[0] is 0**: a room-temperature
   body emits no visible light, so the light ring is dark until P6b's sky.
4. **The light currency has #78's lesson built in.** #78 found the heat sweep's
   per-ordinate terms were single-digit integers near ambient and rounded away.
   So pick L°'s integer scale so that:
   - a single burning tile's light is still ≥ 2^8 counts per ordinate 16 tiles
     away in clear air;
   - the faintest glow the ramp shows is ≥ 2^8 counts per ordinate at its own
     cell.

   Prove the int64 headroom at the table top in the style of #78's §4 (every
   product < 2^58 with margin). The table carries its currency the way
   `EmissiveTable::fine_bits` does. Reuse that pattern; do not invent a second
   one.
5. **Kirchhoff per channel, and bodies do not glow.** The material share
   `a_c` absorbs and emits `a_c · L°_c[T]`. The body share `d_c − a_c` absorbs and
   re-emits the ambient level, which for light is L°[0] = 0. That is design
   row 25's structure with a dark ambient: units block light and do not glow.
   There is no Fleck on light: light has no material feedback (§4.1).
6. **Smoke** follows the heat side's P5a pattern (the sweep's own pre-pass,
   never a stored plane):
   - `a_gas_c = min(ONE, Σ_g absorb_light_q16[g][c] · max(0, N_g) >> 16)` from
     `GasTable.absorption` (RGB), quantized at the gases door, with the `[smoke]`
     dials folded in there: one owner, the same dials the render medium reads
     (§4.4).
   - Glow is `stream × scatter_albedo × density`, render-only, and only where
     gas is.
   - Hot smoke emits through L° like any cell with `a_gas_c > 0`. That is the
     arc's "black-body smoke".
7. **Not digested** until P7 (§8.3). The A/B lockstep harness carries the
   integer light planes (`field_ab_harness.py` SIM_FIELDS) beside the heat
   ones. The sweep zeroes its own new outputs (design row 38), never a caller.

## 3. Properties to pin

Write these from this section, never from the implementation. Each test's
docstring names its property and the change that breaks it. Validate each new
property by breaking the code once. Write the reference first
(`sweep_ref_q.py` + gates), then `radiation_sweep.cpp`, then the CUDA twin.

1. **Gate 0 for light:** the C++ light channels match the reference bit for
   bit, over the gate-0 matrix plus light-specific scenes.
2. **CPU == CUDA at tol 0** on every light plane and counter, and on the heat
   planes with light on.
3. **Heat is untouched:**
   - bit-identical heat planes and counters with light on vs off, on both
     backends;
   - the golden does not move.
4. **Conservation per channel**, exact in int64, on the light channel's own
   books: emitted + entered-from-the-ring == absorbed + left-to-the-ring. Keep it
   as counters if per-cell planes are not needed.
5. **A dark world is exactly dark:** no hot cell (with a sky boundary of 0) gives
   `light_q ≡ 0` everywhere.
6. **Optics:**
   - a wall spanning the grid at `a = ONE` leaves the far side exactly dark
     (0);
   - glass: light and heat each read their own row coefficient (`light_atten`
     vs `heat_atten`, §5); use the shipped glass row;
   - a stamped marine blocks light (MAX stamp);
   - smoke tints by its RGB absorption and dims monotonically with density;
   - with no gas there is no glow.
7. **Colour follows temperature:** a hot cell's emitted chroma equals
   `blackbody.py`'s at that temperature, within quantization, and hotter is
   brighter.
8. **Headroom** (decision 4) on an over-driven scene, and **the resolution floor**
   (decision 4's two ≥ 2^8 checks).
9. **The flux vector** points away from a lone source, on each side of it.

**Timing** (the design's gate, §10): print sweep ms per tick with light off
and on, CPU and CUDA, on the playground and on a 128×256 map. **STOP and report
before merging** if heat + light on the CPU exceeds half the 41.67 ms tick
budget on the playground.

A failing pre-existing test is a finding: report it, never bend the design to
it.

## 4. Systems

Uses, never duplicates:
- the Radiation sweep and its CUDA twin
- the Emissive table (L° lives beside E° in the same module and pattern)
- the Integer reference (spec first)
- the Extinction planes and `optics_fixed.py` (the light RGB planes join it,
  §3)
- `stamp_units` (a new integer output beside `dyn_heat_atten_q`)
- the Gas table (`GasTable.absorption` / `scatter_albedo`)
- Blackbody (the ONE colour map, read offline by the generator)
- the Fixed-point kits
- the A/B lockstep harness
- the RL-batch habits (§A: the resident core is born `(N, h, w)`)

**Draft rules for CLAUDE.md:**
- Extend the Radiation sweep row: *"light is three step-transported RGB
  channels on the same traversal, requested (default off); L° is a checked-in
  table regenerated from blackbody.py by `<tool>`, never computed at load."*
- Extend the Extinction planes row with `light_atten_q` / `dyn_light_atten_q`.

## 5. Out of scope

Out of scope, by the one-system rule:
- the renderer, `LightingPass`, the accessor, cone emitters and the sky
  boundary (all P6b);
- deleting `raycaster.*`, `fire_lights.py` and the float `dyn_light_atten` (P6c);
- the heat channel's arithmetic (#78 just landed; do not touch it);
- the EOS (#72).

Anything else you notice gets one line in the report.

## 6. Execution

- **Where:** branch `12-p6a-light-channels` off `fire-12`, worktree
  `breach-p6a`, one writer. **Model: Opus.**
- **Builds:** run `cpp\build_cpu_home.bat` and `cpp\build_cuda.bat` from the
  worktree, first and after every C++ change.
  - Python is `C:/Users/steen/anaconda3/python.exe`.
  - Run the suite as `pytest tests -q -p no:cacheprovider` from the worktree root.
- **Commits:** `feat(#12): P6a -- …` / `test(#12): …` / `docs(#12): …`, staging
  explicit paths only (never `git add -A`). End each commit with the
  Co-Authored-By line. Do not merge, do not push.
- **Trap:** running `tests/_xarch_perfield_digest.py` as a script overwrites the
  tracked `tests/_xarch_perfield_DESKTOP-0E98HUV.txt`. Restore it from HEAD and
  never commit it.
- **Report:** your FINAL MESSAGE is the report (the harness blocks report
  files). It carries:
  - the §2 decision details you settled (the scale, the headroom table, the L°
    generator)
  - the timing table
  - every test added or rewritten, one line each: its property and what breaks
    it
  - suite totals on both builds
  - the commit list
  - findings outside the patch, one line each

  Keep it as long as the evidence needs, and include no question lists.
