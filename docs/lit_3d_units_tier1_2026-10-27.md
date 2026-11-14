# Lit 3D units, tier 1 — the whole body lit from the unit's light read (#114)

2026-10-27. Graphics track (#93). Render-only: no sim state, no digest, no
golden. **HUMAN-TEST**: Erik plays it before merge. One patch, one Opus
implementer. Tier 2 (#115, a per-object probe from the sweep) is out of scope.

## 1. The problem

A unit's 3×3 footprint is stamped opaque into the light sweep
(`light_atten` 1,1,1, `GameMap.stamp_units`). That is what casts the long
shadows on the floor Erik loves; **it stays**. But every fragment of the
marine reads the light field at the tile under it (`renderer/lit3d.py`
`_FIELD_SAMPLE_GLSL`: `world_uv = fragWorldPos.xz / u_world_px`), and the
sweep holds light only in the footprint's EDGE row facing the lamp (a cell's
light is counted on arrival, before its body absorbs it); the 2×3 behind it
reads 0. So most of the body is dark.

Measured headless on 2026-10-27 (scratch probes, not committed), omni cursor
lamp, playground, marine at (13, 42):

- The sweep field around the block is exactly symmetric N/S/E/W. Lamp 4
  tiles south of the block centre:

  ```
    0.16   0.12   0.05 [ 0.00][ 0.00][ 0.00]  0.05   0.12   0.16
    0.19   0.19   0.10 [ 0.00][ 0.00][ 0.00]  0.10   0.19   0.19
    0.19   0.28   0.25 [ 0.28][ 0.51][ 0.28]  0.25   0.28   0.19
    0.21   0.25   0.45   0.35   0.69   0.35   0.45   0.25   0.21
  ```
  (Σrgb in light units; [ ] = the footprint.)
- The space marine mesh is centred on its block within 0.08 tile at every
  facing; 97–99 % of its vertices are inside the block; the 3D camera is
  orthographic. Model and tiles line up.
- A CPU emulation of the shader's diffuse term (`incoming(tile) ·
  half-Lambert`, up-facing vertices): only 19–32 % of the visible body reads
  any light; lamp north vs south differs 15–30 %, and the difference flips
  with the unit's facing (anatomy).
- **Not explained yet:** in Erik's screenshots a marine lit from the south
  looks lit and one lit from the north looks dark — far more than the CPU
  emulation gives. Something on the GPU path is not in the emulation. See §4.

## 2. The design

**One read per unit, both consumers.** `renderer/unit_shadow.py::read_unit_light`
already reads, once per frame per unit, the light around the unit from the
light accessor's `LightView` (footprint sums of rgb and of the net flux). The
body's light comes from the SAME read, so the lit side of the body always
agrees with the shadow on the floor. Extend that read with what the body needs
(an irradiance estimate per RGB channel, beside the existing brightness /
directionality / direction); do not add a second read or a second pass over
the planes. **The projected shadow's inputs and output stay bit-identical**
(the shadow is not under change in this patch).

**The body is shaded from it, not from the tile under each fragment.** The
marine shader gets per-unit uniforms (set before each unit's draw) and uses
them in place of the per-fragment field sample for `incoming_rgb` and `L`:

- a directional part, `I_dir · halfLambert(N·L)` with `L` the read's direction
  lifted to 3D with `u_light_z` exactly as today, and
- a non-directional part for the read's balanced share (`1 − directionality`),
  so a unit in balanced light (two opposite lamps, the sky) is lit evenly
  instead of black on both sides.

How `I_dir` / `I_iso` are estimated from the footprint sums is the
implementer's call, held by these PROPERTIES (§5): a unit lit from one side
has its lit side about as bright as the floor tile just outside its footprint
on that side; a unit in a uniform, balanced field is about as bright as the
floor around it; the result is the same (within float noise) for a lamp N, S,
E or W at equal distance. Note the footprint-wide sum divided by 9 (today's
`brightness`) is diluted by the dark interior — that is the quantity NOT to use
as the body's irradiance.

Everything else in the shader keeps working off the new `incoming_rgb` / `L`:
the smoke cast shadow stays PER FRAGMENT (`smokeShadow(world_uv, …)` still
multiplies `incoming_rgb`), as do the gloss highlight, rim and normal map.

**The shared GLSL block stays byte-identical.** `_FIELD_SAMPLE_GLSL` is shared
with the props' shader (`lit3d` seam); props are NOT in scope (tier 2). The
override lives in the marine shader's own `main()` after the shared block.
`tests/test_lit3d_extraction.py` gates `MARINE_FS` against an oracle taken
before the lit3d refactor; this patch is a deliberate, approved change to the
marine shader, so that oracle is re-baselined in the same commit with a
one-line rationale naming #114 — and whatever that test asserts about the
SHARED block must still hold unchanged.

**A/B switch.** `config.toml [render.unit_shading] body_light = "unit"`
(this patch) or `"field"` (today's per-fragment sample), read every frame
through `unit_looks.unit_shading` like its siblings, refused when missing or
mistyped. Erik judges with it. Ship `"unit"`.

**The no-shader fallback** (`light_rgb_fn` tint in `_draw_one`) may read the
same per-unit value if that is a one-liner; otherwise leave it.

## 3. Look shots

Add rows to `tools/look_shots.toml` (the one screenshot harness; a scene is a
row): the playground with a marine and the omni flashlight (`spread` 2π) one
tile gap away from its footprint to the N, S, E and W, zoomed on the marine
(and, if the harness allows, a diagonal). Render them BEFORE (with
`body_light = "field"`) and AFTER, and look at the PNGs before claiming the
patch works. Interpreter on this PC: `C:/Users/steen/anaconda3/python.exe`.

## 4. The GPU check (owed from the 2026-10-27 chat)

With the BEFORE shots, say whether the north/south difference Erik saw
reproduces, and if it does, find its cause. If the cause is in how the marine
body is lit (a normal or light-direction sign, the normal map's
`u_normal_y_sign`, the rim/gloss terms), it is this patch's system: fix it and
say so. If it is anywhere else (the light textures, the floor shader, the
sweep), record it in one line and leave it (one system at a time).

## 5. Verification — properties to pin

Each test names its property and the change that breaks it (no snapshots):

1. **The shadow is untouched**: on a real lit scene, the projected shadow's
   read (brightness, directionality, direction) and `shadow_cast` output are
   identical before/after the read is extended. Breaks if the extension
   changes the shadow's inputs.
2. **Lit side ≈ floor beside it**: a unit lit from one side by an omni lamp
   (real sweep, playground): the body read's lit-side irradiance is within a
   stated tolerance of the light on the floor tile just outside the
   footprint on that side. Breaks if the dark interior dilutes it (÷9) or the
   edge is double-counted.
3. **Rotation symmetry**: lamp N, S, E, W at equal distance → equal body
   irradiance and the direction pointing at the lamp each time. Breaks on an
   axis swap or sign flip.
4. **Balanced light lights evenly**: two equal opposite lamps → directionality
   ≈ 0 and irradiance > 0 (the body is not black). Breaks if the body uses
   the directional part only.
5. **Config door**: `body_light` missing or not one of the two strings is
   refused. Breaks if the door is dropped.

The full suite, `pytest tests -q` with this PC's interpreter, must be green;
a pre-existing test that fails is reported, never bent.

## 6. Systems

**(a) Existing canonical systems used:** the light accessor
(`simulation/light_field.read_light`, `LightView`) — never a sweep plane; the
unit light read `renderer/unit_shadow.py::read_unit_light` (extended, not
copied); the lit-3D seam `renderer/lit3d.py` (shared block untouched); 3D
units + `[render.unit_shading]` via `unit_looks.unit_shading`; the look-shot
harness (`tools/look_shots.toml` + `tools/lighting_demo.py --shots`); the one
tone-map (the marine still writes linear HDR).

**(b) New reusable systems:** none. CLAUDE.md's "3D units" row gains one
clause at implementation, draft: *"A unit's BODY is lit from its one per-unit
light read (`read_unit_light`, the same read as its projected shadow), never
from the tile under each fragment (`[render.unit_shading] body_light`, #114);
the smoke cast shadow stays per fragment."*

## 7. Hand-off

Commit as you go on `114-lit-3d-units`. When green: push the branch, and
report: what changed, every test added (property + what breaks it, one line
each), the look-shot PNG paths before/after, the §4 finding, and the exact
copy-paste command for Erik to play it (`C:/Users/steen/anaconda3/python.exe
main.py --level playground`, run from the worktree folder). Do not merge:
HUMAN-TEST.
