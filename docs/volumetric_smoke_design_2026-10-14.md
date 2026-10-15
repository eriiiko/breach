# Volumetric smoke — a baked, looping 4D detail volume raymarched in a tilted view

> **Status**: prototype arc, issue #90. Render-only, determinism-exempt,
> HUMAN-TEST gated (Erik plays it before merge). Design critique deferred to
> the human test by choice: this is an exploration of a new look, not a sim
> or safety surface. 2026-10-14.

## 1. The idea (Erik, 2026-10-14) and its sharpening

Erik's picture: a smoke "movie" whose frames are played not along time but
along a spatial axis — a frame of the yz-movie at every x, a frame of the
xz-movie at every y — the panes transparent except where the sim says there
is smoke, and the movie also advancing in time. Both movies must loop in space
and in time.

Sharpened: the two movies are one object. A yz-frame at x and an xz-frame at
y are both slices of a single periodic 4D field `M(x, y, z, t)`. So we bake
that field once — a periodic 3D volume evolving in time — and let the GPU
slice it any way the camera needs, by **raymarching a 3D texture** in a
fragment shader. No slice stacks, no popping, hardware trilinear filtering,
wrap-around in every axis for free.

The division of labour is the one `shaders/gas_medium.fs` already uses,
extended to 3D: **the sim owns the macro shape** (where smoke is, how thick,
how hot — the per-tile optical depth), **the baked volume owns everything
below tile scale** (wisps, rolling, vertical structure). Height is invented
from density: a thick tile grows a tall column, a thin one a low haze.

This is the *look* the 2.5D multi-layer smoke of
`docs/blackbody_smoke_and_rendering_brainstorm.md` §0.5 reaches for, at zero
sim cost and zero determinism risk: nothing here reads anything but render-
side copies of sim fields.

## 2. Architecture

```
offline                      per frame (render only)
-------                      ------------------------------------------------
tools/bake_smoke_volume.py   upload_state: tau (H,W) = gas_optical_depth(...)
  renderer/volume_bake.py                  -> RGBA16F grid texture (R = tau)
  periodic-box NS solver     compose_world: as today, MINUS the 2D gas layer
  -> assets/volumes/*.npz    tilt view: world RT -> textured ground quad under
     (T, Z, Y, X, 2) uint8              a perspective Camera3D (pitch ~55 deg)
                             gas volume: fullscreen raymarch above the ground
renderer/gl3d.py (ctypes)              sampler3D = the baked volume (RG8)
  GL_TEXTURE_3D upload                 sampler2D = tau, light_tex_a, heat tex
```

### 2.1 The bake — `renderer/volume_bake.py` (pyray-free) + `tools/bake_smoke_volume.py`

Incompressible Navier-Stokes on a fully periodic box (a torus), Stam's
stable-fluids splitting:

- **advection**: semi-Lagrangian with MacCormack correction + clamping
  (Selle et al. 2008) for velocity and scalars — sharp filaments at modest
  grids; `scipy.ndimage.map_coordinates(mode="grid-wrap")` is the periodic
  trilinear lookup;
- **forcing**: a divergence-free low-wavenumber body force (curl of a few
  slowly-rotating Fourier modes) keeps turbulence alive against the
  semi-Lagrangian damping; mild buoyancy on the second scalar gives a
  vertical preference;
- **vorticity confinement** (Fedkiw, Stam, Jensen 2001) re-injects the
  small swirls the advection damps;
- **projection**: exact on the torus via FFT (`u_k -= k (k.u_k)/|k|^2`,
  Stam 2001 "A Simple Fluid Solver based on the FFT");
- **scalars**: `rho` (smoke) and `phi` (a sharper "fire" channel) are
  advected and **relaxed toward slowly drifting low-frequency targets**, so
  contrast never washes out and the statistics are stationary — a
  turbulent passive scalar, which is what a detail volume should be.

**Spatial loop**: free — periodic BCs. **Time loop**: the standard crossfade
of two consecutive periods: with `F(k)` the recorded frames, the output is
`O(k) = lerp(F(k+T), F(k), s(k/T))`, `s` a smoothstep, so `O(0) == O(T) ==
F(T)` exactly and the blend is invisible in smoke.

Asset: `assets/volumes/<name>.npz` — `volume` uint8 `(T, Z, Y, X, 2)`
(R = smoke, G = fire), plus `period_tiles`, `height_tiles`, `loop_seconds`.
Gitignored (tens of MB); the game falls back to a procedural fBm volume so
a fresh clone runs. **Resolution** 128 x 128 x 64 over 4 x 4 x 4 tiles
(32 texels per tile in x/y, 16 in z), 48 frames; `smoke_hd` = 256 x 256 x
64 over the same box for the "how much do we need" comparison.

**Finding (2026-10-14, the movie-only view):** the first bake repeated
every 1 tile vertically. A tilted ray through a 6-tile slab crosses that
repeat several times at slightly shifted x/z, and the sum smears into
streaks that fan out along the view rays. The vertical period must be as
tall as the columns get; z texels may be coarser than x/y.

Looking at the movie itself: `tools/view_smoke_volume.py` (black-and-white
playback, slice / mean / max over z) and F12 → 2 in the game (the slab
filled with the volume under flat grey light).

### 2.2 The 3D texture — `renderer/gl3d.py`

raylib has no 3D textures. Three OpenGL entry points via `ctypes` on
`opengl32.dll` + `wglGetProcAddress` (we are Windows-only on every machine):
`glTexImage3D`, `glBindTexture(GL_TEXTURE_3D)`, `glActiveTexture`. The
volume is uploaded once as `(X, Y, (Z+1)*T)` RG8 with slice `Z` of every
frame a copy of slice 0 (so trilinear wrap in z is exact inside a frame);
`x`/`y` wrap in hardware (`GL_REPEAT`), `z` and `t` are wrapped in the
shader (two trilinear fetches, lerped in t). Bound on a high texture unit
raylib's batch never touches.

### 2.3 The tilted view — `renderer/tilt_view.py`

A **screen-stage** view mode, not a change to the world RT: `compose_world`
keeps producing the top-down world RT exactly as today (walls, floors,
lights, 3D marines and props all baked into it). `TiltView` owns an RT the
size of the map viewport and, in it, draws the world RT as a textured ground
quad under a perspective `Camera3D` (3D X = x_wpx, Z = y_wpx, Y = up — the
`lit3d` convention), then runs the volume pass, then blits to the map area
with the usual y-flip. The camera derives from the existing `Camera2D`:
target = the visible rect's centre on the ground, distance chosen so the
visible width at the target matches the 2D zoom, pitch from config. Panning
and zooming keep their keys. `rlSetClipPlanes` lifts raylib's 1000-unit far
plane (world RTs are up to 5760 px). Cursor->tile unprojects the mouse ray
onto the ground plane so ignite/gas keys keep working.

### 2.4 The volume pass — `renderer/gas_volume.py` + `shaders/gas_volume.fs`

Fullscreen quad; per pixel: ray from the inverse view-projection, intersect
the slab `0 <= y_up <= h_max`, march N jittered steps. Per sample:

- `tau` = bilinear(sim optical depth) at the ground position — **the same
  number `gas_medium.py` builds** (`gas_optical_depth`, single source of
  scale); zero -> skip;
- `d` = trilinear(volume.r), `f` = trilinear(volume.g), looked up at world
  tiles / `period_tiles`, `y_up / height_tiles`, frame = sim-tick phase;
- solidity `s = saturate(tau / tau_solid)`; column height
  `h = h_max * saturate(tau / tau_h)^0.5`; Nubis-style erosion
  `coverage = saturate((d - erode*(1-s)) / (1 - erode*(1-s)))`; the top of
  the column fades with `d` so it is wispy, not a lid;
- extinction `sigma = k_ext * tau * coverage * top_fade`;
  `T *= exp(-sigma ds)`; inscatter `+= T (1 - exp(-sigma ds)) (L + E)`;
- `L` = (ambient + light_tex_a.rgb * gain) * albedo * exp(-k_sh * tau *
  (h - y_up)/h_max) — the baked light field the ship and marines read, with
  a vertical Beer-Lambert self-shadow (dark bottoms, lit tops);
- `E` = heat_tex.rgb * heat_tex.a * f^p * (1 - y_up/h)^q * e_gain — the
  blackbody overlay's colour glowing up from the floor.

Output premultiplied (ACES(inscatter), 1 - T), `BLEND_ALPHA_PREMULTIPLY`
over the ground, exactly the gas medium's compositing operator. Clock = sim
tick via `advected_noise.advection_phase` (replays render identically).
Dials in `[render.gas_volume]`, re-read every frame (Ctrl+R retunes live).

## 3. Scoped OUT of this patch (Erik's request: build the idea only)

Touched imperfectly on purpose, each a later patch:

- **Walls / floors / doors / water** are flat decals on the ground plane in
  the tilted view (3D walls later).
- **3D marines and props** stay baked into the top-down RT: flat, no smoke
  occlusion. Later: redraw them in the tilt camera with a depth texture
  (`rlLoadTextureDepth`) the raymarch reads.
- **The 2D smoke detail pass** is simply not drawn in tilt mode; no
  reconciliation of the two looks.
- **God rays / raycaster inscatter** (`smoke_glow`) unused by the volume;
  lamp shadows through smoke are v2.
- **Per-gas colour within a tile**: v1 lights each tile with its
  tau-weighted mean albedo (`GasMediumOverlay.albedo_rgb` — steam white, soot
  dark, poison green at tile level); separating coexisting gases vertically
  is later.
- **UI overlays** drawn into the RT appear on the floor; only cursor->tile
  is unprojected.
- **Laptop GPU cost** unmeasured; steps/resolution are config dials.
- **Nothing in the sim, the digests or the goldens moves.**

v2 ideas, in payoff order: advect the lookup by the tamed wind with the
two-phase flow-map crossfade (bounded distortion, hidden resets — solves
Erik's continuity worry); a smooth per-tile phase offset from temperature
(hot regions boil faster, no seams); lamp shadows; depth-composited marines
and walls; a 256-res "HD" bake to answer "how much resolution do we need".

## 4. Patches

| # | Patch | Mode | Gate |
|---|---|---|---|
| P1 | `volume_bake.py` solver + baker CLI + mosaic preview | inline | property tests (periodicity, projection, loop closure) + a look at the mosaic |
| P2 | `gl3d.py` 3D-texture shim | inline | upload + read-back of a known pattern in the studio |
| P3 | `gas_volume.py` + `gas_volume.fs` | inline | screenshots in `tools/lighting_demo.py --tilt` |
| P4 | `tilt_view.py` + renderer/main/demo flags + config | inline | screenshots, cursor unproject property test |
| P5 | look tuning + perf numbers + CLAUDE.md rows | inline | HUMAN-TEST (Erik) |

## 5. Systems

**Existing canonical systems this design uses**

- Gas medium (`renderer/gas_medium.py::gas_optical_depth`) — THE per-tile
  optical depth; the volume reads it, never a second density definition.
- LightingPass / LightFieldCtx (`light_tex_a`, ambient, gain) — the light
  the volume is lit by; never a second raycast.
- Blackbody (`renderer/blackbody.py` via `HeatFieldOverlay.tex`) — the
  single T->colour map, consumed as its packed texture.
- Sim-tick clock (`renderer/advected_noise.advection_phase`).
- Camera2D (`renderer/camera.py`) — the tilt camera derives from it.
- `main.py` `_parse_*` launch flags (`--tilt`).
- Config `[render.gas_volume]`, `CFG`, Ctrl+R.
- Lit-3D seam (`renderer/lit3d.py`) — the world-px 3D axis convention
  (X = x_wpx, Z = y_wpx, Y = up); the tilt camera is a screen-stage camera,
  not a second copy of `make_camera` (which frames the world RT).
- Tamed wind (`gas_detail.tame_wind`) — v2's advection input.

**New systems (draft rules, written into CLAUDE.md at P5)**

- Gas volume (`renderer/gas_volume.py` + `shaders/gas_volume.fs`) — the
  volumetric smoke look for the tilted view; consumes the gas medium's tau,
  `light_tex_a` and the blackbody overlay; a second raymarcher is the bug.
- Smoke volume bake (`renderer/volume_bake.py`, `tools/bake_smoke_volume.py`,
  `assets/volumes/*.npz`) — THE periodic-box fluid baker and its asset
  contract; a new detail volume (fire, steam, dust) is a bake config, not a
  new solver.
- GL 3D texture shim (`renderer/gl3d.py`) — the only raw-GL in the repo;
  anything else needing a GL feature raylib lacks extends it.
- Tilt view (`renderer/tilt_view.py`) — the only screen-stage 3D camera;
  anything drawn in perspective over the world RT goes through it.

## 6. Day 2 (2026-10-15): periodicity, the phase field, directional light

Erik's first look (the tilted west hall, `sls_west_v7_mid.png`): "this
definitely looks cool", number-one issue **periodicity** — lingering smoke
shows the 4-tile repeat of the asset.

**Diagnosis.** Not resolution, and not the initial conditions (the warm-up
forgets them). Two choices made every box hold one or two box-sized blobs,
which the eye reads as "same blob every 4 tiles": the forcing drove only
wavelengths one box and half a box (`force_kmax = 2`), and the density
target's steep red spectrum put most of its energy there too. Where the sim
is busy the macro shape hides it; on a uniform floor the asset shows naked.
`q_base` vs `q_mid25` mosaics (scratchpad `bake/`) show it directly.

**Fixes landed (commits 34f0fce, fed04f9):**

1. **The phase field** (`renderer/volume_phase.py`, Erik's idea): each tile
   reads the movie at its own moment of the loop; the GPU's bilinear fetch
   interpolates the phase between tile centres, so it is continuous over the
   floor and, the movie being periodic in time, seamless. Uploaded as
   (cos, sin) so the interpolation crosses the wrap on the short arc.
   `offset` mode: phase = clock + static world-space noise + sim sources
   (temperature, pressure above the room median, tamed wind speed), all
   bounded. `speed` mode: each tile integrates its own rate deviation with a
   leak (`phase_relax_loops`) so the temporal shear between a hot tile and a
   quiet neighbour stays finite. Dials `phase_*`. Result: the repeat is
   clearly weaker (`cmp_off_d2.png` vs `cmp_on_d2.png`) but a lattice
   remains — a time offset cannot hide a structure that persists through
   the whole loop, which the box-scale modes do. Hence:
2. **Band floor on the bake** (`force_kmin`, `rho_kmin`, `phi_kmin`): the
   forcing now drives wavelengths 3–5 per box and the targets cut
   `|k| <= kmin`. `q_mid25` (kmin 2.5) is homogeneous with no visible repeat
   at all; `q_mid15` keeps bigger billows with a faint residual order. The
   128-res bakes `smoke_mid15` / `smoke_mid25` are the candidates.
3. **Directional light**: lamp direction = gradient of the baked light field
   at the ray's ground hit (per pixel, lamps are far against a column),
   smoke normal = gradient of the baked density (three extra samples per
   lit sample, only when `dir_light > 0`); wrap-lit N·L brightens the
   lamp-facing side and darkens the back, a Henyey-Greenstein term gives the
   halo when the lamp sits behind the smoke. Debug 3 paints the direction
   field. Subtle on dark soot (its look is mostly transmittance), visible
   on lit plumes; dials `dir_light`, `dir_eps_texels`, `hg_g`, `hg_mix`.

**Still open about the volume's box:** it wraps in z too (the FFT pressure
solve needs it) — a non-wrapped z with a real floor needs a sine transform
in z and would give honest rising plumes (candidate for the bake sweep). A
bigger box costs quadratically (8 tiles = the 256-res `smoke_hd`); the
shader tricks cost three fetches.

### 6.1 TODO (Erik, 2026-10-15 — keep on the list)

- **Stochastic tiling** in the shader (Heitz & Neyret 2018; Deliot & Heitz
  2019): per-cell random rotation / flip / time offset, three nearest cells
  blended histogram-preserving; plus a slow world-space domain warp; plus a
  dial fading detail amplitude where the local tau is uniform ("homogeneous
  asset, fade where you shouldn't see it"). Can go on top of the phase
  field, whole or in parts.
- **Bake sweep** at 64-res on a small parameter grid (band floor, period 2
  vs 4 vs 8, non-wrapped z, forcing band), judged on the viewer mosaic and
  a movie-only shot, winner rebaked at 128 and locked in.
- **Directional light** refinements: apply the N·L term to the glow
  (inscatter deposit) too; per-lamp direction from `frame_lights` instead
  of the field gradient where several lamps overlap.
- **Speed mode** judged live (a still cannot show it): the shear along
  shockwaves Erik is curious about.
- Earlier list (§3): 3D walls, depth-composited marines, lamp shadows
  through smoke, flow-map wind advection, laptop GPU cost.

### 6.2 Systems (additions)

- Phase field (`renderer/volume_phase.py`) — THE per-tile phase-offset
  state for a looping detail asset; a second "local clock" for another
  looping texture (water surface is a candidate, Erik 2026-10-15) extends
  this module rather than re-deriving the cos/sin upload.
- Tamed wind (`gas_detail.tame_wind`) is now consumed by the volume pass as
  a phase source — still the one render wind seam.
