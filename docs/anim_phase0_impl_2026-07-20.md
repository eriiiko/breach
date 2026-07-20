# Phase 0 — Live 3D marines over the 2D world (impl doc)

**Status: IMPL SPEC (2026-07-20).** Branch `anim-phase0-3d-marines`. Feel-gated → **HUMAN-TEST, no auto-merge.**
Derives from the ML-animation lit search (`docs/research/ml_animation_litsearch_2026-07-20.md`, Phase 0) and the
classical brief (`docs/procedural_animation_brainstorm.md`). This is the render-only "learn a little" rung: a rigged
3D marine drawn on top of the existing 2D top-down world, driven read-only off sim state.

## Goal & non-goals
- **Goal:** load a rigged glTF humanoid, animate a clip chosen from unit state (idle/walk/fire/dead), draw it at the
  unit's world position with yaw = continuous `unit.facing`, composited into the existing world render target, behind a
  toggle (default OFF). Prove the pipeline + get it good enough that later work builds on it.
- **Non-goals (explicitly deferred):** procedural gait/IK, limping (needs a future `Unit` stance field — additive later),
  reactions/impacts (Phase 1), GPU-skinning optimization (see §Skinning), the pre-render-to-sprite fallback.

## Hard constraints (do not violate)
1. **Render-only. No animation/model state on `Unit`.** Per-unit anim phase lives in a renderer-side dict keyed by
   `unit.id`. Putting it on `Unit` would enter the synced sim/digest — forbidden. This is what makes Phase 0 parallel to
   the physics arc and auto-skipped in headless ML training (training never constructs a `GameRenderer`).
2. **No sim / fixed-point / CUDA / determinism surface touched.** Pure render layer.
3. **Toggle-gated.** `RenderConfig.use_3d_units: bool = False`. When False, behavior is byte-for-byte the current sprite
   path. The 3D path is inert until flipped (e.g. a key toggle or config).
4. **Never `git add -A`.** Stage explicit paths. Asset files go under `assets/models/` (new dir) + a LICENSE note.

## The render seam (from recon, main-tree line refs)
- Unit draw loop: `renderer/game_renderer.py` `_draw_units_world(marines, zombies)` (~:532), called from `compose_world`
  (~:487), which runs **inside** the world render target (`WorldComposite.begin()/end()`, `renderer/world_composite.py`).
- Actual sprite blit: `renderer/overlays.py` `draw_unit(...)` (~:471), `draw_texture_pro` at ~:499. The 3D path is a
  **sibling** of this, selected by the toggle — do NOT rip out the sprite path.
- Coords: `renderer/coords.py` `tile_to_world_px(x_tile, wpt)`; unit center in world-px = `(unit.x + footprint/2)*wpt`,
  `(unit.y + footprint/2)*wpt`. `wpt = self.world.world_px_per_tile` (default 24).
- Unit state (`src/simulation/unit.py`): `x`,`y` (float tiles, top-left), `facing` (float radians, 0=E, CCW+),
  `alive` (property), `life_state`, `footprint`, `id`. **No velocity vector** — infer "moving" from per-frame (x,y)
  delta or non-empty `move_path`. Prone via `simulation.status.composed_flags(unit).is_prone`.
- Animation clock: reuse `self._anim_t0 = time.perf_counter()` (already drives the water shader, render-only,
  determinism-exempt). Wall-clock so it animates through pause.

## Design — `renderer/unit_model_renderer.py` (the future-proof seam)
A single cohesive module so every later upgrade (limp, GPU skinning, reactions) is a change *inside* it.

- `class UnitModelRenderer`:
  - `load(...)` — load the glTF model + its `ModelAnimation` clips once, guarded by a live GL context (mirror
    `UnitSprites.load()` at `game_renderer.py:250`). Store clip-name → index map. Hold the model scale/orientation
    correction (glTF up-axis vs Breach top-down) as constants determined during load-verify.
  - Per-unit state: `self._anim: dict[int, _UnitAnimState]` keyed by `unit.id` (phase/frame, last pos for motion infer,
    current clip). Prune entries for absent ids.
  - `select_clip(unit, moving) -> clip_name` — **data-driven state→clip map** (dict), the extension point for limp/
    wounded later. Phase 0 set: `dead`→death/last-frame, `firing`(if detectable, else skip)→fire, `moving`→walk/run,
    else→idle. Keep it a table, not if-chains.
  - `draw_units(units, wpt, clock, camera3d)` — `begin_mode_3d(camera3d)`; for each alive+visible unit: advance its
    clip via `UpdateModelAnimation(model, anim, frame)` (CPU skinning — see §Skinning), set transform (translate to
    world-px center, yaw = `facing`, scale to match footprint), `DrawModelEx`; draw a **blob-shadow** quad/circle under
    it (cheap, reads great top-down — no shadow maps). `end_mode_3d()`.
  - `unload()` — free model + animations.
- **Top-down Camera3D:** a fixed **orthographic** camera looking straight down (−Z/−Y to match world axes), framed to the
  same world-pixel extents as the RT so models land exactly on the 2D floor. Built once; drawn while the world RT is
  still bound (nest `begin_mode_3d` inside `WorldComposite.begin()/end()` at the unit slot).
  - **Depth gotcha (the one structural unknown):** `LoadRenderTexture` must give a usable depth buffer for 3D depth-test.
    Verify during load-verify (draw two overlapping models, confirm correct occlusion). If the RT lacks depth, either
    add a depth attachment or depth-sort models by world-Y and draw back-to-front (painter's) as a fallback.
- **Integration point:** in `game_renderer.py` `_draw_units_world` (or `compose_world` at the unit slot), branch on
  `self.cfg.use_3d_units`: True → `self.unit_models.draw_units(...)`; False → existing sprite path unchanged.
  Load models in the renderer ctor alongside `UnitSprites`, only when `use_3d_units` (don't pay load cost otherwise).

## Skinning (CPU now, GPU-swap seam)
- This pyray/raylib **6.0.1** binding exposes `UpdateModelAnimation` / `UpdateModelAnimationEx` (**CPU skinning**) but
  **not** `UpdateModelAnimationBoneMatrices` (the GPU helper). GPU shader locations (`SHADER_LOC_MATRIX_BONETRANSFORMS`,
  bone IDs/weights) *are* present.
- **Phase 0 uses CPU skinning** — zero shader work. Soft ceiling ~20 animated chars (lower in CPython). Fine for the
  prototype + eye-test. If the ceiling bites, the fix is a self-contained upgrade *inside* `UnitModelRenderer`: compute
  bone matrices + upload via `SetShaderValueV` to the bone-matrices uniform + a skinning vertex shader. **Deferred; keep
  the draw path swappable so this is an internal change.**

## Asset (Quaternius CC0)
- Primary: a **Quaternius CC0** low-poly rigged humanoid + Universal Animation Library (walk/idle/fire/death), glTF/GLB.
  CC0 = commercial-OK, attribution-optional, ML-OK. Place under `assets/models/marine/` with a `LICENSE.txt` noting
  source + CC0. Verify it is genuinely rigged (has `BoneInfo` + `ModelAnimation` clips) on load.
- **Fallback (only if a specific Quaternius GLB won't import cleanly — glTF-import bugs are a flagged risk):** use any
  known-good CC0 rigged glTF to prove the code path (e.g. a raylib example character), flag it clearly, and keep
  Quaternius as the art target — the rig seam is identical, so art swaps without code change. Do NOT block Phase 0 on one
  finicky file.

## Verification (before wiring into the live renderer)
1. **Load-verify script** (throwaway, in `scratchpad/` or `prototypes/`): load model + clips, print bone/clip counts,
   render one animated model into a RenderTexture via the top-down ortho Camera3D, save a PNG. Confirms: import works,
   animation advances, composites into an RT, depth/occlusion correct, facing→yaw + scale look right.
2. **Smoke-launch** the real game with `use_3d_units=True` on a small level: marines appear as animated 3D, turn with
   facing, walk when moving, sprite path unchanged when toggle off. Capture a screenshot.
3. **Regression:** with toggle OFF, the render output/behavior is unchanged (sprite path untouched).

## Gate
**HUMAN-TEST (feel):** Erik toggles `use_3d_units` on, plays, judges the look. **No merge until he blesses it.**
Autonomy is in execution; the merge is his call. Surface any surprise (import trouble, depth issue, scale/feel) rather
than plough ahead.
