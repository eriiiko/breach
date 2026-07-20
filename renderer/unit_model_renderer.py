"""UnitModelRenderer — render-only 3D skinned marines over the 2D world.

Phase 0 (docs/anim_phase0_impl_2026-07-20.md): a rigged glTF humanoid drawn on
top of the existing 2D top-down world, driven READ-ONLY off sim state, behind
``RenderConfig.use_3d_units`` (default OFF). No model/animation state ever lands
on ``Unit`` — per-unit animation phase lives here, keyed by ``unit.id`` — so this
never enters the synced sim/digest and is auto-skipped in headless ML training
(which never builds a GameRenderer).

Asset: Quaternius "Universal Animation Library" (CC0), a rigged character + a
portion of the animation set, one glTF holding the mesh and all 46 clips. See
``assets/models/marine/LICENSE.txt``.

Skinning: CPU (``update_model_animation``) — this pyray/raylib 6.1-dev binding
exposes no ``UpdateModelAnimationBoneMatrices`` GPU helper. The shared model is
re-skinned to each unit's pose immediately before its ``DrawModelEx``; soft
ceiling ~20 animated units in CPython. The GPU-skinning upgrade (compute bone
matrices + a skinning vertex shader) is a self-contained change INSIDE this
module — ``_draw_one`` is the swap seam — so nothing outside changes.

Coordinate mapping (calibrated in prototypes/scratchpad):
  * The world render target is ``world_px_w × world_px_h`` world-pixels, drawn
    top-left origin, y-down. A top-down ORTHOGRAPHIC ``Camera3D`` measured in
    world-pixels maps 3D X = x_wpx, 3D Z = y_wpx, 3D Y = height-up, with
    ``fovy = world_px_h`` and camera up = (0, 0, -1). Verified: 3D primitives
    land exactly on the same-coordinate 2D draws (calib_camera.py).
  * The RT's own depth buffer occludes correctly (verified: a near model drawn
    FIRST fully occludes a far model drawn SECOND), so NO world-Y painter's
    fallback is needed.
  * Facing→yaw: the model's forward at yaw 0 points +Z; the sim facing is
    ``(cosθ, -sinθ)`` in world (x, y-down), i.e. (X, Z), giving
    ``yaw_deg = degrees(facing) + 90`` (``_YAW_OFFSET_DEG``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pyray as rl

# ---------------------------------------------------------------------------
# Asset + tunables
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_MODEL_PATH = (_REPO_ROOT / "assets" / "models" / "marine"
               / "AnimationLibrary_Godot_Standard.gltf")

# Data-driven state -> clip-name map (the extension point for limp/wounded/…).
# Keep it a TABLE, not if-chains: future stances add a key + a clip, nothing
# else. Clip names are the Quaternius Universal Animation Library names present
# in the asset (verified at load — a missing name falls back to idle).
CLIP_MAP: Dict[str, str] = {
    "idle": "Idle_Loop",
    "walk": "Walk_Loop",
    "fire": "Pistol_Shoot",     # dormant in Phase 0 (firing not inferred yet)
    "dead": "Death01",          # dormant in Phase 0 (dead units are skipped)
}

# How many tiles the model's native height represents — the single feel knob
# for on-screen size. Uniform scale, so the top-down footprint scales with it.
_SCALE_TILES_TALL = 3.0
# Keyframe playback rate for the wall-clock animation advance (clips are ~30fps).
_ANIM_FPS = 30.0
# Facing(rad) -> yaw(deg) about the vertical axis. Calibrated: model forward is
# +Z at yaw 0; world facing dir is (cosθ, -sinθ) in (X, Z).
_YAW_OFFSET_DEG = 90.0
_YAW_SIGN = 1.0
# Motion inference: moved at least this many tiles since last frame => "walking"
# (belt-and-braces with a non-empty move_path). No velocity field on Unit.
_MOVE_EPS_TILES = 0.01
# Blob shadow: radius as a fraction of the footprint, and its RGBA.
_SHADOW_RADIUS_FRAC = 0.32       # of footprint side, in world px
_SHADOW_COLOR = (0, 0, 0, 90)
# Prune a unit's anim state once it has gone unseen this many seconds.
_STALE_SECONDS = 1.0
# Top-down camera height above the floor, in world px. Ortho => this does NOT
# affect on-screen size, only near/far framing. It MUST stay below raylib's
# orthographic far-clip (empirically < ~5000 in this build), so it CANNOT scale
# with world size: a tall level (RT up to 5760 px) pushed the old
# max(w,h)*2 height past the far plane and culled every model (the "press M and
# everything vanishes" bug). 500 clears the tallest model (~3*wpt) with margin
# and is safely inside the far plane at every level size (verified 2400x5760).
_CAM_HEIGHT = 500.0


@dataclass
class _UnitAnimState:
    """Renderer-side per-unit animation state (NEVER on Unit)."""
    phase: float = 0.0           # fractional keyframe cursor
    clip: str = "idle"           # current logical clip key (CLIP_MAP key)
    last_x: float = 0.0
    last_y: float = 0.0
    last_seen: float = 0.0       # wall-clock of last draw (for pruning)
    initialised: bool = False


class UnitModelRenderer:
    """Owns the shared skinned model, its clips, and per-unit anim state."""

    def __init__(self) -> None:
        self.model: Optional[rl.Model] = None
        self._anims = None                    # cffi array of ModelAnimation
        self._n_anims = 0
        self._clip_index: Dict[str, int] = {}  # clip-name -> anim index
        self._native_height = 1.83            # overwritten at load (bbox)
        self._loaded = False
        self._anim: Dict[int, _UnitAnimState] = {}
        self._last_clock: Optional[float] = None

    # ------------------------------------------------------------------
    # Load / unload  (mirror UnitSprites.load(): needs a live GL context)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the rigged model + all animation clips once. Must run after
        init_window (OpenGL context required), exactly like UnitSprites.load().
        Failures are logged and leave the renderer inert (draw_units no-ops)."""
        if not rl.is_window_ready():
            print("[unit_model] WARN: no GL context; 3D units disabled")
            return
        path = str(_MODEL_PATH.resolve())
        if not _MODEL_PATH.is_file():
            print(f"[unit_model] WARN: model not found: {path}; 3D units disabled")
            return
        try:
            self.model = rl.load_model(path)
            n_ptr = rl.ffi.new("int *", 0)
            self._anims = rl.load_model_animations(path, n_ptr)
            self._n_anims = int(n_ptr[0])
            # Build the clip-name -> index map.
            for i in range(self._n_anims):
                nm = self._anims[i].name
                name = rl.ffi.string(nm).decode("utf-8", "replace") if nm else f"clip{i}"
                self._clip_index[name] = i
            # Native height for the size scale (top-down ortho => uniform scale).
            bb = rl.get_model_bounding_box(self.model)
            self._native_height = max(1e-3, bb.max.y - bb.min.y)
            rigged = (self._n_anims > 0
                      and any(self.model.meshes[i].boneCount > 0
                              for i in range(self.model.meshCount)))
            if not rigged:
                print("[unit_model] WARN: model is not rigged (no bones/clips)")
            self._loaded = rigged
            print(f"[unit_model] loaded {path}: clips={self._n_anims} "
                  f"native_height={self._native_height:.3f} rigged={rigged}")
        except Exception as exc:  # pragma: no cover - defensive, mirrors sprites
            print(f"[unit_model] WARN: could not load model: {exc}")
            self._loaded = False

    def unload(self) -> None:
        """Free the model + animation clips."""
        if self._anims is not None and self._n_anims:
            rl.unload_model_animations(self._anims, self._n_anims)
        if self.model is not None:
            rl.unload_model(self.model)
        self.model = None
        self._anims = None
        self._n_anims = 0
        self._clip_index.clear()
        self._anim.clear()
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Camera (owned here so the top-down mapping math lives in one place)
    # ------------------------------------------------------------------

    @staticmethod
    def make_camera(world_px_w: int, world_px_h: int) -> rl.Camera3D:
        """Top-down orthographic Camera3D framed to the world RT in world-px.

        3D X = x_wpx, 3D Z = y_wpx, 3D Y = up. ``fovy = world_px_h`` makes the
        ortho view span exactly the RT (aspect = w/h fills the width), so 3D
        world-px coords land on the same texels as the 2D world-px draws.
        Camera up = (0,0,-1) so world +Z (screen-down, i.e. y-down) reads down.
        """
        cam = rl.Camera3D()
        cx, cy = world_px_w / 2.0, world_px_h / 2.0
        # Fixed height (NOT world-size-scaled): ortho => distance doesn't change
        # on-screen size, only near/far framing, and it must stay under raylib's
        # ortho far-clip. See _CAM_HEIGHT (the old max(w,h)*2 culled everything on
        # tall levels).
        cam.position = rl.Vector3(cx, _CAM_HEIGHT, cy)
        cam.target = rl.Vector3(cx, 0.0, cy)
        cam.up = rl.Vector3(0.0, 0.0, -1.0)
        cam.fovy = float(world_px_h)
        cam.projection = rl.CameraProjection.CAMERA_ORTHOGRAPHIC
        return cam

    # ------------------------------------------------------------------
    # Clip selection  (data-driven; the extension point)
    # ------------------------------------------------------------------

    def select_clip(self, unit, moving: bool) -> str:
        """Return a CLIP_MAP key for *unit*. Table-driven, priority top-down.

        Phase 0: dead (dormant — dead units are skipped in draw), firing
        (dormant — not inferred yet), moving -> walk, else idle. Future stances
        (limp, wounded, crouch) add a branch + a CLIP_MAP entry here and a clip
        name to the asset; nothing outside this method changes.
        """
        if not getattr(unit, "alive", True):
            return "dead"
        if moving:
            return "walk"
        return "idle"

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw_units(self, units: Sequence, wpt: float, clock: float,
                   camera3d: rl.Camera3D,
                   base_tint=(255, 255, 255, 255),
                   light_fn: Optional[Callable[[object], float]] = None) -> None:
        """Draw every alive unit as an animated 3D body inside the world RT.

        Nests ``begin_mode_3d`` in the already-open world RT. Per unit: infer
        motion from the (x, y) delta / move_path, pick + advance its clip
        (CPU skin), draw a blob shadow then the model at the unit's world-px
        centre with yaw = facing and a footprint-matched scale.

        ``clock`` is the renderer's wall-clock (self._anim_t0-relative is fine);
        the per-unit phase advances by the real delta so it animates through
        pause — render-only, determinism-irrelevant. ``base_tint`` colours the
        group (marines vs zombies). ``light_fn(unit) -> float`` optionally
        modulates brightness to match the sprite path's local-light tint.

        No-op (leaving the sprite path's world untouched) if the model failed
        to load — the toggle can be on with the asset missing and nothing breaks.
        """
        if not self._loaded or self.model is None:
            return
        dt = 0.0 if self._last_clock is None else max(0.0, clock - self._last_clock)
        self._last_clock = clock

        scale = (_SCALE_TILES_TALL * wpt) / self._native_height
        shadow_r = _SHADOW_RADIUS_FRAC * (3.0 * wpt)  # footprint side ~3 tiles

        rl.begin_mode_3d(camera3d)
        try:
            for u in units:
                if not getattr(u, "alive", True):
                    continue  # Phase 0 parity: dead units skipped (sprite path)
                self._draw_one(u, wpt, dt, clock, scale, shadow_r,
                               base_tint, light_fn)
        finally:
            rl.end_mode_3d()

        self._prune(clock)

    def _draw_one(self, u, wpt: float, dt: float, clock: float, scale: float,
                  shadow_r: float, base_tint, light_fn) -> None:
        """Draw a single unit. THE SWAP SEAM: a future GPU-skinning path
        replaces the update_model_animation + DrawModelEx pair here without
        touching anything else."""
        uid = int(getattr(u, "id", id(u)))
        st = self._anim.get(uid)
        if st is None:
            st = _UnitAnimState(last_x=float(u.x), last_y=float(u.y))
            self._anim[uid] = st

        # --- motion inference (no velocity field on Unit) ---------------
        moved = (abs(float(u.x) - st.last_x) + abs(float(u.y) - st.last_y)
                 > _MOVE_EPS_TILES)
        has_path = bool(getattr(u, "move_path", None))
        moving = moved or has_path
        st.last_x, st.last_y = float(u.x), float(u.y)
        st.last_seen = clock

        # --- clip selection + wall-clock advance ------------------------
        clip = self.select_clip(u, moving)
        if clip != st.clip:
            st.clip = clip
            st.phase = 0.0
        anim_idx = self._clip_index.get(CLIP_MAP.get(clip, ""),
                                        self._clip_index.get(CLIP_MAP["idle"], 0))
        anim = self._anims[anim_idx]
        n_keys = max(1, anim.keyframeCount)
        st.phase += dt * _ANIM_FPS
        frame = int(st.phase) % n_keys
        rl.update_model_animation(self.model, anim, frame)  # CPU skin (swap seam)

        # --- transform: world-px centre, yaw = facing, matched scale ----
        fp = float(getattr(u, "footprint", 3))
        cx = (float(u.x) + fp / 2.0) * wpt      # world px, matches the sprite path
        cy = (float(u.y) + fp / 2.0) * wpt
        facing = float(getattr(u, "facing", math.pi / 2.0))
        yaw = _YAW_SIGN * math.degrees(facing) + _YAW_OFFSET_DEG

        # Blob shadow first (on the floor, under the model — reads great
        # top-down, no shadow maps). A thin filled disc via a short cylinder.
        sr, sg, sb, sa = _SHADOW_COLOR
        rl.draw_cylinder(rl.Vector3(cx, 0.5, cy), shadow_r, shadow_r, 1.0, 16,
                         rl.Color(sr, sg, sb, sa))

        # Tint: group base colour, optionally dimmed by local light (parity with
        # the sprite path, which multiplies the sprite by light_at(unit)).
        br, bg, bb2, ba = base_tint
        if light_fn is not None:
            L = max(0.0, min(1.0, float(light_fn(u))))
            br, bg, bb2 = int(br * L), int(bg * L), int(bb2 * L)
        rl.draw_model_ex(self.model, rl.Vector3(cx, 0.0, cy),
                         rl.Vector3(0.0, 1.0, 0.0), yaw,
                         rl.Vector3(scale, scale, scale),
                         rl.Color(br, bg, bb2, ba))

    def _prune(self, clock: float) -> None:
        """Drop anim state for units unseen for _STALE_SECONDS (dead/despawned)."""
        stale = [uid for uid, st in self._anim.items()
                 if clock - st.last_seen > _STALE_SECONDS]
        for uid in stale:
            del self._anim[uid]


__all__ = ["UnitModelRenderer", "CLIP_MAP"]
