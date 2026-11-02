"""UnitModelRenderer — render-only 3D skinned marines over the 2D world.

Phase 0 (docs/anim_phase0_impl_2026-07-20.md): a rigged glTF humanoid drawn on
top of the existing 2D top-down world, driven READ-ONLY off sim state, behind
``RenderConfig.use_3d_units`` (default ON since #33). No model/animation state ever lands
on ``Unit`` — per-unit animation phase lives here, keyed by ``unit.id`` — so this
never enters the synced sim/digest and is auto-skipped in headless ML training
(which never builds a GameRenderer).

Assets: the scripted characters (#33) -- each one skinned mesh generated through
``charkit/gameready.py`` on the Quaternius "Universal Animation Library" (CC0)
skeleton with its 46 clips converted to the character's own rest pose; one .glb
holds mesh, texture and clips. WHICH model a unit is drawn with is data:
``[render.unit_looks]`` in config.toml (players the space marine, zombies the
workers), dealt and kept per unit id by ``renderer/unit_looks.py`` -- render-
only, never on ``Unit``. All looks share the marine's scale and the one lit
shader. See each ``assets/models/<name>/LICENSE.txt``. The library's own mannequin
(``assets/models/marine/``) is no longer drawn: it is the skeleton and clip
SOURCE the game-ready step reads.

Skinning: CPU (``update_model_animation``). The shared model is re-skinned to
each unit's pose immediately before its ``DrawModelEx``; soft ceiling ~20
animated units in CPython. The bindings in use (a raylib 6.x build where this
was written, 5.5.0.4 on the home desktop) name a clip's frame count and frame
poses differently (read through :func:`anim_frame_count`, #80, and
``pose_skinner.anim_pose_fields``). GPU skinning IS available on 5.5 (each
mesh carries ``boneMatrices``, uploaded by ``DrawMesh`` to a shader that
declares them; measured in the design's §4.5), but nothing here uses it yet:
that is W13.2, a second ``pose_skinner.Skinner`` behind ``_draw_one``'s seam,
and it needs the lit and shadow vertex shaders to skin.

Pose layer (M13, #162; ``[render.pose] enabled``, default OFF): with it on,
each unit's pose is composed by ``renderer/pose_layer.py`` (sub-frame clips,
legs / upper / head layers over bone masks, crossfades) in ONE batch per look
per ``draw_units`` call, and ``pose_skinner.CpuInjectSkinner`` puts it on the
mesh. With it off nothing of it is built or imported, and ``_draw_one``
plays the look's own clip at an integer frame exactly as before. #163: the
sim's events reach the layer per unit (``queue_events``, each acted on
once): a launched shot holds the aim overlay and retriggers the recoil, a
hit plays the flinch additive; and each unit's travel relative to its
facing (the position delta) steers the aim-vs-travel twist, so the legs
walk along the travel while the torso keeps the facing.

Coordinate mapping: the top-down camera + world-px mapping now live in
``renderer/lit3d.py`` (the shared lit-3D-in-world-RT seam, extracted 2026-09
for the props & vegetation arc #60 P1) — see its module docstring for the
full calibration note (``make_camera``, ``_CAM_HEIGHT``, world-px axes,
up=(0,0,-1), depth-buffer occlusion). Unchanged here:
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

from config import CFG

from .lit3d import LightFieldCtx, bind_smoke_cover_map, make_camera
from .marine_shader import (MARINE_NORMAL_MAP_FILENAME, UNIT_NORMAL_Y_SIGN,
                            load_marine_shader)
from .unit_looks import (BODY_LIGHT_UNIT, ROLES, LookAssigner, all_look_names,
                         look_path, looks_table, unit_key, unit_shading)
from .unit_shadow import (NO_SHADOW, ShadowCast, ShadowPainter, UnitLight,
                          read_unit_light, shadow_cast, smooth_shadow,
                          unit_shadow_settings)

# ---------------------------------------------------------------------------
# Asset + tunables
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
# Which model each unit is drawn with is DATA: config.toml [render.unit_looks]
# (renderer/unit_looks.py). The space marine is the SCALE REFERENCE: every look
# is scaled by his native height, so he is drawn _SCALE_TILES_TALL tiles tall
# and every other look at its own true height relative to him.
_SCALE_REFERENCE = "space_marine"
_MODEL_PATH = look_path(_SCALE_REFERENCE)
# Tools only (charkit/preview_in_game.py --model): a model file that every
# unit is drawn with instead of the table. None in the game.
_MODEL_OVERRIDE: Optional[Path] = None

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

# How many tiles the SPACE MARINE's native height represents — the single feel
# knob for on-screen size; every look shares his scale (see _SCALE_REFERENCE).
# Uniform scale, so the top-down footprint scales with it.
_SCALE_TILES_TALL = 6.0
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
# Projected shadow (#33): the time constant of the per-unit low-pass on the
# shadow vector -- a turn of the light fades the shadow through faint rather
# than snapping it across the unit. Render feel, not a setting.
_SHADOW_SMOOTH_TAU_S = 0.12
# #114: the read of a unit with no light view (lighting off): dark.
_NO_LIGHT = UnitLight(0.0, 0.0, 0.0, 0.0)
# Prune a unit's anim state once it has gone unseen this many seconds.
_STALE_SECONDS = 1.0
# _CAM_HEIGHT and LightFieldCtx moved to renderer/lit3d.py (P1 extraction,
# #60); imported above and re-exported here for import-compat.

# The raylib bindings in use name a clip's frame count differently: raylib 5.5
# (the home desktop's pyray 5.5.0.4) ``ModelAnimation.frameCount``, raylib 6.x
# ``keyframeCount``. Read in this order, first present wins.
_FRAME_COUNT_FIELDS = ("frameCount", "keyframeCount")


def anim_frame_count(anim) -> int:
    """A clip's frame count under EITHER raylib binding -- the ONE read of it
    (#80: reading 6.x's ``keyframeCount`` crashed the 3D marines on 5.5).
    Raises AttributeError naming both fields when the struct has neither."""
    for name in _FRAME_COUNT_FIELDS:
        try:
            return int(getattr(anim, name))
        except AttributeError:
            continue
    raise AttributeError(f"ModelAnimation carries none of {_FRAME_COUNT_FIELDS}")


def pose_layer_enabled(cfg) -> bool:
    """``[render.pose] enabled`` -- the ONE switch of the pose layer (#162),
    read when the renderer loads. Raises ValueError when the section or the
    key is missing or not a bool (a typo must not silently pick a path).
    Everything else of the table is read by ``pose_layer.pose_settings``,
    only when this is true."""
    section = getattr(getattr(cfg, "render", None), "pose", None)
    if section is None:
        raise ValueError("config.toml has no [render.pose] section")
    value = getattr(section, "enabled", None)
    if not isinstance(value, bool):
        raise ValueError(f"[render.pose] enabled must be a bool, got {value!r}")
    return value


@dataclass
class _UnitAnimState:
    """Renderer-side per-unit animation state (NEVER on Unit)."""
    phase: float = 0.0           # fractional keyframe cursor
    clip: str = "idle"           # current logical clip key (CLIP_MAP key)
    last_x: float = 0.0
    last_y: float = 0.0
    last_seen: float = 0.0       # wall-clock of last draw (for pruning)
    initialised: bool = False
    shadow: Optional[ShadowCast] = None   # projected shadow, low-passed (#33)
    pose: Optional[object] = None         # pose_layer.PoseState (#162; pose layer on)


@dataclass
class _Look:
    """One loaded unit look: a skinned model, its clips, its own clip-name ->
    index map and native (bind-pose bounding-box) height. Renderer-side only."""
    name: str
    path: Path
    model: object
    anims: object                         # cffi array of ModelAnimation
    n_anims: int
    clip_index: Dict[str, int]
    native_height: float
    normal_tex: object = None             # P2 normal map (ROUGHNESS slot)


class UnitModelRenderer:
    """Owns the loaded unit looks (one per model named in
    ``[render.unit_looks]``), the ONE shared lit shader, which look each unit
    is drawn with (a ``unit_looks.LookAssigner``, keyed by unit id) and the
    per-unit anim state. Nothing here is ever written to a ``Unit``."""

    def __init__(self) -> None:
        self._looks: Dict[str, _Look] = {}     # loaded looks, in load order
        self._resolve: Dict[str, _Look] = {}   # every table name -> a loaded look
        self._assigner: Optional[LookAssigner] = None
        self._scale_height = 1.83              # the scale reference's height
        self._loaded = False
        self._anim: Dict[int, _UnitAnimState] = {}
        self._last_clock: Optional[float] = None
        self._dt = 0.0                   # the current frame's wall-clock step
        # P1 lit marine shader (set up in load() once the GL context exists),
        # SHARED by every look. None in the Patch-0 fallback path (flat per-
        # channel CPU RGB tint); when present, _draw_one lets the shader sample
        # the light field per-fragment and skips the CPU tint.
        self._shader = None
        self._marine_shader = None       # MarineShader wrapper (locs + setters)
        # #33 projected shadows: the flat shader + single-darkening draw.
        self._shadow_painter: Optional[ShadowPainter] = None
        # #162 pose layer + its skinner: built in load() only when
        # [render.pose] enabled; None = the clip player as before.
        self._pose = None
        self._skinner = None
        self._delivered: Dict[int, object] = {}   # the last event batch (#163)
        self._shot_event = self._hit_event = None

    @property
    def model(self):
        """The first loaded look's model (None before load) -- compat for
        callers that only ask whether a model is there."""
        first = next(iter(self._looks.values()), None)
        return first.model if first is not None else None

    # ------------------------------------------------------------------
    # Load / unload  (mirror UnitSprites.load(): needs a live GL context)
    # ------------------------------------------------------------------

    @staticmethod
    def _table() -> Dict[str, list]:
        """The role -> look-name table: ``[render.unit_looks]``, or every role
        drawn with ``_MODEL_OVERRIDE`` when a tool has set it."""
        if _MODEL_OVERRIDE is not None:
            name = Path(_MODEL_OVERRIDE).stem
            return {role: [name] for role in ROLES}
        return looks_table(CFG)

    @staticmethod
    def _path_of(name: str) -> Path:
        if _MODEL_OVERRIDE is not None:
            return Path(_MODEL_OVERRIDE)
        return look_path(name)

    def load(self) -> None:
        """Load every look the table names (model + all its clips) once. Must
        run after init_window (OpenGL context required), exactly like
        UnitSprites.load(). A look whose file is missing or fails to load is
        drawn as the first look that did load (said once, here); if none
        loads the renderer stays inert (draw_units no-ops -> sprites)."""
        if not rl.is_window_ready():
            print("[unit_model] WARN: no GL context; 3D units disabled")
            return
        try:
            table = self._table()
        except ValueError as exc:
            print(f"[unit_model] WARN: {exc}; 3D units disabled")
            return
        names = all_look_names(table)
        for name in names:
            look = self._load_look(name, self._path_of(name))
            if look is not None:
                self._looks[name] = look
        if not self._looks:
            print("[unit_model] WARN: no unit look loaded; 3D units disabled")
            return
        first = next(iter(self._looks.values()))
        for name in names:
            if name not in self._looks:
                print(f"[unit_model] WARN: look {name!r} did not load; its "
                      f"units are drawn as {first.name!r}")
            self._resolve[name] = self._looks.get(name, first)
        self._assigner = LookAssigner(table)
        self._scale_height = self._reference_height(first)
        self._loaded = True
        self._setup_marine_shader()
        self._shadow_painter = ShadowPainter(rl)
        self._setup_pose()

    def _setup_pose(self, skinner=None) -> None:
        """#162: build the pose layer over every loaded look -- only when
        ``[render.pose] enabled``; otherwise nothing of it is imported or
        built and the draw path is the clip player as before. A refused
        setting, a look whose skeleton no mask table fits, or a look whose
        raylib bone table differs from the pose layer's raises: the layer
        was asked for, so it never silently falls back. ``skinner`` defaults
        to the CPU-inject skinner (tests pass their own)."""
        if not pose_layer_enabled(CFG):
            return
        from .pose_layer import PoseConfigError, PoseLayer, pose_settings
        from .pose_skinner import CpuInjectSkinner
        # #163: the events the reactions read (render-side import, as in
        # GameRenderer.consume_events) and the hit causes checked against
        # the sim's closed cause vocabulary
        from simulation.events import CAUSE_NAMES, ShotFiredEvent, UnitHitEvent
        settings = pose_settings(CFG)
        if settings.hit is not None:
            unknown = sorted(set(settings.hit.causes) - set(CAUSE_NAMES))
            if unknown:
                raise PoseConfigError(f"[render.pose.hit] causes {unknown} are not "
                                      f"damage causes {CAUSE_NAMES}")
        self._shot_event, self._hit_event = ShotFiredEvent, UnitHitEvent
        pose = PoseLayer(settings)
        skinner = skinner if skinner is not None else CpuInjectSkinner(rl)
        for look in self._looks.values():
            rig = pose.add_look(look.name, look.path)
            skinner.prepare(look.name, look.model, look.anims[0],
                            rig.bank.skeleton.names)
            print(f"[unit_model] pose layer: {look.name} {rig.bank.n_clips} clips, "
                  f"{rig.bank.skeleton.n_bones} bones, masks "
                  f"[render.pose.skeletons.{rig.skeleton_table}]")
        for warning in pose.warnings:
            print(f"[unit_model] WARN: pose layer: {warning}")
        self._pose = pose
        self._skinner = skinner

    def queue_events(self, events: Sequence) -> None:
        """#162: hand the frame's sim events to the pose layer, one queue per
        unit (``event.unit_id``); the next draw acts on them (#163: a
        launched shot -> the fire reaction, a hit -> the flinch). A no-op
        with the pose layer off.

        The game hands over ``sim.tick_events`` EVERY frame, and that list
        is only cleared by the next sim step -- at 60 fps and 24 ticks/s
        most frames see the previous frame's list again. An event object
        delivered by the previous call is therefore not queued twice (it is
        the same object; the previous batch is held, so its ids cannot be
        reused by new events)."""
        if self._pose is None:
            return
        now = self._last_clock if self._last_clock is not None else 0.0
        seen = self._delivered
        self._delivered = {id(ev): ev for ev in events}
        for ev in events:
            if seen.get(id(ev)) is ev:
                continue
            uid = getattr(ev, "unit_id", None)
            if uid is not None:
                self._pose.queue_event(int(uid), ev, now)

    def _react(self, ps, uid: int) -> None:
        """#163: act on the unit's queued events (oldest first): a round that
        left its barrel fires, a hit flinches (the layer filters the
        cause). Other events are dropped."""
        for ev in self._pose.take_events(uid):
            if isinstance(ev, self._shot_event):
                if ev.launch:
                    self._pose.fire(ps)
            elif isinstance(ev, self._hit_event):
                self._pose.hit(ps, ev.damage, ev.source)

    def _load_look(self, name: str, path: Path) -> Optional[_Look]:
        """Load one look; None (with one warning) if missing, unloadable or
        not rigged."""
        if not path.is_file():
            print(f"[unit_model] WARN: model not found: {path}")
            return None
        spath = str(path.resolve())
        try:
            model = rl.load_model(spath)
            n_ptr = rl.ffi.new("int *", 0)
            anims = rl.load_model_animations(spath, n_ptr)
            n_anims = int(n_ptr[0])
            clip_index: Dict[str, int] = {}
            for i in range(n_anims):
                nm = anims[i].name
                clip = (rl.ffi.string(nm).decode("utf-8", "replace") if nm
                        else f"clip{i}")
                clip_index[clip] = i
            bb = rl.get_model_bounding_box(model)
            native_height = max(1e-3, bb.max.y - bb.min.y)
            rigged = (n_anims > 0
                      and any(model.meshes[i].boneCount > 0
                              for i in range(model.meshCount)))
            print(f"[unit_model] loaded {name}: {spath} clips={n_anims} "
                  f"native_height={native_height:.3f} rigged={rigged}")
            if not rigged:
                print(f"[unit_model] WARN: {name} is not rigged (no bones/clips)")
                if n_anims:
                    rl.unload_model_animations(anims, n_anims)
                rl.unload_model(model)
                return None
            return _Look(name=name, path=path, model=model, anims=anims,
                         n_anims=n_anims, clip_index=clip_index,
                         native_height=native_height)
        except Exception as exc:  # pragma: no cover - defensive, mirrors sprites
            print(f"[unit_model] WARN: could not load {name} ({spath}): {exc}")
            return None

    def _reference_height(self, first: _Look) -> float:
        """The ONE native height every look is scaled by: the space marine's
        (``_MODEL_PATH``), so he is drawn exactly ``_SCALE_TILES_TALL`` tiles
        tall as before #33 and every other look at its own true height
        relative to him. Measured from his file even when he is not a loaded
        look (a preview of another model); the first loaded look's own height
        if even that fails."""
        ref = self._looks.get(_SCALE_REFERENCE)
        height = None
        if ref is not None and ref.path.resolve() == _MODEL_PATH.resolve():
            height = ref.native_height
        else:
            try:
                if _MODEL_PATH.is_file():
                    m = rl.load_model(str(_MODEL_PATH.resolve()))
                    bb = rl.get_model_bounding_box(m)
                    height = max(1e-3, bb.max.y - bb.min.y)
                    rl.unload_model(m)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[unit_model] WARN: could not measure {_MODEL_PATH}: {exc}")
        if height is None:
            height = first.native_height
            print(f"[unit_model] WARN: scale reference {_SCALE_REFERENCE!r} "
                  f"unavailable; scaling by {first.name!r}")
        print(f"[unit_model] scale: {_SCALE_REFERENCE} native_height="
              f"{height:.3f} = {_SCALE_TILES_TALL} tiles; drawn tiles tall: "
              + ", ".join(f"{lk.name} {_SCALE_TILES_TALL * lk.native_height / height:.2f}"
                          for lk in self._looks.values()))
        return height

    @staticmethod
    def _live_materials(model):
        """Material indices the meshes use -- every one but 0 (raylib's glTF
        loader keeps its own default material there and no mesh uses it)."""
        return range(1, model.materialCount)

    def _setup_marine_shader(self) -> None:
        """P1: compile the ONE lit-marine shader and assign it to every look's
        mesh materials. Light-field textures are bound to the material MAP
        slots and per-frame uniforms pushed in draw_units. On a compile failure
        self._shader stays None so _draw_one falls back to the Patch-0 flat CPU
        RGB tint (graceful degrade, never a black unit).
        """
        try:
            ms = load_marine_shader()
            if ms.shader.id == 0:
                print("[unit_model] WARN: marine shader failed to compile; "
                      "falling back to flat RGB tint")
                self._opaque_albedos()
                return
            for look in self._looks.values():
                for mi in self._live_materials(look.model):
                    look.model.materials[mi].shader = ms.shader
                self._bind_normal_map(look)
            self._marine_shader = ms
            self._shader = ms.shader
            print(f"[unit_model] lit unit shader ready (id={ms.shader.id}, "
                  f"shared by {len(self._looks)} looks)")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[unit_model] WARN: marine shader setup failed: {exc}")
            self._shader = None
            self._marine_shader = None
            self._opaque_albedos()

    def _opaque_albedos(self) -> None:
        """Without the lit shader raylib's default one draws the albedo's
        alpha as coverage -- but since #33 that alpha is the GLOSS mask, so a
        matte texel would be see-through. Re-upload each look's albedo without
        its alpha channel (RGB only: sampled alpha reads 1), so the fallback
        path draws the model opaque like the lit one."""
        MM = rl.MaterialMapIndex
        for look in self._looks.values():
            for mi in self._live_materials(look.model):
                maps = look.model.materials[mi].maps
                old = maps[MM.MATERIAL_MAP_ALBEDO].texture
                if old.id <= 1:          # raylib's default 1x1 white
                    continue
                try:
                    img = rl.load_image_from_texture(old)
                    rl.image_format(img, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
                    new = rl.load_texture_from_image(img)
                    rl.unload_image(img)
                    if new.id == 0:
                        continue
                    rl.unload_texture(old)
                    maps[MM.MATERIAL_MAP_ALBEDO].texture = new
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[unit_model] WARN: {look.name}: could not drop the "
                          f"gloss alpha for the fallback path: {exc}")

    def _bind_normal_map(self, look: _Look) -> None:
        """P2: bind the look's normal map into the ROUGHNESS(3) material slot —
        a FREE slot (light textures own METALNESS=1 / NORMAL=2). Static, so bind
        ONCE here; draw_units only rewrites slots 1/2 per frame and never touches
        3. #33: the map is the one baked into the look's own .glb (raylib's
        glTF loader puts it in the NORMAL slot, which the light field takes
        over every frame, so it is moved here first and owned by the look).
        A .glb without one takes a marine_normal_PLACEHOLDER.png next to it;
        without either a 1x1 FLAT normal is bound (0.5,0.5,1 -> tangent +Z),
        never raylib's default WHITE (which would perturb N when the guard is
        on), so the path stays inert.
        """
        MM = rl.MaterialMapIndex
        path = look.path.parent / MARINE_NORMAL_MAP_FILENAME
        try:
            own = None
            for mi in self._live_materials(look.model):
                t = look.model.materials[mi].maps[MM.MATERIAL_MAP_NORMAL].texture
                if t.id > 1:             # 0 none, 1 raylib's default white
                    own = t
                    break
            if own is not None:
                tex = rl.Texture(own.id, own.width, own.height, own.mipmaps,
                                 own.format)
                # Mipmapped + trilinear: a unit is drawn a few dozen pixels
                # tall, and a point-sampled 1024 map would sparkle in the
                # highlight as the model moves.
                rl.gen_texture_mipmaps(tex)
                rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_TRILINEAR)
            elif path.is_file():
                tex = rl.load_texture(str(path.resolve()))
            else:
                fimg = rl.gen_image_color(1, 1, rl.Color(128, 128, 255, 255))
                tex = rl.load_texture_from_image(fimg)
                rl.unload_image(fimg)
            if tex.id == 0:
                print(f"[unit_model] WARN: {look.name} normal map failed to load")
                return
            if own is None:
                # Bilinear + wrap so a tiling detail-normal reads smoothly.
                rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
                rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_REPEAT)
            for mi in self._live_materials(look.model):
                look.model.materials[mi].maps[
                    MM.MATERIAL_MAP_ROUGHNESS].texture = tex
            look.normal_tex = tex
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[unit_model] WARN: {look.name} normal map bind failed: {exc}")
            look.normal_tex = None

    def unload(self) -> None:
        """Free every look (model + clips + normal map) and the shared shader,
        once (raylib's UnloadModel frees only material maps, never a shader or
        texture a material points at -- so sharing the shader is safe)."""
        for look in self._looks.values():
            if look.anims is not None and look.n_anims:
                rl.unload_model_animations(look.anims, look.n_anims)
            if look.normal_tex is not None:
                rl.unload_texture(look.normal_tex)
            rl.unload_model(look.model)
        if self._marine_shader is not None:
            rl.unload_shader(self._marine_shader.shader)
        if self._shadow_painter is not None:
            self._shadow_painter.unload()
            self._shadow_painter = None
        if self._skinner is not None:
            self._skinner.release()
        self._pose = None
        self._skinner = None
        self._looks.clear()
        self._resolve.clear()
        self._assigner = None
        self._anim.clear()
        self._loaded = False
        self._shader = None
        self._marine_shader = None

    @property
    def ready(self) -> bool:
        return self._loaded

    # Camera: moved to renderer/lit3d.py::make_camera (P1 extraction, #60) —
    # bound here as a staticmethod so `UnitModelRenderer.make_camera(...)`
    # keeps working unmodified for every existing caller.
    make_camera = staticmethod(make_camera)

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
                   light_rgb_fn: Optional[Callable[[object], tuple]] = None,
                   light_ctx: Optional[LightFieldCtx] = None,
                   open_mode_3d: bool = True,
                   light_view=None, light_exposure: float = 1.0) -> None:
        """Draw every alive unit as an animated 3D body inside the world RT.

        Nests ``begin_mode_3d`` in the already-open world RT. Per unit: infer
        motion from the (x, y) delta / move_path, pick + advance its clip
        (CPU skin), draw a blob shadow then the model at the unit's world-px
        centre with yaw = facing and a footprint-matched scale.

        ``open_mode_3d=False`` skips the ``begin_mode_3d``/``end_mode_3d``
        pair (props & vegetation arc #60 P3, design §4.3 F23/F25): the
        caller has already opened ONE shared 3D pass for units + props (one
        batch flush, one shared depth buffer) — see
        ``GameRenderer._draw_units_world``.

        ``clock`` is the renderer's wall-clock (self._anim_t0-relative is fine);
        the per-unit phase advances by the real delta so it animates through
        pause — render-only, determinism-irrelevant. ``base_tint`` multiplies
        the draw colour; the game passes none since #33 (Erik's ruling: teams
        are told apart by model and texture, not tint).

        ``light_rgb_fn(unit) -> (r, g, b)`` returns a per-channel brightness
        multiplier in [0, 1] sampled from the SAME baked RGB light field the
        ship is lit by (ambient floor + incoming light colour at the unit's
        foot tile), so a red lamp reddens the marine and an unlit room darkens
        it — colour + occlusion parity with the ship, unlike the old max-
        collapsed grey scalar. It is the FLAT fallback / Patch-0 path.

        ``light_ctx`` (P1) carries the ship's ``light_tex_a`` / ``light_tex_b``
        + world dims + ambient/gain so the lit marine shader samples the field
        PER-FRAGMENT on the marine's real mesh normals (half-Lambert + rim +
        its own grazing key). When the shader is live and light_ctx is given,
        the CPU tint above is skipped (the field sample replaces it).

        ``light_view`` (#33) is the light accessor's ``LightView`` (the one
        ``LightingPass`` last consumed) and ``light_exposure`` the factor that
        takes its light units to the ship shader's (sweep_gain x master_gain).
        With them, and ``[render.unit_shadow] enabled``, each unit casts a
        sharp projected shadow (``renderer/unit_shadow.py``): its skinned model
        once more, flattened onto the floor away from the light, faded by how
        bright and how one-sided the light around it is. None = no shadows.

        #114: with ``[render.unit_shading] body_light = "unit"`` the SAME
        per-unit read also lights the unit's whole body (the lit shader's
        per-unit uniforms, set just before its draw), instead of each
        fragment reading the field tile under it -- one read per unit per
        frame serves both. None = the body gets the ambient floor only.

        No-op (leaving the sprite path's world untouched) if the model failed
        to load — the toggle can be on with the asset missing and nothing breaks.
        """
        if not self._loaded or not self._looks:
            return
        dt = self._frame_dt(clock)

        # ONE scale for every look (the marine's), so heights stay true.
        scale = (_SCALE_TILES_TALL * wpt) / self._scale_height
        shadow_r = _SHADOW_RADIUS_FRAC * (3.0 * wpt)  # footprint side ~3 tiles
        alive = [u for u in units if getattr(u, "alive", True)]
        names = self._assigner.looks_for(alive)
        # #33: [render.unit_shading], read live (Ctrl+R retunes it); a missing
        # or mistyped key raises here rather than drawing a silent default.
        shading = unit_shading(CFG)
        # #33: [render.unit_shadow], read live too (raises naming a bad key).
        shadow_cfg = unit_shadow_settings(CFG)
        shadows_on = (shadow_cfg.enabled and light_view is not None
                      and self._shadow_painter is not None
                      and self._shadow_painter.ready)

        # P1: wire the ship's light field into the marine material + push the
        # per-frame scalar uniforms ONCE (SetShaderValue self-enables the
        # program, so this is safe before begin_mode_3d). Textures go in the
        # material MAP slots (METALNESS -> texture1, NORMAL -> texture2) so
        # DrawMesh auto-binds them every draw and never clobbers the units.
        if self._shader is not None and self._marine_shader is not None \
                and light_ctx is not None:
            MM = rl.MaterialMapIndex
            mats = []
            for look in self._looks.values():
                for mi in self._live_materials(look.model):
                    mat = look.model.materials[mi]
                    mat.maps[MM.MATERIAL_MAP_METALNESS].texture = light_ctx.tex_a
                    mat.maps[MM.MATERIAL_MAP_NORMAL].texture = light_ctx.tex_b
                    mats.append(mat)
            # Graphics G2: the smoke's cast shadow (lit3d's one GLSL copy).
            bind_smoke_cover_map(self._shader, mats, light_ctx)
            self._marine_shader.set_frame_uniforms(
                light_ctx.ambient, light_ctx.light_gain,
                light_ctx.world_px_w, light_ctx.world_px_h,
                normal_y_sign=UNIT_NORMAL_Y_SIGN * light_ctx.normal_y_sign,
                view_dir=(0.0, 1.0, 0.0))
            # #33: rim tint, the gloss highlight and each look's own normal
            # map, all from [render.unit_shading].
            ms = self._marine_shader
            ms.set_rim_albedo(shading.rim_albedo)
            ms.set_gloss(shading.gloss_strength, shading.gloss_shininess)
            ms.set_use_normal(shading.normal_map)
            ms.set_normal_strength(shading.normal_strength)
            ms.set_light_shape(shading.light_height, shading.wrap)   # #114
        # #114: the body lit from the unit's own light read (or, "field",
        # from the tile under each fragment, the A/B).
        body_from_unit = (shading.body_light == BODY_LIGHT_UNIT
                          and self._marine_shader is not None
                          and light_ctx is not None)
        if self._marine_shader is not None:
            self._marine_shader.set_body_light(body_from_unit)

        # #162: with the pose layer on, every unit's pose is composed first,
        # one batch per look; None = the clip player inside _draw_one.
        pose_rows = (self._pose_prepass(alive, names, dt, clock)
                     if self._pose is not None else None)

        if open_mode_3d:
            rl.begin_mode_3d(camera3d)
        try:
            # Phase 0 parity: dead units skipped (sprite path).
            for i, (u, name) in enumerate(zip(alive, names)):
                cast = NO_SHADOW
                # ONE light read per unit, for its shadow and its body.
                light = _NO_LIGHT
                if light_view is not None and (shadows_on or body_from_unit):
                    fp = int(getattr(u, "footprint", 3))
                    light = read_unit_light(light_view, int(u.x), int(u.y), fp,
                                            light_exposure)
                if shadows_on:
                    cast = shadow_cast(light, shadow_cfg)
                if body_from_unit:
                    self._marine_shader.set_unit_light(light)
                self._draw_one(u, self._resolve[name], wpt, dt, clock, scale,
                               shadow_r if shading.blob_shadow else 0.0,
                               base_tint, light_rgb_fn,
                               cast if shadows_on else None,
                               pose_row=None if pose_rows is None else pose_rows[i])
        finally:
            if open_mode_3d:
                rl.end_mode_3d()

        self._prune(clock)

    def _pose_prepass(self, alive: Sequence, names: Sequence[str], dt: float,
                      clock: float) -> list:
        """#162: per unit (read-only): infer motion, pick the state, drive its
        PoseState; then ONE batched compose per look. Returns each unit's
        global-TRS row, in ``alive`` order. #163: before driving, the unit's
        queued events become reactions (fire, flinch) and its travel this
        frame, relative to its facing, steers the aim-vs-travel twist."""
        groups: Dict[str, list] = {}
        for i, (u, name) in enumerate(zip(alive, names)):
            look = self._resolve[name]
            st = self._state_for(u)
            travel = self._relative_travel(u, st)
            moving = self._infer_motion(u, st, clock)
            state = self.select_clip(u, moving)
            if st.pose is None or st.pose.look != look.name:
                st.pose = self._pose.new_state(look.name, state)
            self._react(st.pose, unit_key(u))
            self._pose.steer(st.pose, travel, moving, dt)
            self._pose.drive(st.pose, state, dt)
            groups.setdefault(look.name, []).append((i, st.pose))
        rows = [None] * len(alive)
        for look_name, members in groups.items():
            batch = self._pose.compose(look_name, [ps for _, ps in members])   # + the layer's modifiers
            for j, (i, _) in enumerate(members):
                rows[i] = batch.global_trs[j]
        return rows

    def _state_for(self, u) -> _UnitAnimState:
        """The unit's renderer-side anim state, created on first sight."""
        uid = unit_key(u)
        st = self._anim.get(uid)
        if st is None:
            st = _UnitAnimState(last_x=float(u.x), last_y=float(u.y))
            self._anim[uid] = st
        return st

    @staticmethod
    def _relative_travel(u, st: _UnitAnimState) -> Optional[float]:
        """#163: the direction the unit moved since the last draw RELATIVE to
        its facing, as the model-space yaw the lower body would turn by
        (radians, in (-pi, pi]); None below the motion threshold. Read
        before ``_infer_motion`` moves the remembered position on. Facing
        and travel share the sim's convention (0 = +x, counter-clockwise,
        world y down), so the travel angle is ``atan2(-dy, dx)``."""
        dx = float(u.x) - st.last_x
        dy = float(u.y) - st.last_y
        if abs(dx) + abs(dy) <= _MOVE_EPS_TILES:
            return None
        facing = float(getattr(u, "facing", math.pi / 2.0))
        rel = math.atan2(-dy, dx) - facing
        rel = (rel + math.pi) % (2.0 * math.pi) - math.pi
        return _YAW_SIGN * rel

    @staticmethod
    def _infer_motion(u, st: _UnitAnimState, clock: float) -> bool:
        """Motion inference (no velocity field on Unit): moved since the last
        draw, or a non-empty move_path. Updates the remembered position."""
        moved = (abs(float(u.x) - st.last_x) + abs(float(u.y) - st.last_y)
                 > _MOVE_EPS_TILES)
        has_path = bool(getattr(u, "move_path", None))
        st.last_x, st.last_y = float(u.x), float(u.y)
        st.last_seen = clock
        return moved or has_path

    def _frame_dt(self, clock: float) -> float:
        """The wall-clock step the animation advances by in the frame drawn
        at ``clock``. Every draw_units call of ONE frame (the game makes two:
        players, then zombies, with the same clock) gets the same step; it
        changes only when the clock moves on. (Before this, the second call
        of a frame always got 0, so the zombies never animated.)"""
        if self._last_clock is None:
            self._dt = 0.0
        elif clock != self._last_clock:
            self._dt = max(0.0, clock - self._last_clock)
        self._last_clock = clock
        return self._dt

    def _draw_one(self, u, look: _Look, wpt: float, dt: float, clock: float,
                  scale: float, shadow_r: float, base_tint,
                  light_rgb_fn, shadow: Optional[ShadowCast] = None,
                  pose_row=None) -> None:
        """Draw a single unit with its look. THE SWAP SEAM: a future GPU-
        skinning path replaces the skinning call + DrawModelEx pair here
        without touching anything else. ``shadow`` (#33) is this frame's
        projected-shadow target (None = projected shadows off); it is low-
        passed per unit and drawn from the SAME skinned pose as the body.
        ``pose_row`` (#162) is the unit's composed global pose (pose layer
        on, motion already inferred by ``_pose_prepass``); None = the clip
        player below, exactly as before the pose layer."""
        st = self._state_for(u)

        if pose_row is not None:
            self._skinner.apply(look.name, look.model, pose_row)   # #162
        else:
            moving = self._infer_motion(u, st, clock)
            # --- clip selection + wall-clock advance --------------------
            clip = self.select_clip(u, moving)
            if clip != st.clip:
                st.clip = clip
                st.phase = 0.0
            anim_idx = look.clip_index.get(CLIP_MAP.get(clip, ""),
                                           look.clip_index.get(CLIP_MAP["idle"], 0))
            anim = look.anims[anim_idx]
            n_keys = max(1, anim_frame_count(anim))
            st.phase += dt * _ANIM_FPS
            frame = int(st.phase) % n_keys
            rl.update_model_animation(look.model, anim, frame)  # CPU skin (swap seam)

        # --- transform: world-px centre, yaw = facing, matched scale ----
        fp = float(getattr(u, "footprint", 3))
        cx = (float(u.x) + fp / 2.0) * wpt      # world px, matches the sprite path
        cy = (float(u.y) + fp / 2.0) * wpt
        facing = float(getattr(u, "facing", math.pi / 2.0))
        yaw = _YAW_SIGN * math.degrees(facing) + _YAW_OFFSET_DEG

        # Projected shadow (#33) first, from the pose just skinned (no second
        # skinning). Its fixed far depth keeps it off every body, whichever
        # unit is drawn first, and darkens each floor pixel once.
        if shadow is None:
            st.shadow = None
        else:
            st.shadow = smooth_shadow(st.shadow, shadow, dt, _SHADOW_SMOOTH_TAU_S)
            if st.shadow.alpha > 0.0:
                self._shadow_painter.draw(
                    look.model, self._live_materials(look.model), st.shadow,
                    rl.Vector3(cx, 0.0, cy), yaw, scale)

        # Blob shadow first (on the floor, under the model — reads great
        # top-down, no shadow maps). A thin filled disc via a short cylinder.
        # shadow_r 0 = [render.unit_shading] blob_shadow off (#33).
        if shadow_r > 0.0:
            sr, sg, sb, sa = _SHADOW_COLOR
            rl.draw_cylinder(rl.Vector3(cx, 0.5, cy), shadow_r, shadow_r, 1.0,
                             16, rl.Color(sr, sg, sb, sa))

        # Tint: base colour (white unless a caller passes one), per-CHANNEL modulated by the local baked RGB
        # light (Patch 0 — colour + occlusion parity with the ship: a red lamp
        # reddens the marine, an unlit room darkens it). Replaces the old flat
        # grey scalar multiply. When the P1 lit shader is active it samples the
        # field per-fragment, so this CPU tint is the FALLBACK: skipped so the
        # shader owns the lighting (else double-darkening).
        br, bg, bb2, ba = base_tint
        if self._shader is None and light_rgb_fn is not None:
            mr, mg, mb = light_rgb_fn(u)
            br = int(br * max(0.0, min(1.0, mr)))
            bg = int(bg * max(0.0, min(1.0, mg)))
            bb2 = int(bb2 * max(0.0, min(1.0, mb)))
        rl.draw_model_ex(look.model, rl.Vector3(cx, 0.0, cy),
                         rl.Vector3(0.0, 1.0, 0.0), yaw,
                         rl.Vector3(scale, scale, scale),
                         rl.Color(br, bg, bb2, ba))

    def _prune(self, clock: float) -> None:
        """Drop anim state for units unseen for _STALE_SECONDS (dead/despawned)."""
        stale = [uid for uid, st in self._anim.items()
                 if clock - st.last_seen > _STALE_SECONDS]
        for uid in stale:
            del self._anim[uid]
        if self._pose is not None:
            self._pose.prune_events(self._anim, clock, _STALE_SECONDS)


__all__ = ["UnitModelRenderer", "CLIP_MAP", "anim_frame_count",
           "pose_layer_enabled"]
