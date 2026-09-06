"""static_props — THE path that draws placed 3D props in the world RT.

One ``StaticPropRenderer`` owns:

  * a MODEL CACHE keyed by the look a prop asks for — many placed props that
    share ``(generator, seed, palette, style, decor, decor_density, height)``
    share ONE uploaded model;
  * the PROP SHADER, which samples the ship's baked light field through the
    shared ``renderer/lit3d.py`` seam (``_COMMON_GLSL`` +
    ``_FIELD_SAMPLE_GLSL``, the exact blocks the marine uses) and takes its
    albedo from VERTEX COLORS — no albedo texture, no normal map, no skinning;
  * the draw call, shaped like ``UnitModelRenderer.draw_units``: it is called
    with the world RT already open and nests (or joins) the top-down
    ``begin_mode_3d`` built from ``lit3d.make_camera``.

No prop render state ever lands on a sim entity (digest!), and nothing here
imports ``simulation``.

MESH-OWNERSHIP CONTRACT (design §2, critique F22)
-------------------------------------------------
raylib's ``UnloadMesh`` calls ``RL_FREE`` on ``mesh.vertices`` /
``mesh.normals`` / ``mesh.colors``. The prop_spike prototype pointed those at
numpy buffers via ``ffi.from_buffer`` — which means unloading a model would
hand a numpy-owned pointer to the C allocator: heap corruption. This module
therefore COPIES every vertex array into raylib-owned memory (``rl.mem_alloc``
+ ``ffi.memmove``) before ``upload_mesh``, so ``unload()`` is safe and the
numpy arrays are free to be garbage-collected immediately after upload.

Second ownership trap, same family: raylib's ``UnloadMaterial`` unloads the
material's shader when it is not the default one. With ONE shared prop shader
across N cached models, the first ``unload_model`` would destroy it and the
rest would double-free. ``unload()`` therefore restores raylib's DEFAULT
shader onto every material first, then unloads the models, then unloads the
shared shader exactly once.

VERTEX-COLOR ALPHA IS DATA, NOT OPACITY
---------------------------------------
``propgen`` bakes the wind-flutter weight (0 = rigid trunk … 1 = leaf tuft)
into vertex-color alpha for the P4 sway shader. The VERTEX shader reads it as
the flutter weight; the FRAGMENT shader uses only ``.rgb`` and forces
``finalColor.a = 1.0`` — the world RT is blitted premultiplied, so a
translucent prop would bleed the background through.

WIND SWAY RIDES THE TAMED WIND (P4, design §4.3 F3)
---------------------------------------------------
``gmap.wind_x`` / ``wind_y`` are raw Q16.16 ``-grad(P)``: fire-spiked by orders
of magnitude and NOT a velocity. Props never touch them. The wind a prop sways
by is the product of ``renderer/gas_detail.py::tame_wind`` — THE render-side
wind seam (smooth → direction → saturating speed in tiles/tick), the same
tamed field the smoke-detail pass advects its noise with, so smoke and foliage
in one room always agree about which way the air moves. This module only
SAMPLES that array (one nearest-tile lookup per prop per frame) and turns it
into a model-space displacement vector; it never smooths, never limits, never
re-derives — a parallel taming here would be the bug.

Everything is render-read-only: the tamed array is float (the dequantize-at-
the-render-read convention), nothing is written back, and the sim never learns
that a tree leaned.

SWAY IS A SIGNAL, NOT DECORATION (P4r, Erik's ruling 2026-09-07)
---------------------------------------------------------------
"We're in a spaceship — leaves should be TOTALLY STILL unless there is actual
wind." The shipped ``idle_wind`` floor is therefore **0**: in calm air a prop
draws exactly its rigid P2 mesh, and any motion the player sees means the
atmosphere is really moving there (a blast front, a hull breach, a running
vent). The floor dial survives for experiments; nothing in the shipped path
adds a decorative breeze — the demo's own ``demo_breeze`` was DELETED at P4r
for exactly that reason.

Props & vegetation arc #60 P2 (+ P4 sway). See
``docs/architecture/graphics/props_and_vegetation.md`` §2 / §4.3 / §7.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyray as rl

from .gas_detail import WIND_V_REF
from .lit3d import _COMMON_GLSL, _FIELD_SAMPLE_GLSL, LightFieldCtx, make_camera
from .propgen import GENERATORS

# ---------------------------------------------------------------------------
# Feel knobs (the whole patch is HUMAN-TEST gated)
# ---------------------------------------------------------------------------

# Props get the marine's grazing key rather than the ship's near-overhead one:
# a straight-down lamp flattens a top-down canopy (every visible normal points
# up, N·L ~ 1), an angled key keeps a lit side and a shaded side per lobe.
PROP_LIGHT_Z = 0.35
# Fresnel-ish rim for silhouette pop against the floor. Weaker than the
# marine's (0.35): a canopy is a big soft mass, not a small token, and a strong
# rim reads as a halo on every lobe.
PROP_RIM_STRENGTH = 0.20
PROP_RIM_POWER = 3.0

# Authored height is bucketed before it reaches the generator so a hand-typed
# 2.23 m and 2.25 m share one cached model. The placement is then scaled to its
# EXACT requested height, so the bucket costs no visible precision.
HEIGHT_BUCKET_M = 0.05

# Default metres-per-tile, matching level_loader's own default. Callers that
# know their level pass its real ``tile_size_m``.
DEFAULT_TILE_SIZE_M = 0.333

# Direction the idle breeze blows when the room's tamed wind is (near) zero.
# ERIK'S RULING (2026-09-07, P4r): the SHIPPED idle floor is 0 — "we're in a
# spaceship, leaves should be TOTALLY STILL unless there is actual wind" — so
# this direction is dormant unless someone dials `idle_wind` up for an
# experiment. UNIT vector in (world X, world Z) = (tile col, tile row); it must
# be exactly unit length, or the idle_wind dial would not mean what it says.
_IDLE_WIND_DIR = (math.cos(math.radians(20.0)), math.sin(math.radians(20.0)))


@dataclass(frozen=True)
class SwaySettings:
    """The P4 sway dials — ``[render.props]`` in config.toml, re-read every
    frame so Ctrl+R (config hot-reload) retunes the motion live.

    Erik owes this patch a TUNING PASS ("how much does the tree move"), so
    every number that shapes the motion is here; none is baked into the GLSL.

    * ``strength`` — master amplitude, as a FRACTION OF PROP HEIGHT: the crown
      of a full-wind tree offsets by ``strength × height``. 0 disables sway
      entirely (props draw rigid, and the per-prop wind lookup is skipped).
    * ``flutter`` — leaf-tuft jitter amplitude as a fraction of the bend
      displacement, weighted per-vertex by the baked flutter alpha.
    * ``gust_speed`` / ``gust_depth`` — the slow envelope (rad/s, and how deep
      it dips: ``(1-depth) … (1+depth)``).
    * ``flutter_speed`` — rate multiplier on the high-frequency leaf jitter.
    * ``idle_wind`` — floor on the wind fraction (0..1). **DEFAULT 0 (Erik's
      spaceship-stillness ruling, 2026-09-07)**: sway is a signal that the air
      is actually moving, never ambient decoration, so dead calm is dead still.
      The dial is kept for experiments (a planetside scene, a debug look).
    * ``wind_ref`` — the tamed speed (tiles/tick) that counts as FULL sway;
      defaults to gas_detail's own saturation ceiling, which is the largest
      value ``tame_wind`` can ever return.
    """
    strength: float = 0.06
    flutter: float = 0.38
    gust_speed: float = 1.15
    gust_depth: float = 0.38
    flutter_speed: float = 1.0
    idle_wind: float = 0.0
    wind_ref: float = WIND_V_REF

    @classmethod
    def from_config(cls, cfg) -> "SwaySettings":
        """Build from ``[render.props]`` (getattr-guarded honest defaults, the
        ``GasDetailPass.from_config`` precedent — a config file without the
        section still gets the shipped feel)."""
        render = getattr(cfg, "render", None)
        pr = getattr(render, "props", None)
        g = lambda name, default: float(getattr(pr, name, default))
        return cls(
            strength=g("sway_strength", 0.06),
            flutter=g("flutter_strength", 0.38),
            gust_speed=g("gust_speed", 1.15),
            gust_depth=g("gust_depth", 0.38),
            flutter_speed=g("flutter_speed", 1.0),
            idle_wind=g("idle_wind", 0.0),
            wind_ref=g("wind_ref", WIND_V_REF),
        )


PROP_VS = """#version 330
// Static-prop vertex shader. No skinning; the ONE animation is P4's WIND SWAY,
// a pure vertex displacement in MODEL space (so it rides the placement's yaw +
// scale like the rest of the mesh). We then lift the SWAYED vertex to world
// space for the fragment's field lookup and pass the baked vertex colour on.
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;    // auto-uploaded by DrawMesh (loc auto-populated)
uniform mat4 matNormal;   // = transpose(inverse(matModel)), world-space normals

// ---- P4 sway (design §4.3 "Sway (P4)"; ported from prototypes/prop_spike) --
uniform float u_time;      // seconds on the SIM clock (replay-identical)
uniform vec3  u_wind;      // MODEL-space crown displacement (mesh units)
uniform float u_phase;     // per-prop desync so neighbours never move in step
uniform float u_sway;      // 0 = rigid (pack models / dial off), 1 = sway
uniform float u_height;    // mesh height in model units, for bend normalizing
uniform float u_flutter;   // leaf-flutter amplitude (fraction of |u_wind|)
uniform float u_gust_speed;  // rad/s of the slow gust envelope
uniform float u_gust_depth;  // gust envelope depth (0 = steady, 1 = 0..2x)
uniform float u_flutter_speed;  // rate multiplier for the high-frequency jitter

out vec3 fragWorldPos;
out vec3 fragWorldNormal;
out vec4 fragColor;

void main() {
    // Bend by height SQUARED: the trunk base is rigid, the crown carries the
    // motion. Clamped a little above 1 so a canopy that overshoots the nominal
    // height still bends smoothly instead of running away.
    float hn = clamp(vertexPosition.y / max(u_height, 1e-4), 0.0, 1.3);
    float bend = hn * hn;
    float gust = (1.0 - u_gust_depth) + u_gust_depth * sin(u_time * u_gust_speed
                                                           + u_phase);
    vec3 disp = u_wind * (bend * gust);

    // Leaf flutter: vertexColor.a is the baked flutter WEIGHT (0 rigid trunk ..
    // 1 leaf tuft), scaled by how hard the wind blows, at a higher frequency.
    float fl = vertexColor.a * length(u_wind);
    float ft = u_time * u_flutter_speed;
    disp += fl * u_flutter * vec3(
        sin(ft * 6.1 + u_phase + vertexPosition.x * 3.1),
        0.35 * sin(ft * 5.3 + vertexPosition.z * 2.7),
        cos(ft * 5.6 + u_phase + vertexPosition.y * 2.9));

    vec3 pos = vertexPosition + disp * u_sway;

    // The light-field lookup follows the SWAYED position (a leaning crown reads
    // the light where it actually is). Normals are NOT re-derived from the
    // displacement — the bend is a few percent of the prop's height and the
    // fragment shading is half-Lambert, so a re-derived normal would cost more
    // than it shows.
    fragWorldPos = (matModel * vec4(pos, 1.0)).xyz;
    fragWorldNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    // .a is the wind-flutter WEIGHT (consumed above), never opacity.
    fragColor = vertexColor;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

_PROP_FS_BODY = """#version 330
// Static-prop fragment shader — samples the ship's baked light field on the
// prop's true mesh normals, exactly as the marine does (the sample/unpack
// block below IS the marine's, shared through renderer/lit3d.py), but takes
// its albedo from VERTEX COLORS instead of an albedo texture.
in vec3 fragWorldPos;
in vec3 fragWorldNormal;
in vec4 fragColor;

uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: RGB smoke_glow, A = dir.y
uniform vec4  colDiffuse;     // draw color (per-placement tint)

uniform vec3  u_ambient;       // ship's ambient floor (single source of truth)
uniform float u_light_gain;    // ship's render exposure
uniform float u_light_z;       // prop's own grazing key
uniform vec2  u_world_px;      // (world_px_w, world_px_h) for the field UV
uniform vec3  u_view_dir;      // direction toward the eye (ortho ~ (0,1,0))
uniform float u_rim_strength;
uniform float u_rim_power;
uniform int   u_srgb_decode;

out vec4 finalColor;

// <COMMON>

void main() {
// <FIELD_SAMPLE>

    // Half-Lambert, as the marine: softens the pre-normalized-direction
    // instability that side normals get under near-overhead lamps and reads
    // better on the soft canopy lobes.
    float ndotl = dot(N, L) * 0.5 + 0.5;

    // Albedo from the baked vertex colour. ALPHA IS FLUTTER DATA — never
    // sampled here, never used as opacity.
    vec3 albedo = fragColor.rgb;
    if (u_srgb_decode == 1) albedo = srgb_to_linear(albedo);  // else double-dark
    albedo *= colDiffuse.rgb;

    vec3 lit = albedo * (u_ambient + incoming_rgb * u_light_gain * ndotl);

    // Rim tinted by the LOCAL light, so a prop is never brighter than the room
    // it stands in (dark room -> faint ambient rim only).
    float rim = pow(1.0 - max(0.0, dot(N, u_view_dir)), u_rim_power);
    lit += u_rim_strength * rim * (u_ambient + incoming_rgb * u_light_gain);

    lit = aces_tonemap(lit);
    if (u_srgb_decode == 1) lit = linear_to_srgb(lit);

    // alpha = 1.0: the world RT is blitted premultiplied, and the vertex
    // alpha carries flutter DATA, not transparency.
    finalColor = vec4(lit, 1.0);
}
"""

PROP_FS = (_PROP_FS_BODY
           .replace("// <COMMON>", _COMMON_GLSL)
           .replace("// <FIELD_SAMPLE>", _FIELD_SAMPLE_GLSL))


# ---------------------------------------------------------------------------
# Placement record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropPlacement:
    """One placed prop: WHERE (world px) + WHAT IT LOOKS LIKE (cache key).

    ``x_wpx`` / ``y_wpx`` are the trunk-base position in world pixels — the
    same coordinate space the 2D world draws in (3D X = x_wpx, 3D Z = y_wpx;
    see ``lit3d.make_camera``). The remaining fields are exactly the generator
    parameters plus ``height_m``, and together they form the model cache key.

    ``yaw_deg``, ``tint`` and ``sway`` are per-PLACEMENT (they cost no cache
    entry): two props with the same look but different yaw share one model.

    ``sway`` is the per-prop sway ENABLE (design §4.3: "Pack models draw with
    ``u_sway = 0``") — 1.0 for generated vegetation, 0.0 for a rigid ``kind ==
    "model"`` pack asset, whose geometry carries no flutter weights and whose
    trunk is not authored at the origin.
    """
    x_wpx: float
    y_wpx: float
    generator: str = "tree"
    seed: int = 0
    palette: str = "green"
    height_m: float = 2.2
    style: str = "smooth"
    decor: str = ""
    decor_density: float = 1.0
    yaw_deg: float = 0.0
    tint: Tuple[int, int, int, int] = (255, 255, 255, 255)
    sway: float = 1.0

    def cache_key(self) -> tuple:
        """The look identity — everything the generator consumes, height
        bucketed (see :data:`HEIGHT_BUCKET_M`)."""
        return (self.generator, int(self.seed), self.palette, self.style,
                self.decor or "", round(float(self.decor_density), 3),
                _bucket_height(self.height_m))


def _bucket_height(height_m: float) -> float:
    return round(round(float(height_m) / HEIGHT_BUCKET_M) * HEIGHT_BUCKET_M, 4)


def placements_from_entities(entities, world_px_per_tile: float
                             ) -> list:
    """Build one :class:`PropPlacement` per ``class_name == "prop"`` entity —
    the loader->sim->renderer hand-off (props & vegetation design §4.3).

    Reads entities DUCK-TYPED (``class_name``/``fields``) rather than
    importing ``simulation.entities.prop`` — this module stays renderer-only
    (no ``simulation`` import), matching ``propgen.py``'s contract. Footprint
    fields (``x``/``y``/``stamp_tiles``) are synced entity state; the look
    fields (``generator``/``seed``/``palette``/``height_m``/``style``/
    ``decor``) are the render-only art the prop's digest never sees (F10).
    ``x``/``y`` are the trunk-tile anchor; the placement centres on the
    footprint (``x + stamp_tiles/2``, ``y + stamp_tiles/2`` — v1
    ``stamp_tiles == 1`` so this is just the trunk tile's centre).
    """
    out = []
    for e in entities:
        if getattr(e, "class_name", None) != "prop":
            continue
        f = e.fields
        n = int(f.get("stamp_tiles", 1))
        cx = float(f["x"]) + n / 2.0
        cy = float(f["y"]) + n / 2.0
        try:
            seed = int(f.get("seed", "0"))
        except (TypeError, ValueError):
            seed = 0
        out.append(PropPlacement(
            x_wpx=cx * float(world_px_per_tile),
            y_wpx=cy * float(world_px_per_tile),
            generator=str(f.get("generator", "tree")),
            seed=seed,
            palette=str(f.get("palette", "green")),
            height_m=float(f.get("height_m", 2.2)),
            style=str(f.get("style", "smooth")),
            decor=str(f.get("decor", "")),
            # Sway is for the GENERATED vegetation (its flutter weights are
            # baked in vertex alpha); a `kind == "model"` pack asset draws
            # rigid — design §4.3, "Pack models draw with u_sway = 0".
            sway=0.0 if str(f.get("kind", "generated")) == "model" else 1.0,
        ))
    return out


@dataclass
class _CachedModel:
    """A cached uploaded model + the numbers the budget report needs."""
    model: object                 # rl.Model
    native_height: float          # mesh bbox height in generator units (metres)
    tris: int
    vram_bytes: int
    gen_ms: float


# ---------------------------------------------------------------------------
# Mesh upload — the ownership contract
# ---------------------------------------------------------------------------

def _copy_to_raylib(arr: np.ndarray, ctype: str):
    """Copy *arr* into raylib-owned memory and return a typed pointer.

    ``rl.mem_alloc`` is raylib's ``RL_MALLOC``, which is what ``UnloadMesh``'s
    ``RL_FREE`` expects — so the mesh really owns its vertex data and can be
    unloaded. Handing raylib a numpy buffer instead (the spike's
    ``ffi.from_buffer``) makes ``unload_model`` free a Python-owned pointer.
    """
    arr = np.ascontiguousarray(arr)
    n_bytes = int(arr.nbytes)
    ptr = rl.mem_alloc(n_bytes)
    rl.ffi.memmove(ptr, rl.ffi.from_buffer(arr), n_bytes)
    return rl.ffi.cast(ctype, ptr)


def build_model(verts: np.ndarray, normals: np.ndarray,
                colors: np.ndarray):
    """Upload a propgen triangle soup as a raylib ``Model`` that OWNS its data.

    Requires a live GL context. The returned model is safe to ``unload_model``
    (see the module docstring's mesh-ownership contract).
    """
    mesh = rl.Mesh()
    mesh.vertexCount = int(len(verts))
    mesh.triangleCount = int(len(verts) // 3)
    mesh.vertices = _copy_to_raylib(verts.astype(np.float32, copy=False),
                                    "float *")
    mesh.normals = _copy_to_raylib(normals.astype(np.float32, copy=False),
                                   "float *")
    mesh.colors = _copy_to_raylib(colors.astype(np.uint8, copy=False),
                                  "unsigned char *")
    rl.upload_mesh(mesh, False)
    return rl.load_model_from_mesh(mesh)


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class StaticPropRenderer:
    """Model cache + prop shader + the world-RT draw for placed 3D props."""

    def __init__(self, world_px_per_tile: float,
                 tile_size_m: float = DEFAULT_TILE_SIZE_M) -> None:
        # Scale contract: a prop authored at ``height_m`` metres must land at
        # that many metres on screen. 1 tile = ``tile_size_m`` metres =
        # ``world_px_per_tile`` world pixels, so metres -> world px is one
        # constant. (Uniform scale: the top-down ortho camera measures in
        # world px on X/Z and the same world px on Y/up.)
        self.world_px_per_tile = float(world_px_per_tile)
        self.tile_size_m = float(tile_size_m)
        self.px_per_m = self.world_px_per_tile / max(self.tile_size_m, 1e-6)

        self._cache: Dict[tuple, _CachedModel] = {}
        # P4 sway dials. Plain attribute (the GasDetailPass precedent): the
        # caller re-assigns it from CFG every frame, which is what makes
        # Ctrl+R a live tuning session.
        self.sway = SwaySettings()
        self._shader = None
        self._locs: Dict[str, int] = {}
        self._default_shader_id = 0
        self._default_shader_locs = None
        self._ready = False

    # ------------------------------------------------------------------
    # Load / unload (needs a live GL context, like UnitModelRenderer.load)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Compile the prop shader. Must run after ``init_window``.

        A compile failure leaves the renderer inert (``draw_props`` no-ops)
        rather than drawing untextured black blobs over the ship.
        """
        if not rl.is_window_ready():
            print("[static_props] WARN: no GL context; props disabled")
            return
        try:
            shader = rl.load_shader_from_memory(PROP_VS, PROP_FS)
            if shader.id == 0:
                print("[static_props] WARN: prop shader failed to compile; "
                      "props disabled")
                return
            names = ["u_ambient", "u_light_gain", "u_light_z", "u_world_px",
                     "u_view_dir", "u_rim_strength", "u_rim_power",
                     "u_srgb_decode",
                     # P4 sway
                     "u_time", "u_wind", "u_phase", "u_sway", "u_height",
                     "u_flutter", "u_gust_speed", "u_gust_depth",
                     "u_flutter_speed"]
            self._locs = {n: rl.get_shader_location(shader, n) for n in names}
            self._shader = shader
            # Constant-per-run uniforms, set once.
            self._set_f("u_light_z", PROP_LIGHT_Z)
            self._set_f("u_rim_strength", PROP_RIM_STRENGTH)
            self._set_f("u_rim_power", PROP_RIM_POWER)
            self._set_i("u_srgb_decode", 1)
            # Safe default so a stray draw before set_frame_uniforms cannot
            # divide the field UV by zero.
            rl.set_shader_value(shader, self._locs["u_world_px"],
                                rl.ffi.new("float[2]", [1.0, 1.0]),
                                rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
            # Captured for the unload dance (see the module docstring).
            self._default_shader_id = int(rl.rl_get_shader_id_default())
            self._default_shader_locs = rl.rl_get_shader_locs_default()
            self._ready = True
            print(f"[static_props] prop shader ready (id={shader.id})")
        except Exception as exc:  # pragma: no cover - defensive, mirrors marine
            print(f"[static_props] WARN: prop shader setup failed: {exc}")
            self._shader = None
            self._ready = False

    def unload(self) -> None:
        """Free every cached model and the shared shader, in the ONE order
        that is safe (see the module docstring's second ownership trap)."""
        default = None
        if self._default_shader_locs is not None:
            default = rl.Shader()
            default.id = self._default_shader_id
            default.locs = self._default_shader_locs
        for entry in self._cache.values():
            model = entry.model
            if default is not None:
                # Restore the DEFAULT shader first: UnloadMaterial unloads any
                # non-default shader, and ours is shared across every model.
                for mi in range(model.materialCount):
                    model.materials[mi].shader = default
            rl.unload_model(model)      # frees the raylib-owned vertex arrays
        self._cache.clear()
        if self._shader is not None:
            rl.unload_shader(self._shader)   # exactly once
        self._shader = None
        self._locs = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Model cache
    # ------------------------------------------------------------------

    def get_model(self, placement: PropPlacement) -> Optional[_CachedModel]:
        """Return (building on first use) the cached model for *placement*'s
        look. ``None`` if the generator name is unknown."""
        key = placement.cache_key()
        entry = self._cache.get(key)
        if entry is not None:
            return entry
        gen = GENERATORS.get(placement.generator)
        if gen is None:
            print(f"[static_props] WARN: unknown generator "
                  f"{placement.generator!r}; prop skipped")
            return None
        height = _bucket_height(placement.height_m)
        t0 = time.perf_counter()
        if placement.generator == "palm":
            verts, normals, colors = gen(int(placement.seed),
                                         placement.palette, height)
        else:
            verts, normals, colors = gen(int(placement.seed),
                                         placement.palette, height,
                                         placement.style,
                                         placement.decor or None,
                                         float(placement.decor_density))
        gen_ms = (time.perf_counter() - t0) * 1000.0
        native_h = float(verts[:, 1].max() - verts[:, 1].min())
        entry = _CachedModel(
            model=build_model(verts, normals, colors),
            native_height=max(native_h, 1e-6),
            tris=int(len(verts) // 3),
            # VRAM: position + normal + colour, non-indexed, as uploaded.
            vram_bytes=int(verts.nbytes + normals.nbytes + colors.nbytes),
            gen_ms=gen_ms,
        )
        if self._shader is not None:
            for mi in range(entry.model.materialCount):
                entry.model.materials[mi].shader = self._shader
        self._cache[key] = entry
        return entry

    def budget_report(self) -> str:
        """One line per cached model — the measured tri / VRAM / gen-time
        budget (design §2 asks for these to be MEASURED, not estimated)."""
        lines = ["[static_props] model cache "
                 f"({len(self._cache)} distinct looks):"]
        tot_tris = tot_bytes = 0
        tot_ms = 0.0
        for key, e in self._cache.items():
            tot_tris += e.tris
            tot_bytes += e.vram_bytes
            tot_ms += e.gen_ms
            lines.append(f"    {key} -> {e.tris:6d} tris  "
                         f"{e.vram_bytes / 1024.0:8.1f} KiB  "
                         f"{e.gen_ms:7.1f} ms")
        lines.append(f"    TOTAL {tot_tris:6d} tris  "
                     f"{tot_bytes / 1024.0:8.1f} KiB  {tot_ms:7.1f} ms")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def set_frame_uniforms(self, ctx: LightFieldCtx,
                           view_dir=(0.0, 1.0, 0.0)) -> None:
        """Push the per-frame scalar uniforms. Safe before ``begin_mode_3d``
        (``SetShaderValue`` self-enables the program)."""
        if self._shader is None:
            return
        rl.set_shader_value(
            self._shader, self._locs["u_ambient"],
            rl.ffi.new("float[3]", [float(ctx.ambient[0]),
                                    float(ctx.ambient[1]),
                                    float(ctx.ambient[2])]),
            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
        self._set_f("u_light_gain", ctx.light_gain)
        rl.set_shader_value(
            self._shader, self._locs["u_world_px"],
            rl.ffi.new("float[2]", [float(ctx.world_px_w),
                                    float(ctx.world_px_h)]),
            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
        rl.set_shader_value(
            self._shader, self._locs["u_view_dir"],
            rl.ffi.new("float[3]", [float(view_dir[0]), float(view_dir[1]),
                                    float(view_dir[2])]),
            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

    # ------------------------------------------------------------------
    # P4 sway: the tamed wind -> one model-space displacement per prop
    # ------------------------------------------------------------------

    def sample_wind(self, wind_field: Optional[np.ndarray],
                    x_wpx: float, y_wpx: float) -> Tuple[float, float]:
        """The ONE lookup: nearest tile of the TAMED wind array, clamped.

        *wind_field* is ``(h, w, 2)`` tiles/tick from
        ``renderer.gas_detail.tame_wind`` — never the raw ``gmap.wind_*``
        planes. ``None`` (no sim wind available) reads as dead calm, which at
        the shipped ``idle_wind = 0`` means NO sway at all.
        """
        if wind_field is None:
            return (0.0, 0.0)
        h, w = wind_field.shape[0], wind_field.shape[1]
        wpt = max(self.world_px_per_tile, 1e-6)
        tx = min(max(int(x_wpx / wpt), 0), w - 1)
        ty = min(max(int(y_wpx / wpt), 0), h - 1)
        return (float(wind_field[ty, tx, 0]), float(wind_field[ty, tx, 1]))

    def model_wind(self, p: PropPlacement, native_height: float,
                   wind_field: Optional[np.ndarray]) -> Tuple[float, float]:
        """Turn the tamed wind at *p*'s tile into the shader's ``u_wind``
        (MODEL-space X,Z crown displacement, in mesh units).

        Three steps, all of them dial-driven:
          1. sample + normalize the tamed speed against ``wind_ref`` (which IS
             ``tame_wind``'s saturation ceiling, so the fraction is 0..1) and
             apply the ``idle_wind`` floor (0 by ruling — calm air, no motion);
          2. scale by ``strength × the mesh's own height`` — so sway is a
             FRACTION OF THE PROP, and a shrub and a palm lean by the same
             visual proportion;
          3. rotate by ``-yaw_deg``: the displacement is applied before the
             model matrix, so the wind must be expressed in the prop's own
             frame or a yawed tree would bend the wrong way.
        """
        s = self.sway
        wx, wy = self.sample_wind(wind_field, p.x_wpx, p.y_wpx)
        mag = math.hypot(wx, wy)
        if mag > 1e-12:
            dx, dz = wx / mag, wy / mag
        else:
            dx, dz = _IDLE_WIND_DIR
        frac = mag / max(s.wind_ref, 1e-9)
        frac = min(max(frac, max(s.idle_wind, 0.0)), 1.5)
        amp = s.strength * float(native_height) * frac
        wx_w, wz_w = dx * amp, dz * amp
        if p.yaw_deg:
            a = math.radians(float(p.yaw_deg))
            ca, sa = math.cos(a), math.sin(a)
            # inverse of raylib's Y-rotation (x,z) -> (c*x - s*z, s*x + c*z)
            return (ca * wx_w + sa * wz_w, -sa * wx_w + ca * wz_w)
        return (wx_w, wz_w)

    @staticmethod
    def prop_phase(p: PropPlacement) -> float:
        """Per-prop gust phase (radians) — deterministic in the placement, so
        neighbours desync but a prop's motion is reproducible frame to frame
        and run to run (no RNG, no per-prop state anywhere)."""
        v = p.x_wpx * 0.0173 + p.y_wpx * 0.0291 + int(p.seed) * 1.7
        return float(v % (2.0 * math.pi))

    def draw_props(self, props: Sequence[PropPlacement],
                   camera3d: rl.Camera3D,
                   ctx: Optional[LightFieldCtx] = None,
                   open_mode_3d: bool = True,
                   time_s: float = 0.0,
                   wind_field: Optional[np.ndarray] = None) -> None:
        """Draw every placement inside the ALREADY-OPEN world RT.

        Mirrors ``UnitModelRenderer.draw_units``: nests ``begin_mode_3d``
        (build *camera3d* with ``lit3d.make_camera``), binds the ship's baked
        light field into the material map slots so ``DrawMesh`` re-binds it
        every draw, and pushes the per-frame scalar uniforms once.

        ``open_mode_3d=False`` skips the ``begin_mode_3d`` / ``end_mode_3d``
        pair so P3 can draw props inside the UNITS' 3D pass (design §4.3
        F23/F25: one 3D pass, one batch flush, one shared depth buffer).

        P4 sway: *time_s* is the animation clock — pass the SIM clock
        (``sim_tick × sim_dt``), never wall time, so a replay renders the same
        motion (the ``gas_detail`` crossfade precedent). *wind_field* is the
        ``(h, w, 2)`` TAMED wind from ``gas_detail.tame_wind``; ``None`` means
        dead calm — and at the shipped ``idle_wind = 0`` dead calm means the
        props draw rigid (Erik's spaceship-stillness ruling).

        No-op when the shader failed to compile or no light field is given —
        the ship draws exactly as it does today.
        """
        if not self._ready or self._shader is None or ctx is None:
            return

        # Light field -> material MAP slots (METALNESS -> texture1, NORMAL ->
        # texture2), the same auto-bind mechanism the marine rides, so DrawMesh
        # binds them every draw and never clobbers our texture units. Rebound
        # per frame because the LightingPass may swap its ping-pong targets.
        MM = rl.MaterialMapIndex
        for entry in self._cache.values():
            for mi in range(entry.model.materialCount):
                mat = entry.model.materials[mi]
                mat.maps[MM.MATERIAL_MAP_METALNESS].texture = ctx.tex_a
                mat.maps[MM.MATERIAL_MAP_NORMAL].texture = ctx.tex_b
        self.set_frame_uniforms(ctx)

        # Sway uniforms that are the same for every prop this frame (the dials
        # + the clock); the per-prop ones (wind, phase, height, enable) are set
        # in _draw_one. DrawMesh issues its own draw call per model, so a
        # uniform changed between draws never leaks into the previous one.
        s = self.sway
        self._set_f("u_time", time_s)
        self._set_f("u_flutter", s.flutter)
        self._set_f("u_gust_speed", s.gust_speed)
        self._set_f("u_gust_depth", s.gust_depth)
        self._set_f("u_flutter_speed", s.flutter_speed)
        sway_on = s.strength > 0.0

        if open_mode_3d:
            rl.begin_mode_3d(camera3d)
        try:
            for p in props:
                self._draw_one(p, ctx, wind_field if sway_on else None,
                               sway_on)
        finally:
            if open_mode_3d:
                rl.end_mode_3d()

    def _draw_one(self, p: PropPlacement, ctx: LightFieldCtx,
                  wind_field: Optional[np.ndarray] = None,
                  sway_on: bool = False) -> None:
        entry = self.get_model(p)
        if entry is None:
            return
        # Sway: one wind lookup per prop per frame, in the prop's own frame.
        # u_sway 0 (dial off, or a rigid pack model) draws exactly the P2 mesh.
        if sway_on and p.sway > 0.0:
            wx, wz = self.model_wind(p, entry.native_height, wind_field)
            rl.set_shader_value(self._shader, self._locs["u_wind"],
                                rl.ffi.new("float[3]", [wx, 0.0, wz]),
                                rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
            self._set_f("u_phase", self.prop_phase(p))
            self._set_f("u_height", entry.native_height)
            self._set_f("u_sway", float(p.sway))
        else:
            self._set_f("u_sway", 0.0)
        # Scale contract: the mesh's TRUE bbox height becomes height_m metres
        # of world pixels. (The generator's `height` is a nominal authored
        # size — a canopy overshoots it — so normalizing on the measured bbox
        # is what makes "3 m tall" mean 3 m on screen, seed to seed.)
        target_px = float(p.height_m) * self.px_per_m
        s = target_px / entry.native_height
        r, g, b, a = p.tint
        rl.draw_model_ex(entry.model,
                         rl.Vector3(float(p.x_wpx), 0.0, float(p.y_wpx)),
                         rl.Vector3(0.0, 1.0, 0.0), float(p.yaw_deg),
                         rl.Vector3(s, s, s), rl.Color(r, g, b, a))

    # ------------------------------------------------------------------

    def _set_f(self, name: str, val: float) -> None:
        rl.set_shader_value(self._shader, self._locs[name],
                            rl.ffi.new("float[1]", [float(val)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def _set_i(self, name: str, val: int) -> None:
        rl.set_shader_value(self._shader, self._locs[name],
                            rl.ffi.new("int[1]", [int(val)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)


__all__ = ["StaticPropRenderer", "PropPlacement", "SwaySettings",
           "build_model", "placements_from_entities",
           "PROP_VS", "PROP_FS", "HEIGHT_BUCKET_M", "make_camera"]
