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
into vertex-color alpha for the P4 sway shader. The fragment shader here
therefore uses only ``.rgb`` and forces ``finalColor.a = 1.0`` — the world RT
is blitted premultiplied, so a translucent prop would bleed the background
through.

Props & vegetation arc #60 P2. See
``docs/architecture/graphics/props_and_vegetation.md`` §2 / §4.3 / §7.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyray as rl

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


PROP_VS = """#version 330
// Static-prop vertex shader. No skinning, no sway (P4 adds the sway terms
// here); we only lift the vertex to world space for the fragment's field
// lookup and pass the baked vertex colour through.
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;    // auto-uploaded by DrawMesh (loc auto-populated)
uniform mat4 matNormal;   // = transpose(inverse(matModel)), world-space normals

out vec3 fragWorldPos;
out vec3 fragWorldNormal;
out vec4 fragColor;

void main() {
    fragWorldPos = (matModel * vec4(vertexPosition, 1.0)).xyz;
    fragWorldNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    // .a is the wind-flutter WEIGHT (P4 sway input), never opacity.
    fragColor = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
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

    ``yaw_deg`` and ``tint`` are per-PLACEMENT (they cost no cache entry): two
    props with the same look but different yaw share one model.
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
                     "u_srgb_decode"]
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

    def draw_props(self, props: Sequence[PropPlacement],
                   camera3d: rl.Camera3D,
                   ctx: Optional[LightFieldCtx] = None,
                   open_mode_3d: bool = True) -> None:
        """Draw every placement inside the ALREADY-OPEN world RT.

        Mirrors ``UnitModelRenderer.draw_units``: nests ``begin_mode_3d``
        (build *camera3d* with ``lit3d.make_camera``), binds the ship's baked
        light field into the material map slots so ``DrawMesh`` re-binds it
        every draw, and pushes the per-frame scalar uniforms once.

        ``open_mode_3d=False`` skips the ``begin_mode_3d`` / ``end_mode_3d``
        pair so P3 can draw props inside the UNITS' 3D pass (design §4.3
        F23/F25: one 3D pass, one batch flush, one shared depth buffer).

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

        if open_mode_3d:
            rl.begin_mode_3d(camera3d)
        try:
            for p in props:
                self._draw_one(p, ctx)
        finally:
            if open_mode_3d:
                rl.end_mode_3d()

    def _draw_one(self, p: PropPlacement, ctx: LightFieldCtx) -> None:
        entry = self.get_model(p)
        if entry is None:
            return
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


__all__ = ["StaticPropRenderer", "PropPlacement", "build_model",
           "placements_from_entities",
           "PROP_VS", "PROP_FS", "HEIGHT_BUCKET_M", "make_camera"]
