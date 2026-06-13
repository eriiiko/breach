"""Water surface optics pass (graphics/water_rendering.md §7 steps 1+2).

A SEPARATE GLSL fragment pass (not an extension of lighting.fs, §5) that turns
the sim's water fields into convincing water: perturbed-ripple normal -> a
see-through refraction + Beer-Lambert depth-tint BASE (with a small Fresnel
ambient sheen) + an ADDITIVE GGX glint (reusing the light buffer) that rides on
top as HDR specular light. The glint is added, not Fresnel-blended at ~2% (which
crushed it top-down) — see shaders/water.fs. Caustics / foam / chromatic-
aberration / matcap are the next staging step (§7 step 3/4) and are NOT here.

It draws in the WaterFieldOverlay compose slot (after the lit ship, before
units), REPLACING the CPU-tinted WaterFieldOverlay placeholder. It is
DORMANT-SAFE: the shader emits premultiplied alpha = 0 on every dry tile
(water_depth == 0), so a ship with no standing water (the default level at load)
renders bit-identically to no pass.

Plumbing mirrors LightingPass: one per-frame RGBA16F "water texture" uploaded
via core.update_rgba16f_texture, packed
    R = ripple (m)   G = ripple_v (m/s, reserved for later)
    B = water_depth (m)   A = foam/agitation (reserved, zero)
The light_rgb / light_dir / diffuse textures need NO new plumbing — the pass
binds the LightingPass's existing GPU textures + the level diffuse directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyray as rl

from config import CFG

from . import core
from .lighting import art_src_and_uv_rect


SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"


def _cfg_water():
    """The [graphics.water] config Namespace, or None if absent."""
    graphics = getattr(CFG, "graphics", None)
    return getattr(graphics, "water", None) if graphics is not None else None


class WaterPass:
    """Owns the water shader + the dynamic packed water-field texture."""

    def __init__(self, grid_h: int, grid_w: int):
        self.h = grid_h
        self.w = grid_w

        # CPU-side packed scratch (mirrors LightingPass.packed_a layout).
        #   R = ripple, G = ripple_v, B = water_depth, A = foam/agitation(0)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.float16)
        self.water_tex = core.create_dynamic_rgba16f_texture(grid_w, grid_h)
        # Bilinear so the floor-warp / glints read smoothly between tiles.
        rl.set_texture_filter(self.water_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        # Track whether the last upload wrote an all-zero texture, so a dry
        # ship costs one .any() (mirrors WaterFieldOverlay's fast path). The
        # GPU texture starts all-zero (gen_image_color BLACK), so _tex_blank
        # starts True — the first dry frame uploads nothing.
        self._tex_blank = True

        # [art.align] transform (level format v2 §1.3) — same convention as
        # LightingPass.draw_lit_world. None = legacy full-stretch draw.
        self.art_align = None

        self.shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"),   # reuse the passthrough VS
            str(SHADERS_DIR / "water.fs"),
        )

        # Sampler + scalar uniform locations (warn-but-continue on -1).
        self._loc_diffuse   = self._lookup("u_diffuse")
        self._loc_light_a   = self._lookup("u_light_a")
        self._loc_light_b   = self._lookup("u_light_b")
        self._loc_water     = self._lookup("u_water")
        self._loc_rough_base = self._lookup("u_roughness_base")
        self._loc_rough_agit = self._lookup("u_roughness_agitation")
        self._loc_fog       = self._lookup("u_fog_density")
        self._loc_refract   = self._lookup("u_refract_strength")
        self._loc_r0        = self._lookup("u_r0")
        self._loc_water_col = self._lookup("u_water_color")
        self._loc_light_z   = self._lookup("u_light_z")
        self._loc_srgb      = self._lookup("u_srgb_decode")
        self._loc_ripple_scale = self._lookup("u_ripple_scale")
        self._loc_texel     = self._lookup("u_texel")
        self._loc_art_uv    = self._lookup("u_art_uv_rect")
        self._loc_time      = self._lookup("u_time")
        self._loc_glint     = self._lookup("u_glint_strength")
        self._loc_alpha_scale = self._lookup("u_alpha_scale")
        self._loc_alpha_min = self._lookup("u_alpha_min")
        self._loc_alpha_max = self._lookup("u_alpha_max")
        self._loc_ambient   = self._lookup("u_ambient")
        # Phase 2 (mood pass): caustics / foam / chromatic aberration / waves.
        self._loc_caustic_str   = self._lookup("u_caustic_strength")
        self._loc_caustic_scale = self._lookup("u_caustic_scale")
        self._loc_foam_thresh   = self._lookup("u_foam_threshold")
        self._loc_foam_int      = self._lookup("u_foam_intensity")
        self._loc_ca_amount     = self._lookup("u_ca_amount")
        self._loc_wave_scale    = self._lookup("u_wave_scale")
        self._loc_ambient_amp   = self._lookup("u_ambient_amp")

        # Bind the static [graphics.water] uniforms once (look-tuning lives in
        # config per the graphics README; restart to re-apply, like the other
        # render params). getattr-defaults keep the pass alive if the block is
        # missing — defaults reproduce the §2 reference look.
        wc = _cfg_water()
        self._set_f(self._loc_rough_base,
                    float(getattr(wc, "roughness_base", 0.08)))
        self._set_f(self._loc_rough_agit,
                    float(getattr(wc, "roughness_agitation", 0.6)))
        self._set_f(self._loc_fog,
                    float(getattr(wc, "fog_density", 3.0)))
        self._set_f(self._loc_refract,
                    float(getattr(wc, "refract_strength", 0.02)))
        self._set_f(self._loc_r0,
                    float(getattr(wc, "r0", 0.02)))
        color = getattr(wc, "water_color", (0.03, 0.10, 0.18))
        self._set_vec3(self._loc_water_col,
                       (float(color[0]), float(color[1]), float(color[2])))
        # ripple_scale: metres-of-ripple-height -> screen-readable normal slope.
        self._set_f(self._loc_ripple_scale,
                    float(getattr(wc, "ripple_scale", 8.0)))
        # glint_strength: ADDITIVE GGX-glint multiplier (light off the surface).
        self._set_f(self._loc_glint,
                    float(getattr(wc, "glint_strength", 2.0)))
        # alpha (transparency) ramp dials: alpha = clamp(depth*scale, min, max).
        self._set_f(self._loc_alpha_scale,
                    float(getattr(wc, "alpha_scale", 6.0)))
        self._set_f(self._loc_alpha_min,
                    float(getattr(wc, "alpha_min", 0.15)))
        self._set_f(self._loc_alpha_max,
                    float(getattr(wc, "alpha_max", 0.95)))
        # Phase 2 (mood pass) defaults — caustics / foam / CA / wave size.
        self._set_f(self._loc_caustic_str,
                    float(getattr(wc, "caustic_strength", 2.5)))
        self._set_f(self._loc_caustic_scale,
                    float(getattr(wc, "caustic_scale", 6.0)))
        self._set_f(self._loc_foam_thresh,
                    float(getattr(wc, "foam_threshold", 0.02)))
        self._set_f(self._loc_foam_int,
                    float(getattr(wc, "foam_intensity", 0.6)))
        self._set_f(self._loc_ca_amount,
                    float(getattr(wc, "ca_amount", 0.012)))
        # wave_scale multiplies the ambient-sine spatial frequencies; ambient_amp
        # is the idle-shimmer amplitude base (the old hardcoded 0.06).
        self._set_f(self._loc_wave_scale,
                    float(getattr(wc, "wave_scale", 2.0)))
        self._set_f(self._loc_ambient_amp,
                    float(getattr(wc, "ambient_amp", 0.06)))
        # u_texel = neighbour-tap offset in UV = 1/grid (per axis).
        self._set_vec2(self._loc_texel, (1.0 / grid_w, 1.0 / grid_h))
        # Default art-UV rect — overwritten per-draw when an [art.align] is set.
        self._set_vec4(self._loc_art_uv, (0.0, 0.0, 1.0, 1.0))
        # sRGB decode default matches the lighting pass (PNG art is sRGB).
        self.set_srgb_decode(True)
        # light_z default mirrors LightingPass (overhead-lamp feel).
        self.set_light_z(0.5)
        # Ambient default mirrors LightingPass.set_ambient((0.18,0.18,0.22)) so
        # the water body is lit by `ambient + sources` like the dry floor even
        # before the per-frame push. game_renderer pushes the lighting pass's
        # live ambient each frame, so the demo's ambient sliders drive this too.
        self.set_ambient((0.18, 0.18, 0.22))

    # ---- uniform helpers -----------------------------------------------

    def _lookup(self, name: str) -> int:
        loc = rl.get_shader_location(self.shader, name)
        if loc == -1:
            print(f"[water] WARN: shader uniform '{name}' not found (loc=-1)")
        return loc

    def _set_f(self, loc: int, v: float) -> None:
        val = rl.ffi.new("float[1]", [float(v)])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def _set_vec2(self, loc: int, v) -> None:
        val = rl.ffi.new("float[2]", [float(v[0]), float(v[1])])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

    def _set_vec3(self, loc: int, v) -> None:
        val = rl.ffi.new("float[3]", [float(v[0]), float(v[1]), float(v[2])])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

    def _set_vec4(self, loc: int, v) -> None:
        val = rl.ffi.new("float[4]", [float(x) for x in v])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

    def set_srgb_decode(self, on: bool) -> None:
        val = rl.ffi.new("int[1]", [1 if on else 0])
        rl.set_shader_value(self.shader, self._loc_srgb, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def set_light_z(self, z: float) -> None:
        """Keep the water pass's L-reconstruction z in step with LightingPass."""
        self._set_f(self._loc_light_z, max(0.0, min(1.5, float(z))))

    def set_art_align(self, offset_px, px_per_tile) -> None:
        """Bind the level's [art.align] transform (mirrors LightingPass)."""
        if isinstance(px_per_tile, (list, tuple)):
            ppt_x, ppt_y = float(px_per_tile[0]), float(px_per_tile[1])
        else:
            ppt_x = ppt_y = float(px_per_tile)
        self.art_align = (float(offset_px[0]), float(offset_px[1]),
                          ppt_x, ppt_y)

    # ---- live tunable setters (demo slider hookup) ---------------------
    # The MAIN game binds [graphics.water] once at construction (restart to
    # re-apply, the render-params precedent). These per-frame setters let the
    # tuning DEMO drag each dial live: each just re-binds its uniform (cheap;
    # safe to call every frame). Mirrors LightingPass.set_* / the _set_f helper.

    def set_glint_strength(self, v: float) -> None:
        self._set_f(self._loc_glint, float(v))

    def set_roughness_base(self, v: float) -> None:
        self._set_f(self._loc_rough_base, float(v))

    def set_roughness_agitation(self, v: float) -> None:
        self._set_f(self._loc_rough_agit, float(v))

    def set_fog_density(self, v: float) -> None:
        self._set_f(self._loc_fog, float(v))

    def set_refract_strength(self, v: float) -> None:
        self._set_f(self._loc_refract, float(v))

    def set_r0(self, v: float) -> None:
        self._set_f(self._loc_r0, float(v))

    def set_water_color(self, rgb) -> None:
        self._set_vec3(self._loc_water_col,
                       (float(rgb[0]), float(rgb[1]), float(rgb[2])))

    def set_ambient(self, rgb) -> None:
        """Push the global ambient (mirror LightingPass.set_ambient). Lights
        the refracted floor + the faint surface sheen by `ambient + sources`,
        so water is visible OUTSIDE the flashlight beam. game_renderer keeps
        this in step with the lighting pass's ambient each frame."""
        self._set_vec3(self._loc_ambient,
                       (float(rgb[0]), float(rgb[1]), float(rgb[2])))

    def set_alpha_scale(self, v: float) -> None:
        self._set_f(self._loc_alpha_scale, float(v))

    def set_alpha_min(self, v: float) -> None:
        self._set_f(self._loc_alpha_min, float(v))

    def set_alpha_max(self, v: float) -> None:
        self._set_f(self._loc_alpha_max, float(v))

    def set_ripple_scale(self, v: float) -> None:
        self._set_f(self._loc_ripple_scale, float(v))

    # ---- Phase 2 (mood pass) setters -----------------------------------

    def set_caustic_strength(self, v: float) -> None:
        self._set_f(self._loc_caustic_str, float(v))

    def set_caustic_scale(self, v: float) -> None:
        self._set_f(self._loc_caustic_scale, float(v))

    def set_foam_threshold(self, v: float) -> None:
        self._set_f(self._loc_foam_thresh, float(v))

    def set_foam_intensity(self, v: float) -> None:
        self._set_f(self._loc_foam_int, float(v))

    def set_ca_amount(self, v: float) -> None:
        self._set_f(self._loc_ca_amount, float(v))

    def set_wave_scale(self, v: float) -> None:
        self._set_f(self._loc_wave_scale, float(v))

    def set_ambient_amp(self, v: float) -> None:
        self._set_f(self._loc_ambient_amp, float(v))

    # ---- per-frame upload ----------------------------------------------

    def update(self, water_depth: np.ndarray,
               ripple: Optional[np.ndarray] = None,
               ripple_v: Optional[np.ndarray] = None) -> None:
        """Pack the water fields into the RGBA16F texture and upload.

        R = ripple, G = ripple_v, B = water_depth, A = 0 (foam, reserved).
        Dry-ship fast path: when there is no standing water AND the texture is
        already blank, skip the pack + upload entirely (mirrors the overlay).
        RENDER-ONLY — every field is read, never written.
        """
        has_water = bool(water_depth.any())
        if not has_water and self._tex_blank:
            return
        self.packed[..., 0] = ripple if ripple is not None else 0.0
        self.packed[..., 1] = ripple_v if ripple_v is not None else 0.0
        self.packed[..., 2] = water_depth
        self.packed[..., 3] = 0.0   # foam/agitation reserved (computed in-shader)
        core.update_rgba16f_texture(self.water_tex, self.packed)
        self._tex_blank = not has_water

    # ---- draw ----------------------------------------------------------

    def draw(self, diffuse: rl.Texture, light_tex_a: rl.Texture,
             light_tex_b: rl.Texture, world_px_w: int, world_px_h: int,
             anim_t: float = 0.0) -> None:
        """Draw the water pass over the full world RT.

        Caller must already be inside BeginTextureMode(world_rt), in the
        WaterFieldOverlay compose slot (after the lit ship, before units). The
        shader gates on water_depth and emits premultiplied alpha = 0 on dry
        tiles, so we draw with BLEND_ALPHA_PREMULTIPLY — exactly the smoke /
        old-water-overlay blend (preserves the ship's destination alpha; the
        god-ray-fix discipline §5). The fragment output is
        ``rgb = base*alpha + glint`` with ``a = alpha``: under premultiplied
        blend (out = src.rgb + dst.rgb*(1-src.a)) the base composites OVER the
        floor (transparency-bound) while the GGX `glint` rides on top as pure
        additive HDR light — a true additive specular term that does NOT raise
        alpha, all in one draw (no separate additive sub-pass needed for the v1
        core; that discipline matters once caustics land).

        Sampler bindings are issued INSIDE BeginShaderMode (raylib gotcha: a
        SetShaderValueTexture before BeginShaderMode can target the wrong
        shader — same note as LightingPass.draw_lit_world).
        """
        src_rect, uv_rect = art_src_and_uv_rect(
            self.art_align, self.w, self.h,
            float(diffuse.width), float(diffuse.height))
        src = rl.Rectangle(*src_rect)

        self._set_f(self._loc_time, float(anim_t))

        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.begin_shader_mode(self.shader)
        # texture0 (u_diffuse) is the draw texture; bind the rest by name.
        rl.set_shader_value_texture(self.shader, self._loc_diffuse, diffuse)
        rl.set_shader_value_texture(self.shader, self._loc_light_a, light_tex_a)
        rl.set_shader_value_texture(self.shader, self._loc_light_b, light_tex_b)
        rl.set_shader_value_texture(self.shader, self._loc_water, self.water_tex)
        self._set_vec4(self._loc_art_uv, uv_rect)

        dst = rl.Rectangle(0, 0, float(world_px_w), float(world_px_h))
        rl.draw_texture_pro(diffuse, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_shader_mode()
        rl.end_blend_mode()

    def unload(self) -> None:
        rl.unload_shader(self.shader)
        rl.unload_texture(self.water_tex)


__all__ = ["WaterPass"]
