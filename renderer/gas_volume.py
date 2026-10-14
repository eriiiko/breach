"""The volumetric smoke pass for the tilted view (issue #90) — owns the baked
detail volume (a GL 3D texture), ``shaders/gas_volume.fs`` and the per-frame
optical-depth texture, and draws ONE premultiplied fullscreen raymarch above
the ground plane the tilt view lays down.

Design: docs/volumetric_smoke_design_2026-10-14.md §2.4. RENDER-ONLY,
determinism-EXEMPT: it reads the gas medium's optical depth (the SAME
``gas_optical_depth`` number ``renderer/gas_medium.py`` builds — single source
of scale), the ship's baked light field (``light_tex_a``) and the blackbody
overlay's packed texture; it writes only its own GPU resources.

Division of labour (the P3 detail pass's philosophy, extended to 3D): the SIM
owns the macro shape (per-tile tau), the BAKED VOLUME owns everything below
tile scale (wisps, rolling, the vertical structure). Height is invented from
tau in the shader.

Volume asset: ``assets/volumes/<name>.npz`` from ``tools/bake_smoke_volume.py``
(contract in ``renderer/volume_bake.py``). Missing -> a fast procedural
fallback so a fresh clone runs (and a printed hint to bake).

Clock = SIM TICK (``advected_noise.advection_phase``), never wall time, so
replays render identical smoke — the gas_detail precedent.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyray as rl

from . import core
from . import gl3d
from .advected_noise import advection_phase
from .volume_bake import load_volume, procedural_volume

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "volumes"


def pad_volume_for_gl(volume: np.ndarray) -> np.ndarray:
    """(T, Z, Y, X, C) -> (T*(Z+1), Y, X, C): every frame gains a duplicate of
    its slice 0 on top, so the shader's trilinear lookup wraps in z WITHOUT
    bleeding into the next frame (the shader addresses frames and z itself;
    only x/y wrap in hardware). Pure numpy, headless-testable."""
    T, Z, Y, X, C = volume.shape
    padded = np.empty((T, Z + 1, Y, X, C), dtype=volume.dtype)
    padded[:, :Z] = volume
    padded[:, Z] = volume[:, 0]
    return padded.reshape(T * (Z + 1), Y, X, C)


class VolumeDials:
    """The ``[render.gas_volume]`` dials, re-read from the LIVE config every
    frame (Ctrl+R retunes the look) — getattr-guarded so a config without the
    block gets the design defaults. Plain attributes, so a harness may also
    poke them directly."""

    DEFAULTS = dict(
        enabled=True, asset="smoke_v1", steps=64,
        h_max_tiles=6.0, tau_h=3.0, tau_solid=6.0, erode=0.7,
        detail_gamma=1.0, top_var=0.35, k_ext=1.0, k_shadow=0.6,
        albedo_floor=0.25, emit_gain=3.0, emit_pow=2.0, exposure=1.0,
        rise_turns=1.0,
        loop_seconds=0.0,      # 0 = the asset's own loop length
    )

    enabled: bool
    asset: str
    steps: int
    h_max_tiles: float
    tau_h: float
    tau_solid: float
    erode: float
    detail_gamma: float
    top_var: float
    k_ext: float
    k_shadow: float
    albedo_floor: float
    emit_gain: float
    emit_pow: float
    exposure: float
    rise_turns: float
    loop_seconds: float

    def __init__(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

    def refresh(self, cfg) -> "VolumeDials":
        gv = getattr(getattr(cfg, "render", None), "gas_volume", None)
        for k, v in self.DEFAULTS.items():
            val = getattr(gv, k, v)
            setattr(self, k, type(v)(val) if not isinstance(v, str) else str(val))
        return self


class GasVolumePass:
    """Owns the 3D texture + shader + tau texture; ``draw`` is one premult
    fullscreen quad inside the tilt view's RT."""

    def __init__(self, grid_h: int, grid_w: int, *, world_px_w: int,
                 world_px_h: int, px_per_tile: float,
                 asset: Optional[str] = "smoke_v1"):
        self.h = grid_h
        self.w = grid_w
        self.world_px_w = float(world_px_w)
        self.world_px_h = float(world_px_h)
        self.px_per_tile = float(px_per_tile)
        self.dials = VolumeDials()
        self.ready = False
        self.tex3d: Optional[int] = None
        self.asset_name = ""
        self.vol_z = 1
        self.vol_t = 1
        self.period_tiles = 4.0
        self.height_tiles = 1.0
        self.loop_seconds = 4.0
        self._phase = 0.0
        # Alignment checker (F12 live; BREACH_VOLUME_DEBUG=1 for headless
        # shots): paints the ray's ground hit as a tile checker.
        self.debug = os.environ.get("BREACH_VOLUME_DEBUG", "") == "1"
        self._dials_logged = False

        # --- the per-frame optical-depth texture (R = tau) -----------------
        self.tau_packed = np.zeros((grid_h, grid_w, 4), dtype=np.float16)
        self.tau_tex = core.create_dynamic_rgba16f_texture(grid_w, grid_h)

        # --- shader ---------------------------------------------------------
        self.shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"),
            str(SHADERS_DIR / "gas_volume.fs"))
        self._loc = {}
        for name in ("u_tau", "u_light", "u_light_b", "u_heat", "u_volume",
                     "u_inv_vp",
                     "u_cam_pos", "u_world_px", "u_px_per_tile", "u_vol_zt",
                     "u_phase", "u_period_tiles", "u_height_tiles",
                     "u_rise_turns", "u_steps", "u_h_max", "u_tau_h",
                     "u_tau_solid", "u_erode", "u_detail_gamma", "u_top_var",
                     "u_k_ext", "u_k_shadow", "u_albedo_floor", "u_ambient",
                     "u_light_gain", "u_glow_gain", "u_emit_gain", "u_emit_pow",
                     "u_heat_on",
                     "u_exposure", "u_debug"):
            loc = rl.get_shader_location(self.shader, name)
            if loc == -1:
                print(f"[gas_volume] WARN: uniform '{name}' not found")
            self._loc[name] = loc

        # --- the volume (GL 3D texture) -------------------------------------
        self.load_asset(asset)

    # ---- asset -----------------------------------------------------------

    def load_asset(self, asset: Optional[str]) -> bool:
        """Load ``assets/volumes/<asset>.npz`` (or the procedural fallback)
        into the 3D texture. Returns True when a GL volume is bound."""
        vol = None
        meta: dict = {}
        if asset:
            path = ASSET_DIR / f"{asset}.npz"
            if path.is_file():
                try:
                    vol, meta = load_volume(path)
                    self.asset_name = asset
                except Exception as exc:          # a bad file is a warning
                    print(f"[gas_volume] WARN: {path.name}: {exc}")
            else:
                print(f"[gas_volume] no baked volume at {path} — using the "
                      f"procedural fallback. Bake one with: python "
                      f"tools/bake_smoke_volume.py")
        if vol is None:
            vol = procedural_volume()
            meta = {"period_tiles": 4.0, "height_tiles": 1.0,
                    "loop_seconds": 4.0}
            self.asset_name = "<procedural>"
        return self._upload(vol, meta)

    def _upload(self, vol: np.ndarray, meta: dict) -> bool:
        if self.tex3d is not None:
            try:
                gl3d.delete_texture(self.tex3d)
            except gl3d.GLUnavailable:
                pass
            self.tex3d = None
        T, Z, _Y, _X, _C = vol.shape
        try:
            self.tex3d = gl3d.create_texture_3d(pad_volume_for_gl(vol))
        except (gl3d.GLUnavailable, RuntimeError, ValueError) as exc:
            print(f"[gas_volume] WARN: 3D texture upload failed: {exc} — "
                  f"the volume pass is disabled")
            self.ready = False
            return False
        self.vol_z = int(Z)
        self.vol_t = int(T)
        self.period_tiles = float(meta["period_tiles"])
        self.height_tiles = float(meta["height_tiles"])
        self.loop_seconds = float(meta["loop_seconds"])
        self.ready = True
        print(f"[gas_volume] volume {self.asset_name}: {vol.shape[3]}x"
              f"{vol.shape[2]}x{Z} x {T} frames, period {self.period_tiles} "
              f"tiles, height {self.height_tiles} tiles, loop "
              f"{self.loop_seconds} s ({vol.nbytes / 1e6:.0f} MB)")
        return True

    # ---- per frame -------------------------------------------------------

    def update(self, tau: np.ndarray, albedo_rgb: Optional[np.ndarray] = None,
               *, sim_tick: int, sim_dt: float, cfg=None) -> None:
        """Upload this frame's optical depth (+ per-tile albedo) and advance
        the loop phase.

        ``tau`` : (H, W) float — the gas medium's pre-curve optical depth
        (``GasMediumOverlay.tau``). ``albedo_rgb`` : (H, W, 3) float — its
        tau-weighted scatter albedo (``GasMediumOverlay.albedo_rgb``; None =
        a neutral 1). ``sim_tick`` is the monotonic sim tick (the smoke
        clock). ``cfg`` re-reads the dials (Ctrl+R)."""
        if cfg is not None:
            self.dials.refresh(cfg)
        if not self._dials_logged:
            d = self.dials
            print(f"[gas_volume] dials: steps={d.steps} h_max={d.h_max_tiles} "
                  f"tau_h={d.tau_h} tau_solid={d.tau_solid} erode={d.erode} "
                  f"albedo_floor={d.albedo_floor} emit_gain={d.emit_gain} "
                  f"k_shadow={d.k_shadow} rise={d.rise_turns}")
            self._dials_logged = True
        self.tau_packed[..., 0] = np.asarray(tau, dtype=np.float32).astype(np.float16)
        if albedo_rgb is not None:
            self.tau_packed[..., 1:4] = np.asarray(albedo_rgb,
                                                   dtype=np.float32).astype(np.float16)
        else:
            self.tau_packed[..., 1:4] = np.float16(1.0)
        core.update_rgba16f_texture(self.tau_tex, self.tau_packed)
        loop_s = (float(self.dials.loop_seconds)
                  if float(self.dials.loop_seconds) > 0 else self.loop_seconds)
        tps = 1.0 / float(sim_dt) if sim_dt > 0 else 0.0
        self._phase = advection_phase(int(sim_tick), loop_s, tps).phase

    # ---- draw ------------------------------------------------------------

    def _set_f(self, name, v):
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new("float[1]", [float(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def _set_i(self, name, v):
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new("int[1]", [int(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def _set_vec2(self, name, x, y):
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new("float[2]", [float(x), float(y)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

    def _set_vec3(self, name, v):
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new("float[3]", [float(v[0]), float(v[1]),
                                                    float(v[2])]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

    def _set_mat4(self, name, m: np.ndarray):
        """numpy (4,4) row-major M[r][c] -> raylib Matrix (m[c*4+r])."""
        mat = rl.Matrix()
        for c in range(4):
            for r in range(4):
                setattr(mat, f"m{c * 4 + r}", float(m[r, c]))
        rl.set_shader_value_matrix(self.shader, self._loc[name], mat)

    def draw(self, *, inv_vp: np.ndarray, cam_pos, light_tex_a: rl.Texture,
             light_tex_b: rl.Texture, heat_tex: Optional[rl.Texture],
             heat_on: bool, ambient, light_gain: float, glow_gain: float,
             dst_w: int, dst_h: int) -> None:
        """ONE premultiplied fullscreen raymarch over the (already drawn)
        ground. Call inside the tilt view's RT, after its ground quad."""
        if not self.ready or not self.dials.enabled or self.tex3d is None:
            return
        d = self.dials
        self._set_mat4("u_inv_vp", inv_vp)
        self._set_vec3("u_cam_pos", cam_pos)
        self._set_vec2("u_world_px", self.world_px_w, self.world_px_h)
        self._set_f("u_px_per_tile", self.px_per_tile)
        self._set_vec2("u_vol_zt", self.vol_z, self.vol_t)
        self._set_f("u_phase", self._phase)
        self._set_f("u_period_tiles", self.period_tiles)
        self._set_f("u_height_tiles", self.height_tiles)
        self._set_f("u_rise_turns", d.rise_turns)
        self._set_i("u_steps", max(4, int(d.steps)))
        self._set_f("u_h_max", d.h_max_tiles)
        self._set_f("u_tau_h", max(d.tau_h, 1e-3))
        self._set_f("u_tau_solid", max(d.tau_solid, 1e-3))
        self._set_f("u_erode", d.erode)
        self._set_f("u_detail_gamma", max(d.detail_gamma, 1e-3))
        self._set_f("u_top_var", d.top_var)
        self._set_f("u_k_ext", d.k_ext)
        self._set_f("u_k_shadow", d.k_shadow)
        self._set_f("u_albedo_floor", d.albedo_floor)
        self._set_vec3("u_ambient", ambient)
        self._set_f("u_light_gain", light_gain)
        self._set_f("u_glow_gain", glow_gain)
        self._set_f("u_emit_gain", d.emit_gain)
        self._set_f("u_emit_pow", max(d.emit_pow, 1e-3))
        self._set_i("u_heat_on", 1 if (heat_on and heat_tex is not None) else 0)
        self._set_f("u_exposure", d.exposure)
        self._set_i("u_debug", 1 if self.debug else 0)
        # The sampler3D lives on its own unit (raylib never touches it).
        self._set_i("u_volume", gl3d.VOLUME_TEXTURE_UNIT)
        gl3d.bind_texture_3d(self.tex3d, gl3d.VOLUME_TEXTURE_UNIT)

        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.begin_shader_mode(self.shader)
        rl.set_shader_value_texture(self.shader, self._loc["u_tau"], self.tau_tex)
        rl.set_shader_value_texture(self.shader, self._loc["u_light"], light_tex_a)
        rl.set_shader_value_texture(self.shader, self._loc["u_light_b"], light_tex_b)
        rl.set_shader_value_texture(
            self.shader, self._loc["u_heat"],
            heat_tex if heat_tex is not None else self.tau_tex)
        # Fullscreen quad with fragTexCoord spanning 0..1: draw_texture_pro of
        # a real texture. NEVER draw_rectangle here — raylib's shapes draw
        # samples a tiny white rect of whatever "shapes texture" is set (raygui
        # points it at its font atlas), so its texcoords are ~constant and the
        # reconstructed rays all point the same way (the bands incident,
        # 2026-10-14).
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(0, 0, float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tau_tex, src, dst, rl.Vector2(0, 0), 0.0,
                            rl.WHITE)
        rl.end_shader_mode()
        rl.end_blend_mode()

    def unload(self) -> None:
        rl.unload_shader(self.shader)
        rl.unload_texture(self.tau_tex)
        if self.tex3d is not None:
            try:
                gl3d.delete_texture(self.tex3d)
            except gl3d.GLUnavailable:
                pass
            self.tex3d = None


__all__ = ["GasVolumePass", "VolumeDials", "pad_volume_for_gl", "ASSET_DIR"]
