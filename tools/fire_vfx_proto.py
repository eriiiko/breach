r"""tools/fire_vfx_proto.py -- PNG strips of the flame VFX's numpy spec twin
(graphics arc G3 "Fire", patch F0b; docs/g3_fire_design_2026-10-23.md §3.3,
§3.4, §3.5, §13).

Runs ``renderer/fire_vfx_ref.py`` on five scenes and writes, per scene, one
PNG strip to a scratch folder (NOT committed): row A = "temperature only"
(E = smoothstep(r0, r1, R)), row B = "temperature + thickness" (E = 1 -
exp(-kappa R)), row R = the flame density (grey) for judging shape. Frames
are 0.1 s apart after a warm-up, so the strip shows the motion between ticks.

Scenes (64-ish cells a side at M = 8 cells/tile):
  a  one burning furniture tile, no wind
  b  one wood-wall fire (solid fuel, one open face), no wind
  c  the same wood wall in the REAL tamed studio wind (a window of the
     fire_smoke_studio dump, by its supply vent, where the vent loop blows)
  d  a sealed 6x6 room (hull walls) with a burning furniture tile inside
  e  the studio's real five-tile row and its surroundings, straight from
     the dump (fire I, temperature, wind, masks)

Emission: Le(T) from the ONE ramp (``BlackbodyRamp.emission_at_kelvin``),
composited as the §3.5/§13.4 emitter cover ``Le*E + under*(1 - E*w(T))``
over a dim floor with the burning tiles' body glow, then the RESOLVE's own
curve (``blackbody.aces_curve``, the CPU twin of world_resolve.fs) and its
gamma-2.2 encode -- a preview of what the screen does, not a second tone-map
in the renderer.

Prints per scene: max|p|, max T_vfx vs max T_src, total R and the flame
centroid's offset from its source along the wind (the "lean").

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe tools/fire_vfx_proto.py --out <dir>
    ... --scenes ac --seconds 3
    ... --gl          the same strips from the REAL GL loop (G3 F2)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from renderer import fire_vfx_ref as fv  # noqa: E402
from renderer.blackbody import aces_curve  # noqa: E402
from simulation.materials import MAT_HULL  # noqa: E402,F401  (documentation: hull = solid, not fuel)

DUMP = ROOT / "tests" / "data" / "fire_smoke_studio_flame_inputs.npz"


class Scene:
    """Static per-tile inputs (the twin's per-tick reads, held constant)."""

    def __init__(self, name, fire, t_kelvin, solid, flammable, vacuum, wind,
                 *, tile_size_m=0.333, tps=24.0, scales=None, note=""):
        self.name, self.note = name, note
        self.fire = fire
        self.t_game = scales.tscale.from_kelvin(t_kelvin)
        self.solid, self.flammable, self.vacuum = solid, flammable, vacuum
        self.wind = wind                 # (h, w, 2) TAMED, tiles/tick
        self.tile_size_m, self.tps = tile_size_m, tps

    def tick(self, scales, dials):
        return fv.prepare_tick(self.fire, self.t_game, self.solid,
                               self.flammable, self.vacuum, self.wind,
                               tps=self.tps, tile_size_m=self.tile_size_m,
                               scales=scales, dials=dials)


def _gas_bump(h, w, cy, cx, t_amb, dt_k, sigma):
    yy, xx = np.mgrid[0:h, 0:w]
    return t_amb + dt_k * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def _load_dump():
    return np.load(DUMP)


def build_scenes(scales, which: str):
    t_amb = scales.t_amb_k
    out = []
    n = 10
    z = lambda: np.zeros((n, n), dtype=bool)  # noqa: E731
    still = np.zeros((n, n, 2))
    if "a" in which:
        fire = np.zeros((n, n)); fire[5, 4] = 0.45
        tk = _gas_bump(n, n, 5, 4, t_amb, 250.0, 1.5); tk[5, 4] = 1300.0
        fl = z(); fl[5, 4] = True
        out.append(Scene("a_furniture_still", fire, tk, z(), fl, z(), still,
                         scales=scales, note="one burning furniture tile, no wind"))
    if "b" in which or "c" in which:
        def wood_wall(wind, name, note):
            solid = z(); solid[3, :] = True; solid[4, :] = True   # hull row 3, wood row 4
            fl = z(); fl[4, :] = True
            fire = np.zeros((n, n)); fire[4, 4] = 0.45; fire[4, 5] = 0.40
            tk = _gas_bump(n, n, 5.5, 4.5, t_amb, 250.0, 1.5)
            tk[4, 4] = 1300.0; tk[4, 5] = 1250.0
            return Scene(name, fire, tk, solid, fl, z(), wind, scales=scales, note=note)
        if "b" in which:
            out.append(wood_wall(still, "b_wood_wall_still",
                                 "wood wall (hull behind), two burning tiles, open south face, no wind"))
        if "c" in which:
            d = _load_dump()
            k = list(d["ticks"]).index(192) if 192 in list(d["ticks"]) else 1
            ox, oy = d["origin_xy"]
            # a 10x10 window by the supply vent (level x 5..14, y 17..26):
            # the vent loop's jet, the strongest real wind in the studio
            y0, x0 = 17 - oy, 5 - ox
            wind = np.asarray(d["wind_tamed"][k][y0:y0 + n, x0:x0 + n], np.float64)
            sc = wood_wall(wind, "c_wood_wall_studio_wind",
                           f"wood wall in the REAL tamed studio wind (vent window, tick {int(d['ticks'][k])})")
            out.append(sc)
    if "f" in which:
        d = _load_dump()
        k = list(d["ticks"]).index(192) if 192 in list(d["ticks"]) else 1
        ox, oy = d["origin_xy"]
        y0, x0 = 17 - oy, 5 - ox                       # the same vent window as c
        wind = np.asarray(d["wind_tamed"][k][y0:y0 + n, x0:x0 + n], np.float64)
        fire = np.zeros((n, n)); fire[5, 3] = 0.45     # in the jet core (level row 22)
        tk = _gas_bump(n, n, 5, 3, t_amb, 250.0, 1.5); tk[5, 3] = 1300.0
        fl = z(); fl[5, 3] = True
        out.append(Scene("f_furniture_studio_wind", fire, tk, z(), fl, z(), wind,
                         scales=scales,
                         note=f"one furniture tile in the REAL vent-jet wind (tick {int(d['ticks'][k])})"))
    if "d" in which:
        solid = z(); solid[1, 1:9] = solid[8, 1:9] = True
        solid[1:9, 1] = solid[1:9, 8] = True                  # 6x6 interior: 2..7
        fire = np.zeros((n, n)); fire[4, 4] = 0.45
        fl = z(); fl[4, 4] = True
        tk = _gas_bump(n, n, 4, 4, t_amb, 250.0, 1.5); tk[4, 4] = 1300.0
        out.append(Scene("d_sealed_room", fire, tk, solid, fl, z(), still,
                         scales=scales, note="sealed 6x6 room, one burning furniture tile"))
    if "e" in which:
        d = _load_dump()
        k = list(d["ticks"]).index(192) if 192 in list(d["ticks"]) else 1
        ox, oy = d["origin_xy"]
        ys, xs = slice(17 - oy, 28 - oy), slice(19 - ox, 33 - ox)   # row 22, x 23..27 + margin
        tk = scales.to_kelvin(d["t_game"][k][ys, xs])
        sc = Scene("e_studio_row", np.asarray(d["fire"][k][ys, xs], np.float64), tk,
                   d["solid"][ys, xs], d["flammable"][ys, xs],
                   d["is_vacuum"][ys, xs], np.asarray(d["wind_tamed"][k][ys, xs], np.float64),
                   tile_size_m=float(d["tile_size_m"]), tps=float(d["tps"]),
                   scales=scales,
                   note=f"fire_smoke_studio's five-tile row, dump tick {int(d['ticks'][k])}")
        out.append(sc)
    return out


def compose(state, tick, scene, scales, dials, mode):
    """Linear HDR -> display RGB [0,1] (the resolve's curve + encode)."""
    m = dials.m
    h, w = state.R.shape
    under = np.full((h, w, 3), 0.025)
    up = lambda a: fv._upsample_nearest(a, m)  # noqa: E731
    solid_c = up(scene.solid)
    wood_c = up(scene.solid & scene.flammable)
    under[solid_c] = (0.035, 0.04, 0.05)
    under[wood_c] = (0.06, 0.035, 0.018)
    # the burning tiles' body glow (the heat layer's per-tile emission)
    tk = up(scales.to_kelvin(scene.t_game))
    burn_c = up(tick.sources.burning)
    ch, it = scales.ramp.emission_at_kelvin(tk)
    under = np.where(burn_c[..., None], under + ch * it[..., None], under)
    Le, E, cover = fv.flame_emission(state, scales.ramp, mode, dials)
    lin = Le * E[..., None] + under * (1.0 - cover[..., None])
    return np.power(aces_curve(lin), 1.0 / 2.2)


def lean(state, tick, scene, dials, scales):
    """Offset (tiles) of the TONGUES' emission centroid (flame outside the
    source cells) from the source centroid, projected on the mean wind
    direction over the source (0 if no wind); and its unprojected size."""
    Le, E, _ = fv.flame_emission(state, scales.ramp, "B", dials)
    s = tick.s
    wgt = (Le.sum(-1) * E) * (s <= 0)
    if wgt.sum() <= 0 or s.sum() <= 0:
        return 0.0, 0.0
    h, w = wgt.shape
    yy, xx = np.mgrid[0:h, 0:w] / dials.m
    cx, cy = (wgt * xx).sum() / wgt.sum(), (wgt * yy).sum() / wgt.sum()
    sx, sy = (s * xx).sum() / s.sum(), (s * yy).sum() / s.sum()
    wv = (tick.u_wind * (tick.s > 0)[..., None]).reshape(-1, 2).sum(0)
    nrm = np.hypot(*wv)
    along = ((cx - sx) * wv[0] + (cy - sy) * wv[1]) / nrm if nrm > 1e-9 else 0.0
    return along, float(np.hypot(cx - sx, cy - sy))


class GLScene:
    """One scene run through the REAL GL loop (``renderer.fire_vfx``, the
    whole scene = one active box, a hidden-window GL context required) --
    the ``--gl`` backend of this tool and the GL check's long-run runner.
    :meth:`step` advances one substep like ``fire_vfx_ref.substep``;
    :meth:`state` reads the fields back as a ``FireVFXState``."""

    def __init__(self, scene, scales, dials):
        from config import CFG
        from renderer.blackbody import BlackbodyRamp
        from renderer.fire_vfx import (FireVFXPass, FireVFXSettings,
                                       boxes_mask, pack_tile_inputs)
        h, w = scene.fire.shape
        self.vfx = FireVFXPass(h, w, FireVFXSettings(dials=dials, m_u=dials.m),
                               BlackbodyRamp.from_config(CFG))
        self.d = self.vfx.settings.dials
        self.t = 0.0
        self.vfx.upload_pack(pack_tile_inputs(
            scene.fire, scene.t_game, scene.solid, scene.flammable,
            scene.vacuum, scene.wind, tps=scene.tps,
            tile_size_m=scene.tile_size_m, scales=scales, dials=self.d,
            t_base=0.0))
        self.vfx.region.boxes = [(0, 0, w, h)]
        self.vfx.region.domain = boxes_mask(self.vfx.region.boxes, (h, w))

    def run(self, n: int) -> None:
        from renderer.fire_vfx import plan_frame
        k = 0
        while k < n:
            b = min(64, n - k)
            times = [self.t + j * self.d.dt_s for j in range(b)]
            self.vfx._run_plan(plan_frame(self.vfx.region.boxes, [], times,
                                          self.d.n_jacobi))
            self.t = times[-1] + self.d.dt_s
            k += b

    def fields(self):
        from renderer import core
        return (core.read_texture_rgba32f(self.vfx.vel.front).astype(np.float64),
                core.read_texture_rgba32f(self.vfx.flame.front).astype(np.float64))

    def state(self) -> "fv.FireVFXState":
        v, f = self.fields()
        return fv.FireVFXState(u=v[..., :2], p=v[..., 2], R=f[..., 0],
                               RT=f[..., 1], t=self.t)

    def close(self) -> None:
        self.vfx.unload()


def run_scene(scene, scales, dials, seconds, frames_at, scale_px, gl=False):
    tick = scene.tick(scales, dials)
    modes = fv.curl_noise_modes(dials)
    H, W = tick.s.shape
    st = fv.FireVFXState.zeros(H, W)
    runner = GLScene(scene, scales, dials) if gl else None
    n = int(round(seconds / dials.dt_s))
    want = {int(round(t / dials.dt_s)): t for t in frames_at}
    rows = {"A": [], "B": [], "R": []}
    pmax, tmax, leans = 0.0, 0.0, []
    tsrc_max = float(tick.t_src_k.max())
    t0 = time.time()
    for k in range(1, n + 1):
        if runner is None:
            st = fv.substep(st, tick, dials, modes)
        else:
            runner.run(1)
            st = runner.state()
        pmax = max(pmax, float(np.abs(st.p).max()))
        T = st.temperature(dials.r_eps)
        tmax = max(tmax, float(T[st.R > 1e-4].max()) if (st.R > 1e-4).any() else 0.0)
        assert st.R.min() >= 0.0
        if k * dials.dt_s > 1.0:
            leans.append(lean(st, tick, scene, dials, scales))
        if k in want:
            for mode in ("A", "B"):
                rows[mode].append(compose(st, tick, scene, scales, dials, mode))
            r = np.clip(st.R / max(dials.r1 * 2, 1e-6), 0, 1)
            rows["R"].append(np.repeat(r[..., None] ** 0.6, 3, axis=-1))
    el = time.time() - t0
    if runner is not None:
        runner.close()
    gap = np.ones((H, 2, 3))
    strip = []
    for key in ("A", "B", "R"):
        row = []
        for img in rows[key]:
            row += [img, gap]
        strip.append(np.concatenate(row[:-1], axis=1))
        strip.append(np.ones((2, strip[-1].shape[1], 3)))
    img = np.concatenate(strip[:-1], axis=0)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = np.repeat(np.repeat(img, scale_px, 0), scale_px, 1)
    la = np.array(leans) if leans else np.zeros((1, 2))
    stats = dict(pmax=pmax, tmax=tmax, tsrc_max=tsrc_max,
                 R_total=float(st.R.sum()) / dials.m ** 2,
                 lean_along=float(la[:, 0].mean()), lean_abs=float(la[:, 1].mean()),
                 wind=float(np.hypot(*tick.u_wind[tick.s > 0].mean(0))) if (tick.s > 0).any() else 0.0,
                 ms_per_substep=1000 * el / n)
    return Image.fromarray(img), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="scratch folder for the PNGs")
    ap.add_argument("--scenes", default="abcfde")
    ap.add_argument("--seconds", type=float, default=2.6)
    ap.add_argument("--px", type=int, default=3, help="screen px per cell")
    ap.add_argument("--gl", action="store_true",
                    help="run the REAL GL loop (renderer.fire_vfx, a hidden "
                    "window) instead of the numpy twin -- F2's look check")
    ap.add_argument("--set", action="append", default=[],
                    help="dial override name=value (repeatable)")
    a = ap.parse_args()
    dials = fv.FireVFXDials()
    for kv in a.set:
        k, v = kv.split("=")
        cur = getattr(dials, k)
        dials = dials.with_(**{k: type(cur)(float(v)) if not isinstance(cur, bool)
                               else v.lower() in ("1", "true")})
    scales = fv.load_scales()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [a.seconds - 0.5 + 0.1 * i for i in range(6)]
    if a.gl:
        import pyray as rl
        rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
        rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
        rl.init_window(320, 240, "fire_vfx_proto --gl")
    for sc in build_scenes(scales, a.scenes):
        img, s = run_scene(sc, scales, dials, a.seconds, frames, a.px, gl=a.gl)
        p = out / f"{sc.name}.png"
        img.save(p)
        print(f"{sc.name:26s} |p|max {s['pmax']:8.2f}  T {s['tmax']:6.0f} <= {s['tsrc_max']:6.0f}"
              f"  R {s['R_total']:6.2f}  wind {s['wind']:5.2f} t/s  lean {s['lean_along']:+.2f}"
              f" (|off| {s['lean_abs']:.2f}) tiles  {s['ms_per_substep']:.1f} ms/sub  -> {p}")
        print(f"    {sc.note}")


if __name__ == "__main__":
    main()
