"""The flame VFX's numpy SPEC TWIN -- graphics arc G3 "Fire", patch F0b.

Design: ``docs/g3_fire_design_2026-10-23.md`` (v3, APPROVED) -- §3.2-§3.4 as
amended by the binding §13 (13.1 screened pressure, 13.2 velocity scales and
structure sources, 13.4 no source below the glow floor, 13.5 cooling never
heats, 13.6 vacuum).

WHAT THIS IS. A float64 reference implementation of the flame layer's loop
("Layer B": a render-only 2D fluid per tile column, seen from straight
above). F1/F2 port it to GLSL ping-pong passes; F2 checks the GL loop against
this file ONE PASS AT A TIME (§13.9), so every stage is its own function with
the same inputs and outputs a shader has:

    advect_velocity -> follow_wind -> confine -> divergence -> jacobi_sweep
    (x n) -> project -> advect_flame -> inject -> decay_cool

and :func:`substep` is nothing but those calls in that order. F2's GL passes
are named compositions of these (``renderer/fire_vfx.py`` PASSES: e.g. the
"flame" pass = advect_flame -> inject -> decay_cool); a test asserts the
compositions in pass order ARE :func:`substep`, bit for bit.

GPU PRECISION (F2). The shaders never see the absolute game time: the
puffing is evaluated from phases wrapped at a base time
(:func:`puff_phases`, :func:`puff_factor_rebased`) and the curl noise from
per-mode phases wrapped in float64 (:func:`curl_mode_phases`) -- the same
functions of t, exact after hours of play.

RENDER-ONLY. Reads dequantized read-only copies of sim fields (fire I, the
game temperature, the TAMED wind, the solid/vacuum masks), writes nothing
synced, never imports a sim writer or pyray. Float math is allowed here: this
is the render layer (determinism-exempt, CLAUDE.md iron rule 1).

THE FIELDS (§3.2), on one cell grid of ``m`` cells per tile (velocity and
flame share it, M_u = M):

    u   (H, W, 2)  flame-scale velocity, TILES PER SECOND; [...,0] = +x (east),
                   [...,1] = +y (south, the tile grid's row direction)
    p   (H, W)     screened pressure, tiles^2/s, warm-started across substeps
    R   (H, W)     flame density, "how much flame stands in this column", >= 0
    RT  (H, W)     premultiplied flame temperature R * T_vfx (Kelvin)

``T_vfx = RT / R`` where ``R > r_eps``.

THE SOURCE LAW (§3.3, §13.4), :func:`flame_sources`, per tile:
  * permeable fuel (flammable & not solid: furniture, kindling, foliage, air
    fires) injects over its OWN tile -- where the sim lands its heat and soot;
  * solid fuel (flammable & solid: wood) injects in a strip of the OPEN
    neighbour tile along every face that touches non-blocking air;
  * never in vacuum, never inside a solid, never where the burning tile is
    below the ramp's glow floor (``kelvin_glow_min``): a flame that cannot
    glow is not drawn (§13.4; the ~300 K look-shot air fires make none);
  * rate ``s = s0 * I`` and expansion ``q = q0 * I * (T_src/T_amb - 1)``,
    both monotone in the fire intensity I;
  * ``T_src`` is the burning tile's OWN temperature -- the value the sweep
    emits from and the heat layer glows with, so flame colour and room light
    agree; hence the invariant ``T_vfx <= T`` of the tile that fed it;
  * pool-fire puffing (§13.2): s and q pulse at ``f = 1.5 / sqrt(D)`` Hz
    (Cetegen & Ahmed 1993; D = the burning cluster's equivalent diameter in
    metres), each cluster at its own hashed phase and +-jittered frequency,
    with a slow phase wobble so equal-sized fires never lock in step (the
    flicker no-sync gate, §6.3).

Its output is PER TILE (``FlameSources``), uploaded per tile and sampled
nearest (§13.14 "no CPU upsampling"); :func:`rasterize_sources` is the cell
rule F1's shader re-implements: a tile's ``cover`` code says "whole tile" or
"strips along these faces".

WHERE IT LIVES. ``flame_sources`` sits in this module, the twin, because it IS
part of the spec and F1's GL pass imports it from here (pure numpy, no GL);
F1 adds ``renderer/fire_vfx.py`` beside it for the GPU side.

Credits (repo rule -- archived in ``docs/papers/fire_vfx_references.md``):
  - Jos Stam, "Stable Fluids", SIGGRAPH 1999 -- semi-Lagrangian advection,
    projection.
  - Jos Stam, "Real-Time Fluid Dynamics for Games", GDC 2003.
  - Ronald Fedkiw, Jos Stam, Henrik Wann Jensen, "Visual Simulation of
    Smoke", SIGGRAPH 2001 -- vorticity confinement.
  - Duc Quang Nguyen, Ronald Fedkiw, Henrik Wann Jensen, "Physically Based
    Modeling and Animation of Fire", SIGGRAPH 2002 -- expansion at the
    reaction front as a prescribed divergence.
  - Mark J. Harris, "Fast Fluid Dynamics Simulation on the GPU", GPU Gems
    ch. 38 (2004) -- the collocated texture-grid Jacobi formulation.
  - Andrew Selle, Ronald Fedkiw, ByungMoon Kim, Yingjie Liu, Jarek Rossignac,
    "An Unconditionally Stable MacCormack Method", J. Sci. Comput. 35 (2008)
    -- MacCormack advection with the min/max limiter and SL fallback.
  - Robert Bridson, Jim Hourihan, Marcus Nordenstam, "Curl-Noise for
    Procedural Fluid Flow", SIGGRAPH 2007 -- divergence-free noise forcing.
  - Baki M. Cetegen, Taher A. Ahmed, "Experiments on the periodic instability
    of buoyant plumes and pool fires", Combustion and Flame 93 (1993) --
    the puffing frequency f ~ 1.5 / sqrt(D).
  - Sørb, the "Ignitement" fire-VFX breakdown (Unity blog, 2026) -- the
    approach of a cheap 2D flame fluid driven by gameplay fire.
The screened Poisson solve (§13.1) is the design's own amendment: the
``-p/l^2`` term stands for the plume leaving the plane upward, which makes a
sealed room's problem well-posed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NamedTuple, Optional, Tuple

import numpy as np

# Face bits of a tile's ``cover`` code (which strips of THIS tile a source
# fills). N = -y (row above), S = +y, W = -x, E = +x.
FACE_N, FACE_E, FACE_S, FACE_W = 1, 2, 4, 8
COVER_FULL = 16


# ---------------------------------------------------------------------------
# Dials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireVFXDials:
    """Every dial of the flame loop, with its documented default.

    F1 binds these to ``[render.fire_vfx]``; until then the defaults here ARE
    the tuned values (F0b tuned them by eye on the PNG strips of
    ``tools/fire_vfx_proto.py``; the report records why each sits where it
    does).
    """
    # --- grid and clock (§3.2, §13.3) -------------------------------------
    m: int = 8                      # cells per tile (M = M_u)
    dt_s: float = 1.0 / 120.0       # fixed substep on the sim clock, s

    # --- sources (§3.3, §13.2, §13.4) -------------------------------------
    s0: float = 8.0                 # flame density injected per second at I=1
    q0: float = 8.0                # LOOK DIAL, 1/s: expansion per unit
    #                                 I*(T_src/T_amb-1). ~100x the physical
    #                                 in-plane expansion (~0.2 tiles/s at the
    #                                 tile edge), which is too weak for tongues
    i_min: float = 0.01             # fire intensity below which a tile makes no flame
    strip_frac: float = 0.375       # solid fuel's strip depth into the open tile (of a tile)
    puff_amp: float = 0.45          # pool-fire puffing depth on s and q (0..1)
    puff_wobble: float = 1.6        # rad, slow phase wobble that keeps equal fires out of step
    puff_freq_jitter: float = 0.15  # +- fraction of f per cluster
    puff_c: float = 1.5             # f = puff_c / sqrt(D [m]) Hz (Cetegen & Ahmed 1993)

    # --- velocity (§3.4 steps 1-3, §13.1, §13.2) --------------------------
    k_follow: float = 2.0           # 1/s, relaxation of u toward u_wind = tamed * tps
    vort_eps: float = 2.0           # vorticity confinement strength (Fedkiw 2001), dimensionless
    curl_amp: float = 200.0          # tiles/s^2 of curl-noise forcing at local I = 1
    curl_wavelength_tiles: float = 0.6   # dominant noise wavelength
    curl_rate_hz: float = 1.2       # how fast the noise pattern changes
    curl_modes: int = 32            # random Fourier modes in the stream function
    i_local_sigma_tiles: float = 1.25    # reach of "local I" around a burning tile
    n_jacobi: int = 20              # sweeps per substep, warm-started
    screen_len_tiles: float = 3.0   # l of the screened solve (§13.1)

    # --- flame scalars (§3.4 steps 4-5, §13.5) ----------------------------
    maccormack: bool = True         # False = plain SL (the fallback)
    lifetime_s: float = 0.3        # e-fold of R
    cool_rate: float = 0.9          # 1/s, relaxation of T_vfx toward the local game gas T
    r_eps: float = 1e-6             # R below this has no defined temperature
    r_cut: float = 1e-3             # R below this is cleared to exactly 0 (no stale flame)

    # --- drawing (§3.5; F3 tunes, F0b proposes) ----------------------------
    kappa: float = 2.5              # B: E = 1 - exp(-kappa * R)
    r0: float = 0.10                # A: E = smoothstep(r0, r1, R)
    r1: float = 0.60
    e_min: float = 0.04             # emission threshold: E below e_min draws
    #                                 nothing, full from 2 e_min (F3: faint R
    #                                 fills the margin boxes, and B's tail
    #                                 1 - exp(-kappa R) never reaches 0)

    seed: int = 1234

    def with_(self, **kw) -> "FireVFXDials":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# The canonical scales (never hardcoded here)
# ---------------------------------------------------------------------------

class CanonicalScales(NamedTuple):
    """The canonical maps the flame reads: the game-T -> Kelvin table
    (``src/temperature_scale.py``) and the ONE blackbody ramp
    (``renderer/blackbody.py``, which owns ``kelvin_glow_min``)."""
    tscale: object          # temperature_scale.TemperatureScale
    ramp: object            # blackbody.BlackbodyRamp

    @property
    def t_amb_k(self) -> float:
        return float(self.tscale.kelvin_ambient)

    @property
    def glow_min_k(self) -> float:
        return float(self.ramp.kelvin_glow_min)

    def to_kelvin(self, t_game: np.ndarray) -> np.ndarray:
        """Dequantized game delta-T -> Kelvin through THE map."""
        return np.asarray(self.tscale.to_kelvin(
            np.asarray(t_game, dtype=np.float64)), dtype=np.float64)


def load_scales(cfg=None) -> CanonicalScales:
    """Build the scales from the live config (``config.CFG`` by default)."""
    from temperature_scale import load as load_tscale
    from .blackbody import BlackbodyRamp
    if cfg is None:
        from config import CFG as cfg
    return CanonicalScales(tscale=load_tscale(cfg),
                           ramp=BlackbodyRamp.from_config(cfg))


# ---------------------------------------------------------------------------
# The source law (§3.3, §13.2, §13.4)
# ---------------------------------------------------------------------------

@dataclass
class FlameSources:
    """Per-TILE source texture: what F1 uploads each tick (sampled nearest).

    ``cover`` 0 = no source; ``COVER_FULL`` = the whole tile; else the OR of
    ``FACE_*`` bits = strips of depth ``strip_frac`` along those faces of
    this tile (solid fuel next door). ``s`` (1/s), ``q`` (1/s) and ``t_src_k``
    are the un-puffed rate, expansion and source temperature; ``puff_hz`` /
    ``puff_phase`` give the feeding cluster's puffing, evaluated per substep
    by :func:`puff_factor`. ``feeder`` is the linear index of the burning tile
    that sets this tile's values (-1 where there is no source) -- for tests
    and debug views.
    """
    s: np.ndarray
    q: np.ndarray
    t_src_k: np.ndarray
    cover: np.ndarray
    puff_hz: np.ndarray
    puff_phase: np.ndarray
    feeder: np.ndarray
    burning: np.ndarray          # the tiles that feed a flame (I, glow floor, not vacuum)
    i_eff: np.ndarray            # their I (0 elsewhere): the "local I" basis


def _hash01(key: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic integer hash -> [0, 1) (render-only; no RNG state)."""
    with np.errstate(over="ignore"):         # wrap-around IS the hash
        x = (np.asarray(key, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             + np.uint64(salt) * np.uint64(0xBF58476D1CE4E5B9))
        x ^= x >> np.uint64(31)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(29)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _clusters(burning: np.ndarray) -> np.ndarray:
    """8-connected labels of burning tiles; label = the cluster's smallest
    linear index (stable while that tile burns), -1 elsewhere."""
    h, w = burning.shape
    lab = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.nonzero(burning)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0] >= 0:
            continue
        stack = [(y0, x0)]
        members = []
        lab[y0, x0] = 0
        while stack:
            y, x = stack.pop()
            members.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and burning[yy, xx]
                            and lab[yy, xx] < 0):
                        lab[yy, xx] = 0
                        stack.append((yy, xx))
        key = min(yy * w + xx for yy, xx in members)
        for yy, xx in members:
            lab[yy, xx] = key
    return lab


def flame_sources(fire: np.ndarray, t_kelvin: np.ndarray, solid: np.ndarray,
                  flammable: np.ndarray, vacuum: np.ndarray, *,
                  t_amb_k: float, glow_min_k: float, tile_size_m: float,
                  dials: FireVFXDials = FireVFXDials(),
                  origin: Tuple[int, int] = (0, 0),
                  map_w: Optional[int] = None) -> FlameSources:
    """THE source law, per tile, pure numpy. All inputs are (h, w) tile
    arrays, read only: ``fire`` = dequantized intensity I, ``t_kelvin`` =
    the tile temperature through the canonical map, ``solid`` /
    ``flammable`` / ``vacuum`` = the GameMap masks (``vacuum`` may carry
    any other "nothing burns or flows here" tiles). ``t_amb_k`` and
    ``glow_min_k`` come from :class:`CanonicalScales` -- never literals.

    CROPPING (F2). The inputs may be a WINDOW of the map whose north-west
    tile is ``origin = (y0, x0)`` on a map ``map_w`` tiles wide (default: the
    inputs ARE the map). Cluster keys and ``feeder`` are then MAP linear
    indices, so a cropped call gives every tile exactly the puffing phase and
    frequency of the full-map call -- provided the window holds every
    burning tile's whole cluster and its four neighbours (the caller's
    contract: crop to the burning tiles' bounding box dilated by >= 1).
    The cost then scales with the fire, not the map (F1 measured ~2.3 ms per
    tick full-map on 128 x 256).
    """
    fire = np.asarray(fire, dtype=np.float64)
    tk = np.asarray(t_kelvin, dtype=np.float64)
    solid = np.asarray(solid, dtype=bool)
    flammable = np.asarray(flammable, dtype=bool)
    vacuum = np.asarray(vacuum, dtype=bool)
    h, w = fire.shape

    burning = (flammable & ~vacuum & (fire > dials.i_min) & (tk >= glow_min_k))
    i_eff = np.where(burning, fire, 0.0)
    rate = dials.s0 * i_eff
    expansion = dials.q0 * i_eff * np.maximum(tk / t_amb_k - 1.0, 0.0)

    oy, ox = int(origin[0]), int(origin[1])
    mw = w if map_w is None else int(map_w)

    def to_map(lin):
        """Window linear index -> map linear index (row-major order kept,
        so the smallest member of a cluster is the same tile either way)."""
        lin = np.asarray(lin, dtype=np.int64)
        return (lin // w + oy) * mw + (lin % w + ox)

    # Puffing per cluster (Cetegen & Ahmed): D = equivalent diameter, m.
    lab = _clusters(burning)
    puff_hz = np.zeros((h, w))
    puff_phase = np.zeros((h, w))
    # per cluster on the burning tiles' own indices (F1: never a full-map
    # mask per cluster -- the cost scales with the fire, not the map)
    by, bx = np.nonzero(lab >= 0)
    keys = to_map(lab[by, bx])
    for key in np.unique(keys):
        sel = keys == key
        d_m = np.sqrt(4.0 * sel.sum() / np.pi) * tile_size_m
        jit = 1.0 + dials.puff_freq_jitter * (2.0 * _hash01(key, dials.seed) - 1.0)
        puff_hz[by[sel], bx[sel]] = dials.puff_c / np.sqrt(max(d_m, 1e-3)) * jit
        puff_phase[by[sel], bx[sel]] = 2.0 * np.pi * _hash01(key, dials.seed + 7)

    s = np.zeros((h, w))
    q = np.zeros((h, w))
    tsrc = np.zeros((h, w))
    cover = np.zeros((h, w), dtype=np.uint8)
    feeder = np.full((h, w), -1, dtype=np.int64)
    out_hz = np.zeros((h, w))
    out_ph = np.zeros((h, w))

    def _offer(ty, tx, fy, fx, bits):
        """Tile (ty, tx) receives a source fed by burning tile (fy, fx)."""
        r = rate[fy, fx]
        if r > s[ty, tx]:
            s[ty, tx] = r
            q[ty, tx] = expansion[fy, fx]
            tsrc[ty, tx] = tk[fy, fx]
            feeder[ty, tx] = (fy + oy) * mw + (fx + ox)
            out_hz[ty, tx] = puff_hz[fy, fx]
            out_ph[ty, tx] = puff_phase[fy, fx]
        cover[ty, tx] |= bits

    open_air = ~solid & ~vacuum
    ys, xs = np.nonzero(burning)
    for y, x in zip(ys, xs):
        if not solid[y, x]:
            _offer(y, x, y, x, COVER_FULL)          # permeable fuel: own tile
            continue
        # solid fuel: the open neighbour's strip along the shared face
        for dy, dx, face in ((-1, 0, FACE_S), (1, 0, FACE_N),
                             (0, -1, FACE_E), (0, 1, FACE_W)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and open_air[ny, nx]:
                _offer(ny, nx, y, x, face)
    cover = np.where(cover & COVER_FULL, COVER_FULL, cover).astype(np.uint8)
    return FlameSources(s=s, q=q, t_src_k=tsrc, cover=cover, puff_hz=out_hz,
                        puff_phase=out_ph, feeder=feeder, burning=burning,
                        i_eff=i_eff)


def puff_factor(hz: np.ndarray, phase: np.ndarray, t: float,
                dials: FireVFXDials) -> np.ndarray:
    """``1 + a sin(theta)``: the pool fire's puffing (Cetegen & Ahmed 1993),
    evaluated per substep (cells without a source get 1).

    ``theta = 2 pi f t + phase + b [sin(2 pi g1 t + 1.7 phase)
    + sin(2 pi g2 t + 2.9 phase)]``: a phase WOBBLE of amplitude ``b``
    (``puff_wobble``) at two slow rates ``g1, g2`` that are themselves
    hashed from the cluster's phase -- each cluster wanders on its own
    clock, so two equal fires (same f) do not stay in step (§6.3 no-sync).
    Everything derives from the per-tile ``(f, phase)`` pair, so F1's
    texture needs no third channel."""
    u = phase / (2.0 * np.pi)
    g1 = hz * (0.17 + 0.22 * np.mod(u * 13.0, 1.0))
    g2 = hz * (0.41 + 0.22 * np.mod(u * 29.0, 1.0))
    theta = (2.0 * np.pi * hz * t + phase
             + dials.puff_wobble * (np.sin(2.0 * np.pi * g1 * t + 1.7 * phase)
                                    + np.sin(2.0 * np.pi * g2 * t + 2.9 * phase)))
    return np.where(hz > 0.0, 1.0 + dials.puff_amp * np.sin(theta), 1.0)


def puff_phases(hz: np.ndarray, phase: np.ndarray, t_base: float):
    """The puffing's three phases at ``t_base``, wrapped to [0, 2 pi) in
    float64, plus its two wobble rates: ``(theta0, theta1, theta2, g1, g2)``
    with ``theta0 = 2 pi f t_base + phase``, ``theta1 = 2 pi g1 t_base + 1.7
    phase``, ``theta2 = 2 pi g2 t_base + 2.9 phase`` (see
    :func:`puff_factor`). F2's per-tile puff texture carries them, so the
    GPU evaluates :func:`puff_factor_rebased` at the SMALL ``tau = t -
    t_base`` -- a float32 ``2 pi f t`` at hours of play would have lost the
    phase (F1 open item 3)."""
    hz = np.asarray(hz, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    u = phase / (2.0 * np.pi)
    g1 = hz * (0.17 + 0.22 * np.mod(u * 13.0, 1.0))
    g2 = hz * (0.41 + 0.22 * np.mod(u * 29.0, 1.0))
    tb = float(t_base)
    two_pi = 2.0 * np.pi
    # f * t_base is reduced mod 1 BEFORE the 2 pi
    th0 = np.mod(two_pi * np.mod(hz * tb, 1.0) + phase, two_pi)
    th1 = np.mod(two_pi * np.mod(g1 * tb, 1.0) + 1.7 * phase, two_pi)
    th2 = np.mod(two_pi * np.mod(g2 * tb, 1.0) + 2.9 * phase, two_pi)
    return th0, th1, th2, g1, g2


def puff_factor_rebased(hz, th0, th1, th2, g1, g2, tau, dials: FireVFXDials):
    """:func:`puff_factor` at ``t = t_base + tau`` from :func:`puff_phases`
    at ``t_base`` -- the same function, re-anchored (what F2's shaders
    evaluate)."""
    two_pi = 2.0 * np.pi
    theta = (th0 + two_pi * hz * tau
             + dials.puff_wobble * (np.sin(th1 + two_pi * g1 * tau)
                                    + np.sin(th2 + two_pi * g2 * tau)))
    return np.where(np.asarray(hz) > 0.0,
                    1.0 + dials.puff_amp * np.sin(theta), 1.0)


def rasterize_sources(src: FlameSources, m: int, strip_frac: float):
    """Per-tile sources -> per-cell (H, W) arrays (s, q, t_src_k, puff_hz,
    puff_phase): nearest per tile, masked by the ``cover`` rule. F1's shader
    evaluates exactly this rule per fragment from the per-tile texture."""
    up = lambda a: np.repeat(np.repeat(a, m, axis=0), m, axis=1)  # noqa: E731
    h, w = src.cover.shape
    f = (np.arange(m) + 0.5) / m
    fy = np.tile(f, h)[:, None] * np.ones((1, w * m))
    fx = np.ones((h * m, 1)) * np.tile(f, w)[None, :]
    cov = up(src.cover.astype(np.int64))
    inside = ((cov & COVER_FULL) > 0)
    inside |= ((cov & FACE_N) > 0) & (fy < strip_frac)
    inside |= ((cov & FACE_S) > 0) & (fy > 1.0 - strip_frac)
    inside |= ((cov & FACE_W) > 0) & (fx < strip_frac)
    inside |= ((cov & FACE_E) > 0) & (fx > 1.0 - strip_frac)
    pick = lambda a: np.where(inside, up(a), 0.0)  # noqa: E731
    return (pick(src.s), pick(src.q), pick(src.t_src_k),
            pick(src.puff_hz), pick(src.puff_phase))


def local_intensity(i_eff: np.ndarray, sigma_tiles: float) -> np.ndarray:
    """"Local I" (§13.2) per tile: the strongest burning tile nearby,
    weighted by a Gaussian of its distance -- so curl noise acts on the
    flame around a fire, not only on the fuel tile. Per tick, tile res."""
    h, w = i_eff.shape
    r = int(np.ceil(2.0 * sigma_tiles))
    out = np.zeros((h, w))
    ys, xs = np.nonzero(i_eff > 0.0)
    if ys.size == 0:
        return out
    # scattered from the burning tiles only (F2: the cost scales with the
    # fire, not the map; the same max as shifting the whole map)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    dy, dx = dy.reshape(-1), dx.reshape(-1)
    wgt = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_tiles ** 2))
    ty = ys[:, None] + dy[None, :]
    tx = xs[:, None] + dx[None, :]
    val = i_eff[ys, xs][:, None] * wgt[None, :]
    ok = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
    np.maximum.at(out, (ty[ok], tx[ok]), val[ok])
    return out


# ---------------------------------------------------------------------------
# Per-tick inputs at cell resolution
# ---------------------------------------------------------------------------

@dataclass
class TickInputs:
    """Everything one tick hands the loop, at cell resolution (H, W)."""
    s: np.ndarray
    q: np.ndarray
    t_src_k: np.ndarray
    puff_hz: np.ndarray
    puff_phase: np.ndarray
    solid: np.ndarray            # no-through, R = 0
    blocked: np.ndarray          # solid | vacuum: R = 0 (§13.6)
    vacuum: np.ndarray           # open (p = 0), R = 0
    u_wind: np.ndarray           # (H, W, 2) tiles/s
    t_gas_k: np.ndarray          # local game temperature, cooling target
    i_local: np.ndarray          # local I for the curl noise
    sources: FlameSources = field(repr=False, default=None)


def _upsample_nearest(a: np.ndarray, m: int) -> np.ndarray:
    return np.repeat(np.repeat(a, m, axis=0), m, axis=1)


def _upsample_bilinear(a: np.ndarray, m: int) -> np.ndarray:
    """Tile field -> cell centres, bilinear with clamped edges (what a
    linearly-filtered per-tile texture returns)."""
    h, w = a.shape
    c = (np.arange(m) + 0.5) / m - 0.5
    ty = (np.arange(h)[:, None] + c[None, :]).reshape(-1)
    tx = (np.arange(w)[:, None] + c[None, :]).reshape(-1)
    yy, xx = np.meshgrid(ty, tx, indexing="ij")
    return _bilinear(a, xx, yy, "clamp")


def prepare_tick(fire, t_game, solid, flammable, vacuum, wind_tamed, *,
                 tps: float, tile_size_m: float, scales: CanonicalScales,
                 dials: FireVFXDials = FireVFXDials()) -> TickInputs:
    """One sim tick's per-tile reads -> the loop's cell inputs.

    ``wind_tamed`` is THE tamed-wind product (``gas_detail.tame_wind``,
    tiles PER TICK, (h, w, 2)); it enters as ``u_wind = tamed * tps`` in
    tiles/s (§13.2). ``t_game`` is the dequantized game temperature; it goes
    through the canonical Kelvin map once, here.
    """
    m = dials.m
    tk = scales.to_kelvin(t_game)
    solid = np.asarray(solid, dtype=bool)
    vacuum = np.asarray(vacuum, dtype=bool)
    src = flame_sources(fire, tk, solid, flammable, vacuum,
                        t_amb_k=scales.t_amb_k, glow_min_k=scales.glow_min_k,
                        tile_size_m=tile_size_m, dials=dials)
    s, q, tsrc, hz, ph = rasterize_sources(src, m, dials.strip_frac)
    wt = np.asarray(wind_tamed, dtype=np.float64) * float(tps)
    u_wind = np.stack([_upsample_bilinear(wt[..., 0], m),
                       _upsample_bilinear(wt[..., 1], m)], axis=-1)
    solid_c = _upsample_nearest(solid, m)
    vac_c = _upsample_nearest(vacuum, m)
    return TickInputs(
        s=s, q=q, t_src_k=tsrc, puff_hz=hz, puff_phase=ph,
        solid=solid_c, blocked=solid_c | vac_c, vacuum=vac_c,
        u_wind=np.where((solid_c | vac_c)[..., None], 0.0, u_wind),
        t_gas_k=_upsample_nearest(tk, m),
        i_local=_upsample_bilinear(
            local_intensity(src.i_eff, dials.i_local_sigma_tiles), m),
        sources=src)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class FireVFXState:
    u: np.ndarray
    p: np.ndarray
    R: np.ndarray
    RT: np.ndarray
    t: float = 0.0               # t_vfx (+ t_idle), s: the noise clock

    @classmethod
    def zeros(cls, h_cells: int, w_cells: int) -> "FireVFXState":
        z = lambda: np.zeros((h_cells, w_cells))  # noqa: E731
        return cls(u=np.zeros((h_cells, w_cells, 2)), p=z(), R=z(), RT=z())

    def temperature(self, r_eps: float) -> np.ndarray:
        """T_vfx = RT / R where R > r_eps, else 0."""
        return np.where(self.R > r_eps, self.RT / np.maximum(self.R, r_eps), 0.0)


# ---------------------------------------------------------------------------
# Sampling helpers (texel-space, cell centre i at coordinate i)
# ---------------------------------------------------------------------------

class _Taps(NamedTuple):
    """The four texel taps of a bilinear lookup at continuous cell coords:
    flat indices (edge-clamped), in-domain flags and the weights. Built once
    per trace, shared by every field sampled along it."""
    idx: tuple
    ok: tuple
    fx: np.ndarray
    fy: np.ndarray


def _taps(x: np.ndarray, y: np.ndarray, h: int, w: int) -> _Taps:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    idx, ok = [], []
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            ok.append((xi >= 0) & (xi < w) & (yi >= 0) & (yi < h))
            idx.append(np.minimum(np.maximum(yi, 0), h - 1) * w
                       + np.minimum(np.maximum(xi, 0), w - 1))
    return _Taps(tuple(idx), tuple(ok), x - x0, y - y0)


def _sample(f: np.ndarray, taps: _Taps, mode: str, stencil: bool = False):
    """Bilinear sample of ``f`` through ``taps``.

    ``mode`` 'clamp' = edge texels extend (velocity, tile fields);
    'zero' = outside the domain is empty (flame scalars: nothing flows in
    from beyond the active region). With ``stencil`` also returns the four
    taps' min and max -- the Selle limiter's stencil."""
    flat = f.reshape(-1)
    vals = [flat[i] if mode == "clamp" else np.where(o, flat[i], 0.0)
            for i, o in zip(taps.idx, taps.ok)]
    v00, v10, v01, v11 = vals
    fx, fy = taps.fx, taps.fy
    val = ((v00 * (1 - fx) + v10 * fx) * (1 - fy)
           + (v01 * (1 - fx) + v11 * fx) * fy)
    if not stencil:
        return val
    lo = np.minimum(np.minimum(v00, v10), np.minimum(v01, v11))
    hi = np.maximum(np.maximum(v00, v10), np.maximum(v01, v11))
    return val, lo, hi


def _bilinear(f: np.ndarray, x: np.ndarray, y: np.ndarray, mode: str):
    """One-off bilinear sample (value only) at continuous cell coords."""
    return _sample(f, _taps(x, y, *f.shape), mode)


_GRID_CACHE: dict = {}


def _cell_grid(h: int, w: int):
    """(jj, ii) cell-index coordinate planes, cached per shape (read only)."""
    g = _GRID_CACHE.get((h, w))
    if g is None:
        jj, ii = np.meshgrid(np.arange(h, dtype=np.float64),
                             np.arange(w, dtype=np.float64), indexing="ij")
        jj.flags.writeable = False
        ii.flags.writeable = False
        g = _GRID_CACHE[(h, w)] = (jj, ii)
    return g


def _backtrace(u: np.ndarray, dt: float, m: int) -> _Taps:
    """Taps at ``x - u dt`` (in cells) for every cell centre."""
    h, w = u.shape[:2]
    jj, ii = _cell_grid(h, w)
    return _taps(ii - u[..., 0] * dt * m, jj - u[..., 1] * dt * m, h, w)


def _nb(a: np.ndarray, dy: int, dx: int, edge: str, fill=0.0):
    """Neighbour values ``a[y+dy, x+dx]`` for a one-axis unit shift; outside
    the domain = the cell's own value ('replicate') or ``fill`` ('const')."""
    out = np.empty_like(a)
    if dx == 1:
        out[:, :-1] = a[:, 1:]
        out[:, -1] = a[:, -1] if edge == "replicate" else fill
    elif dx == -1:
        out[:, 1:] = a[:, :-1]
        out[:, 0] = a[:, 0] if edge == "replicate" else fill
    elif dy == 1:
        out[:-1] = a[1:]
        out[-1] = a[-1] if edge == "replicate" else fill
    elif dy == -1:
        out[1:] = a[:-1]
        out[0] = a[0] if edge == "replicate" else fill
    else:
        raise ValueError("one-axis unit shifts only")
    return out


def _nb_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Neighbour of a bool mask; outside the domain = False (open)."""
    return _nb(np.asarray(mask, dtype=bool), dy, dx, "const", False)


_SOLID_NB_CACHE: dict = {}


def _solid_nb(solid: np.ndarray):
    """{(dy, dx): neighbour-is-solid} for the four faces, plus the count of
    open neighbours -- a pure function of the solid mask, memoised on its
    bytes (the mask changes per tick at most; the solve runs it ~20x per
    substep). F1's shader reads the solid texture's neighbours instead."""
    solid = np.asarray(solid, dtype=bool)
    key = (solid.shape, solid.tobytes())
    hit = _SOLID_NB_CACHE.get(key)
    if hit is None:
        nb = {d: _nb_mask(solid, *d) for d in ((0, 1), (0, -1), (1, 0), (-1, 0))}
        n_open = sum((~v).astype(np.float64) for v in nb.values())
        if len(_SOLID_NB_CACHE) > 32:
            _SOLID_NB_CACHE.clear()
        hit = _SOLID_NB_CACHE[key] = (nb, n_open)
    return hit


# ---------------------------------------------------------------------------
# The stages (§3.4 + §13) -- one function per GL pass
# ---------------------------------------------------------------------------

def advect_velocity(u: np.ndarray, dt: float, m: int) -> np.ndarray:
    """Stage 1a: semi-Lagrangian self-advection of u (Stam 1999)."""
    taps = _backtrace(u, dt, m)
    ux = _sample(u[..., 0], taps, "clamp")
    uy = _sample(u[..., 1], taps, "clamp")
    return np.stack([ux, uy], axis=-1)


def follow_wind(u: np.ndarray, u_wind: np.ndarray, k_follow: float,
                dt: float) -> np.ndarray:
    """Stage 1b: relax u toward the tamed wind (tiles/s), exponentially --
    unconditionally stable for any k_follow * dt."""
    a = 1.0 - np.exp(-k_follow * dt)
    return u + a * (u_wind - u)


def curl_noise_modes(dials: FireVFXDials):
    """The stream function's random Fourier modes (seeded, fixed): wave
    vectors around ``curl_wavelength_tiles``, temporal rates around
    ``curl_rate_hz``, amplitudes normalised so the curl's RMS is ~1. F1
    uploads this table as uniforms."""
    rng = np.random.default_rng(dials.seed)
    n = dials.curl_modes
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    lam = dials.curl_wavelength_tiles * rng.uniform(0.6, 1.5, n)
    k = 2.0 * np.pi / lam
    kx, ky = k * np.cos(ang), k * np.sin(ang)
    omega = 2.0 * np.pi * dials.curl_rate_hz * rng.uniform(0.5, 1.5, n) \
        * rng.choice([-1.0, 1.0], n)
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    amp = 1.0 / (k * np.sqrt(n / 2.0))
    return kx, ky, omega, phase, amp


def curl_mode_phases(modes, t: float) -> np.ndarray:
    """Each mode's temporal phase ``(omega t + phase) mod 2 pi`` in float64
    (the cycle count reduced mod 1 first): F2 uploads these per substep, so
    the shader's float32 argument stays small however long the game runs
    (F1 open item 3)."""
    _, _, om, ph, _ = modes
    two_pi = 2.0 * np.pi
    cyc = np.mod(np.asarray(om, dtype=np.float64) / two_pi * float(t), 1.0)
    return np.mod(two_pi * cyc + np.asarray(ph, dtype=np.float64), two_pi)


def curl_noise(h: int, w: int, m: int, t: float, modes) -> np.ndarray:
    """(H, W, 2) divergence-free noise ``(d psi/dy, -d psi/dx)`` (Bridson
    2007) at cell centres, positions in tiles."""
    kx, ky, om, ph, amp = modes
    y = (np.arange(h) + 0.5) / m
    x = (np.arange(w) + 0.5) / m
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cx = np.zeros((h, w))
    cy = np.zeros((h, w))
    phase_t = curl_mode_phases(modes, t)
    for i in range(len(kx)):
        c = amp[i] * np.cos(kx[i] * xx + ky[i] * yy + phase_t[i])
        cx += ky[i] * c
        cy -= kx[i] * c
    return np.stack([cx, cy], axis=-1)


def confine(u: np.ndarray, i_local: np.ndarray, solid: np.ndarray, t: float,
            dt: float, dials: FireVFXDials, modes) -> np.ndarray:
    """Stage 2: vorticity confinement (Fedkiw, Stam & Jensen 2001) plus curl
    noise scaled by local I (Bridson 2007, §13.2). Both are accelerations,
    applied over ``dt``. Solid cells stay at rest."""
    m = dials.m
    hc = 1.0 / m                                     # cell size, tiles
    ux, uy = u[..., 0], u[..., 1]
    w_ = ((_nb(uy, 0, 1, "replicate") - _nb(uy, 0, -1, "replicate"))
          - (_nb(ux, 1, 0, "replicate") - _nb(ux, -1, 0, "replicate"))) / (2 * hc)
    aw = np.abs(w_)
    gx = (_nb(aw, 0, 1, "replicate") - _nb(aw, 0, -1, "replicate")) / (2 * hc)
    gy = (_nb(aw, 1, 0, "replicate") - _nb(aw, -1, 0, "replicate")) / (2 * hc)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-9
    nx, ny = gx / gn, gy / gn
    fx = dials.vort_eps * hc * (ny * w_)
    fy = dials.vort_eps * hc * (-nx * w_)
    h, w = ux.shape
    cn = curl_noise(h, w, m, t, modes)
    amp = dials.curl_amp * i_local
    fx = fx + amp * cn[..., 0]
    fy = fy + amp * cn[..., 1]
    out = u + dt * np.stack([fx, fy], axis=-1)
    return np.where(solid[..., None], 0.0, out)


def divergence(u: np.ndarray, solid: np.ndarray, m: int) -> np.ndarray:
    """Stage 3a: central-difference divergence (Harris 2004), 1/s. A solid
    neighbour mirrors the normal component (the face between them carries
    no flow); beyond the domain edge the velocity extends (open outflow)."""
    hc = 1.0 / m
    ux, uy = u[..., 0], u[..., 1]
    nb, _ = _solid_nb(solid)
    ue = np.where(nb[(0, 1)], -ux, _nb(ux, 0, 1, "replicate"))
    uw = np.where(nb[(0, -1)], -ux, _nb(ux, 0, -1, "replicate"))
    vs = np.where(nb[(1, 0)], -uy, _nb(uy, 1, 0, "replicate"))
    vn = np.where(nb[(-1, 0)], -uy, _nb(uy, -1, 0, "replicate"))
    d = ((ue - uw) + (vs - vn)) / (2 * hc)
    return np.where(solid, 0.0, d)


def jacobi_sweep(p: np.ndarray, rhs: np.ndarray, solid: np.ndarray,
                 vacuum: np.ndarray, m: int, screen_len_tiles: float) -> np.ndarray:
    """Stage 3b: ONE Jacobi sweep of the SCREENED Poisson problem (§13.1)

        (lap - 1/l^2) p = rhs = div(u*) - q

    p = 0 beyond the domain edge and in vacuum (open: nothing to push
    against); a solid neighbour is Neumann (drops out of the stencil). The
    screening makes every enclosed region's problem unique and bounded --
    a sealed room's warm-started p cannot grow without limit."""
    hc2 = (1.0 / m) ** 2
    scr = hc2 / (screen_len_tiles ** 2)
    nb, n_open = _solid_nb(solid)
    acc = np.zeros_like(p)
    for (dy, dx), nb_solid in nb.items():
        acc += np.where(nb_solid, 0.0, _nb(p, dy, dx, "const", 0.0))
    new = (acc - hc2 * rhs) / (n_open + scr)
    return np.where(solid | vacuum, 0.0, new)


def project(u: np.ndarray, p: np.ndarray, solid: np.ndarray,
            vacuum: np.ndarray, m: int) -> np.ndarray:
    """Stage 3c: subtract grad p (central, same boundary rules as the
    sweep), then enforce no-through: no velocity component points into a
    solid neighbour; solid cells at rest."""
    hc = 1.0 / m
    nb, _ = _solid_nb(solid)

    def pn(dy, dx):
        return np.where(nb[(dy, dx)], p, _nb(p, dy, dx, "const", 0.0))

    gx = (pn(0, 1) - pn(0, -1)) / (2 * hc)
    gy = (pn(1, 0) - pn(-1, 0)) / (2 * hc)
    ux = u[..., 0] - gx
    uy = u[..., 1] - gy
    ux = np.where(nb[(0, 1)], np.minimum(ux, 0.0), ux)
    ux = np.where(nb[(0, -1)], np.maximum(ux, 0.0), ux)
    uy = np.where(nb[(1, 0)], np.minimum(uy, 0.0), uy)
    uy = np.where(nb[(-1, 0)], np.maximum(uy, 0.0), uy)
    out = np.stack([ux, uy], axis=-1)
    return np.where(solid[..., None], 0.0, out)


def advect_flame(R: np.ndarray, RT: np.ndarray, u: np.ndarray, dt: float,
                 dials: FireVFXDials) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 4a: advect R and the premultiplied R*T.

    MacCormack (Selle et al. 2008): forward SL, backward SL, corrected by
    half the round-trip error; where the corrected value leaves the min/max
    of the four texels the forward trace landed between, it falls back to
    the SL value (Selle's revert). Then the TEMPERATURE limiter: R*T may
    not exceed R times the hottest stencil temperature -- R and R*T are
    limited separately above, which bounds each but not their ratio. Both
    together give R >= 0 and no new temperature maxima (the
    ``T_vfx <= T_src`` invariant)."""
    m = dials.m
    taps = _backtrace(u, dt, m)
    T_old = np.where(R > dials.r_eps, RT / np.maximum(R, dials.r_eps), 0.0)
    _, _, t_max = _sample(T_old, taps, "zero", stencil=True)
    taps_fwd = _backtrace(u, -dt, m) if dials.maccormack else None
    out = []
    for phi in (R, RT):
        fwd, lo, hi = _sample(phi, taps, "zero", stencil=True)
        if dials.maccormack:
            back = _sample(fwd, taps_fwd, "zero")
            mc = fwd + 0.5 * (phi - back)
            fwd = np.where((mc < lo) | (mc > hi), fwd, mc)
        out.append(np.maximum(fwd, 0.0))
    R_new, RT_new = out
    RT_new = np.minimum(RT_new, R_new * t_max)
    return R_new, RT_new


def inject(R: np.ndarray, RT: np.ndarray, tick: TickInputs, t: float,
           dt: float, dials: FireVFXDials) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 4b: add ``s * puff * dt`` of flame at ``T_src`` (premultiplied:
    ``s * puff * T_src * dt``) -- a mix, so it never exceeds T_src."""
    add = tick.s * puff_factor(tick.puff_hz, tick.puff_phase, t, dials) * dt
    return R + add, RT + add * tick.t_src_k


def decay_cool(R: np.ndarray, RT: np.ndarray, tick: TickInputs, dt: float,
               dials: FireVFXDials) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 5: R e-folds over ``lifetime_s``; T_vfx relaxes toward the
    local game gas temperature but NEVER rises (§13.5: ``T <- min(T,
    relax(T, T_gas))``); R below ``r_cut`` is cleared to exactly 0; R = 0
    in solids and vacuum (§13.6)."""
    f = np.exp(-dt / dials.lifetime_s)
    R = R * f
    T = np.where(R > dials.r_eps, RT * f / np.maximum(R, dials.r_eps), 0.0)
    a = 1.0 - np.exp(-dials.cool_rate * dt)
    T = np.minimum(T, T + a * (tick.t_gas_k - T))
    dead = tick.blocked | (R < dials.r_cut)
    R = np.where(dead, 0.0, R)
    return R, np.where(dead, 0.0, R * T)


def pressure_rhs(u: np.ndarray, tick: TickInputs, t: float,
                 dials: FireVFXDials) -> np.ndarray:
    """``div(u*) - q * puff`` (§3.3: the expansion survives the projection
    because it is IN the solve, not a kick the solve would remove)."""
    q = tick.q * puff_factor(tick.puff_hz, tick.puff_phase, t, dials)
    return divergence(u, tick.solid, dials.m) - q


def substep(state: FireVFXState, tick: TickInputs, dials: FireVFXDials,
            modes=None) -> FireVFXState:
    """One fixed substep of the loop (§3.4 as amended by §13): exactly the
    stage functions above, in order."""
    if modes is None:
        modes = curl_noise_modes(dials)
    dt, m, t = dials.dt_s, dials.m, state.t
    # 1. advect u (SL); relax toward the tamed wind
    u = advect_velocity(state.u, dt, m)
    u = follow_wind(u, tick.u_wind, dials.k_follow, dt)
    # 2. vorticity confinement + curl noise (local I)
    u = confine(u, tick.i_local, tick.solid, t, dt, dials, modes)
    # 3. divergence - q -> screened Jacobi (warm-started) -> subtract grad p
    rhs = pressure_rhs(u, tick, t, dials)
    p = state.p
    for _ in range(dials.n_jacobi):
        p = jacobi_sweep(p, rhs, tick.solid, tick.vacuum, m, dials.screen_len_tiles)
    u = project(u, p, tick.solid, tick.vacuum, m)
    # 4. advect R and R*T (MacCormack + limiter); inject s, s*T_src
    R, RT = advect_flame(state.R, state.RT, u, dt, dials)
    R, RT = inject(R, RT, tick, t, dt, dials)
    # 5. decay R; cool T toward the local gas T (never heating); R = 0 blocked
    R, RT = decay_cool(R, RT, tick, dt, dials)
    return FireVFXState(u=u, p=p, R=R, RT=RT, t=t + dt)


def extinction_time_s(dials: FireVFXDials, r_peak: float) -> float:
    """Upper bound on how long flame density ``r_peak`` takes to reach
    ``r_cut`` by decay alone, once its sources stop: ``lifetime *
    ln(r_peak / r_cut)``. After that no flame is left anywhere."""
    return dials.lifetime_s * np.log(max(r_peak, dials.r_cut) / dials.r_cut)


# ---------------------------------------------------------------------------
# Emission (§3.5) -- linear HDR; the resolve tone-maps, never this module
# ---------------------------------------------------------------------------

def flame_emission(state: FireVFXState, ramp, mode: str,
                   dials: FireVFXDials):
    """``L_flame = Le(T_vfx) * E(R)``, linear HDR, through the ONE ramp.

    Returns ``(Le (H, W, 3), E (H, W), cover (H, W))``: Le is the ramp's
    chroma times its intensity at T_vfx (``BlackbodyRamp.emission_at_kelvin``
    -- no second T->colour map), E the opacity for mode 'A' (temperature
    only: ``smoothstep(r0, r1, R)``) or 'B' (temperature + thickness:
    ``1 - exp(-kappa R)``), clipped by the emission threshold
    (:func:`emission_clip`), and ``cover = E * w(T)`` the §13.4 emitter-cover
    weight (w = ``BlackbodyRamp.cover_weight``: the ramp's intensity
    capped at 1, so a dim flame covers little, a cold one nothing). The GL
    draw (``shaders/fire_glow.fs``) is this law on the upsampled, edge-
    eroded field."""
    T = state.temperature(dials.r_eps)
    chroma, inten = ramp.emission_at_kelvin(np.where(T > 0, T, ramp.kelvin_floor))
    inten = np.where(T > 0, inten, 0.0)
    Le = chroma * inten[..., None]
    R = state.R
    if mode == "A":
        z = np.clip((R - dials.r0) / max(dials.r1 - dials.r0, 1e-9), 0.0, 1.0)
        E = z * z * (3.0 - 2.0 * z)
    elif mode == "B":
        E = 1.0 - np.exp(-dials.kappa * R)
    else:
        raise ValueError(f"mode must be 'A' or 'B', not {mode!r}")
    E = E * emission_clip(E, dials.e_min)
    cover = E * ramp.cover_weight(np.where(T > 0, T, 0.0))
    return Le, E, cover


def emission_clip(E, e_min: float):
    """The emission threshold (F3): a weight that is 0 for ``E <= e_min``,
    1 for ``E >= 2 e_min``, smoothstep between -- faint flame density draws
    nothing (B's ``1 - exp(-kappa R)`` tail would otherwise fill the active
    boxes with a faint glow). ``shaders/fire_glow.fs`` applies the same."""
    if e_min <= 0.0:
        return np.ones_like(np.asarray(E, dtype=np.float64))
    z = np.clip((np.asarray(E, dtype=np.float64) - e_min) / e_min, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)
