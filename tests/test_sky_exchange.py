"""Sky exchange — planetside volumetric O2 replenishment (gates a/b/c).

docs/sky_exchange_design_2026-07-24.md (Fable, fire-tuning §7 Q2 → Option A).
Every sky-connected interior air tile relaxes its gas COMPOSITION toward ambient
at FIXED local N_total, once per tick, on the host mirror after combustion.

Gates pinned here:
  a. Dormancy / byte-identity — λ == 0, empty mask, and space maps are no-ops;
     a space map's sky_mask is all-false.
  b. Sealed-room identity — the flood fill excludes a room walled off from the
     ring (no sky); a breach EXPANDS the mask (the room starts breathing).
  c. Conservation — N_total per tile is invariant to the LSB; the sky_flux rail
     equals the actual applied per-plane delta and is a pure transfer
     (flux[O2] == -flux[inert]); only the O2/inert planes move.
Plus: linear relaxation toward mul_q16(o2_frac, N_total) with time-constant ≈ τ,
sign-symmetric rounding, and the [0, N_total] clamp. The pass is exercised
against a Python oracle that mirrors the C++ integer math bit-for-bit.

Run:
    conda run -n data python -m pytest tests/test_sky_exchange.py -q
"""
from __future__ import annotations

import struct
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import level_loader  # noqa: E402
from simulation import gas_fixed  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402

FP_ONE = 1 << 16
O2_FRAC_Q = int(gas_fixed.quantize_scalar(0.21))    # 13763 — one truth w/ the ring
SPACE = 9                                            # v2 SPACE tile code
HULL = 1
AIR = 0


# ---------------------------------------------------------------------------
# Python oracle — the EXACT integer math of cpp/src/sky_exchange.cpp
# ---------------------------------------------------------------------------
def _mul_q16_floor(a: int, n_tot: int) -> int:
    """mul_q16(a, n_tot) for non-negative operands: (a*n_tot) >> 16 (floor)."""
    return (int(a) * int(n_tot)) >> 16


def _narrow_round_signed(wide: int) -> int:
    """fixedpoint::narrow_round_signed — round |wide|/2^16 half-up, re-sign."""
    HALF = 1 << 15
    if wide >= 0:
        return (wide + HALF) >> 16
    return -(((-wide) + HALF) >> 16)


def _ref_sky_step(gas, o2_idx, inert_idx, mask, o2_frac_q, lambda_q):
    """Apply one sky-exchange step to a COPY, returning (gas_out, sky_flux).
    Bit-for-bit mirror of the C++ pass — the oracle the binding must match."""
    gas = gas.copy()
    n_gases = gas.shape[0]
    flux = np.zeros(n_gases, dtype=np.int64)
    if lambda_q == 0:
        return gas, flux
    m = mask.reshape(-1)
    o2 = gas[o2_idx].reshape(-1)
    inr = gas[inert_idx].reshape(-1)
    tot_flux = 0
    for i in range(m.size):
        if not m[i]:
            continue
        o2_old = int(o2[i]); in_old = int(inr[i])
        n_tot = o2_old + in_old
        target = _mul_q16_floor(o2_frac_q, n_tot)
        diff = target - o2_old
        dN = _narrow_round_signed(lambda_q * diff)
        o2_new = o2_old + dN
        if o2_new < 0:
            o2_new = 0
        if o2_new > n_tot:
            o2_new = n_tot
        o2[i] = o2_new
        inr[i] = n_tot - o2_new
        tot_flux += (o2_new - o2_old)
    flux[o2_idx] += tot_flux
    flux[inert_idx] -= tot_flux
    return gas, flux


def _run(gas, mask, o2_frac_q, lambda_q, o2_idx=O2, inert_idx=INERT_N2):
    """Drive the real binding in place; return the accumulated sky_flux."""
    flux = np.zeros(gas.shape[0], dtype=np.int64)
    bp.sky_exchange_step(gas, o2_idx, inert_idx, mask.astype(bool),
                         int(o2_frac_q), int(lambda_q), flux)
    return flux


def _rand_scene(rng, h=12, w=10, n_gases=7):
    """A random gas field + mask. Gas planes are non-negative Q16 counts; the
    O2/inert split is arbitrary (not necessarily ambient) so relaxation moves."""
    gas = rng.integers(0, 4 * FP_ONE, size=(n_gases, h, w), dtype=np.int64
                       ).astype(np.int32)
    mask = rng.integers(0, 2, size=(h, w), dtype=np.int8).astype(bool)
    return gas, mask


# ---------------------------------------------------------------------------
# The binding matches the oracle (the master gate — determinism reference)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_binding_matches_oracle(seed):
    rng = np.random.default_rng(seed)
    gas, mask = _rand_scene(rng)
    lam = int(gas_fixed.quantize_scalar((1.0 / 24.0) / 60.0))   # τ=60 @ 24 Hz
    got = gas.copy()
    flux = _run(got, mask, O2_FRAC_Q, lam)
    exp, exp_flux = _ref_sky_step(gas, O2, INERT_N2, mask, O2_FRAC_Q, lam)
    assert np.array_equal(got, exp), "binding diverged from the integer oracle"
    assert np.array_equal(flux, exp_flux), "sky_flux rail diverged from oracle"


def test_multi_tick_matches_oracle():
    """Bit-identity holds across a long run (the resident tick reuses this same
    host call, so this is also the CPU==CUDA lockstep reference)."""
    rng = np.random.default_rng(99)
    gas, mask = _rand_scene(rng, h=20, w=16)
    lam = int(gas_fixed.quantize_scalar((1.0 / 24.0) / 45.0))
    got = gas.copy()
    ref = gas.copy()
    for _ in range(200):
        _run(got, mask, O2_FRAC_Q, lam)
        ref, _ = _ref_sky_step(ref, O2, INERT_N2, mask, O2_FRAC_Q, lam)
    assert np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# Gate a — dormancy / no-op
# ---------------------------------------------------------------------------
def test_lambda_zero_is_noop():
    rng = np.random.default_rng(1)
    gas, mask = _rand_scene(rng)
    before = gas.copy()
    flux = _run(gas, mask, O2_FRAC_Q, 0)
    assert np.array_equal(gas, before)
    assert not flux.any()


def test_empty_mask_is_noop():
    rng = np.random.default_rng(2)
    gas, _ = _rand_scene(rng)
    mask = np.zeros(gas.shape[1:], dtype=bool)
    before = gas.copy()
    flux = _run(gas, mask, O2_FRAC_Q, 4096)
    assert np.array_equal(gas, before)
    assert not flux.any()


# ---------------------------------------------------------------------------
# Gate c — conservation + the rail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_n_total_invariant_per_tile(seed):
    rng = np.random.default_rng(seed)
    gas, mask = _rand_scene(rng)
    lam = int(gas_fixed.quantize_scalar((1.0 / 24.0) / 60.0))
    tot_before = gas[O2].astype(np.int64) + gas[INERT_N2].astype(np.int64)
    _run(gas, mask, O2_FRAC_Q, lam)
    tot_after = gas[O2].astype(np.int64) + gas[INERT_N2].astype(np.int64)
    assert np.array_equal(tot_before, tot_after), "N_total per tile not invariant"


@pytest.mark.parametrize("seed", range(6))
def test_rail_is_exact_and_a_pure_transfer(seed):
    rng = np.random.default_rng(seed + 10)
    gas, mask = _rand_scene(rng)
    lam = int(gas_fixed.quantize_scalar((1.0 / 24.0) / 30.0))
    o2_before = gas[O2].astype(np.int64).copy()
    flux = _run(gas, mask, O2_FRAC_Q, lam)
    applied = (gas[O2].astype(np.int64) - o2_before).sum()
    assert flux[O2] == applied                       # rail == actual O2 delta
    assert flux[INERT_N2] == -applied                # a pure transfer
    # rail closes the per-plane budget: Δtotal(plane) == sky_flux[plane]
    assert flux[O2] == (gas[O2].astype(np.int64).sum() - o2_before.sum())


def test_only_o2_and_inert_planes_move():
    rng = np.random.default_rng(7)
    gas, mask = _rand_scene(rng, n_gases=7)
    lam = int(gas_fixed.quantize_scalar((1.0 / 24.0) / 60.0))
    others = [p for p in range(gas.shape[0]) if p not in (O2, INERT_N2)]
    before = {p: gas[p].copy() for p in others}
    flux = _run(gas, mask, O2_FRAC_Q, lam)
    for p in others:
        assert np.array_equal(gas[p], before[p]), f"plane {p} moved"
        assert flux[p] == 0


# ---------------------------------------------------------------------------
# Relaxation toward the ambient composition + time-constant
# ---------------------------------------------------------------------------
def _one_tile(o2, inert, h=1, w=1, n_gases=7):
    gas = np.zeros((n_gases, h, w), dtype=np.int32)
    gas[O2, 0, 0] = o2
    gas[INERT_N2, 0, 0] = inert
    mask = np.ones((h, w), dtype=bool)
    return gas, mask


def test_relaxes_into_the_ambient_deadband():
    """A fully depleted tile (O2 == 0) at ambient N_total gains O2 monotonically
    toward target = mul_q16(o2_frac, N_total), and settles JUST inside the
    integer relaxation's round-to-nearest DEADBAND.

    That deadband is inherent to the design's per-tick Q16 relaxation: once
    |diff| < 0.5/λ counts, λ·diff rounds to 0 and the tile stops. It is ≈
    0.5·N_total/λ counts (~712 at λ=46 / τ=60 s → the far field rests at ~0.199
    mole fraction, not exactly 0.21 — still ≥ 0.19, gate d). It GROWS with τ, so
    P3's τ sweep must weigh it (τ=120 → ~0.188, under the 0.19 floor at rest)."""
    n_tot = FP_ONE
    target = _mul_q16_floor(O2_FRAC_Q, n_tot)         # 13763
    gas, mask = _one_tile(0, n_tot)
    lam = int(gas_fixed.quantize_scalar((1.0 / 24.0) / 60.0))   # 46
    prev = -1
    for _ in range(20000):                            # ≫ enough to hit the floor
        _run(gas, mask, O2_FRAC_Q, lam)
        cur = int(gas[O2, 0, 0])
        assert cur >= prev                            # monotone up toward target
        assert 0 <= cur <= n_tot
        prev = cur
    o2 = int(gas[O2, 0, 0])
    deadband = n_tot // (2 * lam) + 1                 # ≈ 0.5·N_total/λ (+1 LSB)
    assert 0 < target - o2 <= deadband                # settled just inside it
    assert o2 + int(gas[INERT_N2, 0, 0]) == n_tot     # complement stays exact


def test_time_constant_is_about_tau():
    """After τ seconds the gap to ambient should fall to ≈ e^-1 (~0.368) of its
    initial value — the linear relaxation's defining property."""
    n_tot = FP_ONE
    target = _mul_q16_floor(O2_FRAC_Q, n_tot)
    gas, mask = _one_tile(0, n_tot)
    dt = 1.0 / 24.0
    tau = 60.0
    lam = int(gas_fixed.quantize_scalar(dt / tau))
    n_steps = int(round(tau / dt))                    # one time-constant
    for _ in range(n_steps):
        _run(gas, mask, O2_FRAC_Q, lam)
    gap_frac = (target - int(gas[O2, 0, 0])) / target
    assert 0.30 <= gap_frac <= 0.42, f"gap after τ = {gap_frac:.3f}, expected ~0.368"


def test_sign_symmetric_rounding():
    """A tile ENRICHED by the same magnitude it would otherwise be depleted
    loses exactly what the depleted tile gains — no directional (DC) bias."""
    n_tot = 4 * FP_ONE
    target = _mul_q16_floor(O2_FRAC_Q, n_tot)
    lam = 5000
    # depleted: diff = +target ; enriched: diff = -target (o2 = 2*target)
    dep, m = _one_tile(0, n_tot)
    enr, _ = _one_tile(2 * target, n_tot - 2 * target)
    _run(dep, m, O2_FRAC_Q, lam)
    _run(enr, m, O2_FRAC_Q, lam)
    d_dep = int(dep[O2, 0, 0]) - 0
    d_enr = int(enr[O2, 0, 0]) - 2 * target
    assert d_dep == -d_enr, "rounding is not sign-symmetric"


def test_clamp_never_leaves_bounds():
    """A near-instant λ (≈1.0) drives O2 to the target in one step but never
    outside [0, N_total]; the inert plane restates the exact complement."""
    n_tot = 3 * FP_ONE
    lam = FP_ONE - 1                                  # ≈ 1.0 relaxation
    for o2_start in (0, n_tot, n_tot // 3, n_tot - 1):
        gas, mask = _one_tile(o2_start, n_tot - o2_start)
        _run(gas, mask, O2_FRAC_Q, lam)
        o2 = int(gas[O2, 0, 0]); inr = int(gas[INERT_N2, 0, 0])
        assert 0 <= o2 <= n_tot
        assert inr == n_tot - o2
        assert o2 + inr == n_tot


# ---------------------------------------------------------------------------
# Mask gates (b) — via a real planetside GameMap
# ---------------------------------------------------------------------------
def _write_png(path: Path, w: int = 8, h: int = 6) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


_PREFIX = ('version = "2"\nname = "sky"\ntilemap = "tilemap.csv"\n'
           'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n')


def _level(tmp_path: Path, tilemap: np.ndarray, *, boundary="ambient",
           body="") -> str:
    d = tmp_path / "lvl"
    d.mkdir()
    np.savetxt(d / "tilemap.csv", tilemap, fmt="%d", delimiter=",")
    _write_png(d / "diffuse.png")
    prefix = _PREFIX + (f'boundary = "{boundary}"\n' if boundary else "")
    (d / "level.toml").write_text(prefix + "\n" + body, encoding="utf-8",
                                  newline="\n")
    return str(d)


def _open_ambient_tilemap(n=16):
    tm = np.full((n, n), SPACE, dtype=np.int32)
    tm[1:-1, 1:-1] = AIR
    return tm


def _sealed_pocket_tilemap(n=16):
    """Open ambient room with a hollow HULL box enclosing a 2x2 air pocket."""
    tm = _open_ambient_tilemap(n)
    r0, r1, c0, c1 = 6, 9, 6, 9                       # box wall extents (inclusive)
    tm[r0:r1 + 1, c0] = HULL
    tm[r0:r1 + 1, c1] = HULL
    tm[r0, c0:c1 + 1] = HULL
    tm[r1, c0:c1 + 1] = HULL
    tm[r0 + 1:r1, c0 + 1:c1] = AIR                    # 2x2 sealed pocket
    return tm, (7, 7)                                 # a pocket-interior coord


def test_space_map_sky_mask_empty(tmp_path):
    """Gate a: a space map has no ring → sky_mask all-false → pass dead."""
    tm = np.full((12, 12), SPACE, dtype=np.int32)
    tm[1:-1, 1:-1] = AIR
    g = GameMap(level_loader.load(_level(tmp_path, tm, boundary="space")))
    assert not g.is_ambient.any()
    assert not g.sky_mask.any()


def test_open_ambient_mask_is_interior_air(tmp_path):
    """Every open interior air tile is sky-connected; the ring itself and solids
    are excluded."""
    g = GameMap(level_loader.load(_level(tmp_path, _open_ambient_tilemap())))
    expected = (~g.solid) & (~g.is_ambient) & (~g.is_vacuum)
    assert np.array_equal(g.sky_mask, expected)
    assert not g.sky_mask[g.is_ambient].any()         # ring excluded
    assert g.sky_mask.any()


def test_sealed_pocket_excluded_from_mask(tmp_path):
    """Gate b: air walled off from the ring gets no sky."""
    tm, (py, px) = _sealed_pocket_tilemap()
    g = GameMap(level_loader.load(_level(tmp_path, tm)))
    assert not g.sky_mask[py, px], "sealed pocket must have no sky"
    assert g.sky_mask[2, 2], "open interior air must be sky-connected"


def test_breach_expands_the_mask(tmp_path):
    """Gate b: breaching a wall of the sealed pocket makes it start breathing —
    the dirty flag drives a rebuild at ensure_sky_mask, and the pocket joins."""
    tm, (py, px) = _sealed_pocket_tilemap()
    g = GameMap(level_loader.load(_level(tmp_path, tm)))
    assert not g.sky_mask[py, px]
    g.destroy_wall(6, 7)                              # top wall of the box → opens up
    assert g._sky_mask_dirty, "structural edit must dirty the sky mask"
    mask = g.ensure_sky_mask()
    assert mask[py, px], "breached pocket must join the sky mask"
    assert not g._sky_mask_dirty, "rebuild must clear the dirty flag"


# ---------------------------------------------------------------------------
# Loader — sky_tau_s parse + validation (mirrors the sponge-dial house style)
# ---------------------------------------------------------------------------
def _load_ambient(tmp_path, body):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return level_loader.load(_level(tmp_path, _open_ambient_tilemap(),
                                        body=body))


def test_sky_tau_s_defaults_to_dormant(tmp_path):
    lvl = _load_ambient(tmp_path, "")
    assert lvl.ambient.sky_tau_s == 0.0               # absent → dormant (back-compat)


def test_sky_tau_s_explicit_parses(tmp_path):
    lvl = _load_ambient(tmp_path, "[ambient]\nsky_tau_s = 60.0\n")
    assert lvl.ambient.sky_tau_s == 60.0
    assert lvl.ambient.o2_frac_q == O2_FRAC_Q         # composition target source


def test_sky_tau_s_negative_hard_error(tmp_path):
    with pytest.raises(ValueError, match="sky_tau_s"):
        _load_ambient(tmp_path, "[ambient]\nsky_tau_s = -1.0\n")
