"""#217 finding 1 / #72 -- THE EOS CHECKERBOARD BENCH (measurement only).

The gas solver carries an unstable cell-to-cell (odd-even, "checkerboard")
pattern: neighbouring tiles drift apart in density while the stored pressure
stays flat. This harness reproduces it on ``levels/bench_two_room`` (two
sealed 12x12 rooms, one door tile, one crate) and measures it per tick.

Scenarios (``--case``):
  still   all ambient, no seed: the control, must stay perfectly still.
  cb      every other tile +/-``--amp`` (default 0.1 %) bulk air, T ambient.
  warm    the left room +40 degrees, its air thinned to equal pressure
          (#217's first comment).

Every case runs at any tile size (``--tile``, metres; the level's tilemap is
kept and only ``tile_size_m`` changes, as the issue's probes did with their
TILE variable) and with EOS dial overrides (``--eos adiabatic_index=1.0001``
switches step 6's flow work off: k_work = (gamma-1)*T_AMB).

Metrics per tick, over the accountable gas cells (``gmap._gas_energy_
accountable()``):
  gap_rms    rms of (P_stored - P_state)/P_state, P_stored = ``atmosphere``,
             P_state = C*N*T_abs (the air's own pressure)
  gap_corr   lag-1 neighbour correlation of that gap (over every
             horizontally / vertically adjacent accountable pair)
  cb(f)      the amplitude of the (-1)^(x+y) component of f: the projection
             sum(s*(f - room mean))/sum(s^2) with s = (-1)^(x+y); for f =
             f0 + a*s it is a. Reported for N (relative), u_x, u_y (m/s),
             P_stored (relative), T (game degrees) and gas_energy/N (= T_abs)
  T min/max  game degrees, relative to ambient, over the accountable cells
  |u| max    m/s

``--steps`` adds the per-step attribution (deliverable 2 of the #217 patch):
each tick is split, from the engine's own debug outputs and test-only
references, into the increments of step 1 (substeps: SL advection of u +
the donor-cell mass flux), step 3 (the solve: p* -> P), step 4 (the kick +
its absorb / sponge / cap / drag chain) and step 6+7 (energy), each
projected on the checkerboard mode. See ``attribute_tick``.

Seeding goes through the scenario-builder seam only: the bulk planes are
written BEFORE the first tick, then ``reseed_gas_energy`` /
``seed_gas_temperature`` restricted to the cells written (the gas energy
seam's documented builder route; a bare ``temperature[...] =`` goes vacuous,
#82).

HARNESS, not a pytest gate (``_`` prefix). The collected property test is
``tests/test_eos_checkerboard.py``.

Run:
    <breach-py> tests/_eos_checkerboard_bench.py --case cb --tile 0.333 --seconds 20
    <breach-py> tests/_eos_checkerboard_bench.py --table            # the issue's table
    <breach-py> tests/_eos_checkerboard_bench.py --case cb --steps --ticks 40
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import temperature_scale  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402

FIXTURE = "bench_two_room"
Q = 65536.0
TPS = int(getattr(CFG.clock, "ticks_per_second", 24))


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------
def build(tile_m: float = 0.333, case: str = "cb", amp: float = 1e-3,
          warm_deg: float = 40.0, eos: dict | None = None, seed: int = 1):
    """Load the fixture at ``tile_m`` and seed ``case``. Returns the sim.

    Seeding happens before the first tick: the bulk planes are scaled
    directly and the stored energy is re-derived on exactly the cells
    written (``reseed_gas_energy`` / ``seed_gas_temperature``)."""
    level = load_level(FIXTURE, levels_dir=str(ROOT / "levels"))
    level = dataclasses.replace(level, tile_size_m=float(tile_m))
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    for k, v in (eos or {}).items():
        setattr(sim.physics_runner.eos, k, type(getattr(sim.physics_runner.eos, k))(v))
    g = sim.gmap
    acc = g._gas_energy_accountable()
    cons = np.flatnonzero(g.gases.conservative)
    if case == "still":
        pass
    elif case == "cb":
        yy, xx = np.indices(acc.shape)
        s = np.where((xx + yy) % 2 == 0, 1.0, -1.0)
        fac = 1.0 + amp * s
        for gi in cons:
            g.gas[gi][acc] = np.rint(g.gas[gi][acc].astype(np.float64)
                                     * fac[acc]).astype(g.gas.dtype)
        g.reseed_gas_energy(acc)
    elif case == "warm":
        left = acc & (np.indices(acc.shape)[1] < door_column(g))
        t_amb = temperature_scale.load(CFG).eos_t_amb_k
        thin = t_amb / (t_amb + warm_deg)
        for gi in cons:
            g.gas[gi][left] = np.rint(g.gas[gi][left].astype(np.float64)
                                      * thin).astype(g.gas.dtype)
        g.seed_gas_temperature(left, int(round(warm_deg * Q)))
    else:
        raise ValueError(f"unknown case {case!r}")
    return sim


def door_column(g) -> int:
    """The partition's column: the interior column that is solid on all rows
    but one (the door). Read from the map, never assumed."""
    sol = g.solid
    h, w = sol.shape
    best, best_n = w // 2, -1
    for x in range(1, w - 1):
        n = int(sol[1:h - 1, x].sum())
        if n > best_n:
            best, best_n = x, n
    return best


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class Probe:
    """The per-tick metrics, over the accountable cells."""

    def __init__(self, sim):
        self.sim = sim
        g = sim.gmap
        self.acc = g._gas_energy_accountable()
        h, w = self.acc.shape
        yy, xx = np.indices((h, w))
        self.s = np.where((xx + yy) % 2 == 0, 1.0, -1.0)
        self.t_amb = temperature_scale.load(CFG).eos_t_amb_k
        # rooms: connected components of the accountable set with the door
        # tile cut out, so a per-room mean can be removed before projecting
        # (a warm room's mean is not a checkerboard).
        dc = door_column(g)
        rooms = np.zeros((h, w), dtype=np.int32)
        rooms[self.acc & (xx < dc)] = 1
        rooms[self.acc & (xx > dc)] = 2
        self.rooms = rooms
        # neighbour pairs (both accountable), for the lag-1 correlation
        a = self.acc
        self.pairs_e = a[:, :-1] & a[:, 1:]
        self.pairs_s = a[:-1, :] & a[1:, :]

    # --- raw planes, in real units -----------------------------------------
    def planes(self):
        g = self.sim.gmap
        n = g._gas_bulk_n_raw().astype(np.float64) / Q          # N (1.0 = ambient)
        t = g.temperature.astype(np.float64) / Q                 # game deg rel.
        p = g.atmosphere.astype(np.float64) / Q                  # atm
        ux = g.wind_x.astype(np.float64) / Q
        uy = g.wind_y.astype(np.float64) / Q
        e = g.gas_energy.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            eabs = np.where(n > 0, e / (n * Q) / Q, 0.0)         # E/N = T_abs (K)
        return dict(N=n, T=t, P=p, ux=ux, uy=uy, Eabs=eabs)

    def cb(self, f, rel=False):
        """Amplitude of the (-1)^(x+y) component of f over the rooms
        (each room's own mean removed first)."""
        num = den = 0.0
        mean_all = 0.0
        for r in (1, 2):
            m = self.rooms == r
            if not m.any():
                continue
            fr = f[m]
            mu = fr.mean()
            num += float((self.s[m] * (fr - mu)).sum())
            den += float(m.sum())
            mean_all += mu * m.sum()
        amp = num / den if den else 0.0
        if rel:
            amp /= (mean_all / den)
        return amp

    def gap(self, pl=None):
        pl = pl or self.planes()
        p_state = pl["N"] * (pl["T"] + self.t_amb) / self.t_amb
        with np.errstate(divide="ignore", invalid="ignore"):
            gp = np.where(self.acc & (p_state > 0), (pl["P"] - p_state) / p_state, 0.0)
        return gp

    def corr1(self, f):
        a = np.concatenate([f[:, :-1][self.pairs_e], f[:-1, :][self.pairs_s]])
        b = np.concatenate([f[:, 1:][self.pairs_e], f[1:, :][self.pairs_s]])
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    def row(self):
        pl = self.planes()
        gp = self.gap(pl)
        acc = self.acc
        umag = np.hypot(pl["ux"], pl["uy"])
        return dict(
            gap_rms=float(np.sqrt((gp[acc] ** 2).mean())),
            gap_corr=self.corr1(gp),
            cbN=self.cb(pl["N"], rel=True),
            cbux=self.cb(pl["ux"]), cbuy=self.cb(pl["uy"]),
            cbP=self.cb(pl["P"], rel=True),
            cbT=self.cb(pl["T"]),
            cbE=self.cb(pl["Eabs"]),
            Tmin=float(pl["T"][acc].min()), Tmax=float(pl["T"][acc].max()),
            umax=float(umag[acc].max()),
            Pspread=float(pl["P"][acc].max() - pl["P"][acc].min()),
        )


def step(sim):
    sim.set_paused(False)
    sim.step()


def run(tile_m=0.333, case="cb", amp=1e-3, ticks=480, eos=None, every=1,
        quiet=False):
    sim = build(tile_m, case, amp, eos=eos)
    pr = Probe(sim)
    rows = [dict(tick=0, **pr.row())]
    for t in range(1, ticks + 1):
        step(sim)
        if t % every == 0 or t == ticks:
            rows.append(dict(tick=t, **pr.row()))
    if not quiet:
        print_rows(rows, f"case={case} tile={tile_m} amp={amp} eos={eos or {}}")
    return rows, sim


def print_rows(rows, title):
    print(f"# {title}")
    print(f"{'tick':>5} {'gap_rms%':>9} {'corr':>6} {'cbN%':>9} {'cbP%':>9} "
          f"{'cbux':>8} {'cbuy':>8} {'cbT':>8} {'Tmin':>7} {'Tmax':>7} {'|u|max':>7}")
    for r in rows:
        print(f"{r['tick']:5d} {100*r['gap_rms']:9.4f} {r['gap_corr']:6.2f} "
              f"{100*r['cbN']:9.4f} {100*r['cbP']:9.5f} {r['cbux']:8.4f} "
              f"{r['cbuy']:8.4f} {r['cbT']:8.3f} {r['Tmin']:7.2f} {r['Tmax']:7.2f} "
              f"{r['umax']:7.3f}")


def parse_eos(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="cb", choices=("still", "cb", "warm"))
    ap.add_argument("--tile", type=float, default=0.333)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("--eos", action="append", default=[],
                    help="EOS dial override key=value (repeatable)")
    a = ap.parse_args(argv)
    ticks = a.ticks if a.ticks is not None else int(round((a.seconds or 20.0) * TPS))
    run(a.tile, a.case, a.amp, ticks, eos=parse_eos(a.eos), every=a.every)


if __name__ == "__main__":
    main()
