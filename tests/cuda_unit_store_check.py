"""#29 P6.1 -- THE DEVICE UNIT STORE GATE: the round trip and the SHADOW
LOCKSTEP of slot 1c (exposure decay), tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_store.py). The store is ``simulation.unit_store``'s
``DeviceStore`` (an (N, U, W) int64 buffer, ``cpp/src/cuda_unit_store.cu``);
the reference slot is ``exposure.decay_units``; the harness is
``tests/cuda_unit_lockstep.py``.

PART 1 -- ROUND TRIP. At N = 1, 2 and 8: rows of live units (ship_demo's
match, packed) and random int64 rows over the whole range go up and come back
bit for bit; two uploads back to back land the second (the pinned staging is
never overwritten under a copy in flight); a store whose width is not the
compiled layout's is refused.

PART 1b -- THE KERNEL ON SYNTHETIC ROWS: the exposure kernel against
``exposure.decay_step`` over the whole int64 range (0, negatives, 1, values
near 2^63) at eight decays, on valid rows, while padding rows (valid 0)
carrying the same garbage come back untouched; the coverage counter equals
the rows the law changed; a decay outside [0, 65536] is refused.

PART 2 -- SHADOW LOCKSTEP, N = 1, ship_demo's bots match (seed 1, the horde,
the shipped ``[damage.exposure]``: k_expose 1, a 3-tick half life -- the
carbine's hits raise exposure from tick ~437 and it decays to 0 between
volleys). Every tick of the hooked match: the device's rows after the kernel
equal the host's rows after ``decay_units`` (every column); the kernel's
coverage counter equals the rows the host changed, and is > 0 on part of the
window; and the hooked match plays EXACTLY as an unhooked twin (per-tick unit
hash and every-tick field digest equal) -- hooking cannot change the
reference.

PART 3 -- SHADOW LOCKSTEP, BATCHED, N = 2 and N = 8: the focus-fire bench's
arena (tools/bench_focus_fire.py), env e with a different number of shooters
(so U differs and padding rows ride along) firing in "continuous" or "single"
mode at a marine of 10 000 hp; every tick env i of the batch equals the host
AND equals env i run alone through an N = 1 store.

PROPERTY: the device kernel is ``exposure.decay_step`` on every valid row,
integer for integer, in a batch as alone. BREAKS IF the kernel's decay
differs from the law (a rounding, a floor, the max(1, .) step), it touches a
padding row, it reads another env's rows, a copy loses or reorders bytes, or
hooking a slot changes the game. Each validated by breaking it (the patch's
report).

Prints TIMING lines and ``UNIT_STORE_RESULT: PASS`` / ``FAIL``; exits 0 / 1.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import DeviceSlot, ShadowLockstep  # noqa: E402
from simulation import exposure, unit_store  # noqa: E402

FAILS: list = []
SHIP_TICKS = int(os.environ.get("UNIT_STORE_SHIP_TICKS", "700"))
ARENA_TICKS = int(os.environ.get("UNIT_STORE_ARENA_TICKS", "240"))


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


def exposure_slot():
    return DeviceSlot(label="1c exposure decay",
                      run=lambda store: store.exposure_decay(
                          exposure.get_settings().decay_q))


# ---------------------------------------------------------------------------
# PART 1
# ---------------------------------------------------------------------------
def part1_round_trip(ship_units):
    rng = np.random.default_rng(2611)
    lay = unit_store.layout()
    for n in (1, 2, 8):
        U = len(ship_units) + 3
        st = unit_store.DeviceStore(n, U)
        live = unit_store.pack_batch([ship_units] * n, U)
        rand = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                            size=st.shape, dtype=np.int64, endpoint=True)
        for name, a in (("live", live), ("random", rand)):
            st.upload(a)
            b = st.download()
            if not np.array_equal(a, b):
                fail(f"part1 N={n} {name}: round trip differs at "
                     f"{np.argwhere(a != b)[0].tolist()}")
        st.upload(rand)
        st.upload(live)                   # back to back: the second lands
        if not np.array_equal(st.download(), live):
            fail(f"part1 N={n}: a second upload did not land intact")
        print(f"part1 N={n} U={U} W={lay.width}: round trip ok")
    try:
        bp.UnitStoreBuffer(1, 1, lay.width + 1)
        fail("part1: a width other than the compiled layout's was accepted")
    except ValueError as e:
        print("part1 width refusal:", str(e)[:80])


def part1b_kernel_on_synthetic_rows():
    """The kernel against ``exposure.decay_step`` over the whole int64 range
    (zero, negative, one, values near 2^63) and every decay in a sweep, on
    VALID rows; PADDING rows (valid 0) carry the same garbage and must come
    back untouched -- live matches never fill a padding row, so this is
    where touching one would show."""
    rng = np.random.default_rng(1404)
    lay = unit_store.layout()
    ex = lay.column("exposure")
    n, U = 3, 64
    st = unit_store.DeviceStore(n, U)
    special = np.array([0, 1, -1, 2, 65535, 65536, 65537, 2 ** 31,
                        np.iinfo(np.int64).max, np.iinfo(np.int64).min,
                        np.iinfo(np.int64).max - 1], dtype=np.int64)
    for decay_q in (0, 1, 2, 1000, 21844, 32768, 65535, 65536):
        a = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                         size=st.shape, dtype=np.int64, endpoint=True)
        blk = a[:, :, ex.offset:ex.offset + ex.width]
        blk[...] = np.where(rng.random(blk.shape) < 0.3, 0,
                            rng.choice(special, size=blk.shape))
        small = rng.random(blk.shape) < 0.3
        blk[small] = rng.integers(-5, 300000, size=int(small.sum()))
        a[:, :, lay.valid] = (rng.random((n, U)) < 0.7).astype(np.int64)
        want = a.copy()
        changed = np.zeros(n, dtype=np.int64)
        for e in range(n):
            for r in range(U):
                if not want[e, r, lay.valid]:
                    continue
                row_changed = False
                for t in range(ex.width):
                    x = int(want[e, r, ex.offset + t])
                    if x:
                        nx = exposure.decay_step(x, decay_q)
                        if nx != x:
                            row_changed = True
                        want[e, r, ex.offset + t] = nx
                changed[e] += int(row_changed)
        st.upload(a)
        cov = st.exposure_decay(decay_q)
        got = st.download()
        if not np.array_equal(got, want):
            e, r, c = np.argwhere(got != want)[0].tolist()
            fail(f"part1b decay_q={decay_q}: env {e} row {r} "
                 f"{lay.column_at(c)} (valid {a[e, r, lay.valid]}): device "
                 f"{got[e, r, c]} != law {want[e, r, c]} (from {a[e, r, c]})")
            return
        if not np.array_equal(cov, changed):
            fail(f"part1b decay_q={decay_q}: coverage {cov.tolist()} != "
                 f"{changed.tolist()}")
            return
    for bad in (-1, 65537):
        try:
            st.exposure_decay(bad)
            fail(f"part1b: decay_q {bad} was accepted")
        except ValueError:
            pass
    print("part1b: the kernel equals exposure.decay_step over the int64 "
          "range at 8 decays; padding rows untouched; bad decay refused")


# ---------------------------------------------------------------------------
# PART 2
# ---------------------------------------------------------------------------
def _unit_hash(sim):
    from field_ab_harness import _capture_unit_state
    return _capture_unit_state(sim)["hash"]


def _field_digest(sim):
    from field_ab_harness import SIM_FIELDS, _snapshot
    from field_digest import field_digest
    return field_digest(_snapshot(sim.gmap, SIM_FIELDS))


def part2_ship_lockstep():
    from simulation.match_session import MatchSession
    level = level_loader.load("ship_demo", str(_ROOT / "levels"))
    hooked = MatchSession(level, seed=1, breach_physics=bp)
    plain = MatchSession(level, seed=1, breach_physics=bp)
    t_pack = []
    t_unpack = []
    cov_ticks = 0
    n = 0
    with ShadowLockstep([hooked], exposure_slot()) as ls:
        for _ in range(SHIP_TICKS):
            if hooked.is_terminal():
                break
            rep = ls.tick()
            plain.tick()
            n += 1
            if not rep.equal:
                fail(f"part2: {ls.divergences[-1]}")
                break
            if int(rep.coverage[0]) != int(rep.host_changed[0]):
                fail(f"part2 tick {rep.tick}: coverage {rep.coverage[0]} != "
                     f"host-changed rows {rep.host_changed[0]}")
                break
            cov_ticks += int(rep.coverage[0] > 0)
            if _unit_hash(hooked.sim) != _unit_hash(plain.sim) or \
                    _field_digest(hooked.sim) != _field_digest(plain.sim):
                fail(f"part2 tick {hooked.sim.tick}: the hooked match "
                     f"diverged from the unhooked one")
                break
            # the bridge's wall clock on this match's units
            t0 = time.perf_counter()
            rows = unit_store.pack(hooked.sim.units)
            t1 = time.perf_counter()
            unit_store.unpack(rows, unit_store.blank_units(len(rows)))
            t2 = time.perf_counter()
            t_pack.append(t1 - t0)
            t_unpack.append(t2 - t1)
    print(f"part2 ship_demo: {n} ticks, every tick equal; coverage > 0 on "
          f"{cov_ticks} ticks ({100.0 * cov_ticks / max(1, n):.1f} %); "
          f"{len(hooked.sim.units)} units")
    if cov_ticks == 0:
        fail("part2: the kernel changed nothing on any tick (vacuous scene)")
    print(f"TIMING pack per tick ({len(hooked.sim.units)} units): median "
          f"{1e3 * float(np.median(t_pack)):.3f} ms; unpack (into blank "
          f"units) median {1e3 * float(np.median(t_unpack)):.3f} ms")
    return hooked.sim.units


# ---------------------------------------------------------------------------
# PART 3
# ---------------------------------------------------------------------------
class FocusFireArena:
    """The focus-fire bench's arena (``tools/bench_focus_fire.py``): n
    marines on an arc fire at a marine of 10 000 hp (it lives through the
    window), ``continuous`` (trigger held) or ``single`` (one pull a
    second). Steps its own sim (``.sim`` / ``.tick()``)."""

    def __init__(self, n, mode, seed):
        import bench_focus_fire as ff
        from config import CFG
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        from simulation.unit import Unit
        self.tps = int(CFG.clock.ticks_per_second)
        sim = Simulation(ff._arena(), seed=seed, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.hand_zombies_to_controller()
        tx, ty = ff.TARGET_XY
        tgt = sim.get_unit(sim.add_unit(
            Unit("target", x=tx, y=ty, team=1, env_profile="default")))
        tgt.current_hp = 10000.0
        self.aims = {}
        for i in range(n):
            b = math.radians(ff.BEARINGS_DEG[i])
            sx = round(tx + ff.DISTANCE_TILES * math.cos(b))
            sy = round(ty + ff.DISTANCE_TILES * math.sin(b))
            u = sim.get_unit(sim.add_unit(
                Unit(f"S{i}", x=sx, y=sy, team=0, env_profile="default")))
            self.aims[u.id] = ff._unit_vector_q(tx - sx, ty - sy)
        sim.set_paused(False)
        self.sim, self.mode, self.k = sim, mode, 0

    def tick(self):
        held = self.mode == "continuous" or self.k % self.tps == 0
        for uid, (ax, ay) in self.aims.items():
            self.sim.set_aim(uid, ax, ay)
            self.sim.set_trigger(uid, held)
        self.sim.step()
        self.sim.set_paused(False)
        self.k += 1


def part3_batch(n_env):
    modes = ("continuous", "single")
    drivers = [FocusFireArena(1 + (e % 4) + (e // 4), modes[e % 2], 181 + e)
               for e in range(n_env)]
    counts = [len(d.sim.units) for d in drivers]
    cov_env_ticks = 0
    t_dev = []
    with ShadowLockstep(drivers, exposure_slot()) as ls:
        for _ in range(ARENA_TICKS):
            rep = ls.tick()
            if not rep.equal:
                fail(f"part3 N={n_env}: {ls.divergences[-1]}")
                break
            if not np.array_equal(rep.coverage, rep.host_changed):
                fail(f"part3 N={n_env} tick {rep.tick}: coverage "
                     f"{rep.coverage.tolist()} != host-changed "
                     f"{rep.host_changed.tolist()}")
                break
            cov_env_ticks += int((rep.coverage > 0).sum())
        # the device step's wall clock: upload + kernel + download
        pre = unit_store.pack_batch([d.sim.units for d in drivers], ls.U)
        for _ in range(50):
            t0 = time.perf_counter()
            ls.store.upload(pre)
            ls.store.exposure_decay(exposure.get_settings().decay_q)
            ls.store.download()
            t_dev.append(time.perf_counter() - t0)
    total = n_env * len(ls.reports)
    print(f"part3 N={n_env}: units per env {counts} (U {ls.U}), "
          f"{len(ls.reports)} ticks, every tick equal and env i == env i "
          f"alone; coverage > 0 on {cov_env_ticks} of {total} env-ticks "
          f"({100.0 * cov_env_ticks / max(1, total):.1f} %)")
    if cov_env_ticks == 0:
        fail(f"part3 N={n_env}: the kernel changed nothing (vacuous scene)")
    if len(set(counts)) < 2:
        fail(f"part3 N={n_env}: every env has the same unit count (no padding)")
    print(f"TIMING device step N={n_env} U={ls.U}: upload + kernel + "
          f"download median {1e3 * float(np.median(t_dev)):.3f} ms")


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_STORE_RESULT: FAIL (not the CUDA build)")
        return 1
    ship_units = part2_ship_lockstep()
    part1_round_trip(ship_units)
    part1b_kernel_on_synthetic_rows()
    st = unit_store.DeviceStore(1, len(ship_units))
    rows = unit_store.pack_batch([ship_units])
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        st.upload(rows)
        st.exposure_decay(exposure.get_settings().decay_q)
        st.download()
        ts.append(time.perf_counter() - t0)
    print(f"TIMING device step N=1 U={len(ship_units)}: upload + kernel + "
          f"download median {1e3 * float(np.median(ts)):.3f} ms")
    for n in (2, 8):
        part3_batch(n)
    ok = not FAILS
    print("UNIT_STORE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
