"""#29 reference debt (the P6 slot inventory, Part E): on the continuous path
``Simulation.projectiles`` holds only the grenades still in flight.

Before this patch a detonated ``Projectile`` was flagged and stayed in the
list for ever (``Simulation._update_projectiles`` skipping it every tick);
only the two legacy WEGO paths pruned it (TwoPhaseWEGO's round end,
OnePhaseWEGO's round boundary). ``ContinuousRealtime.on_tick_end`` (slot 10,
so ``MatchRules`` too) now drops it, after slot 2 has detonated it and before
slot 11's team picture and every between-tick reader (the renderer through
``get_state()``, the observation through ``team_picture.live_objects``) --
each of which already skipped a detonated grenade. The device form (P7b)
needs the list bounded by what is in flight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.ruleset import ContinuousRealtime, MatchRules  # noqa: E402
from simulation.team_picture import live_objects  # noqa: E402
from simulation.turning import place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402


def _arena(h=40, w=40) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="proj_bound", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


@pytest.mark.parametrize("make_ruleset", [
    ContinuousRealtime, lambda: MatchRules(time_limit_s=600.0)],
    ids=["continuous", "match"])
def test_a_detonated_grenade_leaves_the_projectile_list(make_ruleset):
    """Property: on the continuous path (ContinuousRealtime and the match's
    MatchRules over it), between ticks every entry of ``sim.projectiles`` is
    a grenade still in flight -- so the list is bounded by what is in the
    air, not by how many grenades the match has thrown -- and the readers'
    view (``live_objects``) is the whole list. Not vacuous: grenades of
    several kinds are thrown on overlapping fuses, each detonates (one
    ``ExplosionEvent`` per grenade) and some are in flight at once.
    Breaks if the continuous ruleset stops pruning detonated projectiles
    (the pre-patch growth: the first grenade is still listed, flagged,
    ticks after its blast), or prunes one still in flight (fewer blasts
    than throws)."""
    sim = Simulation(_arena(), seed=3, breach_physics=bp,
                     enable_recorder=False, ruleset=make_ruleset())
    sim.set_paused(False)
    u = sim.get_unit(sim.add_unit(Unit("T", x=18, y=18, team=0)))
    place(u, 0)
    u.current_hp = 10000.0          # its own frags must not end the throwing
    kinds = len(get_tables().hand_grenade_rows)
    for k in range(kinds):
        u.grenade_counts[k] = 3
    throws, blasts, most_in_air = 0, 0, 0
    for t in range(260):
        if t % 25 == 0 and t < 200:
            # an alternating direction, a short fuse: several in the air
            dx = FP_ONE if (t // 25) % 2 == 0 else -FP_ONE
            sim.throw_grenade_intent(u.id, dx, 0, 30, 6 * FP_ONE,
                                     (t // 25) % kinds)
            throws += 1
        sim.step()
        blasts += sum(isinstance(e, ExplosionEvent) for e in sim.tick_events)
        assert all(not p.detonated for p in sim.projectiles), t
        assert list(live_objects(sim)) == list(sim.projectiles), t
        most_in_air = max(most_in_air, len(sim.projectiles))
    assert blasts == throws >= 4, (blasts, throws)
    assert most_in_air >= 2
    assert sim.projectiles == []
