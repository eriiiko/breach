"""G3p E2E probe (a harness, not a test): ``tools/e2e_drive.py`` -- the real
``main.py`` in a hidden window -- with a recorder on ``Simulation.step`` that
logs, at every tick something changes, the possessed marine's grenade counts,
the ``ThrowEvent``s of the tick, the grenades in flight (``thrown_tick``, the
round's row name) and any BLINDED status on any unit. Run as
``python tests/_e2e_throw_probe.py <e2e_drive args>``; prints one
``THROWPROBE <json>`` line, a list of
``[tick, counts, [[kind, ticks], ...], [[thrown_tick, ammo], ...],
[[unit id, BLINDED ticks left], ...], [status kinds on the marine]]``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT):
    sys.path.insert(0, str(_p))

import e2e_drive  # noqa: E402
import breach_physics  # noqa: E402,F401
from simulation import simulation as S  # noqa: E402
from simulation.events import ThrowEvent  # noqa: E402
from simulation.status import BLINDED  # noqa: E402

LOG: list = []
_last = [None]
_orig = S.Simulation.step


def _step(self, *a, **k):
    t = self.tick
    r = _orig(self, *a, **k)
    u = next((u for u in self.units
              if u.team == self.player_team and u.alive), None)
    if u is not None:
        events = [[e.kind, e.ticks] for e in self.tick_events
                  if isinstance(e, ThrowEvent) and e.unit_id == u.id]
        flying = [[p.thrown_tick, p.ammo_name] for p in self.projectiles]
        blind = [[v.id, s.remaining_ticks] for v in self.units
                 for s in v.statuses if s.kind == BLINDED]
        mine = [st.kind for st in u.statuses]
        cur = (list(u.grenade_counts), flying, blind, mine)
        if events or cur != _last[0]:
            LOG.append([t, cur[0], events, flying, blind, mine])
            _last[0] = cur
    return r


S.Simulation.step = _step

import control_direct as _cd  # noqa: E402

KEYS: list = []
_orig_emit = _cd.DirectControl._emit


def _emit(self, sim, uid, frame):
    before = self.grenade_kind
    _orig_emit(self, sim, uid, frame)
    if frame.cycle_grenade or frame.throw is not None:
        KEYS.append([sim.tick, "cycle" if frame.cycle_grenade else "throw",
                     before, self.grenade_kind])


_cd.DirectControl._emit = _emit
rc = e2e_drive.main(sys.argv[1:])
print("THROWKEYS " + json.dumps(KEYS))
print("THROWPROBE " + json.dumps(LOG))
sys.exit(rc)
