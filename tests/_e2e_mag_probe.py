"""G2p E2E probe (a harness, not a test): ``tools/e2e_drive.py`` -- the real
``main.py`` in a hidden window -- with a recorder on ``Simulation.step`` that
logs the possessed marine's magazine state at every tick it changes or a
round leaves. Run as ``python tests/_e2e_mag_probe.py <e2e_drive args>``;
prints one ``MAGPROBE <json>`` line, a list of
``[tick, rounds_this_tick, current_mag, spare_mags, reload_done_tick]``."""
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
from simulation.events import ShotFiredEvent  # noqa: E402

LOG: list = []
_last = [None]
_orig = S.Simulation.step


def _step(self, *a, **k):
    t = self.tick
    r = _orig(self, *a, **k)
    u = next((u for u in self.units
              if u.team == self.player_team and u.alive), None)
    if u is not None:
        n = sum(1 for e in self.tick_events
                if isinstance(e, ShotFiredEvent) and e.launch
                and e.unit_id == u.id)
        cur = (u.current_mag, u.spare_mags, u.reload_done_tick)
        if cur != _last[0] or n:
            LOG.append([t, n, *cur])
            _last[0] = cur
    return r


S.Simulation.step = _step
rc = e2e_drive.main(sys.argv[1:])
print("MAGPROBE " + json.dumps(LOG))
sys.exit(rc)
