"""Game-ready worker_female: static skin, rigged to the game's skeleton, evidence renders.
Options: `charkit/workergame.py` (and `charkit/gameready.py`).

    blender -b --factory-startup -P scripts/game.py -- [--rig-only] [--variant white_clean] [--evidence-only]

The rigged model is the game asset `assets/models/worker_female/worker_female.glb`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import rig_spec  # noqa: E402
import worker  # noqa: E402
import workergame  # noqa: E402

workergame.run(os.path.normpath(os.path.join(HERE, "..")), "worker_female", worker, rig_spec)
