"""Game-ready space marine: static skin (stages 1-6), then rigged to the game's skeleton (stage 7).
Options: see `charkit/gameready.py`.

    blender -b --factory-startup -P scripts/game.py -- [--tris 10000] [--rig-only] [--no-rig]

The rigged model is the game asset `assets/models/space_marine/space_marine.glb`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import gameready  # noqa: E402
import marine  # noqa: E402
import refkit  # noqa: E402
import rig_spec  # noqa: E402
import scene  # noqa: E402

REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
gameready.run(os.path.normpath(os.path.join(HERE, "..")), "space_marine", marine.build, scene.make_materials,
              refkit.HEIGHT_M, scene.BEAUTY, rig_spec=rig_spec,
              rig_out=os.path.join(REPO, "assets", "models", "space_marine", "space_marine.glb"))
