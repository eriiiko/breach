"""Game-ready static mesh of the space marine. Options: see `charkit/gameready.py`.

    blender -b --factory-startup -P scripts/game.py -- [--tris 10000]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import gameready  # noqa: E402
import marine  # noqa: E402
import refkit  # noqa: E402
import scene  # noqa: E402

gameready.run(os.path.normpath(os.path.join(HERE, "..")), "space_marine", marine.build, scene.make_materials,
              refkit.HEIGHT_M, scene.BEAUTY)
