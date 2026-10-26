"""Game-ready Kessler: static skin (gameready stages 1-6), then rigged to the game's skeleton (stage 7).
Options: see `charkit/gameready.py`.

    blender -b --factory-startup -P art/characters/kessler/scripts/game.py -- [--rig-only] [--no-rig]

The "build" is `kessler_source.build`: the AI model imported, scaled to HEIGHT_M, its visor
replaced by the smooth green cap and its mesh split into named regions (the GEAR table's parts).
The rigged model is the game asset `assets/models/kessler/kessler.glb`; the pictures land in
`previews/game/` (and `scripts/evidence.py` adds the pose / helmet / hand set there).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import gameready  # noqa: E402
import kessler_source  # noqa: E402
import rig_spec  # noqa: E402

REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# Studio views (gameready/studio.render_beauty): his arm span is 2.05 m, wider than the marine's.
BEAUTY = {
    "hero": (0.0, 32.0, 8.0, 6.2, 0.93, 85.0, (1200, 1600)),
    "top": (0.0, 0.0, 89.0, 5.8, 0.9, 85.0, (1200, 1200)),
}

gameready.run(ROOT, "kessler", lambda materials: kessler_source.build(), lambda: None,
              kessler_source.HEIGHT_M, BEAUTY, rig_spec=rig_spec,
              rig_out=os.path.join(REPO, "assets", "models", "kessler", "kessler.glb"),
              previews_dir=os.path.join(ROOT, "previews", "game"))
