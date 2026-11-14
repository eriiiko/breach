"""Beatrice posed for stills (B1m): the saved build (`source/Beatrice.blend`) posed on a copy and rendered.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/posed.py -- [options]
    (from art/characters; then `python Beatrice/scripts/posed_sheets.py` lays out the comparison pictures)

Options and how the posing works: `charkit/suitpose.py`. The pose, the joints, the weight bands, the shots and the
extra light setups are `beatrice.POSE`. Needs a saved build (`bash charkit/run.sh Beatrice --save`); never changes it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import beatrice  # noqa: E402
import suitpose  # noqa: E402

suitpose.run(os.path.normpath(os.path.join(HERE, "..")), "Beatrice", beatrice)
