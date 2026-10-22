"""Build Beatrice from scratch, compare with the front view, render, save.

    blender -b --factory-startup -P scripts/build.py -- [options]
    (or, from art/characters:  bash charkit/run.sh Beatrice [options])

Options and outputs: `charkit/suitbuild.py`. The tables are `beatrice.py`, the reference's
frame `sheet_ref.py`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import beatrice  # noqa: E402
import sheet_ref  # noqa: E402
import suitbuild  # noqa: E402

suitbuild.run(os.path.normpath(os.path.join(HERE, "..")), "Beatrice", beatrice, sheet_ref)
