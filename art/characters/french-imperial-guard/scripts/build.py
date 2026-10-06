"""Build the guard officer from scratch, compare with the sheet, render, save.

    blender -b --factory-startup -P scripts/build.py -- [options]
    (or, from art/characters:  bash charkit/run.sh french-imperial-guard [options])

Options and outputs: `charkit/workerbuild.py` (the shared driver; `--variant green` is the
green dolman with one red-tipped plume). The tables are `guard.py`, the assembly
`assemble.py`, the reference sheet's frame `sheet_ref.py`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import assemble  # noqa: E402
import guard  # noqa: E402
import sheet_ref  # noqa: E402
import workerbuild  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
variant = argv[argv.index("--variant") + 1] if "--variant" in argv else None  # the plumes are geometry: the builder needs it too
workerbuild.run(os.path.normpath(os.path.join(HERE, "..")), "french-imperial-guard", guard, sheet_ref, builder=assemble.Builder(guard, variant))
