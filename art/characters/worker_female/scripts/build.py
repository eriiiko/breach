"""Build the female worker from scratch, compare with the sheet, render, save.

    blender -b --factory-startup -P scripts/build.py -- [options]
    (or, from art/characters:  bash charkit/run.sh worker_female [options])

Options and outputs: `charkit/workerbuild.py`. The tables are `worker.py`, the
reference sheet's frame `sheet_ref.py`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import sheet_ref  # noqa: E402
import worker  # noqa: E402
import workerbuild  # noqa: E402

workerbuild.run(os.path.normpath(os.path.join(HERE, "..")), "worker_female", worker, sheet_ref)
