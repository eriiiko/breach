"""#80 -- the 3D marines read a clip's frame count through ONE helper that
accepts either raylib binding's field name.

The home desktop's pyray is raylib 5.5.0.4, whose ``ModelAnimation`` carries
``frameCount``; the 3D-unit renderer was written against a raylib 6.x build,
whose field is ``keyframeCount``. Reading the 6.x name directly crashed the game
the moment M turned the 3D marines on (reproduced end to end by
``tools/e2e_drive.py --level playground --press M@40 --frames 120``). Headless:
imports pyray (no window) and builds a real cffi ``ModelAnimation`` struct of
the INSTALLED binding.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_model_frame_count.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pyray as rl  # noqa: E402

from renderer.unit_model_renderer import anim_frame_count  # noqa: E402


def test_the_installed_bindings_clip_struct_reads_its_frame_count():
    """PROPERTY (#80): on the raylib binding this machine actually has, a real
    ``ModelAnimation`` struct's frame count reads back through the helper as
    the value the struct holds -- whichever of the two field names the binding
    uses.

    BREAKS IF: the helper (or the draw path) reads one binding's field name
    only -- e.g. ``anim.keyframeCount`` on raylib 5.5, the #80 crash.
    """
    fields = {f[0] for f in rl.ffi.typeof("ModelAnimation").fields}
    name = "frameCount" if "frameCount" in fields else "keyframeCount"
    assert name in fields, f"the binding's ModelAnimation has neither name: {fields}"
    anim = rl.ffi.new("ModelAnimation *")[0]
    setattr(anim, name, 37)
    assert anim_frame_count(anim) == 37


@pytest.mark.parametrize("field", ["frameCount", "keyframeCount"])
def test_either_bindings_field_name_is_read(field):
    """PROPERTY (#80): a clip carrying only raylib 5.5's ``frameCount`` or only
    6.x's ``keyframeCount`` yields its count, and a struct with neither raises
    AttributeError naming both (never a silent 0 or 1 frame clip).

    BREAKS IF: the helper drops one name from its list, or swallows the
    missing-field case into a default.
    """
    assert anim_frame_count(SimpleNamespace(**{field: 12})) == 12
    with pytest.raises(AttributeError, match="frameCount"):
        anim_frame_count(SimpleNamespace(boneCount=3))
