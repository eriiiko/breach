"""#33 -- every unit model the game names is in the repo and carries the clips
the renderer asks for.

Which model a unit is drawn with is the ``[render.unit_looks]`` table in
config.toml. Both failures are SILENT in the game: a missing model file logs one
warning and its units are drawn as another look (or, with none loaded, as the
2D sprites); a clip name the file does not carry falls back to idle. So every
named file and its clip names are checked here, headless (the .glb's JSON chunk
is read directly -- no window, no raylib load).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_model_asset.py -q
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CFG  # noqa: E402
from renderer.unit_looks import all_look_names, look_path, looks_table  # noqa: E402
from renderer.unit_model_renderer import (  # noqa: E402
    CLIP_MAP, _MODEL_PATH, _SCALE_REFERENCE)

_GLB_MAGIC = 0x46546C67   # "glTF"
_CHUNK_JSON = 0x4E4F534A  # "JSON"


def _gltf_json(path: Path) -> dict:
    """The glTF document of a .glb (its first chunk) or a .gltf."""
    data = path.read_bytes()
    if path.suffix.lower() != ".glb":
        return json.loads(data)
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    assert magic == _GLB_MAGIC, f"{path} is not a binary glTF"
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == _CHUNK_JSON, f"{path}: first chunk is not JSON"
    return json.loads(data[20:20 + chunk_len])


def _shipped_look_paths():
    return [look_path(n) for n in all_look_names(looks_table(CFG))]


def test_every_shipped_look_file_exists():
    """PROPERTY: every model named in the shipped ``[render.unit_looks]`` is
    present in the repo, and so is the scale reference (the space marine).

    BREAKS IF: an asset is moved, renamed or deleted without the table
    following it, or a name in the table is misspelt (the game would then
    draw those units as another look, with one warning in the log), or the
    marine whose height sets every look's scale goes missing.
    """
    missing = [str(p) for p in _shipped_look_paths() if not p.is_file()]
    assert not missing, f"unit models missing: {missing}"
    assert _MODEL_PATH == look_path(_SCALE_REFERENCE)
    assert _MODEL_PATH.is_file(), f"scale reference missing: {_MODEL_PATH}"


def test_every_shipped_look_carries_every_clip_the_renderer_names():
    """PROPERTY: for every model named in ``[render.unit_looks]``, every clip
    name in ``CLIP_MAP`` is one of its animations, on a skinned mesh.

    BREAKS IF: a model is regenerated without its clips or with renamed ones,
    a ``CLIP_MAP`` row names a clip some look does not have (that state would
    silently play idle on those units), or a file loses its skin (the
    renderer then reports it as not rigged and draws another look).
    """
    for path in _shipped_look_paths():
        doc = _gltf_json(path)
        clips = {a.get("name") for a in doc.get("animations", [])}
        missing = sorted(set(CLIP_MAP.values()) - clips)
        assert not missing, f"{path.name} lacks the clips {missing}"
        assert doc.get("skins"), f"{path.name} has no skin"
