"""#33 -- the unit model the renderer names is in the repo and carries the
clips the renderer asks for.

Both failures are SILENT in the game: a missing model file logs one warning and
the units fall back to the 2D sprites; a clip name the file does not carry
falls back to idle. So the file and its clip names are checked here, headless
(the .glb's JSON chunk is read directly -- no window, no raylib load).

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

from renderer.unit_model_renderer import CLIP_MAP, _MODEL_PATH  # noqa: E402

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


def test_the_unit_model_file_exists():
    """PROPERTY: the model file the 3D-unit renderer loads is present in the
    repo.

    BREAKS IF: the asset is moved, renamed or deleted without the renderer's
    ``_MODEL_PATH`` following it (the game would then draw sprites, with one
    warning in the log).
    """
    assert _MODEL_PATH.is_file(), f"unit model missing: {_MODEL_PATH}"


def test_the_unit_model_carries_every_clip_the_renderer_names():
    """PROPERTY: every clip name in ``CLIP_MAP`` is an animation of the unit
    model, on a skinned mesh.

    BREAKS IF: the model is regenerated without its clips or with renamed
    ones, a ``CLIP_MAP`` row names a clip the file does not have (that state
    would silently play idle), or the file loses its skin (the renderer then
    reports the model as not rigged and draws sprites).
    """
    doc = _gltf_json(_MODEL_PATH)
    clips = {a.get("name") for a in doc.get("animations", [])}
    missing = sorted(set(CLIP_MAP.values()) - clips)
    assert not missing, f"{_MODEL_PATH.name} lacks the clips {missing}"
    assert doc.get("skins"), f"{_MODEL_PATH.name} has no skin"
