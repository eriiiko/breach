"""Look-shot table gates (graphics arc G0 P3; tools/look_shots.py).

The shot set is the shared instrument every graphics patch is judged on, so
its table must stay loadable and honest. Each test names the property it
protects and the change that would break it. No test pins the number or the
order of shots: the set is designed to grow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.look_shots import (load_shots, level_exists, parse_shot,  # noqa: E402
                              shot_path)


def test_every_shipped_shot_parses_and_names_a_real_level():
    """Property: every row of tools/look_shots.toml loads and its level exists
    on disk. Breaks when a row is malformed or a level it names is renamed or
    deleted (the shot would silently stop being rendered)."""
    table = load_shots()
    assert table.shots, "the look-shot table is empty"
    for s in table.shots:
        assert level_exists(s.level), f"{s.name}: no level {s.level!r}"
        if s.detonate is not None:
            assert 0 <= s.detonate[2] < s.frames


def test_reference_shots_of_the_direction_doc_are_present():
    """Property: the §7 / §10 reference shots (dark corridor, fire + plume,
    steam in a beam, the explosion series, the smoke-light studio) stay in
    the set. A SUPERSET check -- new shots never break it. Breaks when one
    of the references is deleted or renamed without updating the doc."""
    names = {s.name for s in load_shots().shots}
    required = {"dark_corridor_lamp", "fire_plume", "steam_flashlight",
                "explosion_t04", "explosion_t12", "explosion_t40",
                "smoke_light_studio"}
    assert required <= names, f"missing: {sorted(required - names)}"


def test_duplicate_names_are_rejected(tmp_path):
    """Property: shot names are unique (they name the output files). Breaks
    if load_shots stops checking, letting one shot overwrite another's PNG."""
    p = tmp_path / "shots.toml"
    p.write_text('[[shot]]\nname = "a"\nlevel = "fire_studio"\nframes = 1\n'
                 '[[shot]]\nname = "a"\nlevel = "fire_studio"\nframes = 2\n',
                 encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_shots(p)


def test_detonation_outside_the_shot_is_rejected():
    """Property: a scripted detonation must fall inside the shot's frames.
    Breaks if the validator is dropped (the blast would silently never fire
    and the 'explosion' shot would show a quiet room)."""
    with pytest.raises(ValueError, match="detonate"):
        parse_shot({"name": "x", "level": "fire_studio", "frames": 10,
                    "detonate": [1, 1, 10]})


def test_output_is_named_by_shot_and_tag(tmp_path):
    """Property: the PNG lands at <dir>/<name>_<tag>.png (§8.1), so before/
    after renders of different commits sit side by side. Breaks if the
    naming drops the tag (before and after would overwrite each other)."""
    p = shot_path(tmp_path, "fire_plume", "abc123")
    assert p.parent == tmp_path
    assert p.name.startswith("fire_plume_") and "abc123" in p.name
