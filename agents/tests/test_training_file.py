"""The training file (agents/training_file.py): rung 2 note 2c, 4. Each test
names the property it protects and the change that must break it."""
from __future__ import annotations

import dataclasses
import tomllib

import pytest

from agents.settings import Settings
from agents.spec import reduced, v1
from agents.training_file import _OWN_TABLE, TrainingFileRefused, load, parse
from simulation.reward_v1 import REWARD_GROUPS

#: The note's training file (section 4), the design's table (4.4) written
#: out; planned_rounds a placeholder.
DESIGN_FILE = """
[network]
spec = "v1"
memory = false
[train]
planned_rounds = 1200
decisions = 128
gamma = 0.99
lam = 0.95
clip = 0.2
passes = 4
max_distance = 0.02
piece = 2048
lump = 512
step_size = 3e-4
variety = 0.01
max_grad_norm = 0.5
stretch = 16
value_first_rounds = 0
[weights]
outcome = 1.0
own = 1.0
team = 0.0
friendly_fire = 0.0
order = 0.0
[stop]
wall_hours = 2.5
[save]
save_every = 10
keep_every = 0
"""


def _raw(text=DESIGN_FILE):
    return tomllib.loads(text)


def test_the_design_file_gives_the_settings_defaults():
    """Protects: the note's file, which writes the design's table, reads to
    exactly ``Settings``' defaults (with its planned rounds and wall stop)
    and the full-width network without memory; a file naming only the
    required keys reads to the same. Breaks when a default drifts from the
    design's table, or a key is read into the wrong field."""
    s, net = parse(_raw())
    assert s == Settings(planned_rounds=1200, wall_hours=2.5)
    assert net == v1(memory=False)
    s2, net2 = parse(_raw('[network]\nspec = "v1"\nmemory = false\n'
                          '[train]\nplanned_rounds = 1200\n'))
    assert s2 == Settings(planned_rounds=1200)
    assert net2 == net


def test_every_settings_field_is_a_key_and_a_value_reaches_it():
    """Protects: every field of ``Settings`` can be set from the file (one
    key per field, in ``[train]``, ``[weights]``, ``[stop]`` or ``[save]``)
    and a value
    written there is the value the trainer gets. Breaks when a field is
    added to Settings without a way in, or a key is dropped on the way."""
    s0, _ = parse(_raw())
    for f in dataclasses.fields(Settings):
        if f.name in ("weights", "wall_hours", "planned_rounds"):
            continue
        v = getattr(s0, f.name)
        new = v * 2 if isinstance(v, int) else v * 0.5
        raw = _raw()
        table = raw[_OWN_TABLE.get(f.name, "train")]
        table[f.name] = new
        if f.name == "piece":
            table["piece"] = 4096
            new = 4096
        if f.name in ("value_first_rounds", "keep_every"):  # 0 doubles to 0
            table[f.name] = new = 5
        assert new != v, f"{f.name}: the written value must differ"
        s, _ = parse(raw)
        assert getattr(s, f.name) == new, f.name
    raw = _raw()
    raw["weights"]["team"] = 0.5
    assert parse(raw)[0].weights[REWARD_GROUPS.index("team")] == 0.5
    raw["network"]["width"] = "reduced"
    raw["network"]["memory"] = True
    assert parse(raw)[1] == reduced(memory=True)


@pytest.mark.parametrize("edit, name", [
    (lambda r: r["train"].update(passses=4), "passses"),
    (lambda r: r.update(extra={"a": 1}), "extra"),
    (lambda r: r["network"].update(depth=3), "depth"),
    (lambda r: r["stop"].update(rounds=3), "rounds"),
    (lambda r: r["train"].pop("planned_rounds"), "planned_rounds"),
    (lambda r: r["network"].pop("memory"), "memory"),
    (lambda r: r["train"].update(decisions=128.0), "decisions"),
    (lambda r: r["train"].update(gamma=True), "gamma"),
    (lambda r: r["train"].update(clip="0.2"), "clip"),
    (lambda r: r["stop"].update(wall_hours="2.5"), "wall_hours"),
    (lambda r: r["save"].update(save_every=0), "save_every"),
    (lambda r: r["save"].update(keep_every=2.0), "keep_every"),
    (lambda r: r["save"].update(best=1), "best"),
    (lambda r: r["network"].update(spec="v2"), "spec"),
    (lambda r: r["network"].update(width="half"), "width"),
    (lambda r: r["network"].update(memory=0), "memory"),
    (lambda r: r["weights"].pop("order"), "order"),
    (lambda r: r["weights"].update(zombie=1.0), "zombie"),
    (lambda r: r["train"].update(lump=4096), "lump"),
])
def test_a_bad_file_is_refused_by_name(edit, name):
    """Protects: an unknown table or key, a missing planned_rounds or
    memory, a wrong type (a float for an integer, a bool or a string for a
    number), an unknown spec or width, a weights table that does not name
    every group, and what Settings itself refuses, are each refused with
    the key named. Breaks when the door lets one through silently (a typo
    would run the design's default and the sweep would measure nothing)."""
    raw = _raw()
    edit(raw)
    with pytest.raises(TrainingFileRefused, match=name):
        parse(raw)


def test_load_keeps_the_bytes_and_their_hash(tmp_path):
    """Protects: ``load`` returns the file's own text and sha256 beside the
    settings (the run copies the file it read). Breaks when the text kept
    is a re-serialisation."""
    import hashlib
    p = tmp_path / "train.toml"
    p.write_text(DESIGN_FILE, encoding="utf-8")
    tf = load(p)
    assert tf.text == p.read_bytes().decode("utf-8")
    assert tf.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert tf.settings.planned_rounds == 1200
