"""The saved run (agents/saved.py), the batch's header and the resume
through agents/run.py: rung 2 note 2d, 9 to 12 and 14. On the stand-in, the
processor. Each test names the property it protects and the change that
must break it."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from agents import plot_curve as P
from agents import run as R
from agents import saved as SV
from agents.batch import ForeignBatch
from agents.network import SoldierNet
from agents.settings import Settings
from agents.sizes import sizes_for
from agents.spec import reduced
from agents.standin import StandIn
from agents.tasks import networks
from agents.tests._scripted import ScriptTask
from agents.trainer import Trainer
from agents.value_scale import ValueScale

ROOT = Path(__file__).resolve().parents[2]
N = 2
IDS = {"run_sha256": "r" * 64, "training_sha256": "t" * 64,
       "build_id": None}
HASHES = {"run_file": "r" * 64, "training_file": "t" * 64,
          "build_id": None, "commit": "c" * 40}
MANIFEST = {"format": 1, "training_file_sha256": "t" * 64,
            "run_file_sha256": "r" * 64, "levels_sha256": None,
            "room_generator_commit": None, "seed_lists_sha256": None,
            "commit": "c" * 40, "dirty": False}


def _trainer(seed=0, net_seed=None, memory=True, rounds=10):
    task = ScriptTask(3)
    env = StandIn(task, N, seed=5)
    task.bind(env.sizes)
    soldier, value = networks(env, seed if net_seed is None else net_seed,
                              reduced(memory))
    s = Settings(planned_rounds=rounds, decisions=32, piece=64, lump=32,
                 stretch=16)
    return Trainer(env, soldier, value, s, seed=seed, ids=IDS)


def _save(tr, path):
    return SV.save(path, tr, hashes=HASHES, manifest=MANIFEST,
                   started_from="nothing")


def _same(a, b, where="") -> None:
    """``a`` and ``b`` equal to the bit (tensors, arrays, nested tables)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype, where
        assert torch.equal(a.cpu(), b.cpu()), where
    elif isinstance(a, dict):
        assert set(a) == set(b), where
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def _everything(tr) -> dict:
    """All a resume must carry, read off a trainer."""
    ln, col, sc = tr.learner, tr.collector, tr.learner.scale
    return {"soldier": ln.soldier.state_dict(), "value": ln.value.state_dict(),
            "collector_soldier": col.soldier.state_dict(),
            "opt_soldier": ln.opt_soldier.state_dict(),
            "opt_value": ln.opt_value.state_dict(),
            "scale": [sc._m1, sc._m2, sc.mean, sc.spread, sc.rounds],
            "rng": ln.rng.bit_generator.state,
            "generator": col.generator.get_state(),
            "counters": [tr.rounds, ln.round, ln.version, col.version,
                         col.decisions]}


# -- the state ----------------------------------------------------------------
def test_save_and_restore_carry_every_state_the_learner_owns(tmp_path):
    """Protects: saved at round 2 and restored into a FRESH trainer (other
    initial weights, same files), every weight of both networks, both
    optimisers' state, the value scale (its moments, mean, spread and
    rounds), the learner's generator, the collector's generator, the
    counters and the weights' version are equal to the bit, and the
    collector plays the restored weights. Breaks when any of them is left
    out of ``state`` or ``restore`` (e.g. the optimisers rebuilt fresh, or
    the scale's ``rounds`` not restored)."""
    tr = _trainer()
    for _ in range(2):
        tr.round()
    _save(tr, tmp_path / "last.pt")
    fresh = _trainer(net_seed=7)
    SV.restore(fresh, SV.load(tmp_path / "last.pt"))
    _same(_everything(tr), _everything(fresh))


def test_every_attribute_is_saved_or_named_as_not_saved():
    """Protects: every attribute of the trainer, learner, collector and
    value scale is either in the saved file (``SAVED``) or named with why
    it is not (``NOT_SAVED``), so a new state joins the saved file in the
    patch that adds it. Breaks when an attribute is added to one of them
    (a new learner state, say) without a decision about saving it."""
    tr = _trainer()
    for obj in (tr, tr.learner, tr.collector, tr.learner.scale):
        name = type(obj).__name__
        have = set(vars(obj))
        named = set(SV.SAVED[name]) | set(SV.NOT_SAVED[name])
        assert have == named, (name, sorted(have - named),
                               sorted(named - have))
        assert not set(SV.SAVED[name]) & set(SV.NOT_SAVED[name]), name
    assert type(tr.learner.scale) is ValueScale


def test_a_resume_is_deterministic_and_the_step_size_continues(tmp_path):
    """Protects: two fresh trainers restored from the same file, each over a
    fresh stand-in of the same seed, end two more rounds with bit-identical
    weights and optimiser states (the resume is deterministic); the first
    resumed round's step size is the plan's for that round. Breaks when the
    resume depends on anything not in the file (the networks' fresh
    initial weights, an unrestored generator), or the round counter is not
    restored (the step size would restart at the top of the plan)."""
    tr = _trainer()
    for _ in range(3):
        tr.round()
    _save(tr, tmp_path / "last.pt")
    saved = SV.load(tmp_path / "last.pt")
    runs = []
    for net_seed in (11, 12):
        t = _trainer(net_seed=net_seed)
        SV.restore(t, saved)
        stats = t.round()
        assert stats["step_size"] == t.settings.step_size_at(3)
        assert stats["step_size"] != t.settings.step_size_at(0)
        t.round()
        runs.append(_everything(t))
    _same(runs[0], runs[1])


# -- the refusals -------------------------------------------------------------
def _resume_case(tr, saved, edit):
    """``edit`` applied to (saved, hashes, manifest) copies."""
    s = dict(saved)
    s["specs"] = dict(saved["specs"])
    s["hashes"] = dict(saved["hashes"])
    s["manifest"] = dict(saved["manifest"])
    s["soldier_record"] = dict(saved["soldier_record"])
    s["value_record"] = dict(saved["value_record"])
    s["net"] = dict(saved["net"])
    h, m = dict(HASHES), dict(MANIFEST)
    edit(s, h, m)
    return s, h, m


REFUSALS = [
    ("format", lambda s, h, m: s.update(format=2)),
    ("soldier record", lambda s, h, m: s["soldier_record"].update(
        n_kinds=s["soldier_record"]["n_kinds"] + 1)),
    ("value record", lambda s, h, m: s["value_record"].update(n_teams=9)),
    ("network widths", lambda s, h, m: s["net"].update(head_hidden=7)),
    ("observation spec", lambda s, h, m: s["specs"].update(
        observation="v2")),
    ("action spec", lambda s, h, m: s["specs"].update(action="v2")),
    ("reward_profile spec", lambda s, h, m: s["specs"].update(
        reward_profile="duel")),
    ("run_file sha256", lambda s, h, m: h.update(run_file="x" * 64)),
    ("training_file sha256", lambda s, h, m: h.update(
        training_file="x" * 64)),
    ("training manifest levels_sha256", lambda s, h, m: m.update(
        levels_sha256="x" * 64)),
    ("training manifest room_generator_commit", lambda s, h, m: m.update(
        room_generator_commit="x" * 40)),
    ("code changed: build_id", lambda s, h, m: h.update(build_id="b" * 40)),
    ("code changed: commit", lambda s, h, m: h.update(commit="d" * 40)),
]


@pytest.fixture(scope="module")
def saved_two(tmp_path_factory):
    tr = _trainer()
    tr.round()
    path = tmp_path_factory.mktemp("saved") / "last.pt"
    _save(tr, path)
    return tr, SV.load(path)


def test_an_unchanged_resume_passes(saved_two):
    """Protects: the same files, specs, manifest and code pass the resume's
    checks with no code change reported (``commit`` / ``dirty`` of the
    manifest may differ). Breaks when a check refuses a valid resume."""
    tr, saved = saved_two
    assert SV.check_resume(saved, tr, hashes=HASHES,
                           manifest=dict(MANIFEST, dirty=True,
                                         commit="e" * 40)) is None


@pytest.mark.parametrize("name, edit", REFUSALS, ids=[r[0] for r in
                                                       REFUSALS])
def test_a_resume_refuses_each_difference_by_name(saved_two, name, edit):
    """Protects: a resume refuses, naming the field, a file of another
    format, either network's record or widths, another observation /
    action spec or reward profile, a changed run or training file, a
    changed training-manifest field, and a changed build id or commit.
    Breaks when the check of that field is removed (each case is one
    field)."""
    tr, saved = saved_two
    s, h, m = _resume_case(tr, saved, edit)
    with pytest.raises(SV.ResumeRefused, match=name):
        SV.check_resume(s, tr, hashes=h, manifest=m)


def test_allow_code_change_lets_exactly_the_two_code_fields_differ(
        saved_two):
    """Protects: ``allow_code_change`` lets the build id and the commit
    differ and RETURNS the change (to be recorded); it lets nothing else
    through. Breaks when the flag silences another check, or the change is
    not reported."""
    tr, saved = saved_two
    s, h, m = _resume_case(tr, saved, lambda s, h, m: h.update(
        build_id="b" * 40, commit="d" * 40))
    got = SV.check_resume(s, tr, hashes=h, manifest=m,
                          allow_code_change=True)
    assert got == {"build_id": [None, "b" * 40],
                   "commit": ["c" * 40, "d" * 40]}
    for name, edit in REFUSALS:
        if name.startswith("code changed"):
            continue
        s, h, m = _resume_case(tr, saved, edit)
        with pytest.raises(SV.ResumeRefused, match=name):
            SV.check_resume(s, tr, hashes=h, manifest=m,
                            allow_code_change=True)


# -- the atomic write ---------------------------------------------------------
KILL = r"""
import sys
sys.path[:0] = [{root!r}]
import os, torch
from agents import saved as SV
def half(path, data):
    with open(path, "wb") as f:
        f.write(data[: len(data) // 2])
        f.flush()
    os._exit(9)                       # killed in the middle of the write
SV._write_file = half
SV.write_atomic({path!r}, {{"format": 1, "new": torch.ones(100000)}})
"""


def test_a_save_killed_half_way_leaves_the_old_file_whole(tmp_path):
    """Protects: a process killed in the middle of writing a save leaves
    the previous ``last.pt`` whole and loadable (the bytes go to a
    temporary name and are renamed only once written). Breaks when the
    save writes straight to ``last.pt`` (half a file would be left)."""
    path = tmp_path / "last.pt"
    SV.write_atomic(path, {"format": 1, "old": torch.arange(10)})
    before = path.read_bytes()
    r = subprocess.run([sys.executable, "-c",
                        KILL.format(root=str(ROOT), path=str(path))],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 9, r.stderr
    assert path.read_bytes() == before
    assert torch.equal(SV.load(path)["old"], torch.arange(10))
    # and the next save goes through over the leftover
    SV.write_atomic(path, {"format": 1, "old": torch.arange(3)})
    assert torch.equal(SV.load(path)["old"], torch.arange(3))


# -- the soldier's export -----------------------------------------------------
def test_the_soldier_export_loads_into_its_layout_and_is_refused_else(
        tmp_path):
    """Protects: ``export_soldier`` writes the soldier's weights, record,
    widths and spec names and nothing of the learner; it loads into a
    ``SoldierNet`` built for the same layout (equal weights) and is refused
    by name for a network of another layout (``check_fits``) or other
    widths. Breaks when the export carries learner state, or loading skips
    the record or the widths."""
    tr = _trainer()
    tr.round()
    path = tmp_path / "soldier.pt"
    SV.export_soldier(path, tr.learner.soldier, tr.env.layout)
    d = SV.load(path)
    assert set(d) == {"format", "kind", "soldier", "soldier_record", "net",
                      "specs"}
    sizes = tr.env.sizes
    other = SoldierNet(reduced(True), sizes)
    specs = SV.load_soldier(path, other)
    assert specs == SV.specs_of(tr.env.layout)
    _same(other.state_dict(), tr.learner.soldier.state_dict())
    wrong_layout = SoldierNet(reduced(True), sizes_for(
        sizes.n_slots, sizes.n_teams, sizes.n_kinds + 1))
    with pytest.raises(SV.SoldierRefused, match="soldier record"):
        SV.load_soldier(path, wrong_layout)
    with pytest.raises(SV.SoldierRefused, match="network widths"):
        SV.load_soldier(path, SoldierNet(reduced(False), sizes))


# -- the batch's header -------------------------------------------------------
def test_the_learner_refuses_a_foreign_batch_by_name():
    """Protects: the learner refuses a batch played with weights other than
    its current ones (the same batch handed again after a step: its version
    is behind by one) and a batch whose header names another training file,
    each naming the field; a batch without a header is refused. Breaks when
    ``learn`` stops reading the header, or the version is not raised when
    the weights move."""
    tr = _trainer()
    batch = tr.collector.collect(tr.batch)
    assert batch.header["version"] == tr.learner.version == 0
    tr.learner.learn(batch)
    with pytest.raises(ForeignBatch, match="version: batch 0, learner 1"):
        tr.learner.learn(batch)
    tr.collector.set_weights(tr.learner.weights(), tr.learner.version)
    batch = tr.collector.collect(tr.batch)
    batch.header["training_sha256"] = "x" * 64
    with pytest.raises(ForeignBatch, match="training_sha256"):
        tr.learner.learn(batch)
    batch.header = None
    with pytest.raises(ForeignBatch, match="no header"):
        tr.learner.learn(batch)


# -- the training manifest ----------------------------------------------------
def _levels(root: Path, names) -> Path:
    for i, n in enumerate(names):
        (root / n / "sub").mkdir(parents=True)
        (root / n / "level.toml").write_bytes(b"name = '%s'\n" % n.encode())
        (root / n / "sub" / "cells.npy").write_bytes(bytes(range(i, i + 50)))
    return root


def test_the_levels_hash_follows_the_bytes_and_not_the_folder(tmp_path):
    """Protects: ``levels_sha256`` changes when one byte of one level file
    changes, and not when the level folders are copied elsewhere; a level
    the run does not name is not hashed. Breaks when the hash reads the
    absolute path, skips a nested file, or hashes the whole levels
    folder."""
    a = _levels(tmp_path / "a", ["room_one", "room_two"])
    h = SV.levels_sha256(a, ["room_two", "room_one"])
    b = tmp_path / "elsewhere" / "b"
    shutil.copytree(a, b)
    assert SV.levels_sha256(b, ["room_one", "room_two"]) == h
    (b / "unnamed").mkdir()
    (b / "unnamed" / "x.toml").write_text("x", encoding="utf-8")
    assert SV.levels_sha256(b, ["room_one", "room_two"]) == h
    f = b / "room_two" / "sub" / "cells.npy"
    data = bytearray(f.read_bytes())
    data[7] ^= 1
    f.write_bytes(bytes(data))
    assert SV.levels_sha256(b, ["room_one", "room_two"]) != h


def test_the_manifest_has_its_fields_and_names_the_room_generator(tmp_path):
    """Protects: the training manifest carries the note's fields (both
    files' hashes, the levels' hash, the room generator's commit when a
    generated room is named and None otherwise, the seed lists' None, the
    commit and the dirty flag) and is written as JSON beside the curve.
    Breaks when a field is dropped or the room generator's commit is
    recorded for a run of no generated room."""
    import rl_env
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_training_rooms
    name = gen_training_rooms.level_names()[0]
    lv = _levels(tmp_path / "levels", [name, "not_a_generated_room"])
    run, train = tmp_path / "run.toml", tmp_path / "train.toml"
    run.write_text("x", encoding="utf-8")
    train.write_text("y", encoding="utf-8")

    def cfg(levels):
        return rl_env.RunConfig(
            seed=1, n_envs=1, decimation=1, levels=tuple(levels),
            policy="policy", levels_dir=str(lv), channels=(),
            crop_radius=0, vector=(), reward=())
    rec = SV.manifest_record(run, train, cfg([name]))
    assert set(rec) == set(MANIFEST)
    assert rec["run_file_sha256"] == SV.sha256_file(run)
    assert rec["training_file_sha256"] == SV.sha256_file(train)
    assert rec["levels_sha256"] == SV.levels_sha256(lv, [name])
    assert rec["room_generator_commit"] and len(
        rec["room_generator_commit"]) == 40
    assert rec["seed_lists_sha256"] is None
    assert rec["commit"] == SV.git_head()
    assert isinstance(rec["dirty"], bool)
    assert SV.manifest_record(run, train, cfg(["not_a_generated_room"]))[
        "room_generator_commit"] is None
    SV.write_manifest(tmp_path, rec)
    assert SV.read_manifest(tmp_path) == rec


# -- the resume through the run script ------------------------------------------
def _files(tmp_path, rounds, extra=""):
    run = tmp_path / "run.toml"
    run.write_text("# the stand-in's run\n", encoding="utf-8")
    train = tmp_path / "train.toml"
    train.write_text(
        "[network]\nspec = \"v1\"\nwidth = \"reduced\"\nmemory = true\n"
        f"[train]\nplanned_rounds = {rounds}\ndecisions = 32\n"
        f"piece = 64\nlump = 32\nstretch = 16\n[save]\nsave_every = 2\n"
        f"{extra}", encoding="utf-8")
    return run, train


def _standin_env(monkeypatch):
    import rl_env

    def make(run, n_workers=None):
        task = ScriptTask(3)
        env = StandIn(task, N, seed=5)
        task.bind(env.sizes)
        return env
    monkeypatch.setattr(rl_env, "BatchedEnv", make)


def _curve(out):
    return [json.loads(x) for x in
            (out / R.CURVE).read_text(encoding="utf-8").splitlines()]


def test_a_killed_run_resumes_and_its_curve_goes_on_without_a_seam(
        tmp_path, monkeypatch):
    """Protects: a run saved every 2 rounds and killed during round 4 is
    resumed from ``last.pt`` (round 2) through ``agents.run --resume``: the
    line of round 3 written before the kill moves to the discarded curve,
    the curve then reads rounds 1 to 4 in order, the first resumed line
    carries ``resumed_from`` 2 and ``matches_restarted``, the decision
    total goes on from the saved one, ``last.pt`` names what it was started
    from, and a run that is not resumed is refused on the same folder.
    Breaks when the resume restarts the round count, appends after the
    stale line (a seam), or saves without ``started_from``."""
    _standin_env(monkeypatch)
    run, train = _files(tmp_path, rounds=4)
    out = tmp_path / "out"
    real = Trainer.round

    def killed(self):
        if self.rounds == 3:
            raise KeyboardInterrupt("killed")
        return real(self)
    monkeypatch.setattr(Trainer, "round", killed)
    with pytest.raises(KeyboardInterrupt):
        R.run(run, train, out, device="cpu")
    assert [d["round"] for d in _curve(out)] == [1, 2, 3]
    assert SV.load(out / SV.LAST)["counters"]["rounds"] == 2
    monkeypatch.setattr(Trainer, "round", real)
    with pytest.raises(ValueError, match="refused"):
        R.run(run, None, None, resume=out, device="cpu")
    s = R.run(resume=out, device="cpu")
    lines = _curve(out)
    assert [d["round"] for d in lines] == [1, 2, 3, 4] and s["rounds"] == 4
    assert lines[2]["resumed_from"] == 2
    assert lines[2]["matches_restarted"] is True
    assert "resumed_from" not in lines[3]
    assert lines[2]["decisions_total"] == (lines[1]["decisions_total"]
                                           + lines[2]["decisions"])
    gone = (out / R.DISCARDED).read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["round"] for x in gone] == [3]
    saved = SV.load(out / SV.LAST)
    assert saved["counters"]["rounds"] == 4
    assert saved["started_from"]["round"] == 2
    assert saved["started_from"]["from"] == "nothing"
    with pytest.raises(ValueError, match="resume the run"):
        R.run(run, train, out, device="cpu")


def test_the_run_script_refuses_a_changed_copy_and_records_a_code_change(
        tmp_path, monkeypatch):
    """Protects: ``--resume`` refuses a folder whose training-file copy was
    edited (by the field's name), refuses a changed commit, and with
    ``allow_code_change`` resumes and writes the change into the next
    line and into ``started_from`` of the later save. Breaks when the
    script resumes from an edited copy or a code change goes unrecorded."""
    _standin_env(monkeypatch)
    run, train = _files(tmp_path, rounds=4)
    out = tmp_path / "out"
    R.run(run, train, out, device="cpu", max_rounds=2)
    copy = out / R.TRAIN_COPY
    text = copy.read_text(encoding="utf-8")
    copy.write_text(text + "# edited\n", encoding="utf-8")
    with pytest.raises(SV.ResumeRefused, match="training_file sha256"):
        R.run(resume=out, device="cpu")
    copy.write_text(text, encoding="utf-8")
    monkeypatch.setattr(SV, "git_head", lambda repo=None: "f" * 40)
    with pytest.raises(SV.ResumeRefused, match="code changed: commit"):
        R.run(resume=out, device="cpu")
    R.run(resume=out, device="cpu", allow_code_change=True, max_rounds=3)
    line = _curve(out)[2]
    assert line["code_change"]["commit"][1] == "f" * 40
    assert SV.load(out / SV.LAST)["started_from"]["code_change"][
        "commit"][1] == "f" * 40


# -- the curves script ----------------------------------------------------------
def test_the_curves_script_marks_resumes_and_stops_at_a_stop(tmp_path,
                                                             monkeypatch):
    """Protects: ``plot_curve`` reads a curve up to its first stopped line
    and no further, names the stop, draws its five figures with a mark at
    every resume, and draws the memory's live-check number when the line
    has one. Breaks when a stopped line is skipped and the lines after it
    drawn as if nothing happened, or a resume leaves no mark."""
    _standin_env(monkeypatch)
    run, train = _files(tmp_path, rounds=4)
    out = tmp_path / "out"
    R.run(run, train, out, device="cpu", max_rounds=2)
    R.run(resume=out, device="cpu", max_rounds=3)
    lines = _curve(out)
    assert lines[0]["live"]["max_memory_gap"] <= 1e-4
    with open(out / R.CURVE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"round": 4, "stopped": "LiveCheckFailed: x"})
                + "\n" + json.dumps(dict(lines[-1], round=5)) + "\n")
    got, stopped = P.read(out)
    assert [d["round"] for d in got] == [1, 2, 3]
    assert stopped["round"] == 4 and "round 4" in P.stop_text(stopped)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = [d["decisions_total"] for d in got]
    P._resumes(ax, x, got)
    assert len(ax.lines) == 1 and ax.lines[0].get_xdata()[0] == x[2]
    plt.close(fig)
    paths = P.plot(got, tmp_path / "fig", 2, stopped)
    assert len(paths) == 5 and all(p.is_file() for p in paths)
    assert P.main([str(out), "--out", str(tmp_path / "fig2")]) == 0
