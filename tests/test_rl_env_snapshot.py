"""#29 P1b -- the training snapshot door and the run manifest (``src/rl_env.py``,
residency design §6).

Every refusal of :func:`rl_env.require_snapshot` is exercised once by
CONSTRUCTING its condition in a throwaway git repo (``tests/_git_fixture``),
never by touching the real checkout; a temp repo cannot import a real
``breach_physics``, so the module-path and build-id checks read an injected
probe. The run manifest is written and re-checked against a changed field;
``[run] out_dir`` is refused inside the checkout; the farm wires both
(``BatchedEnv(..., snapshot=)`` and the manifest at construction).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rl_env  # noqa: E402
from _git_fixture import commit, git, make_repo, write  # noqa: E402

TAG = "train/t1"
LEVEL_MANIFEST = """[corpus]
level = "x"
seed = 1
ticks = 1
[class_p]
build = []
[twins.w]
class = "P"
kind = "per-call"
device = ["cpp/src/cuda_w.cu"]
slots = ["7 physics"]
level = LEVEL
level_at = "AT"
files = ["cpp/src/w.cpp"]
"""


def _run_cfg(path, sha="ab" * 32):
    """A RunConfig as ``load_run`` returns it (sha256 + path set)."""
    return rl_env.RunConfig(
        seed=1, n_envs=1, decimation=1, levels=("x",), policy="policy",
        levels_dir="", channels=(), crop_radius=0, vector=(), reward=(),
        sha256=sha, path=str(path))


@pytest.fixture
def snap(tmp_path):
    """A snapshot checkout: tracked src/, levels/, config.toml, cpp/ (the
    build dir ignored, as in the real repo), an annotated tag at HEAD, a
    probe that passes, and a run file outside it."""
    repo = make_repo(tmp_path / "train", {
        ".gitignore": "cpp/build*/\n__pycache__/\n",
        "src/simulation/__init__.py": "",
        "config.toml": "[x]\na = 1\n",
        "levels/x/level.toml": "name = 'x'\n",
        "cpp/src/w.cpp": "int w;\n",
        "cpp/src/cuda_w.cu": "int cw;\n",
    })
    write(repo, {"cpp/build/Release/breach_physics.pyd": "fake"})
    git(repo, "tag", "-a", TAG, "-m", "snapshot")
    head = git(repo, "rev-parse", "HEAD")
    probe = {"simulation_file": str(repo / "src/simulation/__init__.py"),
             "bp_file": str(repo / "cpp/build/Release/breach_physics.pyd"),
             "build_id": head}
    run = _run_cfg(tmp_path / "outside" / "run.toml")
    return repo, probe, run, head


def _refused(check, repo, probe, run, backend="cpu_farm", tag=TAG):
    with pytest.raises(rl_env.SnapshotRefused, match=f"require_snapshot: "
                       f"{check}:") as e:
        rl_env.require_snapshot(tag, backend, run, repo=repo, probe=probe)
    return str(e.value)


def test_a_clean_snapshot_passes_and_reports_what_it_checked(snap):
    """Property: an annotated tag at HEAD, a clean tree, this checkout's
    modules, a build id equal to HEAD and a recorded run file pass the farm's
    door, which returns the tag's commit and the run's sha256. Breaks if a
    check refuses a valid snapshot or the record names another commit."""
    repo, probe, run, head = snap
    rec = rl_env.require_snapshot(TAG, "cpu_farm", run, repo=repo,
                                  probe=probe)
    assert rec == {"tag": TAG, "commit": head, "build_id": head,
                   "run_sha256": run.sha256, "backend": "cpu_farm"}


def test_a_lightweight_or_missing_tag_is_refused(snap):
    """Property: a lightweight tag (and a tag that does not exist) is
    refused by the annotated-tag check, even at HEAD. Breaks if a tag
    without a tagger -- which ``git describe`` skips and nobody placed
    deliberately -- could stand for a snapshot."""
    repo, probe, run, _ = snap
    git(repo, "tag", "train/light")
    msg = _refused("annotated tag", repo, probe, run, tag="train/light")
    assert "lightweight" in msg
    assert "no tag" in _refused("annotated tag", repo, probe, run,
                                tag="train/none")


def test_head_past_the_tag_is_refused(snap):
    """Property: HEAD must BE the tag's commit -- one commit later is
    refused. Breaks if a run could train on code the tag does not name."""
    repo, probe, run, _ = snap
    commit(repo, {"src/simulation/more.py": "x = 1\n"})
    probe = dict(probe, build_id=git(repo, "rev-parse", "HEAD"))
    _refused("head at tag", repo, probe, run)


@pytest.mark.parametrize("path", ["levels/x/air_init.npy",
                                  "src/simulation/extra.py",
                                  "cpp/src/extra.cpp"])
def test_an_untracked_file_under_a_snapshot_path_is_refused(snap, path):
    """Property: an UNTRACKED file under src/, levels/ or cpp/ dirties the
    snapshot (a per-tile seed found by presence changes the game with no
    tracked diff). Breaks if the clean check ignores untracked files there."""
    repo, probe, run, _ = snap
    write(repo, {path: "x"})
    assert path in _refused("clean tree", repo, probe, run)


def test_untracked_files_elsewhere_are_allowed(snap):
    """Property: untracked art and notes outside the snapshot paths (and
    ignored build output) do not dirty a snapshot. Breaks if the door
    refuses the real tree's deliberate untracked art."""
    repo, probe, run, _ = snap
    write(repo, {"art/new.png": "x", "notes.txt": "x",
                 "cpp/build_cuda/breach_physics.pyd": "x"})
    rl_env.require_snapshot(TAG, "cpu_farm", run, repo=repo, probe=probe)


def test_a_modified_tracked_file_is_refused(snap):
    """Property: a tracked file differing from HEAD (config.toml) is refused
    by the clean check. Breaks if an uncommitted dial could train under the
    tag's name."""
    repo, probe, run, _ = snap
    write(repo, {"config.toml": "[x]\na = 2\n"})
    assert "config.toml" in _refused("clean tree", repo, probe, run)


@pytest.mark.parametrize("field,value", [
    ("simulation_file", "C:/elsewhere/src/simulation/__init__.py"),
    ("bp_file", "C:/elsewhere/cpp/build/Release/breach_physics.pyd"),
    ("bp_file", "{repo}/cpp/breach_physics.pyd"),
    ("bp_file", "{repo}/src/breach_physics.pyd"),
])
def test_a_module_from_another_checkout_is_refused(snap, field, value):
    """Property: ``simulation`` must come from this checkout's src/ and
    ``breach_physics`` from one of its cpp/build*/ dirs. Breaks if a
    worktree importing main's (or another worktree's) build could pass."""
    repo, probe, run, _ = snap
    probe = dict(probe, **{field: value.format(repo=repo.as_posix())})
    _refused("module paths", repo, probe, run)


@pytest.mark.parametrize("bid", [None, "{head}-dirty", "0" * 40])
def test_a_build_id_other_than_head_is_refused(snap, bid):
    """Property: the module's build-time id must be HEAD exactly -- absent
    (a pre-P1b build), dirty, or another commit is refused. Breaks if a
    stale incremental build or one built from uncommitted edits passes."""
    repo, probe, run, head = snap
    probe = dict(probe, build_id=bid.format(head=head) if bid else None)
    _refused("build id", repo, probe, run)


def test_a_run_without_a_recorded_file_is_refused(snap, tmp_path):
    """Property: the run must come from a file (its sha256 recorded) that
    lies OUTSIDE the checkout. Breaks if a run built in memory, or kept in
    the snapshot tree, could train without a hash to name it."""
    repo, probe, run, _ = snap
    _refused("run file", repo, probe, _run_cfg(run.path, sha=""))
    _refused("run file", repo, probe, _run_cfg(repo / "run.toml"))


def test_an_unknown_backend_is_refused(snap):
    """Property: a backend name the door does not know is refused by name.
    Breaks if a typo silently ran the farm's (weaker) checks."""
    repo, probe, run, _ = snap
    _refused("backend", repo, probe, run, backend="gpu")


def _with_ledger(repo, level: bool, touch: bool):
    base = git(repo, "rev-parse", "HEAD")
    text = (LEVEL_MANIFEST.replace("LEVEL", "true" if level else "false")
            .replace("AT", base))
    files = {"tests/twin_manifest.toml": text}
    if touch:
        files["cpp/src/w.cpp"] = "int w2;\n"
    commit(repo, files, "ledger")
    git(repo, "tag", "-a", "train/t2", "-m", "s2")
    return dict(build_id=git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize("level,touch", [(False, False), (True, True)])
def test_the_device_backend_refuses_while_the_ledger_has_debt(snap, level,
                                                              touch):
    """Property: ``backend="device"`` refuses on twin debt -- a twin marked
    not level, or one whose reference files changed since ``level_at`` --
    and the farm, which needs no ledger, passes the same snapshot. Breaks if
    the device door ignores tools/twin_debt.py or the farm is blocked by
    debt."""
    repo, probe, run, _ = snap
    probe = dict(probe, **_with_ledger(repo, level, touch))
    msg = _refused("twin debt", repo, probe, run, backend="device",
                   tag="train/t2")
    assert "w (" in msg
    rl_env.require_snapshot("train/t2", "cpu_farm", run, repo=repo,
                            probe=probe)


def test_the_device_backend_refuses_outright_with_an_empty_ledger(snap):
    """Property: with every check green and no debt, ``device`` still
    refuses -- no device backend exists in P1b. Breaks if the door claimed
    a backend that is not built."""
    repo, probe, run, _ = snap
    probe = dict(probe, **_with_ledger(repo, True, False))
    msg = _refused("backend", repo, probe, run, backend="device",
                   tag="train/t2")
    assert "no device backend yet" in msg


# ---------------------------------------------------------------------------
# The run manifest
# ---------------------------------------------------------------------------
_KW = dict(run_sha256="ab" * 32, build_id="1" * 40, backend="cpu_farm",
           tag=TAG)


def test_the_manifest_records_the_run_and_its_environment(tmp_path):
    """Property: the manifest holds the commit, tag, build id, run and bank
    sha256 and the environment fields (python, numpy, torch, CuPy, CUDA
    runtime and driver, GPU, host), and a resume with nothing changed passes.
    Breaks if a field the resume door compares is missing from the file."""
    path = tmp_path / "out" / rl_env.RUN_MANIFEST
    rec = rl_env.write_run_manifest(path, **_KW)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == rec
    for k in ("commit", "tag", "build_id", "run_sha256", "bank_sha256",
              "python", "numpy", "torch", "cupy", "cuda_runtime",
              "cuda_driver", "gpu", "hostname"):
        assert k in on_disk, k
    assert on_disk["bank_sha256"] is None          # P8a's
    rl_env.check_run_manifest(path, **_KW)


@pytest.mark.parametrize("field,value", [("run_sha256", "cd" * 32),
                                         ("build_id", "2" * 40),
                                         ("tag", None),
                                         ("bank_sha256", "ef" * 32)])
def test_a_resume_refuses_a_changed_field_by_name(tmp_path, field, value):
    """Property: a resume refuses when the run file, the module, the
    snapshot tag or the bank changed, naming the field. Breaks if a resumed
    run could continue on other code or another reward experiment."""
    path = tmp_path / rl_env.RUN_MANIFEST
    rl_env.write_run_manifest(path, **_KW)
    with pytest.raises(rl_env.RunManifestChanged, match=field):
        rl_env.check_run_manifest(path, **dict(_KW, **{field: value}))


def test_a_resume_refuses_a_changed_environment_but_not_a_new_host(
        tmp_path, monkeypatch):
    """Property: a changed library version (numpy here) refuses the resume
    by name; a different host name alone does not (a resumed job lands on
    another node of the same image). Breaks if a numpy upgrade -- whose
    Generator streams are not promised stable -- went unnoticed, or if
    every resume on a new node were refused."""
    path = tmp_path / rl_env.RUN_MANIFEST
    rl_env.write_run_manifest(path, **_KW)
    real = rl_env.environment_versions
    monkeypatch.setattr(rl_env, "environment_versions",
                        lambda: dict(real(), hostname="another-node"))
    rl_env.check_run_manifest(path, **_KW)
    monkeypatch.setattr(rl_env, "environment_versions",
                        lambda: dict(real(), numpy="0.0.0"))
    with pytest.raises(rl_env.RunManifestChanged, match="numpy"):
        rl_env.check_run_manifest(path, **_KW)


def test_reading_the_environment_imports_no_cuda_library():
    """Property: writing the manifest's environment imports neither torch
    nor CuPy (a farm process creates no CUDA context). Breaks if the
    versions were read by importing the libraries."""
    code = ("import sys; sys.path[:0] = [r'%s', r'%s']; import rl_env; "
            "rl_env.environment_versions(); "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'cupy')))" % (ROOT, ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "[]"


# ---------------------------------------------------------------------------
# [run] out_dir and the farm's wiring
# ---------------------------------------------------------------------------
def test_out_dir_inside_the_checkout_is_refused(tmp_path):
    """Property: ``[run] out_dir`` inside the checkout is refused by name, a
    relative one resolves against the run file. Breaks if checkpoints and
    logs could be written into a snapshot's tree (dirtying it)."""
    from test_rl_env import _OK, _box, _run_toml
    _box(tmp_path, "abox")
    bad = _run_toml(tmp_path / "bad.toml", levels_dir=tmp_path,
                    extra_run=f"out_dir = '{(ROOT / 'out' / 'r').as_posix()}'"
                              "\n", **_OK)
    with pytest.raises(ValueError, match="out_dir"):
        rl_env.load_run(bad)
    ok = _run_toml(tmp_path / "ok.toml", levels_dir=tmp_path,
                   extra_run="out_dir = 'runs/r1'\n", **_OK)
    assert Path(rl_env.load_run(ok).out_dir) == (tmp_path / "runs" /
                                                 "r1").resolve()


def test_the_farm_writes_the_manifest_and_refuses_a_bad_snapshot(tmp_path):
    """Property: a BatchedEnv whose run names ``out_dir`` writes the run
    manifest there with the WORKERS' module build id and the run's sha256;
    a second env on the same out_dir after the run file changed refuses (a
    resume); ``snapshot=`` a tag that does not exist refuses before any
    episode. Breaks if the farm skips the manifest, records the driver's
    module instead of the workers', or lets a bad snapshot start."""
    from test_rl_env import _OK, _box, _run_toml
    _box(tmp_path, "abox")
    out = tmp_path / "runs" / "r1"
    p = _run_toml(tmp_path / "run.toml", levels_dir=tmp_path, n_envs=1,
                  extra_run=f"out_dir = '{out.as_posix()}'\n", **_OK)
    with rl_env.BatchedEnv(p, n_workers=1) as env:
        bid = env.probe()[0]["build_id"]
        assert env.manifest_path == str(out / rl_env.RUN_MANIFEST)
    rec = json.loads((out / rl_env.RUN_MANIFEST).read_text(encoding="utf-8"))
    assert rec["build_id"] == bid and bid
    assert rec["run_sha256"] == rl_env.load_run(p).sha256
    assert rec["tag"] is None and rec["backend"] == "cpu_farm"
    p.write_text(p.read_text(encoding="utf-8").replace("seed = 11",
                                                       "seed = 12"),
                 encoding="utf-8")
    with pytest.raises(rl_env.RunManifestChanged, match="run_sha256"):
        rl_env.BatchedEnv(p, n_workers=1)
    q = _run_toml(tmp_path / "run2.toml", levels_dir=tmp_path, n_envs=1,
                  **_OK)
    with pytest.raises(rl_env.SnapshotRefused, match="annotated tag"):
        rl_env.BatchedEnv(q, n_workers=1, snapshot="train/no-such-tag")
