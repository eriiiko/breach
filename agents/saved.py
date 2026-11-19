"""agents/saved.py -- THE saved network, a soldier's export and the training
manifest (rung 2 note 2d, 9 to 11; design 4.7 and 1): the ONE writer and
reader of all three. A driver of a saved network loads through it, never
its own ``torch.load``.

THE SAVED FILE (:func:`save` / :func:`load`), one ``torch.save`` of plain
tables (read back with ``weights_only``: nothing but tensors, numbers,
strings, lists and dicts), written ATOMICALLY: the bytes go to a temporary
name in the same folder, are flushed to the disk, and only then renamed
over the old file -- a kill during a save leaves the previous file whole.
Its fields (the note's table): ``format``; ``soldier`` and ``value`` (both
networks' weights); ``soldier_record`` / ``value_record`` (what each fits,
compared by ``spec.check_fits``); ``net`` (the network spec's widths);
``learner`` (both optimisers' state, the value scale, the learner's
generator, its round and the weights' version); ``collector`` (its
generator and its decision count -- nothing of a match); ``counters``
(rounds, learning decisions, batches, the weights' version); ``settings``
(the ``Settings`` record as a table); ``specs`` (observation, action,
reward profile); ``hashes`` (the run file's and the training file's
sha256, the build id, the checkout's commit); ``manifest`` (the training
manifest's record); ``started_from`` (``"nothing"``; ``{"copy": ...}``,
patch 2e's; or ``{"saved": <sha256 of the file resumed>, "round": n,
"from": <its own started_from>}``, with ``"code_change"`` when
``--allow-code-change`` let the build id or commit differ).

WHAT A RESUME RESTORES (:func:`restore`): both networks, both optimisers,
the scale, both generators, the counters; the step size then continues at
``step_size_at(rounds done)``. WHAT IT DOES NOT: the matches in flight.
Every match starts again at the environment's first reset, and with the
memory on every row's memory is zero (design 4.7: the card is not
repeatable anyway). What is exact is everything the learner owns.

WHAT A RESUME REFUSES (:func:`check_resume`, :class:`ResumeRefused` naming
the field): a file of another ``format``; either network's record
(``check_fits``) or widths; the observation / action spec or reward
profile against the environment's layout; the run file's or the training
file's sha256 against the folder's copies; any field of the training
manifest but ``commit`` / ``dirty``; the build id or the commit, unless
``allow_code_change`` (then returned, written into the curve and into every
later save's ``started_from``).

THE STATE A SAVE MUST HOLD is listed, per object, in :data:`SAVED` and
:data:`NOT_SAVED` (with why); ``agents/tests/test_saved.py`` holds every
attribute of the learner, collector, trainer and scale to one of the two,
so a new state joins the saved file in the patch that adds it.

THE TRAINING MANIFEST (:func:`manifest_record` / :func:`write_manifest`),
``training_manifest.json`` beside the curve: ``format``; both files'
sha256; ``levels_sha256`` (one hash over the level folders the run names:
every file's path relative to the levels folder and its bytes, in sorted
order -- a copy of the folder elsewhere hashes the same);
``room_generator_commit`` (the last commit that touched
``tools/gen_training_rooms.py`` when the run names one of its rooms, else
None); ``seed_lists_sha256`` (None until patch 2g); ``commit`` and
``dirty`` (``rl_env.check_clean_tree``'s verdict on the checkout).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from agents import _game
from agents.spec import NetSpec, SpecRecord, check_fits

FORMAT = 1
MANIFEST_FORMAT = 1
LAST = "last.pt"
MANIFEST = "training_manifest.json"
#: The fields a code change may alter, only with ``allow_code_change``.
CODE_FIELDS = ("build_id", "commit")
#: The manifest's fields a resume does not compare (section 10's rule).
MANIFEST_FREE = ("commit", "dirty")
#: The room generator whose commit the manifest names.
ROOM_GENERATOR = "tools/gen_training_rooms.py"

#: Per object, the attributes a saved file holds ...
SAVED = {
    "Trainer": ("rounds",),
    "Learner": ("soldier", "value", "opt_soldier", "opt_value", "scale",
                "rng", "round", "version"),
    "Collector": ("generator", "decisions", "version"),
    "ValueScale": ("_m1", "_m2", "mean", "spread", "rounds"),
}
#: ... and those it does not, with why.
NOT_SAVED = {
    "Trainer": {"env": "the environment is built again from the run file",
                "settings": "read again from the folder's training file "
                            "(its hash is compared)",
                "rows": "derived from the environment",
                "batch": "overwritten every round",
                "learner": "its own entries", "collector": "its own entries"},
    "Learner": {"settings": "the training file", "device": "the machine's",
                "ids": "the run's names, compared at a resume",
                "_segs": "one round's stretches, cut afresh from every "
                         "batch"},
    "Collector": {
        "env": "the environment", "rows": "derived",
        "device": "the machine's", "ids": "the run's names",
        "soldier": "the learner's weights, handed over at the restore",
        "env_of_row": "derived", "_names": "the layout's counters",
        "_obs": "a match in flight: matches start again",
        "_full": "a match in flight", "_memory": "a match in flight (zero "
        "at the new start)", "_first": "a match in flight",
        "_story": "a match in flight", "_level": "a match in flight",
        "_paid": "a match in flight",
        "seconds": "one round's", "log": "one round's"},
    "ValueScale": {"layer": "the value network's (saved with it)",
                   "share": "Settings", "floor": "Settings"},
}


class ResumeRefused(ValueError):
    """A resume is refused (the message names the field)."""


class SoldierRefused(ValueError):
    """A soldier's export does not fit the network offered (the message
    names the field)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo, *args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def git_head(repo=None) -> str | None:
    """The checkout's ``HEAD`` (None outside a git checkout)."""
    return _git(repo or _game.ROOT, "rev-parse", "HEAD")


def build_id_of(env) -> str | None:
    """The game's build id the environment's workers loaded (one for all),
    or None for an environment without workers (the stand-in)."""
    probe = getattr(env, "probe", None)
    if probe is None:
        return None
    ids = {p.get("build_id") for p in probe()}
    if len(ids) != 1:
        raise RuntimeError(f"farm workers loaded different builds: "
                           f"{sorted(map(str, ids))}")
    return ids.pop()


def specs_of(layout) -> dict:
    """The three spec names of an environment's layout."""
    return {"observation": layout.spec, "action": layout.action_spec,
            "reward_profile": layout.reward_profile}


def _rec(d: dict) -> SpecRecord:
    def tup(x):
        return tuple(tup(y) for y in x) if isinstance(x, (list, tuple)) else x
    return SpecRecord(**{k: tup(v) for k, v in d.items()})


def _net(d: dict) -> NetSpec:
    return NetSpec(**{k: tuple(v) if isinstance(v, list) else v
                      for k, v in d.items()})


# ---------------------------------------------------------------------------
# the saved file
# ---------------------------------------------------------------------------
def state(trainer, *, hashes: dict, manifest: dict | None,
          started_from) -> dict:
    """The saved file's tables for ``trainer`` as it stands (module text)."""
    ln, col = trainer.learner, trainer.collector
    sc = ln.scale
    return {
        "format": FORMAT,
        "soldier": {k: v.detach().cpu() for k, v in
                    ln.soldier.state_dict().items()},
        "value": {k: v.detach().cpu() for k, v in
                  ln.value.state_dict().items()},
        "soldier_record": ln.soldier.record.as_dict(),
        "value_record": ln.value.record.as_dict(),
        "net": dataclasses.asdict(ln.soldier.spec),
        "learner": {
            "opt_soldier": ln.opt_soldier.state_dict(),
            "opt_value": ln.opt_value.state_dict(),
            "scale": {"m1": sc._m1.cpu(), "m2": sc._m2.cpu(),
                      "mean": sc.mean.cpu(), "spread": sc.spread.cpu(),
                      "rounds": int(sc.rounds)},
            "rng": ln.rng.bit_generator.state,
            "round": int(ln.round),
            "version": int(ln.version),
        },
        "collector": {"generator": col.generator.get_state(),
                      "decisions": int(col.decisions)},
        "counters": {"rounds": int(trainer.rounds),
                     "decisions": int(col.decisions),
                     "batches": int(ln.round),
                     "version": int(ln.version)},
        "settings": dataclasses.asdict(trainer.settings),
        "specs": specs_of(trainer.env.layout),
        "hashes": {k: hashes.get(k) for k in
                   ("run_file", "training_file") + CODE_FIELDS},
        "manifest": manifest,
        "started_from": started_from,
    }


def _write_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` and flush it to the disk."""
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_atomic(path, obj) -> str:
    """``torch.save`` ``obj`` to ``path`` atomically (a temporary name in
    the same folder, flushed, then renamed over ``path``); returns the
    written bytes' sha256."""
    path = Path(path)
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    tmp = path.with_name(path.name + ".partial")
    _write_file(tmp, data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def save(path, trainer, *, hashes: dict, manifest: dict | None,
         started_from) -> str:
    """Save ``trainer``'s run at ``path`` (atomically); returns the file's
    sha256."""
    return write_atomic(path, state(trainer, hashes=hashes, manifest=manifest,
                                    started_from=started_from))


def load(path) -> dict:
    """A saved file's tables (plain data only: ``weights_only``)."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def check_resume(saved: dict, trainer, *, hashes: dict, manifest: dict,
                 allow_code_change: bool = False) -> dict | None:
    """Refuse (:class:`ResumeRefused`, naming the field) a resume of
    ``saved`` into ``trainer`` (built from the folder's files) unless every
    compared field agrees (module text). ``hashes`` and ``manifest`` are
    what the checkout and the folder give NOW. Returns the code change
    allowed (``{field: [saved, now]}``) or None."""
    fmt = saved.get("format")
    if fmt != FORMAT:
        raise ResumeRefused(f"format: the saved file is format {fmt!r}, "
                            f"this code reads format {FORMAT}")
    ln = trainer.learner
    for name, have, net in (("soldier", saved["soldier_record"], ln.soldier),
                            ("value", saved["value_record"], ln.value)):
        try:
            check_fits(_rec(have), net.record)
        except ValueError as e:
            raise ResumeRefused(f"{name} record: {e}") from None
    if _net(saved["net"]) != ln.soldier.spec:
        raise ResumeRefused(f"network widths: saved {saved['net']}, the "
                            f"training file's "
                            f"{dataclasses.asdict(ln.soldier.spec)}")
    now = specs_of(trainer.env.layout)
    for k in now:
        if saved["specs"].get(k) != now[k]:
            raise ResumeRefused(f"{k} spec: saved {saved['specs'].get(k)!r}, "
                                f"the environment's {now[k]!r}")
    for k in ("run_file", "training_file"):
        if saved["hashes"].get(k) != hashes.get(k):
            raise ResumeRefused(f"{k} sha256: saved {saved['hashes'].get(k)}"
                                f", the folder's copy {hashes.get(k)}")
    old_m = saved.get("manifest") or {}
    bad = [f"{k}: saved {old_m.get(k)!r}, now {manifest.get(k)!r}"
           for k in sorted(set(old_m) | set(manifest))
           if k not in MANIFEST_FREE and old_m.get(k) != manifest.get(k)]
    if bad:
        raise ResumeRefused("training manifest " + "; ".join(bad))
    change = {k: [saved["hashes"].get(k), hashes.get(k)] for k in CODE_FIELDS
              if saved["hashes"].get(k) != hashes.get(k)}
    if change and not allow_code_change:
        raise ResumeRefused(
            "code changed: " + "; ".join(f"{k}: saved {a}, now {b}"
                                         for k, (a, b) in change.items())
            + " (--allow-code-change lets these two differ, and records it)")
    return change or None


def restore(trainer, saved: dict) -> None:
    """Put ``saved``'s state into ``trainer`` (built from the same files):
    both networks, both optimisers, the scale, both generators, the
    counters; the collector is handed the soldier's weights with their
    version. The matches are not restored (module text)."""
    ln, col = trainer.learner, trainer.collector
    ln.soldier.load_state_dict(saved["soldier"])
    ln.value.load_state_dict(saved["value"])
    # the optimisers KEEP the tensors they are given (on the processor no
    # copy is made) and step them in place: they get their own copies, or
    # two trainers restored from one loaded file would share their moments
    L = copy.deepcopy(saved["learner"])
    ln.opt_soldier.load_state_dict(L["opt_soldier"])
    ln.opt_value.load_state_dict(L["opt_value"])
    sc, s = ln.scale, L["scale"]
    dev = sc.mean.device
    sc._m1 = s["m1"].to(dev)
    sc._m2 = s["m2"].to(dev)
    sc.mean = s["mean"].to(dev)
    sc.spread = s["spread"].to(dev)
    sc.rounds = int(s["rounds"])
    ln.rng.bit_generator.state = L["rng"]
    ln.round = int(L["round"])
    ln.version = int(L["version"])
    col.generator.set_state(saved["collector"]["generator"])
    col.decisions = int(saved["collector"]["decisions"])
    trainer.rounds = int(saved["counters"]["rounds"])
    col.set_weights(ln.weights(), ln.version)


# ---------------------------------------------------------------------------
# a soldier's export
# ---------------------------------------------------------------------------
def export_soldier(path, soldier, layout) -> str:
    """What a driver of a saved soldier needs and nothing of the learner:
    its weights, its record, its widths and the three spec names (written
    atomically); returns the file's sha256."""
    return write_atomic(path, {
        "format": FORMAT, "kind": "soldier",
        "soldier": {k: v.detach().cpu() for k, v in
                    soldier.state_dict().items()},
        "soldier_record": soldier.record.as_dict(),
        "net": dataclasses.asdict(soldier.spec),
        "specs": specs_of(layout)})


def load_soldier(path, soldier) -> dict:
    """Load an export (:func:`export_soldier`) into ``soldier`` (a
    ``SoldierNet``), refusing by name (:class:`SoldierRefused`) a file of
    another format, other widths, or a record that does not fit
    (``check_fits``). Returns the export's spec names."""
    d = load(path)
    if d.get("format") != FORMAT or d.get("kind") != "soldier":
        raise SoldierRefused(f"format: {path} is not a soldier's export of "
                             f"format {FORMAT}")
    if _net(d["net"]) != soldier.spec:
        raise SoldierRefused(f"network widths: export {d['net']}, network "
                             f"{dataclasses.asdict(soldier.spec)}")
    try:
        check_fits(_rec(d["soldier_record"]), soldier.record)
    except ValueError as e:
        raise SoldierRefused(f"soldier record: {e}") from None
    soldier.load_state_dict(d["soldier"])
    return dict(d["specs"])


# ---------------------------------------------------------------------------
# the training manifest
# ---------------------------------------------------------------------------
def levels_sha256(levels_dir, levels) -> str | None:
    """One sha256 over the level folders ``levels`` under ``levels_dir``:
    every file's path relative to ``levels_dir`` and its bytes, in sorted
    order (None when the run names no levels folder)."""
    if not levels_dir:
        return None
    root = Path(levels_dir)
    files = []
    for name in sorted(set(levels)):
        d = root / name
        if not d.is_dir():
            raise ValueError(f"level folder {d} does not exist")
        files += [p for p in d.rglob("*") if p.is_file()]
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: q.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix().encode("utf-8")
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, "little") + rel)
        h.update(len(data).to_bytes(8, "little") + data)
    return h.hexdigest()


def _generated_rooms() -> tuple:
    import importlib.util
    path = _game.ROOT / ROOM_GENERATOR
    spec = importlib.util.spec_from_file_location("_breach_training_rooms",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # its dataclasses look it up
    spec.loader.exec_module(mod)
    return tuple(mod.level_names())


def dirty(repo=None) -> bool:
    """Whether the checkout has a tracked change, or an untracked file under
    the run check's paths (``rl_env.check_clean_tree``, the one check)."""
    import rl_env
    try:
        rl_env.check_clean_tree(repo or _game.ROOT)
    except rl_env.SnapshotRefused:
        return True
    return False


def manifest_record(run_file, training_file, run, repo=None) -> dict:
    """The training manifest (module text) for the folder's ``run_file``
    and ``training_file`` copies and the loaded run config ``run``."""
    repo = Path(repo) if repo is not None else _game.ROOT
    gen = None
    if run.levels_dir and set(run.levels) & set(_generated_rooms()):
        gen = _git(repo, "log", "-1", "--format=%H", "--", ROOM_GENERATOR)
    return {"format": MANIFEST_FORMAT,
            "training_file_sha256": sha256_file(training_file),
            "run_file_sha256": sha256_file(run_file),
            "levels_sha256": levels_sha256(run.levels_dir, run.levels),
            "room_generator_commit": gen,
            "seed_lists_sha256": None,
            "commit": git_head(repo),
            "dirty": dirty(repo)}


def write_manifest(folder, record: dict) -> Path:
    """Write ``training_manifest.json`` into ``folder`` (atomically)."""
    path = Path(folder) / MANIFEST
    tmp = path.with_name(path.name + ".partial")
    _write_file(tmp, (json.dumps(record, indent=2, sort_keys=True) + "\n")
                .encode("utf-8"))
    os.replace(tmp, path)
    return path


def read_manifest(folder) -> dict:
    return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))


__all__ = ["FORMAT", "MANIFEST_FORMAT", "LAST", "MANIFEST", "CODE_FIELDS",
           "MANIFEST_FREE", "SAVED", "NOT_SAVED", "ResumeRefused",
           "SoldierRefused", "sha256_file", "git_head", "build_id_of",
           "specs_of", "state", "write_atomic", "save", "load",
           "check_resume", "restore", "export_soldier", "load_soldier",
           "levels_sha256", "dirty", "manifest_record", "write_manifest",
           "read_manifest"]
