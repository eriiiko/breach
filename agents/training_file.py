"""agents/training_file.py -- the training file: THE way a run's trainer
numbers reach :class:`agents.settings.Settings` from disk (rung 2 note 2c,
4; design 4.4).

One TOML file, five tables, every key a field that exists::

    [network]
    spec = "v1"          # the network spec (agents.spec.NETWORK_SPEC)
    width = "full"       # "full" (spec.v1) or "reduced" (spec.reduced,
                         # the tests' width); optional, default "full"
    memory = false       # the soldier's memory (the GRU); required
    [train]
    planned_rounds = 1200   # REQUIRED: Settings has no default for it
    decisions = 128         # any other Settings field, by its name
    ...
    [weights]               # optional; when present it names EVERY group
    outcome = 1.0           # of reward_v1.REWARD_GROUPS, nothing else
    own = 1.0
    ...
    [stop]
    wall_hours = 2.5        # Settings.wall_hours; optional
    [save]
    save_every = 10         # Settings.save_every / keep_every; optional
    keep_every = 0

REFUSED BY NAME (:class:`TrainingFileRefused`): an unknown table or key, a
missing ``planned_rounds`` or ``[network] memory``, a value of the wrong
type (a bool where a number belongs, a float where an integer belongs, a
string where a number belongs), a ``[weights]`` table that does not name
every group, an unknown network spec or width, and anything
:class:`Settings` itself refuses. A key left out takes ``Settings``'
default (the design's starting value), so a file that writes the design's
table is that table and nothing is typed anywhere else.

The file's bytes and sha256 are kept (:class:`TrainingFile`) so the run
script copies the file it read beside the curve.
"""
from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from agents import _game  # noqa: F401
from agents.settings import Settings
from agents.spec import NETWORK_SPEC, NetSpec, reduced, v1
from simulation.reward_v1 import REWARD_GROUPS

#: The widths ``[network] width`` names: the spec's constructor for each.
WIDTHS = {"full": v1, "reduced": reduced}
#: ``[network]``'s keys and whether each is required.
NETWORK_KEYS = {"spec": True, "width": False, "memory": True}
#: The Settings fields that live in their own tables, not in ``[train]``.
_OWN_TABLE = {"weights": "weights", "wall_hours": "stop",
              "save_every": "save", "keep_every": "save"}
TABLES = ("network", "train", "weights", "stop", "save")


class TrainingFileRefused(ValueError):
    """The training file is refused (the message names the table and key)."""


@dataclass(frozen=True)
class TrainingFile:
    """A read training file: the settings, the network spec, its bytes."""
    settings: Settings
    net: NetSpec
    text: str
    sha256: str
    path: str = ""


def _types() -> dict:
    """Settings' field name -> "int" / "float" / "float?" / "tuple"."""
    out = {}
    for f in fields(Settings):
        t = str(f.type).replace(" ", "")
        out[f.name] = {"int": "int", "float": "float", "tuple": "tuple",
                       "float|None": "float?"}.get(t, t)
    return out


def _number(v, kind: str, where: str):
    if isinstance(v, bool):
        raise TrainingFileRefused(f"{where} must be a number, got {v!r}")
    if kind == "int":
        if not isinstance(v, int):
            raise TrainingFileRefused(f"{where} must be an integer, got "
                                      f"{v!r}")
        return int(v)
    if not isinstance(v, (int, float)):
        raise TrainingFileRefused(f"{where} must be a number, got {v!r}")
    return float(v)


def _table(raw: dict, name: str, where: str) -> dict:
    t = raw.get(name, {})
    if not isinstance(t, dict):
        raise TrainingFileRefused(f"{where}: [{name}] must be a table")
    return t


def parse(raw: dict, where: str = "training file") -> tuple:
    """A parsed TOML dict -> ``(Settings, NetSpec)``, refusing by name
    (module text)."""
    unknown = sorted(set(raw) - set(TABLES))
    if unknown:
        raise TrainingFileRefused(f"{where}: unknown table(s) {unknown} "
                                  f"(a training file has {list(TABLES)})")
    # [network]
    net = _table(raw, "network", where)
    bad = sorted(set(net) - set(NETWORK_KEYS))
    if bad:
        raise TrainingFileRefused(f"{where}: [network] unknown key(s) {bad} "
                                  f"(keys: {list(NETWORK_KEYS)})")
    for k, req in NETWORK_KEYS.items():
        if req and k not in net:
            raise TrainingFileRefused(f"{where}: [network] {k} is missing")
    if net["spec"] != NETWORK_SPEC:
        raise TrainingFileRefused(f"{where}: [network] spec must be "
                                  f"{NETWORK_SPEC!r}, got {net['spec']!r}")
    width = net.get("width", "full")
    if width not in WIDTHS:
        raise TrainingFileRefused(f"{where}: [network] width must be one of "
                                  f"{list(WIDTHS)}, got {width!r}")
    if not isinstance(net["memory"], bool):
        raise TrainingFileRefused(f"{where}: [network] memory must be true "
                                  f"or false, got {net['memory']!r}")
    spec = WIDTHS[width](memory=net["memory"])
    # [train]
    types = _types()
    train_keys = {k for k in types if k not in _OWN_TABLE}
    train = _table(raw, "train", where)
    bad = sorted(set(train) - train_keys)
    if bad:
        raise TrainingFileRefused(
            f"{where}: [train] unknown key(s) {bad} (each key is a field of "
            f"agents.settings.Settings: {sorted(train_keys)})")
    if "planned_rounds" not in train:
        raise TrainingFileRefused(f"{where}: [train] planned_rounds is "
                                  f"missing (every run plans its own)")
    kw = {k: _number(v, types[k], f"{where}: [train] {k}")
          for k, v in train.items()}
    # [weights]
    if "weights" in raw:
        w = _table(raw, "weights", where)
        if set(w) != set(REWARD_GROUPS):
            raise TrainingFileRefused(
                f"{where}: [weights] must name every reward group "
                f"{list(REWARD_GROUPS)} and nothing else; missing "
                f"{sorted(set(REWARD_GROUPS) - set(w))}, unknown "
                f"{sorted(set(w) - set(REWARD_GROUPS))}")
        kw["weights"] = tuple(_number(w[g], "float", f"{where}: [weights] {g}")
                              for g in REWARD_GROUPS)
    # [stop] and [save]: their fields by name
    for tname in ("stop", "save"):
        table = _table(raw, tname, where)
        keys = sorted(k for k, t in _OWN_TABLE.items() if t == tname)
        bad = sorted(set(table) - set(keys))
        if bad:
            raise TrainingFileRefused(f"{where}: [{tname}] unknown key(s) "
                                      f"{bad} (keys: {keys})")
        for k, v in table.items():
            kind = "float" if types[k] == "float?" else types[k]
            kw[k] = _number(v, kind, f"{where}: [{tname}] {k}")
    try:
        settings = Settings(**kw)
    except ValueError as e:
        raise TrainingFileRefused(f"{where}: {e}") from None
    return settings, spec


def load(path) -> TrainingFile:
    """Read and check the training file at ``path`` (module text)."""
    p = Path(path)
    data = p.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise TrainingFileRefused(f"{p}: {e}") from None
    settings, net = parse(raw, str(p))
    return TrainingFile(settings=settings, net=net,
                        text=data.decode("utf-8"),
                        sha256=hashlib.sha256(data).hexdigest(),
                        path=str(p.resolve()))


__all__ = ["WIDTHS", "NETWORK_KEYS", "TABLES", "TrainingFileRefused",
           "TrainingFile", "parse", "load"]
