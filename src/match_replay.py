"""Match recording and replay by RE-SIMULATION (#126, milestone 10 Replay).

The sim is deterministic in integer, so a match is fully named by what went
INTO it: the level, the parsed match spec, the seed, and every intent any
controller handed the sim, tick by tick. A recording is exactly that, plus a
``(field_digest, tick_digest)`` checkpoint every ``checkpoint_every_ticks``
ticks (and at tick 0 and the last tick) that proves a replay is the same
match and fails loudly on build drift. The field recorder
(``simulation/recorder.py``) is NOT the replay path: it stays the blow-up
debug ring.

The parts:

- **The tap** lives in the sim (``Simulation.attach_intent_log``): the eight
  intent entry points hand every call to the attached sink BEFORE acting,
  with the tick it was issued at. :class:`IntentLog` is that sink. No
  controller logs anything itself.
- :class:`MatchRecorder` owns a recording in progress: it attaches the log,
  takes the checkpoints as the runner steps, and closes into a
  :class:`Recording`. ``tools/run_match.py --record`` drives it.
- :func:`write_recording` / :func:`read_recording`: the on-disk format v1,
  one gzip'd JSON-lines file per match (below).
- :class:`ReplayControl`: the registered match controller ``"replay"``
  (``control_source``), built from a recording by
  ``create_match_controllers(sim, spec, recording=...)``. Called once per
  tick it first verifies that tick's checkpoint, then feeds that tick's
  logged intents back in their original order; :meth:`ReplayControl.finish`
  verifies the last one. A mismatch raises :class:`ReplayDivergence` naming
  the tick. ``tools/run_match.py --replay`` drives it headless; #127 puts it
  behind ``main.py --replay``.

Format v1 (gzip, mtime 0, so one recording is one byte string), UTF-8 JSON,
one object per line::

    {"format": "breach-match-replay", "version": 1, "level": "ship_demo",
     "seed": 1, "checkpoint_every_ticks": 60, "build": "<git describe>",
     "spec": {...MatchSpec as a dict...}}
    {"t": 0, "i": [["set_move_dir", 0, 65536, 0, 1], ["drive_zombies", [8, 9]]]}
    {"c": 0, "f": "<field_digest>", "k": "<tick_digest>"}
    ...
    {"end": 240, "result": {"outcome": "draw", "team": null, "ticks": 240}}

An ``"i"`` row is one tick's intents in issue order, each ``[kind, *args]``
with the arguments exactly as the sim's method normalises them: ints stay
ints (Q16.16 values are their raw ints), ``set_trigger``'s ``held`` a bool,
``throw_grenade_intent``'s ``fuse_seconds`` its exact IEEE value as
``float.hex()``, its ``reach_q`` an int or null. A tick with no intent has no
row. ``apply_action`` orders are REFUSED at record time (v1 matches issue
only the per-tick intents; an ``Order`` has no serializer yet): recording a
match that places orders raises :class:`RecordingRefused`.

Determinism, verified for #126 (not assumed): the only random draw in a
match is ``Simulation(seed=)``'s ``sim.rng``; nothing outside
``src/simulation`` reads it on the match path (bots and zombies draw nothing:
``control_bots`` / ``control_zombies`` headers; ``run_match`` passes the seed
only). A replay spawns through the same ``Simulation.spawn_match`` +
``MatchRules`` path as the recorded run, so unit ids agree.

Pyray-free; imports only the stdlib, ``control_source`` and
``simulation.match`` at module level.
"""
from __future__ import annotations

import gzip
import io
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from control_source import ControlSource
from simulation.match import MatchSpec, MatchTeam

FORMAT_NAME = "breach-match-replay"
FORMAT_VERSION = 1

#: The recordable intent kinds (the sim's entry point names) -> the type of
#: each positional argument after the kind. "int" is a Python int (never a
#: bool), "bool" a bool, "fhex" a float stored as float.hex(), "int?" an int
#: or None, "ints" a list of ints.
INTENT_KINDS = {
    "set_move_dir": ("int", "int", "int", "int"),
    "clear_move_dir": ("int",),
    "set_aim": ("int", "int", "int"),
    "set_trigger": ("int", "bool"),
    "throw_grenade_intent": ("int", "int", "int", "fhex", "int?"),
    "use_intent": ("int",),
    "drive_zombies": ("ints",),
}
#: Entry points the tap sees but format v1 refuses to record.
REFUSED_KINDS = ("apply_action",)


class RecordingRefused(ValueError):
    """The match did something format v1 cannot record."""


class ReplayRefused(ValueError):
    """A recording cannot be replayed here (wrong format, level, seed or
    spec): refused by its header, before any tick runs."""


class ReplayDivergence(RuntimeError):
    """The replay is not the recorded match: a checkpoint's digests differ,
    or the match ended at another tick than the recording. ``tick`` names
    the first tick found different."""

    def __init__(self, tick: int, message: str):
        super().__init__(f"replay diverged at tick {tick}: {message}")
        self.tick = tick


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def replay_settings(cfg) -> int:
    """``[replay] checkpoint_every_ticks`` of a loaded config (``CFG``): an
    int >= 1. Raises ValueError naming the key when the section or key is
    missing or mistyped -- a silent default would hide a typo."""
    section = getattr(cfg, "replay", None)
    if section is None:
        raise ValueError("config.toml has no [replay] section")
    if not hasattr(section, "checkpoint_every_ticks"):
        raise ValueError("[replay] is missing 'checkpoint_every_ticks'")
    v = section.checkpoint_every_ticks
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"[replay] checkpoint_every_ticks must be an int "
                         f">= 1, got {v!r}")
    return v


# ---------------------------------------------------------------------------
# Digests (the checkpoint) -- the same snapshot run_match reports
# ---------------------------------------------------------------------------
def match_digests(sim) -> tuple:
    """``(field_digest, tick_digest)`` of ``sim`` as it stands, through the
    determinism gates' one snapshot (``field_ab_harness.capture_snapshot``).
    Those live in ``tests/``; the caller puts it on ``sys.path`` (as
    ``tools/run_match.py`` does)."""
    try:
        from field_ab_harness import capture_snapshot
        from field_digest import field_digest, tick_digest
    except ImportError as e:     # pragma: no cover - a path set-up error
        raise ImportError("match_digests needs the repo's tests/ directory "
                          "on sys.path (field_ab_harness, field_digest)") \
            from e
    snap = capture_snapshot(sim)
    return field_digest(snap), tick_digest(snap)


def level_id(level) -> str:
    """The name a recording knows its level by: the level FOLDER's name, the
    one ``level_loader.load`` takes (``level.name`` is level.toml's display
    name, which can differ: ship_demo's is ``ship_s1_medium``)."""
    return Path(level.path).name


def build_id() -> str | None:
    """``git describe --always --dirty`` of this checkout, or None when git
    is not at hand. Informational only: a replay never refuses on it (the
    checkpoints are what prove a build replays a match)."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=str(Path(__file__).resolve().parent), capture_output=True,
            text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------
def spec_to_dict(spec: MatchSpec) -> dict:
    d = asdict(spec)
    d["teams"] = [dict(t) for t in d["teams"]]
    return d


def spec_from_dict(d: dict) -> MatchSpec:
    d = dict(d)
    d["teams"] = tuple(MatchTeam(**t) for t in d["teams"])
    return MatchSpec(**d)


@dataclass
class Recording:
    """One recorded match. ``intents`` maps a tick to its intents in issue
    order, each ``(kind, args)`` with ``args`` a tuple as the sim normalised
    it; ``checkpoints`` maps a tick to ``(field_digest, tick_digest)`` taken
    with ``sim.tick`` at that value; ``end_tick`` is the tick the match ended
    at (its last checkpoint)."""
    level: str
    spec: MatchSpec
    seed: int
    checkpoint_every_ticks: int
    build: str | None = None
    intents: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    end_tick: int | None = None
    result: dict | None = None

    def n_intents(self) -> int:
        return sum(len(v) for v in self.intents.values())


class IntentLog:
    """The tap's sink (``Simulation.attach_intent_log``): appends every
    recordable call to ``recording.intents[tick]`` in issue order; refuses
    the kinds format v1 cannot record."""

    def __init__(self, recording: Recording):
        self.recording = recording

    def record(self, tick: int, kind: str, args: tuple) -> None:
        if kind in REFUSED_KINDS:
            raise RecordingRefused(
                f"tick {tick}: {kind} is not recordable in replay format "
                f"v{FORMAT_VERSION} (a match records the per-tick intents "
                f"only; an Order has no serializer)")
        if kind not in INTENT_KINDS:
            raise RecordingRefused(f"tick {tick}: unknown intent {kind!r}")
        if kind == "drive_zombies":
            args = (list(args[0]),)
        self.recording.intents.setdefault(int(tick), []).append(
            (kind, tuple(args)))


class MatchRecorder:
    """A recording in progress, owned by the match runner:
    :meth:`start` after spawning and building the controllers (attaches the
    tap, checkpoints tick 0), :meth:`after_step` after every ``step()``
    (checkpoints every ``checkpoint_every_ticks``), :meth:`finish` once the
    match is terminal (last checkpoint, detaches, returns the Recording)."""

    def __init__(self, level_name: str, spec: MatchSpec, seed: int, *,
                 checkpoint_every_ticks: int, build: str | None = None,
                 digest=match_digests):
        if isinstance(checkpoint_every_ticks, bool) or \
                not isinstance(checkpoint_every_ticks, int) or \
                checkpoint_every_ticks < 1:
            raise ValueError(f"checkpoint_every_ticks must be an int >= 1, "
                             f"got {checkpoint_every_ticks!r}")
        self.recording = Recording(
            level=level_name, spec=spec, seed=int(seed),
            checkpoint_every_ticks=checkpoint_every_ticks, build=build)
        self._digest = digest

    def _checkpoint(self, sim) -> None:
        self.recording.checkpoints[int(sim.tick)] = tuple(self._digest(sim))

    def start(self, sim) -> None:
        if sim.tick != 0:
            raise ValueError(f"a recording starts at tick 0 (sim is at "
                             f"{sim.tick})")
        sim.attach_intent_log(IntentLog(self.recording))
        self._checkpoint(sim)

    def after_step(self, sim) -> None:
        if sim.tick % self.recording.checkpoint_every_ticks == 0:
            self._checkpoint(sim)

    def finish(self, sim, result=None) -> Recording:
        sim.attach_intent_log(None)
        if int(sim.tick) not in self.recording.checkpoints:
            self._checkpoint(sim)
        self.recording.end_tick = int(sim.tick)
        if result is not None:
            self.recording.result = {"outcome": result.outcome,
                                     "team": result.team,
                                     "ticks": result.ticks}
        return self.recording


# ---------------------------------------------------------------------------
# Format v1: writer + reader
# ---------------------------------------------------------------------------
def _encode_args(kind: str, args: tuple) -> list:
    out = []
    for typ, a in zip(INTENT_KINDS[kind], args, strict=True):
        if typ == "fhex":
            out.append(float(a).hex())
        elif typ == "ints":
            out.append([int(i) for i in a])
        else:
            out.append(a)
    return out


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _decode_args(kind: str, raw: list, where: str) -> tuple:
    types = INTENT_KINDS.get(kind)
    if types is None:
        raise ReplayRefused(f"{where}: unknown intent kind {kind!r}")
    if len(raw) != len(types):
        raise ReplayRefused(f"{where}: {kind} takes {len(types)} argument(s), "
                            f"got {len(raw)}")
    out = []
    for typ, v in zip(types, raw):
        ok = ((typ == "int" and _is_int(v))
              or (typ == "bool" and isinstance(v, bool))
              or (typ == "int?" and (v is None or _is_int(v)))
              or (typ == "ints" and isinstance(v, list)
                  and all(_is_int(i) for i in v))
              or (typ == "fhex" and isinstance(v, str)))
        if not ok:
            raise ReplayRefused(f"{where}: {kind} argument {v!r} is not "
                                f"a {typ}")
        if typ == "fhex":
            try:
                v = float.fromhex(v)
            except ValueError:
                raise ReplayRefused(f"{where}: {kind} float {v!r} is not "
                                    f"float.hex()") from None
        out.append(v)
    return tuple(out)


def _dumps(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def encode_recording(rec: Recording) -> bytes:
    """The recording as format-v1 bytes (gzip, mtime 0: deterministic)."""
    if rec.end_tick is None:
        raise ValueError("an unfinished recording (MatchRecorder.finish "
                         "was not called)")
    raw = io.BytesIO()
    raw.write(_dumps({
        "format": FORMAT_NAME, "version": FORMAT_VERSION,
        "level": rec.level, "seed": rec.seed,
        "checkpoint_every_ticks": rec.checkpoint_every_ticks,
        "build": rec.build, "spec": spec_to_dict(rec.spec)}))
    ticks = sorted(set(rec.intents) | set(rec.checkpoints))
    for t in ticks:
        if t in rec.intents:
            raw.write(_dumps({"t": t, "i": [[k, *_encode_args(k, a)]
                                            for k, a in rec.intents[t]]}))
        if t in rec.checkpoints:
            f, k = rec.checkpoints[t]
            raw.write(_dumps({"c": t, "f": f, "k": k}))
    raw.write(_dumps({"end": rec.end_tick, "result": rec.result}))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def write_recording(rec: Recording, path) -> int:
    """Write ``rec`` to ``path``; returns the bytes written."""
    data = encode_recording(rec)
    Path(path).write_bytes(data)
    return len(data)


def decode_recording(data: bytes, where: str = "recording") -> Recording:
    """Format-v1 bytes -> a :class:`Recording`. Refuses another format or
    version, a malformed row, an unknown intent kind, or a missing end."""
    try:
        text = gzip.decompress(data).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as e:
        raise ReplayRefused(f"{where}: not a gzip'd replay file ({e})") \
            from None
    lines = text.splitlines()
    if not lines:
        raise ReplayRefused(f"{where}: empty")
    try:
        rows = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as e:
        raise ReplayRefused(f"{where}: malformed JSON ({e})") from None
    head = rows[0]
    if not isinstance(head, dict) or head.get("format") != FORMAT_NAME:
        raise ReplayRefused(f"{where}: not a {FORMAT_NAME} file")
    if head.get("version") != FORMAT_VERSION:
        raise ReplayRefused(f"{where}: format version {head.get('version')!r}"
                            f", this build reads v{FORMAT_VERSION}")
    try:
        rec = Recording(level=head["level"], spec=spec_from_dict(head["spec"]),
                        seed=head["seed"],
                        checkpoint_every_ticks=head["checkpoint_every_ticks"],
                        build=head.get("build"))
    except (KeyError, TypeError) as e:
        raise ReplayRefused(f"{where}: malformed header ({e})") from None
    for n, row in enumerate(rows[1:], start=2):
        w = f"{where} line {n}"
        if rec.end_tick is not None:
            raise ReplayRefused(f"{w}: a row after the end row")
        if not isinstance(row, dict):
            raise ReplayRefused(f"{w}: not an object")
        if "t" in row:
            t = row["t"]
            if not _is_int(t) or t in rec.intents:
                raise ReplayRefused(f"{w}: bad or repeated tick {t!r}")
            items = []
            for item in row["i"]:
                if not isinstance(item, list) or not item:
                    raise ReplayRefused(f"{w}: bad intent {item!r}")
                items.append((item[0], _decode_args(item[0], item[1:], w)))
            rec.intents[t] = items
        elif "c" in row:
            t = row["c"]
            if not _is_int(t) or t in rec.checkpoints:
                raise ReplayRefused(f"{w}: bad or repeated checkpoint {t!r}")
            rec.checkpoints[t] = (row["f"], row["k"])
        elif "end" in row:
            rec.end_tick = row["end"]
            rec.result = row.get("result")
        else:
            raise ReplayRefused(f"{w}: unknown row {sorted(row)}")
    if rec.end_tick is None:
        raise ReplayRefused(f"{where}: no end row (truncated?)")
    if rec.end_tick not in rec.checkpoints:
        raise ReplayRefused(f"{where}: no checkpoint at the end tick "
                            f"{rec.end_tick}")
    return rec


def read_recording(path) -> Recording:
    return decode_recording(Path(path).read_bytes(), where=str(path))


def check_header(rec: Recording, level_name: str, *, seed=None,
                 spec=None) -> None:
    """Refuse (:class:`ReplayRefused`) replaying ``rec`` on another level,
    seed or match spec than it was recorded with."""
    if rec.level != level_name:
        raise ReplayRefused(f"the recording is of level {rec.level!r}, not "
                            f"{level_name!r}")
    if seed is not None and int(seed) != rec.seed:
        raise ReplayRefused(f"the recording was made with seed {rec.seed}, "
                            f"not {seed}")
    if spec is not None and spec != rec.spec:
        raise ReplayRefused(f"the recording's match spec {rec.spec} differs "
                            f"from {spec}")


# ---------------------------------------------------------------------------
# The replay controller
# ---------------------------------------------------------------------------
class ReplayControl(ControlSource):
    """Feeds a :class:`Recording` back into the sim, once per tick: verifies
    the checkpoint recorded at this tick (if any), then issues this tick's
    logged intents in their original order through the same entry points.
    It is the match's ONE controller (every team and the horde): the
    recording already holds all of their calls interleaved as issued.
    Calling it twice in one tick is a no-op. :meth:`finish` checks the
    match ended where the recording did. Works with ``renderer=None``."""

    def __init__(self, recording: Recording, team=None, *,
                 digest=match_digests):
        self.recording = recording
        self.team = team
        self._digest = digest
        self._last_tick = None
        self.checkpoints_verified = 0

    def _verify(self, sim, tick: int) -> None:
        want = self.recording.checkpoints.get(tick)
        if want is None:
            return
        got = tuple(self._digest(sim))
        if got != tuple(want):
            which = [n for n, a, b in zip(("field_digest", "tick_digest"),
                                          got, want) if a != b]
            raise ReplayDivergence(tick, f"{' and '.join(which)} differ from "
                                         f"the recording's checkpoint")
        self.checkpoints_verified += 1

    def handle_frame(self, sim, renderer=None) -> None:
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        end = self.recording.end_tick
        if tick > end:
            raise ReplayDivergence(tick, f"the recorded match ended at tick "
                                         f"{end}, this one goes on")
        self._verify(sim, tick)
        if tick == end:
            raise ReplayDivergence(tick, "the recorded match ended at this "
                                         "tick, this one goes on")
        for kind, args in self.recording.intents.get(tick, ()):
            getattr(sim, kind)(*args)

    def finish(self, sim) -> None:
        """After the match is terminal: it must end at the recording's last
        tick, whose checkpoint must match."""
        tick = int(sim.tick)
        if tick != self.recording.end_tick:
            raise ReplayDivergence(tick, f"the match ended at tick {tick}, "
                                         f"the recording at "
                                         f"{self.recording.end_tick}")
        self._verify(sim, tick)


__all__ = [
    "FORMAT_NAME", "FORMAT_VERSION", "INTENT_KINDS", "REFUSED_KINDS",
    "RecordingRefused", "ReplayRefused", "ReplayDivergence",
    "replay_settings", "match_digests", "level_id", "build_id",
    "Recording", "IntentLog",
    "MatchRecorder", "encode_recording", "decode_recording",
    "write_recording", "read_recording", "check_header", "ReplayControl",
    "spec_to_dict", "spec_from_dict",
]
