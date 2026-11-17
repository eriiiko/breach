"""The run script and its per-round line (agents/run.py, the collector's
round log, the learner's measurements): rung 2 note 2c, 5 and 7. On the
stand-in, and one smoke on the real training rooms. Each test names the
property it protects and the change that must break it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from agents import run as R
from agents.collector import OUTCOMES, Collector, _outcome
from agents.settings import Settings
from agents.standin import StandIn
from agents.tasks import networks
from agents.tests._scripted import ScriptTask
from agents.trainer import Trainer
from simulation import action_spec_v1 as A
from simulation.reward_v1 import REWARD_GROUPS

ROOT = Path(__file__).resolve().parents[2]
N, T, MATCH = 2, 8, 3

#: Every field of the note's section 5 the line must carry.
LINE_FIELDS = (
    "round", "decisions", "decisions_total", "wall_seconds", "step_size",
    "matches_ended", "outcomes", "per_level", "reward_per_match",
    "reward_per_match_groups", "group_sizes", "hand_counted", "hand_verbs",
    "trigger_counted", "trigger_share", "aim_counted", "aim_on_unit_share",
    "hits_dealt", "hits_per_match", "deaths_by_cause", "explained",
    "variety_heads", "distances", "distance_pass", "stopped_early",
    "clipped", "policy_loss", "value_loss", "scale_spread", "live",
    "masked_choices", "seconds", "card", "host", "stopped")


def _trainer(rounds=4):
    task = ScriptTask(MATCH)
    env = StandIn(task, N, seed=5)
    task.bind(env.sizes)
    soldier, value = networks(env, 0)
    return Trainer(env, soldier, value,
                   Settings(planned_rounds=rounds, decisions=T, piece=64,
                            lump=32), seed=1)


def test_a_rounds_line_has_every_field_and_its_parts_add_up():
    """Protects: the line of a round has every field of note 2c, 5; it is
    JSON; the seconds' parts (environment step, answers, copy, grading,
    passes, hand-over) add up to the round's wall within a tolerance; the
    hand verbs' counts add up to the decisions that counted a hand, and
    those are living stored decisions; the trigger's share is a share.
    Breaks when a part of the round goes unmeasured (or is counted twice),
    a verb is counted on a row where the hand did not count, or a field is
    dropped."""
    tr = _trainer()
    total = 0
    for _ in range(2):
        stats = tr.round()
        total += stats["decisions"]
        line = R.round_line(tr, stats, wall=1.0, decisions_total=total)
        json.loads(json.dumps(line))
        for k in LINE_FIELDS:
            assert k in line, k
        sec = line["seconds"]
        parts = sum(sec[k] for k in R.SECOND_PARTS)
        assert abs(parts - sec["round"]) <= 0.05 * sec["round"] + 0.02, sec
        b = tr.batch
        assert sum(line["hand_verbs"].values()) == line["hand_counted"]
        assert line["hand_counted"] == int(((b.valid != 0)
                                            & (b.used[..., 3] != 0)).sum())
        assert line["hand_counted"] <= line["decisions"]
        assert 0.0 <= line["trigger_share"] <= 1.0
        assert set(line["variety_heads"]) == set(A.HEADS)
        assert line["masked_choices"] == 0
        assert line["live"]["max_logp_gap"] <= 1e-4
        assert set(line["group_sizes"]) == set(REWARD_GROUPS)
        assert line["matches_ended"] == sum(line["outcomes"].values())


def test_the_ended_matches_carry_all_the_pay_and_nothing_twice():
    """Protects: every reward the batch stored for a learning row is
    either in exactly one ended match's reward (the team's rows summed over
    the whole match, rounds before included) or still open in a match that
    has not ended; each ended match is counted once per env and team, on
    the level it was played on. Breaks when the per-match sums are not
    reset at a match's end, are keyed by the wrong rows, or the level is
    read after the reset (the next match's)."""
    tr = _trainer()
    stored = np.zeros(len(REWARD_GROUPS))
    in_ended = np.zeros(len(REWARD_GROUPS))
    n_ended = 0
    for _ in range(3):
        tr.round()
        b = tr.batch
        stored += b.reward.sum(axis=(0, 1))
        for m in tr.collector.log["ended"]:
            in_ended += m.reward
            assert m.level == 0 and m.team in tr.rows.teams
            assert m.outcome in OUTCOMES
        n_ended += len(tr.collector.log["ended"])
    still = tr.collector._paid.sum(axis=0)
    assert np.allclose(stored, in_ended + still, atol=1e-6)
    # every MATCH decisions each env ends once; two learning teams each
    assert n_ended == N * len(tr.rows.teams) * ((3 * T) // MATCH)


def test_the_outcome_is_read_off_the_books_counters():
    """Protects: how a match ended for a team is read off its rows'
    counters at the last step, "won" before "clock" before "eliminated",
    "other" when none was counted. Breaks when the precedence or a counter
    name changes."""
    names = ("kills", "won", "eliminated", "clock")
    assert _outcome(np.array([0, 1, 0, 0]), names) == "won"
    assert _outcome(np.array([3, 0, 0, 1]), names) == "clock"
    assert _outcome(np.array([0, 0, 1, 0]), names) == "eliminated"
    assert _outcome(np.array([2, 0, 0, 0]), names) == "other"


def _files(tmp_path, *, rounds=2, extra="", levels_dir=None, n_envs=4):
    run = tmp_path / "run.toml"
    if levels_dir is not None:
        lv = ", ".join(f'"{n}"' for n in _rooms().level_names())
        run.write_text(
            f"[run]\nseed = 1\nn_envs = {n_envs}\ndecimation = 4\n"
            f"levels = [{lv}]\npolicy = [\"policy\", \"idle\"]\n"
            f"levels_dir = '{Path(levels_dir).as_posix()}'\n"
            "[obs]\nspec = \"v1\"\nfull_view = true\n[action]\nspec = "
            "\"v1\"\n[reward]\nprofile = \"range\"\n", encoding="utf-8")
    else:
        run.write_text("# the stand-in's run\n", encoding="utf-8")
    train = tmp_path / "train.toml"
    train.write_text(
        "[network]\nspec = \"v1\"\nwidth = \"reduced\"\nmemory = false\n"
        f"[train]\nplanned_rounds = {rounds}\ndecisions = {T}\n"
        f"piece = 64\nlump = 32\n{extra}", encoding="utf-8")
    return run, train


def _standin_env(monkeypatch):
    """``rl_env.BatchedEnv`` replaced by the stand-in for one test."""
    import rl_env

    def make(run, n_workers=None):
        task = ScriptTask(MATCH)
        env = StandIn(task, N, seed=5)
        task.bind(env.sizes)
        return env
    monkeypatch.setattr(rl_env, "BatchedEnv", make)


def test_the_run_stops_at_the_planned_rounds_and_writes_its_files(
        tmp_path, monkeypatch):
    """Protects: ``agents.run`` runs exactly the planned rounds, appends one
    line per round, copies both files it read into the output folder and
    writes both networks' weights under the not-a-saved-run names; it
    refuses an output folder inside the checkout and one that already holds
    a curve. Breaks when the loop runs past the plan, a line is lost, or a
    resume is silently allowed."""
    _standin_env(monkeypatch)
    run, train = _files(tmp_path, rounds=3)
    out = tmp_path / "out"
    s = R.run(run, train, out, device="cpu")
    assert s["rounds"] == 3
    lines = (out / R.CURVE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["round"] for x in lines] == [1, 2, 3]
    assert (out / R.RUN_COPY).read_bytes() == run.read_bytes()
    assert (out / R.TRAIN_COPY).read_bytes() == train.read_bytes()
    assert (out / R.SOLDIER_WEIGHTS).is_file()
    assert (out / R.VALUE_WEIGHTS).is_file()
    with pytest.raises(ValueError, match="resuming"):
        R.run(run, train, out, device="cpu")
    with pytest.raises(ValueError, match="inside the checkout"):
        R.run(run, train, ROOT / "never_here", device="cpu")
    assert not (ROOT / "never_here").exists()


def test_the_wall_stop_ends_the_run_before_the_plan(tmp_path, monkeypatch):
    """Protects: ``[stop] wall_hours`` ends the run at the wall even when
    the planned rounds are far off. Breaks when the wall stop is not
    read."""
    _standin_env(monkeypatch)
    run, train = _files(tmp_path, rounds=1000,
                        extra="[stop]\nwall_hours = 0.0003\n")
    s = R.run(run, train, tmp_path / "out", device="cpu")
    assert 1 <= s["rounds"] < 1000


def test_a_masked_choice_writes_the_line_and_stops(tmp_path, monkeypatch):
    """Protects: a masked choice counted on a driven living row stops the
    run with the round's line written, naming the stop and the count.
    Breaks when the stop is swallowed or leaves no line."""
    from agents.collector import MaskedChoiceSent
    _standin_env(monkeypatch)
    real = Collector._act

    def bad(self, obs):
        d = real(self, obs)
        d["actions"][:, 3] = A.HAND_FLASHLIGHT     # reserved: always masked
        return d
    monkeypatch.setattr(Collector, "_act", bad)
    run, train = _files(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(MaskedChoiceSent):
        R.run(run, train, out, device="cpu")
    last = json.loads((out / R.CURVE).read_text(
        encoding="utf-8").splitlines()[-1])
    assert "MaskedChoiceSent" in last["stopped"]
    assert last["masked_choices"] > 0


def _rooms():
    p = str(ROOT / "tools")
    if p not in sys.path:
        sys.path.insert(0, p)
    import gen_training_rooms
    return gen_training_rooms


def test_a_smoke_on_the_real_rooms(tmp_path):
    """Protects: ``agents.run`` on the real training rooms through the
    game's farm (2 rounds, 4 matches, 2 workers, the reduced width, on the
    processor) writes a line per round with the live check silent, no
    masked choice, and decisions on the learning rows only. Breaks when the
    run and the farm disagree on a name, a shape or a stop."""
    if not (ROOT / "cpp" / "build" / "Release").is_dir():
        pytest.skip("needs the processor build of the game's module")
    rooms = tmp_path / "rooms"
    _rooms().write_all(rooms)
    run, train = _files(tmp_path, levels_dir=rooms)
    out = tmp_path / "out"
    s = R.run(run, train, out, workers=2, device="cpu")
    lines = [json.loads(x) for x in
             (out / R.CURVE).read_text(encoding="utf-8").splitlines()]
    assert s["rounds"] == 2 and len(lines) == 2
    for line in lines:
        assert line["stopped"] is None
        assert line["masked_choices"] == 0
        assert line["live"]["max_logp_gap"] <= 1e-4
        assert abs(line["live"]["distance"]) <= 1e-4
        assert 0 < line["decisions"] <= 4 * T
        assert line["hand_counted"] > 0
