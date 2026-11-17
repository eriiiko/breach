"""agents/run.py -- a real training run (rung 2 note 2c, 5; design 7.1):
the game's farm (``rl_env.BatchedEnv``) under a run file, the trainer's
numbers and network from a training file (:mod:`agents.training_file`),
rounds of the existing :class:`agents.trainer.Trainer` until the PLANNED
rounds or the training file's wall stop, whichever first, and ONE line of
JSON per round appended to ``curve.jsonl`` in the output folder.

    python -m agents.run --run <run.toml> --train <train.toml> --out <folder>
        [--workers 16] [--device cuda] [--threads 4] [--card-fraction 0.9]

The script joins, it does not re-implement: the round is
``Trainer.round``; every number of the line is read from the trainer's own
arrays (the batch, the collector's round log over the environment's
``info``, the learner's measurements), never from a second bookkeeping
(:func:`round_line` says where each comes from). ``agents/real_run.py``
stays the small smoke.

THE OUTPUT FOLDER (outside the checkout; refused inside it and refused when
it already holds a ``curve.jsonl``: resuming is patch 2d's): ``run.toml`` and
``train.toml`` (copies of the two files read), ``curve.jsonl``, and at the
end both networks' weights under names that say they are NOT the saved-run
format (patch 2d's): ``soldier_weights_NOT_A_SAVED_RUN.pt``,
``value_weights_NOT_A_SAVED_RUN.pt`` (with the value targets' scale).

A STOP IS A LINE. A masked choice sent on a driven living row, or a live
check that fails, writes the round's line with ``stopped`` naming it (and
``masked_choices`` the count) and ends the run with an error.

THE CARD'S SPILL. On Windows a card that runs out may spill into the
computer's memory and run several times slower instead of failing (design
4.4). The driver setting that turns this off ("CUDA sysmem fallback
policy") is per program and cannot be read or set from here, so on the card
this script caps the process with ``torch.cuda.set_per_process_memory_
fraction``: at most ``--card-fraction`` of the card and never more than 97 %
of what is FREE at the start (the desktop holds the rest), so running out
is an error, never a silent spill. Measured on the Home Desktop
(2026-11-17, RTX 3070, driver 610.88): without a cap the training Python
allocated 9 216 MiB on the 8 191 MiB card without an error -- the spill is
ON for it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.collector import OUTCOMES, MaskedChoiceSent
from agents.learner import LiveCheckFailed
from agents.training_file import load as load_training_file
from simulation import action_spec_v1 as A
from simulation.reward_v1 import REWARD_GROUPS

CURVE = "curve.jsonl"
RUN_COPY = "run.toml"
TRAIN_COPY = "train.toml"
SOLDIER_WEIGHTS = "soldier_weights_NOT_A_SAVED_RUN.pt"
VALUE_WEIGHTS = "value_weights_NOT_A_SAVED_RUN.pt"
#: The seconds of a round, in the order the line lists them; together they
#: are the round (the test holds the sum to ``round`` within a tolerance).
SECOND_PARTS = ("env_step", "answers", "copy", "grading", "passes",
                "handover")


def _hand_names(env) -> tuple:
    from simulation.weapons import get_tables
    names = A.hand_names(get_tables().hand_grenade_rows)
    if len(names) != env.layout.head_sizes[3]:
        raise RuntimeError(f"hand names {names} against a hand head of "
                           f"{env.layout.head_sizes[3]}")
    return names


def _f(x):
    """A JSON-safe float (None stays None)."""
    return None if x is None else float(x)


def round_line(trainer, stats: dict, *, wall: float, decisions_total: int,
               card: dict | None = None, host: dict | None = None) -> dict:
    """The per-round line (note 2c, 5) from the trainer's own arrays:

    - ``batch`` (the round's stored decisions): the reward groups' sizes,
      the hand verbs' counts, the trigger's share pulled, the aim's share
      on a unit;
    - the collector's round log (``info`` per step: ``ended``, ``level``,
      ``counters``, ``masked_choices``) and its seconds;
    - the learner's measurements (``stats``) and the trainer's seconds.

    NOT in the line, because nothing the trainer holds reads it honestly:
    the rounds FIRED (the env counts hits, not shots: it would need a
    counter in ``rl_env``), the geometric "aim on the target" (only the
    CHOICE of aiming at a unit entry is stored: ``aim_on_unit_share``, which
    in a one-target room is the choice of aiming at the target), and hits
    per match (the round's hits include matches still running).
    """
    env, batch = trainer.env, trainer.batch
    col, settings = trainer.collector, trainer.settings
    log = col.log
    levels = list(env.run.levels)
    names = tuple(getattr(env.layout, "counters", ()))
    valid = batch.valid != 0
    value_rows = batch.value_row != 0
    # the matches that ended this round, per level
    per_level = {}
    outcomes = {o: 0 for o in OUTCOMES}
    w = np.asarray(settings.weights, dtype=np.float64)
    pays, pays_g = [], []
    for m in log["ended"]:
        lv = levels[m.level] if 0 <= m.level < len(levels) else str(m.level)
        d = per_level.setdefault(lv, {"ended": 0, **{o: 0 for o in OUTCOMES}})
        d["ended"] += 1
        d[m.outcome] += 1
        outcomes[m.outcome] += 1
        pays.append(float((m.reward * w).sum()))
        pays_g.append(m.reward)
    # the reward groups' sizes: mean |pay| per decision on the value rows
    if value_rows.any():
        sizes = np.abs(batch.reward[value_rows]).mean(axis=0)
    else:
        sizes = np.zeros(len(REWARD_GROUPS))
    # the hand verbs and the trigger, on the decisions where they counted
    hand_used = valid & (batch.used[..., 3] != 0)
    hand_names = _hand_names(env)
    counts = np.bincount(batch.actions[..., 3][hand_used].astype(np.int64),
                         minlength=len(hand_names))
    trig_used = valid & (batch.used[..., 2] != 0)
    n_trig = int(trig_used.sum())
    aim_used = valid & (batch.used[..., 1] != 0)
    n_aim = int(aim_used.sum())
    on_unit = int((batch.actions[..., 1][aim_used] >= A.AIM_TOKENS).sum())
    cnt = dict(zip(names, (float(x) for x in log["counters"])))
    deaths = {k[len("deaths_"):]: v for k, v in cnt.items()
              if k.startswith("deaths_") and v}
    sec = dict(col.seconds)
    sec.update(stats["seconds"])
    sec["handover"] = stats["handover_seconds"]
    sec["round"] = stats["round_seconds"]
    live = stats.get("live")
    n_ended = len(log["ended"])
    return {
        "round": int(stats["round"]),
        "decisions": int(stats["decisions"]),
        "decisions_total": int(decisions_total),
        "wall_seconds": float(wall),
        "step_size": float(stats["step_size"]),
        "matches_ended": n_ended,
        "outcomes": outcomes,
        "per_level": per_level,
        "reward_per_match": (float(np.mean(pays)) if pays else None),
        "reward_per_match_groups": (
            dict(zip(REWARD_GROUPS, np.mean(pays_g, axis=0).tolist()))
            if pays_g else None),
        "group_sizes": dict(zip(REWARD_GROUPS, sizes.tolist())),
        "hand_counted": int(hand_used.sum()),
        "hand_verbs": dict(zip(hand_names, counts.tolist())),
        "trigger_counted": n_trig,
        "trigger_share": (float((batch.actions[..., 2][trig_used] == 1)
                                .mean()) if n_trig else None),
        "aim_counted": n_aim,
        "aim_on_unit_share": (on_unit / n_aim if n_aim else None),
        "hits_dealt": cnt.get("hits_dealt", 0.0),
        "damage_dealt": cnt.get("damage_dealt", 0.0),
        "kills": cnt.get("kills", 0.0),
        "hits_taken": cnt.get("hits_taken", 0.0),
        "deaths_by_cause": deaths,
        "explained": dict(zip(REWARD_GROUPS,
                              [_f(x) for x in stats["explained"]])),
        "variety_heads": dict(zip(A.HEADS, [_f(x)
                                            for x in stats["variety_heads"]])),
        "variety": float(stats["variety"]),
        "distances": [float(x) for x in stats["distances"]],
        "distance_pass": [int(x) for x in stats["distance_pass"]],
        "stopped_early": (list(stats["stopped"]) if stats["stopped"]
                          else None),
        "soldier_steps": int(stats["soldier_steps"]),
        "value_steps": int(stats["value_steps"]),
        "clipped": float(stats["clipped"]),
        "policy_loss": float(stats["policy_loss"]),
        "value_loss": float(stats["value_loss"]),
        "scale_spread": dict(zip(REWARD_GROUPS,
                                 np.asarray(stats["scale_spread"]).tolist())),
        "live": ({"max_logp_gap": float(live["max_logp_gap"]),
                  "distance": float(live["distance"])} if live else None),
        "masked_choices": int(log["masked"]),
        "seconds": {k: float(sec[k]) for k in SECOND_PARTS + ("round",)},
        "card": card,
        "host": host,
        "stopped": None,
    }


def _card(device) -> dict | None:
    if torch.device(device).type != "cuda":
        return None
    return {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def _host() -> dict | None:
    try:
        import psutil
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    return {"used_gib": (vm.total - vm.available) / 2**30,
            "total_gib": vm.total / 2**30}


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def _write(out: Path, line: dict) -> None:
    with open(out / CURVE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=False) + "\n")


def run(run_path, train_path, out, *, workers=None, device="cpu",
        threads=4, card_fraction=0.9, print_every=1, max_rounds=None) -> dict:
    """The run (module text). Returns a summary: rounds, decisions, wall
    seconds, the output folder. ``max_rounds`` (tests only) stops earlier
    than the plan without changing it."""
    import rl_env
    from agents.network import SoldierNet, ValueNet
    from agents.trainer import Trainer, layout_sizes
    out = Path(out)
    if _inside(out, _game.ROOT):
        raise ValueError(f"--out {out} is inside the checkout: a run's output "
                         f"lives outside it")
    if (out / CURVE).exists():
        raise ValueError(f"{out / CURVE} exists: resuming a run is patch "
                         f"2d's; choose a new folder")
    tf = load_training_file(train_path)
    s = tf.settings
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_path, out / RUN_COPY)
    shutil.copyfile(train_path, out / TRAIN_COPY)
    torch.set_num_threads(int(threads))
    dev = torch.device(device)
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        free, total_mem = torch.cuda.mem_get_info(idx)
        cap = min(float(card_fraction), 0.97 * free / total_mem)
        torch.cuda.set_per_process_memory_fraction(cap, idx)
        print(f"card capped at {cap:.3f} of {total_mem / 2**20:.0f} MiB "
              f"({cap * total_mem / 2**20:.0f} MiB; {free / 2**20:.0f} MiB "
              f"free at the start)", flush=True)
    t0 = time.perf_counter()
    env = rl_env.BatchedEnv(str(run_path), n_workers=workers)
    try:
        seed = int(env.run.seed)
        sizes = layout_sizes(env.layout)
        torch.manual_seed(seed)
        soldier = SoldierNet(tf.net, sizes)
        value = ValueNet(tf.net, sizes)
        trainer = Trainer(env, soldier, value, s, device=dev, seed=seed)
        print(f"farm up in {time.perf_counter() - t0:.1f} s: N = "
              f"{env.n_envs} on {getattr(env, 'n_workers', 1)} workers, levels "
              f"{list(env.run.levels)}, batch {trainer.batch.T} x "
              f"{trainer.batch.R} ({trainer.batch.nbytes() / 2**20:.0f} MiB),"
              f" device {dev}, planned rounds {s.planned_rounds}, wall stop "
              f"{s.wall_hours} h", flush=True)
        t_run = time.perf_counter()
        total = 0
        limit = s.planned_rounds if max_rounds is None else min(
            s.planned_rounds, int(max_rounds))
        wall_limit = (None if s.wall_hours is None
                      else float(s.wall_hours) * 3600.0)
        while trainer.rounds < limit:
            if wall_limit is not None and time.perf_counter() - t_run \
                    >= wall_limit:
                break
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                stats = trainer.round()
            except (MaskedChoiceSent, LiveCheckFailed) as e:
                _write(out, {"round": trainer.rounds + 1,
                             "decisions_total": total,
                             "wall_seconds": time.perf_counter() - t_run,
                             "masked_choices": int(getattr(e, "count", 0)
                                                   or trainer.collector
                                                   .log["masked"]),
                             "stopped": f"{type(e).__name__}: {e}"})
                raise
            total += int(stats["decisions"])
            line = round_line(trainer, stats,
                              wall=time.perf_counter() - t_run,
                              decisions_total=total, card=_card(dev),
                              host=_host())
            _write(out, line)
            if print_every and line["round"] % print_every == 0:
                sec = line["seconds"]
                print(f"round {line['round']}: {line['decisions']} dec, "
                      f"{line['decisions'] / sec['round']:.0f} dec/s, ended "
                      f"{line['matches_ended']} {line['outcomes']}, reward/"
                      f"match {line['reward_per_match']}, trigger "
                      f"{line['trigger_share']}, live gap "
                      f"{(line['live'] or {}).get('max_logp_gap')}",
                      flush=True)
        wall = time.perf_counter() - t_run
        torch.save(trainer.learner.soldier.state_dict(), out / SOLDIER_WEIGHTS)
        sc = trainer.learner.scale
        torch.save({"state_dict": trainer.learner.value.state_dict(),
                    "scale_mean": sc.mean.cpu(), "scale_spread": sc.spread.cpu(),
                    "scale_rounds": sc.rounds}, out / VALUE_WEIGHTS)
        summary = {"rounds": trainer.rounds, "decisions": total,
                   "wall_seconds": wall, "out": str(out)}
        print(f"done: {summary}", flush=True)
        return summary
    finally:
        env.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="the run file (rl_env)")
    ap.add_argument("--train", required=True, help="the training file")
    ap.add_argument("--out", required=True,
                    help="the run's output folder, outside the checkout")
    ap.add_argument("--workers", type=int, default=None,
                    help="farm workers (default: rl_env's, threads - 1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--threads", type=int, default=4,
                    help="PyTorch's processor threads in this process")
    ap.add_argument("--card-fraction", type=float, default=0.9,
                    help="the share of the card this process may take "
                         "(running out is an error, never a spill)")
    args = ap.parse_args(argv)
    run(args.run, args.train, args.out, workers=args.workers,
        device=args.device, threads=args.threads,
        card_fraction=args.card_fraction)
    return 0


if __name__ == "__main__":
    sys.exit(main())
