"""agents/measure_card.py -- one measurement on the card (rung 2 note 2b,
10 gate 4): the FULL-WIDTH networks (memory off) learning a made-up task on
the stand-in for a few rounds of the design's size (64 matches x 128
decisions); prints the learning decisions a second for collecting and for
learning apart, and the card's peak memory. A number for the patch's
report, not a pass or a fail. The collecting time includes the stand-in's
own Python (it is no farm); the learning time is the learner's alone.

    python -m agents.measure_card                    # pointer, 3 rounds
    python -m agents.measure_card --task scale --rounds 4
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.settings import Settings
from agents.spec import v1
from agents.tasks import TASKS, make_env, networks
from agents.trainer import Trainer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="pointer", choices=sorted(TASKS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--decisions", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("no card visible: nothing measured")
        return 1
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda")
    task = TASKS[args.task]()
    env = make_env(task, args.envs, seed=0)
    soldier, value = networks(env, 0, spec=v1(memory=False))
    settings = Settings(planned_rounds=args.rounds, decisions=args.decisions)
    torch.cuda.reset_peak_memory_stats(dev)
    trainer = Trainer(env, soldier, value, settings, device=dev, seed=0)
    print(f"{torch.cuda.get_device_name(dev)}; task {task.name}; "
          f"{args.envs} matches x {args.decisions} decisions; batch "
          f"{trainer.batch.nbytes() / 2**20:.0f} MiB in the computer's "
          f"memory; piece {settings.piece}, lump {settings.lump}")
    rows = []
    for _ in range(args.rounds):
        s = trainer.round()
        c = s["decisions"] / s["collect_seconds"]
        lr = s["decisions"] / s["learn_seconds"]
        rows.append((c, lr))
        print(f"round {s['round']}: {s['decisions']} learning decisions; "
              f"collect {s['collect_seconds']:.1f} s ({c:,.0f} a second), "
              f"learn {s['learn_seconds']:.1f} s ({lr:,.0f} a second), "
              f"soldier steps {s['soldier_steps']}, value steps "
              f"{s['value_steps']}, live gap "
              f"{s['live']['max_logp_gap']:.1e}")
    peak = torch.cuda.max_memory_allocated(dev) / 2**20
    reserved = torch.cuda.max_memory_reserved(dev) / 2**20
    later = np.array(rows[1:] or rows)
    print(f"after the first round: collect {later[:, 0].mean():,.0f} and "
          f"learn {later[:, 1].mean():,.0f} learning decisions a second; "
          f"peak card memory {peak:,.0f} MiB allocated, {reserved:,.0f} MiB "
          f"reserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
