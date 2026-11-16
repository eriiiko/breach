"""agents/real_run.py -- the trainer's first real run, as a script (rung 2
note 2b, 10 gate 3): a few rounds on ``ship_demo`` through the game's own
``BatchedEnv`` (the farm, one worker), team 0 learning and team 1 on the
bots, the reduced-width networks, on the processor. It must collect, grade
and learn with the live check silent and no masked choice counted; it
prints what each round measured.

    python -m agents.real_run                 # 2 rounds, 2 matches, 16 decisions
    python -m agents.real_run --rounds 3 --envs 2 --decisions 32

Needs the processor build of the game's module (``cpp/build/Release``).
The run file is written to a temporary folder outside the checkout.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.settings import Settings
from agents.spec import reduced
from agents.trainer import Trainer, layout_sizes
from simulation.reward_v1 import REWARD_GROUPS

RUN_TEXT = """[run]
seed = {seed}
n_envs = {envs}
decimation = 4
levels = ["ship_demo"]
policy = ["policy", "bots"]
levels_dir = '{levels}'
[obs]
spec = "v1"
full_view = true
[action]
spec = "v1"
[reward]
profile = "fireteam"
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--decisions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    import rl_env
    from agents.network import SoldierNet, ValueNet
    with tempfile.TemporaryDirectory(prefix="breach_2b_") as tmp:
        run = Path(tmp) / "run.toml"
        run.write_text(RUN_TEXT.format(
            seed=args.seed, envs=args.envs,
            levels=(_game.ROOT / "levels").as_posix()), encoding="utf-8")
        t0 = time.perf_counter()
        env = rl_env.BatchedEnv(str(run), n_workers=1)
        try:
            lay = env.layout
            print(f"farm up in {time.perf_counter() - t0:.1f} s: N = "
                  f"{lay.n_envs}, T = {lay.n_teams} teams, S = {lay.n_slots},"
                  f" E = {lay.n_entries}, policy {env.run.policy}")
            sizes = layout_sizes(lay)
            torch.manual_seed(args.seed)
            soldier = SoldierNet(reduced(memory=False), sizes)
            value = ValueNet(reduced(memory=False), sizes)
            settings = Settings(planned_rounds=args.rounds,
                                decisions=args.decisions)
            trainer = Trainer(env, soldier, value, settings, seed=args.seed)
            b = trainer.batch
            print(f"batch: T = {b.T} decisions x R = {b.R} rows (learning "
                  f"teams {trainer.rows.teams}), {b.nbytes() / 2**20:.0f} "
                  f"MiB")
            for _ in range(args.rounds):
                s = trainer.round()
                paid = (b.reward * (b.value_row[..., None] != 0)).sum(
                    axis=(0, 1))
                print(f"round {s['round']}: {s['decisions']} learning "
                      f"decisions ({s['value_rows']} value rows), collect "
                      f"{s['collect_seconds']:.1f} s, learn "
                      f"{s['learn_seconds']:.1f} s")
                print(f"  live check: max log-probability gap "
                      f"{s['live']['max_logp_gap']:.2e}, distance "
                      f"{s['live']['distance']:.2e} (silent)")
                print(f"  distances per piece "
                      f"{[round(x, 5) for x in s['distances']]}, stopped "
                      f"{s['stopped']}, steps {s['soldier_steps']} / "
                      f"{s['value_steps']}")
                print(f"  policy loss {s['policy_loss']:.4f}, value loss "
                      f"{s['value_loss']:.4f}, variety {s['variety']:.3f}, "
                      f"clipped {s['clipped']:.3f}")
                print("  paid per group: " + ", ".join(
                    f"{g} {p:.3f}" for g, p in zip(REWARD_GROUPS, paid)))
                print(f"  matches ended in the round: "
                      f"{int(b.first[1:].any(axis=1).sum())} step(s) with a "
                      f"restart; scale spread "
                      f"{np.round(s['scale_spread'], 3).tolist()}")
            print(f"no masked choice counted on a driven row; done in "
                  f"{time.perf_counter() - t0:.1f} s")
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
