"""agents/print_network.py -- the table of parts and weights of both
networks (rung 2 design 2; plan section 4 item 8), and, when run with a
card (``--card``), the peak card memory of one learning step at 512 and
2 048 decisions, both networks, 32-bit and 16-bit.

One learning step here is: both networks forward over the decisions (the
soldier as stretches of 16 with its memory), the probability of drawn
choices and the five values into one made-up loss, and its backward. No
optimiser and no grading (patch 2b's): the step is the memory's shape, not
its learning.

    python -m agents.print_network                  # the tables, processor
    python -m agents.print_network --card           # and the card's peaks
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from agents import choosing, fake, network
from agents import spec as net_spec
from agents.sizes import sizes_for

#: How the soldier's learning step cuts its decisions (design 4.5).
STRETCH = 16


def parts(model: torch.nn.Module) -> list:
    """``[(part, weights), ...]``: the body's parts one by one, then every
    other top-level part of ``model``; the rows add up to every weight."""
    rows = []
    for name, child in model.named_children():
        if name == "body":
            for sub, part in child.named_children():
                rows.append((sub, sum(p.numel() for p in part.parameters())))
        else:
            rows.append((name, sum(p.numel() for p in child.parameters())))
    for name, p in model.named_parameters(recurse=False):
        rows.append((name, p.numel()))
    return rows


def table(title: str, model: torch.nn.Module) -> str:
    rows = parts(model)
    total = sum(n for _, n in rows)
    width = max(len(r[0]) for r in rows + [("total", 0)])
    lines = [title] + [f"  {name:<{width}}  {n:>10,}" for name, n in rows]
    lines.append(f"  {'total':<{width}}  {total:>10,}")
    return "\n".join(lines)


def learning_step(soldier, value, sizes, decisions: int, device,
                  half: bool = False, seed: int = 0):
    """One learning step over ``decisions`` made-up decisions on
    ``device``; returns the card's peak bytes (None on the processor)."""
    rng = np.random.default_rng(seed)
    n = max(1, decisions // STRETCH)
    L = STRETCH if decisions >= STRETCH else decisions
    obs = fake.make_obs(sizes, L * n, rng)
    full = fake.make_full(sizes, L * n, rng)
    seq = {k: torch.as_tensor(v).to(device).reshape((L, n) + v.shape[1:])
           for k, v in obs.items()}
    flat = {k: torch.as_tensor(v).to(device) for k, v in obs.items()}
    fullt = {k: torch.as_tensor(v).to(device) for k, v in full.items()}
    soldier.to(device).train()
    value.to(device).train()
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type=torch.device(device).type,
                        dtype=torch.float16 if cuda else torch.bfloat16,
                        enabled=bool(half)):
        out, _ = soldier.forward_sequence(
            {k: seq[k] for k in network.SOLDIER_KEYS},
            soldier.initial_state(n, device))
        masks = choosing.masks_of({k: flat[k] for k in choosing.MASK_KEYS})
        with torch.no_grad():
            acts = choosing.draw(out, masks).actions
        d = choosing.evaluate(out, masks, acts)
        v = value({k: flat[k] for k in network.VALUE_OBS_KEYS}, fullt)
        loss = -d.logp.mean() - d.variety.mean() + v.float().pow(2).mean()
    loss.backward()
    soldier.zero_grad(set_to_none=True)
    value.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slots", type=int, default=6,
                    help="team size S (the weights do not depend on it)")
    ap.add_argument("--card", action="store_true",
                    help="also measure the card's peak memory")
    args = ap.parse_args(argv)
    sizes = sizes_for(args.slots)
    for label, net in (("full width", net_spec.v1()),
                       ("reduced width", net_spec.reduced())):
        print(table(f"soldier, network spec {net.name}, {label} "
                    f"(memory on; S = {args.slots})",
                    network.SoldierNet(net, sizes)))
        print(table(f"value network, network spec {net.name}, {label}",
                    network.ValueNet(net, sizes)))
    if args.card:
        if not torch.cuda.is_available():
            print("no card visible: no memory measured")
            return 1
        net = net_spec.v1()
        for decisions in (512, 2048):
            for half in (False, True):
                torch.manual_seed(0)
                peak = learning_step(network.SoldierNet(net, sizes),
                                     network.ValueNet(net, sizes), sizes,
                                     decisions, "cuda", half)
                print(f"card peak, one learning step, {decisions} decisions, "
                      f"{'16' if half else '32'}-bit: {peak / 2**20:,.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
