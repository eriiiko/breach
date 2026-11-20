"""agents/plot_curve.py -- the report's figures from a run's ``curve.jsonl``
(rung 2 note 2c, 5; design 7.1). Patch 2d's curves script extends this
one, never a second.

    python -m agents.plot_curve <run folder or curve.jsonl> [--out <folder>]
        [--smooth 5]

Writes five PNG figures beside the curve (or into ``--out``):

- ``curve_rooms.png``: the share of ended matches won per level, and the
  reward per ended match, over learning decisions;
- ``curve_hand.png``: each hand verb's share of the decisions that counted
  a hand, and the trigger's share pulled, over learning decisions;
- ``curve_groups.png``: the reward groups' sizes (mean absolute pay per
  decision) and the share of the return each value output explains;
- ``curve_seconds.png``: a round's seconds, split, and learning decisions
  a second;
- ``curve_live.png``: the live check's numbers every round -- the largest
  gap of a stored choice's probability and, when the soldier has a memory,
  the memory's (the second clause) -- against the tolerance.

PATCH 2d.2's additions: a dashed vertical mark on every figure at every
resume (a line carrying ``resumed_from``: the matches started again there);
the memory's live-check number (above); and a STOPPED line refuses the rest
of the curve -- the figures are drawn up to the stop, say so in their title,
and the script names the stop.

Every number is read from the line as the run wrote it; a running mean over
``--smooth`` rounds is drawn over the raw points where the raw points are
noisy (rates).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CURVE = "curve.jsonl"
#: The live check's tolerance as drawn (``Settings.live_tolerance``'s
#: default; the line does not carry it).
LIVE_TOLERANCE = 1e-4


def _finish(fig, x, lines, stopped) -> None:
    """Every figure: the resume marks on each axis, the stop in the title,
    then the layout."""
    for ax in fig.axes:
        _resumes(ax, x, lines)
    t = stop_text(stopped)
    if t:
        fig.suptitle(t, fontsize=8, color="C3")
    fig.tight_layout()


def read(path) -> tuple:
    """``(lines, stopped)``: the round lines of ``path`` (a run folder or a
    ``curve.jsonl``) up to the first stop line, and that line (None when
    the run did not stop). Nothing after a stop is read: the curve is
    refused past it."""
    p = Path(path)
    if p.is_dir():
        p = p / CURVE
    lines = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.get("stopped"):
            return lines, d
        lines.append(d)
    return lines, None


def stop_text(stopped) -> str | None:
    """The words a figure carries for a stopped curve (None: not stopped)."""
    if not stopped:
        return None
    return (f"STOPPED at round {stopped['round']}: drawn up to it -- "
            f"{str(stopped['stopped'])[:90]}")


def _resumes(ax, x, lines) -> None:
    """A dashed vertical mark at every resume."""
    for xi, d in zip(x, lines):
        if d.get("resumed_from") is not None:
            ax.axvline(xi, ls="--", lw=0.8, color="0.35", zorder=0)


def _smooth(y, k: int):
    y = np.asarray(y, dtype=float)
    if k <= 1 or len(y) < 2:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if w.size else np.nan
    return out


def _style(plt):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False})


def plot(lines, out: Path, smooth: int = 5, stopped=None) -> list:
    """Draw the five figures of ``lines`` into ``out`` (with ``stopped``,
    the stop line, named in every title); returns the paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x = np.array([d["decisions_total"] for d in lines], dtype=float) / 1e6
    xl = "learning decisions (millions)"
    paths = []

    # 1. rooms cleared per level; reward per ended match
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    levels = sorted({lv for d in lines for lv in d["per_level"]})
    for lv in levels:
        won = np.array([d["per_level"].get(lv, {}).get("won", 0)
                        for d in lines], float)
        ended = np.array([d["per_level"].get(lv, {}).get("ended", 0)
                          for d in lines], float)
        # pooled over the window: wins over ended matches
        k = max(1, smooth)
        rate = [won[max(0, i - k + 1):i + 1].sum()
                / max(1.0, ended[max(0, i - k + 1):i + 1].sum())
                for i in range(len(x))]
        a1.plot(x, rate, label=lv, lw=1.2)
    a1.set_ylabel(f"rooms cleared\n(share of ended matches,\n{smooth} rounds pooled)")
    a1.set_ylim(-0.02, 1.02)
    if levels:
        a1.legend(fontsize=7, ncol=2)
    r = np.array([np.nan if d["reward_per_match"] is None
                  else d["reward_per_match"] for d in lines])
    a2.plot(x, r, ".", ms=2, color="0.6", label="per round")
    a2.plot(x, _smooth(r, smooth), lw=1.4, color="C0",
            label=f"mean of {smooth} rounds")
    a2.set_ylabel("reward per ended match\n(weighted groups)")
    a2.set_xlabel(xl)
    a2.legend()
    _finish(fig, x, lines, stopped)
    paths.append(out / "curve_rooms.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # 2. hand verbs and the trigger
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    verbs = list(lines[0]["hand_verbs"]) if lines else []
    for v in verbs:
        share = np.array([d["hand_verbs"][v] / d["hand_counted"]
                          if d["hand_counted"] else np.nan for d in lines])
        if np.nanmax(np.nan_to_num(share)) > 0:
            a1.plot(x, _smooth(share, smooth), lw=1.1, label=v)
    a1.set_ylabel("share of decisions\nthat counted a hand")
    a1.set_yscale("symlog", linthresh=1e-3)
    a1.legend(fontsize=7, ncol=2)
    t = np.array([np.nan if d["trigger_share"] is None
                  else d["trigger_share"] for d in lines])
    a2.plot(x, _smooth(t, smooth), lw=1.4, label="trigger pulled")
    au = np.array([np.nan if d["aim_on_unit_share"] is None
                   else d["aim_on_unit_share"] for d in lines])
    a2.plot(x, _smooth(au, smooth), lw=1.4, label="aim on a unit")
    a2.set_ylabel("share of decisions\nwhere the head counted")
    a2.set_ylim(-0.02, 1.02)
    a2.set_xlabel(xl)
    a2.legend()
    _finish(fig, x, lines, stopped)
    paths.append(out / "curve_hand.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # 3. reward groups: sizes and explained share
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    groups = list(lines[0]["group_sizes"]) if lines else []
    for g in groups:
        a1.plot(x, [d["group_sizes"][g] for d in lines], lw=1.1, label=g)
        e = np.array([np.nan if d["explained"][g] is None
                      else d["explained"][g] for d in lines])
        if not np.all(np.isnan(e)):
            a2.plot(x, _smooth(e, smooth), lw=1.1, label=g)
    a1.set_ylabel("mean |pay| per decision")
    a1.set_yscale("symlog", linthresh=1e-4)
    a1.legend(fontsize=7, ncol=3)
    a2.set_ylabel("share of the return\nthe value output explains")
    a2.set_ylim(-1.0, 1.02)
    a2.set_xlabel(xl)
    if a2.lines:
        a2.legend(fontsize=7, ncol=3)
    _finish(fig, x, lines, stopped)
    paths.append(out / "curve_groups.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # 4. seconds
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    parts = ["env_step", "answers", "copy", "grading", "passes", "handover"]
    bottom = np.zeros(len(lines))
    for p in parts:
        y = np.array([d["seconds"][p] for d in lines])
        a1.fill_between(x, bottom, bottom + y, step="mid", alpha=0.8,
                        label=p.replace("_", " "))
        bottom += y
    a1.set_ylabel("seconds per round")
    a1.legend(fontsize=7, ncol=3)
    rate = np.array([d["decisions"] / d["seconds"]["round"] for d in lines])
    a2.plot(x, rate, lw=1.2)
    a2.set_ylabel("learning decisions\nper second")
    a2.set_xlabel(xl)
    _finish(fig, x, lines, stopped)
    paths.append(out / "curve_seconds.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # 5. the live check
    fig, a1 = plt.subplots(1, 1, figsize=(6.4, 3.2))
    for key, label in (("max_logp_gap", "a stored choice's log-probability"),
                       ("max_memory_gap", "the memory at a stretch's end")):
        y = np.array([np.nan if not d.get("live") or key not in d["live"]
                      else d["live"][key] for d in lines], float)
        if not np.all(np.isnan(y)):
            a1.plot(x, np.maximum(y, 1e-12), lw=1.1, label=label)
    a1.axhline(LIVE_TOLERANCE, color="C3", lw=0.8, ls=":",
               label="tolerance")
    a1.set_yscale("log")
    a1.set_ylabel("live check: largest\ngap recomputed")
    a1.set_xlabel(xl)
    a1.legend(fontsize=7)
    _finish(fig, x, lines, stopped)
    paths.append(out / "curve_live.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("curve", help="a run folder or its curve.jsonl")
    ap.add_argument("--out", default=None, help="where the figures go "
                    "(default: beside the curve)")
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args(argv)
    lines, stopped = read(args.curve)
    if stopped:
        print(f"refused past round {stopped['round']}: the curve holds a "
              f"stopped line ({stopped['stopped']}); drawn up to it")
    if not lines:
        print("no round lines")
        return 1
    p = Path(args.curve)
    out = Path(args.out) if args.out else (p if p.is_dir() else p.parent)
    for f in plot(lines, out, args.smooth, stopped):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
