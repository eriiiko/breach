"""tools/bot_tournament.py -- THE measure of a bot change (#183 G8p).

``docs/bots_zombies_speeds_design_2026-11-10.md`` §2.5: one bot controller
against another (``--a`` / ``--b``, default the layered ``bots`` against
``bots_v1``) and each against the horde, over many seeds and levels, through
the one headless match loop (``run_match.run_match`` and its ``on_tick``
watcher -- never a second loop). Win rate and friendly fire over seeds,
never an opinion.

Pairings, per level and seed:

* ``duel``: A as team 0 against B as team 1, and MIRRORED (B team 0, A team
  1), the level's horde in both (match.toml's ``zombies``);
* ``horde:X`` (``--horde``, for X in A and B): X against an ``idle`` team,
  both sides, the horde in -- X must cross the horde to win, so its
  survivors measure how it fares against the horde.

Per pairing it reports: wins / losses / draws and the win rate of the
decided matches, mean survivors per side (living marines at the end), rounds
fired per kill (launched rounds over kills credited to the side, zombies
included), FRIENDLY-FIRE hits per match (a credited hit whose attacker and
victim are marines of one team, the attacker not the victim; self hits apart),
mean match length and the wall clock per match. ``--json PATH`` writes every
match's record. ``--workers N`` runs matches in N processes (each imports the
CPU build; a match is a pure function of level, spec and seed, so the
answers do not depend on N).

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bot_tournament.py \\
        --levels ship_demo --seeds 1-20 --horde --workers 12

THE DOOR BOTS (#195 1j, ``control_bots_door``): a match with a door team
always runs under a process-local ``RecordSink``, so its record gains
``rec["door"]`` (per door team: decisions, the translation's loss counts,
throws sent and started, ``masked_sent`` -- always 0, the controller raises
first); matches without one gain nothing. ``--records DIR`` (refused inside
the checkout) also writes, per door team, its imitation record
(``DIR/<level>/<c0>-<c1>__z<zombies>__s<seed>__t<team>.npz``) and, per
match, the replay recording beside it (``...__s<seed>.replay.gz``).

Death causes reuse ``bench_bot_matches.DeathLog`` (the bench's watcher); the
team bookkeeping here is :class:`TeamLog`. Measurement only: it reads the
sim, never writes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bench_bot_matches import DeathLog  # noqa: E402
from simulation.events import (  # noqa: E402
    NO_ATTACKER, ShotFiredEvent, ThrowEvent, UnitHitEvent, UnitKilledEvent)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402


class TeamLog:
    """``on_tick`` watcher: per marine team, the rounds launched, the kills
    credited (and of zombies), the friendly-fire and self hits. A unit's
    team is the one it ACTED in: its team at the tick before (a marine
    killed and converted this tick still shot as a marine)."""

    def __init__(self):
        self.shots = Counter()
        self.kills = Counter()
        self.zombie_kills = Counter()
        self.ff_hits = Counter()
        self.self_hits = Counter()
        self.hits = Counter()          # hits on another team's units
        self.throws = Counter()        # (team, hand-grenade kind) -> throws
        self._prev = None              # id -> (alive, is_zombie, team)

    def _acting(self, uid):
        p = self._prev.get(uid)
        if p is None or p[1]:
            return None                # unknown or a zombie
        return p[2]

    def __call__(self, sim):
        if self._prev is None:
            self._prev = {u.id: (True, u.is_zombie, u.team)
                          for u in sim.units}
        for ev in sim.tick_events:
            if isinstance(ev, ShotFiredEvent):
                if ev.launch:
                    t = self._acting(ev.unit_id)
                    if t is not None:
                        self.shots[t] += 1
            elif isinstance(ev, ThrowEvent):
                t = self._acting(ev.unit_id)
                if t is not None:
                    self.throws[(t, int(ev.kind))] += 1
            elif isinstance(ev, UnitHitEvent):
                a = ev.attacker_id
                if a == NO_ATTACKER:
                    continue
                ta = self._acting(a)
                if ta is None:
                    continue
                if a == ev.unit_id:
                    self.self_hits[ta] += 1
                    continue
                v = self._prev.get(ev.unit_id)
                if v is not None and not v[1] and v[2] == ta:
                    self.ff_hits[ta] += 1
                else:
                    self.hits[ta] += 1
            elif isinstance(ev, UnitKilledEvent):
                ta = self._acting(ev.attacker_id)
                v = self._prev.get(ev.unit_id)
                if ta is None or v is None:
                    continue
                if v[1]:
                    self.kills[ta] += 1
                    self.zombie_kills[ta] += 1
                elif v[2] != ta:
                    self.kills[ta] += 1
        self._prev = {u.id: (u.alive, u.is_zombie, u.team)
                      for u in sim.units}


# ---------------------------------------------------------------------------
# One match (a worker's job)
# ---------------------------------------------------------------------------
_LEVELS = {}


def _level(name, levels_dir):
    key = (name, levels_dir)
    if key not in _LEVELS:
        import level_loader
        _LEVELS[key] = level_loader.load(name, levels_dir)
    return _LEVELS[key]


def play(job) -> dict:
    """Run one match: ``job`` = (level, levels_dir, controllers (team 0,
    team 1), seed, time_limit_s or None, zombies or None, ``--set``
    overrides[, the sight rule or None[, the records directory or None]]).
    Returns its record."""
    name, levels_dir, ctrls, seed, limit, zombies, sets = job[:7]
    sight = job[7] if len(job) > 7 else None
    records = job[8] if len(job) > 8 else None
    apply_sets(sets)
    from run_match import run_match
    level = _level(name, levels_dir)
    spec = level.match
    spec = replace(spec, teams=tuple(replace(t, controller=c)
                                     for t, c in zip(spec.teams, ctrls)))
    if limit is not None:
        spec = replace(spec, time_limit_s=float(limit))
    if zombies is not None:
        spec = replace(spec, zombies=int(zombies))
    if sight is not None:
        spec = replace(spec, sight=str(sight))
    tlog, dlog = TeamLog(), DeathLog()

    def watch(sim):
        tlog(sim)
        dlog(sim)
        watch.sim = sim
    sink = None
    import control_bots_door
    if any(c in control_bots_door.DOOR_VARIANTS for c in ctrls):
        # #195 1j: a door team's stats (and, with --records, its records)
        sink = control_bots_door.RecordSink(records)
    t0 = time.perf_counter()
    control_bots_door.install_sink(sink)
    try:
        o = run_match(level, spec, seed=seed, on_tick=watch,
                      record=sink is not None and sink.records)
    finally:
        control_bots_door.install_sink(None)
    door = None
    if sink is not None:
        door = sink.close(watch.sim, o.result,
                          (o.field_digest, o.tick_digest), seed=seed,
                          recording=o.recording)
    wall = time.perf_counter() - t0
    sim = watch.sim
    survivors = Counter(u.team for u in sim.units
                        if u.alive and not u.is_zombie)
    rec = {"level": name, "controllers": list(ctrls), "seed": seed,
           "outcome": o.result.outcome, "team": o.result.team,
           "ticks": o.result.ticks, "wall": wall,
           "sight": spec.sight, "field_digest": o.field_digest,
           "tick_digest": o.tick_digest}
    for t in (0, 1):
        rec[f"t{t}"] = {"survivors": survivors[t], "shots": tlog.shots[t],
                        "kills": tlog.kills[t],
                        "zombie_kills": tlog.zombie_kills[t],
                        "ff_hits": tlog.ff_hits[t],
                        "self_hits": tlog.self_hits[t],
                        "hits": tlog.hits[t],
                        "throws": {str(k): n for (tt, k), n
                                   in sorted(tlog.throws.items()) if tt == t}}
    rec["marine_deaths"] = dict(dlog.marine)
    if door is not None:
        rec["zombies"] = int(spec.zombies)
        rec["door"] = door
    return rec


def door_summary(records) -> list:
    """Per (level, door controller, kind of match): the door stats summed
    over the matches (``rec["door"]``), with each loss count per slot
    decision."""
    from control_bots_door import LOSS_BITS
    groups = defaultdict(Counter)
    n = Counter()
    for r in records:
        for team, st in (r.get("door") or {}).items():
            kind = ("horde" if "idle" in r["controllers"]
                    else "nohorde" if r.get("zombies") == 0 else "duel")
            key = (r["level"], r["controllers"][int(team)], kind)
            n[key] += 1
            groups[key].update({k: v for k, v in st.items()
                                if isinstance(v, (int, float))
                                and k != "bytes"})
    rows = []
    for key in sorted(groups):
        g = groups[key]
        sd = max(g["slot_decisions"], 1)
        rows.append({"level": key[0], "door": key[1], "kind": key[2],
                     "matches": n[key], "totals": dict(g),
                     "per_decision": {b: g[b] / sd for b in LOSS_BITS}})
    return rows


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
def summarise(records, a, b) -> list:
    """Rows per (level, pairing): see the module docstring. ``side`` X's
    numbers are gathered from whichever team X played."""
    groups = defaultdict(list)
    for r in records:
        c0, c1 = r["controllers"]
        if {c0, c1} == {a, b} and a != b:
            kind = "duel"
        elif a == b and c0 == c1 == a:
            kind = "duel"
        elif "idle" in (c0, c1):
            kind = "horde:" + (c1 if c0 == "idle" else c0)
        else:
            kind = f"{c0}-{c1}"
        groups[(r["level"], kind)].append(r)
    rows = []
    for (level, kind), rs in sorted(groups.items()):
        sides = (a, b) if kind == "duel" else (kind.split(":", 1)[1],)
        row = {"level": level, "pairing": kind, "n": len(rs),
               "ticks": sum(r["ticks"] for r in rs) / len(rs),
               "wall": sum(r["wall"] for r in rs) / len(rs)}
        draws = sum(1 for r in rs if r["outcome"] != "win")
        row["draws"] = draws
        for x in sides:
            wins = surv = shots = kills = ff = selfh = 0
            throws = Counter()
            for r in rs:
                for t in (0, 1):
                    if r["controllers"][t] != x:
                        continue
                    if a == b and kind == "duel" and t == 1:
                        continue           # a mirror match: count team 0
                    d = r[f"t{t}"]
                    wins += r["outcome"] == "win" and r["team"] == t
                    surv += d["survivors"]
                    shots += d["shots"]
                    kills += d["kills"]
                    ff += d["ff_hits"]
                    selfh += d["self_hits"]
                    throws.update(d.get("throws", {}))
            row[x] = {"wins": wins, "survivors": surv / len(rs),
                      "rounds_per_kill": (shots / kills) if kills else None,
                      "ff_per_match": ff / len(rs),
                      "self_per_match": selfh / len(rs),
                      "throws_per_match": {k: n / len(rs) for k, n
                                           in sorted(throws.items())}}
        if kind == "duel" and a != b:
            decided = row[a]["wins"] + row[b]["wins"]
            row["a_rate"] = row[a]["wins"] / decided if decided else None
        rows.append(row)
    return rows


def format_rows(rows, a, b) -> str:
    out = []
    for row in rows:
        head = (f"{row['level']:38s} {row['pairing']:16s} n={row['n']:3d} "
                f"draws={row['draws']:3d} ticks={row['ticks']:7.0f} "
                f"wall={row['wall']:5.1f}s")
        if row.get("a_rate") is not None:
            head += f"  {a} won {row['a_rate'] * 100:5.1f}% of decided"
        out.append(head)
        for x in dict.fromkeys((a, b)):
            if x not in row:
                continue
            d = row[x]
            rpk = ("  -  " if d["rounds_per_kill"] is None
                   else f"{d['rounds_per_kill']:5.1f}")
            out.append(f"    {x:10s} wins {d['wins']:3d}  survivors "
                       f"{d['survivors']:4.2f}  rounds/kill {rpk}  FF hits/"
                       f"match {d['ff_per_match']:5.2f}  self "
                       f"{d['self_per_match']:4.2f}  throws/match "
                       + " ".join(f"{_kind_name(k)} {v:.2f}" for k, v
                                  in d["throws_per_match"].items()))
    return "\n".join(out)


def apply_sets(sets) -> None:
    """``--set`` overrides of the ``[bots]`` table (``key=value`` or
    ``features.spaced=1.5``), applied to this process's config before a
    match: a tuning experiment without editing config.toml (the shipped
    values stay the file's)."""
    from config import CFG
    for item in sets or ():
        key, val = item.split("=", 1)
        ns = CFG.bots
        *path, last = key.split(".")
        for p in path:
            ns = getattr(ns, p)
        old = getattr(ns, last)
        setattr(ns, last, type(old)(float(val)) if not isinstance(old, str)
                else val)


def _kind_name(k) -> str:
    """A hand-grenade kind index -> its row name without ``grenade_``."""
    from simulation.weapons import get_tables
    rows = get_tables().hand_grenade_rows
    i = int(k)
    return rows[i].replace("grenade_", "") if i < len(rows) else str(k)


def jobs_for(levels, levels_dir, seeds, a, b, horde, limit,
             zombies=None, sets=(), duel=True, sight=None,
             records=None) -> list:
    sets = tuple(sets)
    jobs = []
    for name in levels:
        for seed in seeds:
            if duel:
                jobs.append((name, levels_dir, (a, b), seed, limit, zombies,
                             sets))
            if duel and a != b:
                jobs.append((name, levels_dir, (b, a), seed, limit, zombies,
                             sets))
            if horde:
                for x in ((a,) if a == b else (a, b)):
                    jobs.append((name, levels_dir, (x, "idle"), seed, limit,
                                 zombies, sets))
                    jobs.append((name, levels_dir, ("idle", x), seed, limit,
                                 zombies, sets))
    if records is not None:
        return [j + (sight, str(records)) for j in jobs]
    return [j + (sight,) for j in jobs] if sight is not None else jobs


def _seeds(text: str) -> list:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bot_tournament")
    ap.add_argument("--levels", default="ship_demo",
                    help="comma-separated level folders (each a match.toml)")
    ap.add_argument("--levels-dir", default=str(ROOT / "levels"))
    ap.add_argument("--seeds", default="1-20", help="'1-20' or '1,4,7'")
    ap.add_argument("--a", default="bots")
    ap.add_argument("--b", default="bots_v1")
    ap.add_argument("--horde", action="store_true",
                    help="each controller against an idle team + the horde")
    ap.add_argument("--no-duel", action="store_true",
                    help="with --horde: the horde pairings only")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="replace the match's time limit (seconds)")
    ap.add_argument("--zombies", type=int, default=None,
                    help="replace the match's horde size (0 = no horde)")
    ap.add_argument("--set", action="append", default=[],
                    help="override a [bots] value for this run "
                         "(key=value, features.spaced=1.5); repeatable")
    ap.add_argument("--sight", default=None, choices=("cone", "all_round"),
                    help="replace the match's sight rule (#195 1d)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", default=None, help="write every match record")
    ap.add_argument("--records", default=None, metavar="DIR",
                    help="write the door teams' imitation records and the "
                         "matches' replays under DIR (outside the checkout; "
                         "#195 1j)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.records is not None:
        rd = Path(args.records).resolve()
        if rd == ROOT or ROOT in rd.parents:
            raise SystemExit(f"--records {args.records}: inside the checkout "
                             f"({ROOT}); records live outside the repo")
        rd.mkdir(parents=True, exist_ok=True)
    levels = [s for s in args.levels.split(",") if s]
    jobs = jobs_for(levels, args.levels_dir, _seeds(args.seeds), args.a,
                    args.b, args.horde, args.time_limit, args.zombies,
                    args.set, duel=not args.no_duel, sight=args.sight,
                    records=args.records)
    t0 = time.perf_counter()
    records = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap(play, jobs)):
                records.append(rec)
                print(f"[{i + 1}/{len(jobs)}] {rec['level']} "
                      f"{rec['controllers']} seed {rec['seed']}: "
                      f"{rec['outcome']} {rec['team']} at {rec['ticks']}",
                      flush=True)
    else:
        for i, job in enumerate(jobs):
            rec = play(job)
            records.append(rec)
            print(f"[{i + 1}/{len(jobs)}] {rec['level']} "
                  f"{rec['controllers']} seed {rec['seed']}: "
                  f"{rec['outcome']} {rec['team']} at {rec['ticks']}",
                  flush=True)
    rows = summarise(records, args.a, args.b)
    print(format_rows(rows, args.a, args.b))
    door_rows = door_summary(records)
    for d in door_rows:
        t = d["totals"]
        print(f"door {d['level']:38s} {d['door']:22s} {d['kind']:7s} "
              f"matches {d['matches']:3d} slot decisions "
              f"{t.get('slot_decisions', 0)} masked_sent "
              f"{t.get('masked_sent', 0)} throws sent "
              f"{t.get('throws_sent', 0)} started "
              f"{t.get('throws_started', 0)}")
        print("    per decision: " + " ".join(
            f"{k} {v:.3f}" for k, v in d["per_decision"].items()))
    print(f"{len(records)} matches in {time.perf_counter() - t0:.0f} s")
    if args.json:
        out = {"args": vars(args), "rows": rows, "records": records}
        if door_rows:
            out["door_rows"] = door_rows
        Path(args.json).write_text(json.dumps(out, indent=1),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
