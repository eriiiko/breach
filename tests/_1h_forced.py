"""#195 1h's FORCED-MASKED-CHOICE gate: a masked choice issued by force does
nothing (``docs/rl_agent_rung1_note_1h_2026-11-12.md`` section 5).

THE PROPERTY: for every masked option of every head, on a living row,
sending it leaves the match's ``field_digest`` and ``tick_digest`` where the
row with that option's DECLARED FALLBACK leaves them -- hand "nothing" for a
masked hand verb or a masked argument, "where I walk" for a masked aim --
and anything sent on an invalid row (dead, turned, an empty roster place)
leaves them where the all-default row does.

THE DRIVER: two in-process ``EnvRunner``s (the CPU build, ``record=False``)
on the same run (``[obs] spec = "v1"``, ``[action] spec = "v1"``, every team
on ``policy``, all-round sight, decimation 4), the same level and seed. At
every decision a seeded sampler draws row A among A's LEGAL options of each
head (the realised ``arg_mask`` included), the hand biased toward the
non-"nothing" legal verbs so throws, plants, wind-ups, cool-downs and
reloads happen. Runner B gets the same rows except ONE living row, which
gets ONE masked option of one head -- chosen to cover the least-covered
(option, state) bucket so far: the hand table's masked entries, the aim's
masked places (an empty slot, a mate, an object, a door, "keep" outside a
wind-up) and the masked argument -- while A gets that option's fallback in
the same place; and every INVALID row gets a random in-range row in B, the
default row in A. After every TICK the two ``(field_digest, tick_digest)``
must be equal; a difference stops the run naming the decision, the tick, the
forced rows, heads, options and the units' gate inputs. Both runners stay
identical while the promise holds, so one pair tests every forced choice of
the run in sequence (episodes roll over: both reset to the same draw).

    <py> tests/_1h_forced.py [--seed N ...] [--decisions N] [--floor N]

FAILS (vacuous) unless every masked hand entry was forced at least
``floor`` times (20 by default) and every required state was hit at least
once: throw during a wind-up, of a kind at 0, while stunned; plant during a
plant, during a wind-up, while stunned; melee during a wind-up, during a
plant, on cool-down, while stunned; shove on cool-down, while stunned;
reload on a never-bound full magazine, on a full magazine, during a stall;
use away from every door; the reserved flashlight; a forced hand verb on a
dead row and on a turned row (whose ``pending_*`` latches must stay at
their defaults). ``main(...)`` returns the coverage table; the pytest
wrapper is ``tests/test_action_spec_v1.py::test_forced_masked_choices_do_
nothing``.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import breach_physics as bp  # noqa: E402
import rl_env  # noqa: E402
from simulation import action_spec_v1 as A  # noqa: E402
from simulation import combat, unit_impulse  # noqa: E402
from simulation.status import composed_flags  # noqa: E402
from simulation.unit_schema import MAG_UNBOUND  # noqa: E402

#: The (hand option, state) buckets the full run must hit at least once.
REQUIRED = (
    ("throw", "winding"), ("throw", "kind0"), ("throw", "stunned"),
    ("plant", "planting"), ("plant", "winding"), ("plant", "stunned"),
    ("melee", "winding"), ("melee", "planting"), ("melee", "cooldown"),
    ("melee", "stunned"), ("shove", "cooldown"), ("shove", "stunned"),
    ("reload", "never_bound"), ("reload", "full"), ("reload", "stall"),
    ("use", "no_door"), ("flashlight", "reserved"),
    ("invalid", "dead"), ("invalid", "turned"),
)
_PENDING = ("pending_throw", "pending_reload", "pending_shove",
            "pending_melee", "pending_plant", "pending_use")


class Forced(AssertionError):
    """The gate failed: a forced masked choice moved a digest."""


#: The default runs: ship_demo at two seeds (the note's gate), then a small
#: box whose horde stands next to team 0 (``test_rl_env.ROOMS``) -- random
#: play on ship_demo never brings a blow home nor a horde to a marine, so the
#: box supplies the melee cool-down and the TURNED row (a CHOICE beyond the
#: note: coverage, the same gate).
SCENARIOS = (("ship_demo", 1, 400), ("ship_demo", 2, 400), ("zbox", 1, 150))


def zbox(root: Path):
    """``test_rl_env``'s zombie box with a one-room graph (a level a "v1"
    run accepts), written under ``root``; returns its name."""
    from dataclasses import replace
    from test_obs_spec_v1 import _graph_box
    from test_rl_env import BOX_SPEC
    spec = replace(BOX_SPEC, time_limit_s=60.0)
    _graph_box(root, "zbox", spec)
    return "zbox"


def run_text(seed: int, level: str = "ship_demo", decimation: int = 4,
             levels_dir=None) -> str:
    ld = ("" if levels_dir is None
          else f"levels_dir = '{Path(levels_dir).as_posix()}'\n")
    return ("[run]\n"
            f"seed = {int(seed)}\nn_envs = 1\ndecimation = {decimation}\n"
            f"levels = [\"{level}\"]\npolicy = \"policy\"\n" + ld +
            "sight = \"all_round\"\n"
            "[obs]\nspec = \"v1\"\nfull_view = false\n"
            "[action]\nspec = \"v1\"\n[reward]\nkills = 1.0\n")


def _verb(h: int) -> str:
    if h >= A.HAND_THROW0:
        return "throw"
    return A.FIXED_HANDS[h]


def _states(sim, u, h: int) -> tuple:
    """The gate-input states of unit ``u`` that bucket a forced hand option
    ``h`` (the coverage table's columns)."""
    T = int(sim.tick)
    v = _verb(h)
    out = []
    if not composed_flags(u).can_act:
        out.append("stunned")
    if int(u.throw_release_tick) >= 0:
        out.append("winding")
    if int(u.plant_done_tick) >= 0:
        out.append("planting")
    if v == "throw" and u.grenade_counts[h - A.HAND_THROW0] <= 0:
        out.append("kind0")
    if v == "melee" and u.melee_id:
        w = sim.weapons_tables.weapons.by_name[u.melee_id]
        if u.last_blow_tick >= 0 and T - u.last_blow_tick < \
                w.rof_interval_ticks:
            out.append("cooldown")
    if v == "shove" and not unit_impulse.shove_cooled(sim.impulse_table, u,
                                                      T):
        out.append("cooldown")
    if v == "reload":
        w = sim.weapons_tables.weapons.by_name.get(u.weapon_id)
        if int(getattr(u, "current_mag", MAG_UNBOUND)) == MAG_UNBOUND:
            out.append("never_bound")
        elif w is not None and w.mag_size > 0:
            st = combat.mag_status(u, w, T)
            if st[2]:
                out.append("stall")
            elif st[0] >= w.mag_size:
                out.append("full")
    if v == "use":
        out.append("no_door")       # masked: the gate found no door in reach
    if v == "flashlight":
        out.append("reserved")
    return tuple(out) or ("other",)


def _gate_inputs(sim, u) -> dict:
    T = int(sim.tick)
    w = sim.weapons_tables.weapons.by_name.get(getattr(u, "weapon_id", ""))
    return {
        "uid": int(u.id), "alive": bool(u.alive), "zombie": bool(u.is_zombie),
        "can_act": bool(composed_flags(u).can_act),
        "throw_release_tick": int(u.throw_release_tick),
        "plant_done_tick": int(u.plant_done_tick),
        "mag_status": (combat.mag_status(u, w, T) if w is not None
                       and w.mag_size > 0 else None),
        "last_blow_tick": int(u.last_blow_tick),
        "last_shove_tick": int(getattr(u, "last_shove_tick", -1)),
        "grenades": tuple(int(c) for c in u.grenade_counts),
        "has_explosive": int(u.has_explosive),
    }


class _Pair:
    """Two lockstep runners and their observation buffers."""

    def __init__(self, run, levels, layout):
        self.layout = layout
        self.r = [rl_env.EnvRunner(0, run, layout, levels, breach_physics=bp)
                  for _ in range(2)]
        self.out = [{k: np.zeros(s[1:], d) for k, (s, d)
                     in layout.obs_shapes().items()} for _ in range(2)]

    def reset(self):
        for r, o in zip(self.r, self.out):
            r.reset()
            r.observe(o)

    def digests(self):
        return [r.session.digests() for r in self.r]


def _legal(rng, m):
    return int(rng.choice(np.flatnonzero(m)))


def _draw(rng, out, i, bias: float, near=None) -> np.ndarray:
    """Row ``i`` drawn among its legal options: the hand biased toward the
    legal verbs other than "nothing" (a verb CLASS uniformly, the throws one
    class), and -- when an enemy stands within reach (``near``, its aim
    index) and the blow is legal -- aimed at it with a melee blow, so blows
    land and their cadence runs."""
    row = np.zeros(5, np.int64)
    row[0] = _legal(rng, out["mask_feet"][i])
    row[1] = _legal(rng, out["mask_aim"][i])
    row[2] = _legal(rng, out["mask_trigger"][i])
    hm = out["mask_hand"][i].copy()
    if (near is not None and out["mask_aim"][i][near]
            and hm[A.HAND_MELEE] and rng.random() < 0.8):
        row[1], row[3] = near, A.HAND_MELEE
    else:
        fixed = [h for h in range(1, A.HAND_THROW0) if hm[h]]
        throws = [h for h in range(A.HAND_THROW0, len(hm)) if hm[h]]
        classes = fixed + (["throw"] if throws else [])
        if classes and rng.random() < bias:
            c = classes[int(rng.integers(len(classes)))]
            row[3] = int(rng.choice(throws)) if c == "throw" else c
        else:
            row[3] = _legal(rng, hm)
    row[4] = _legal(rng, A.arg_mask(row[3], row[1]))
    return row


def _near_enemy(sim, u, out, i, reach2: int = 16):
    """The aim index of the nearest enemy (any other team, the horde
    included) whose centre lies within ``sqrt(reach2)`` tiles of ``u``'s and
    which the list holds as an aimable entry, or ``None``."""
    ux, uy = u.center_tile_x(), u.center_tile_y()
    best = None
    for e in sim.units:
        if not e.alive or int(e.team) == int(u.team):
            continue
        d2 = (e.center_tile_x() - ux) ** 2 + (e.center_tile_y() - uy) ** 2
        if d2 <= reach2 and (best is None or d2 < best[0]):
            best = (d2, int(e.id))
    if best is None:
        return None
    return A.aim_index_for_unit(best[1], out["list_uid"][i],
                                out["list"][i, :, 0])


def _aim_options(out, i, S):
    """The masked aim places of row ``i``, each ``(label, index)``: an empty
    slot, a mate, an object, a door, "keep" outside a wind-up."""
    m = out["mask_aim"][i]
    if m[A.AIM_KEEP]:
        return []                   # a hold: the note forces none of these
    lm, lt = out["list_mask"][i], out["list"][i, :, 0].astype(int)
    opts = [("keep", A.AIM_KEEP)]
    E = len(lm)
    empty = [k for k in range(E) if not lm[k]]
    if empty:
        opts.append(("aim_empty", A.AIM_TOKENS + empty[0]))
    for t, label in ((0, "aim_mate"), (3, "aim_object"), (4, "aim_door")):
        ks = [k for k in range(E) if lm[k] and lt[k] == t]
        if ks:
            opts.append((label, A.AIM_TOKENS + ks[0]))
    return opts


def main(scenarios=SCENARIOS, floor: int = 20, bias: float = 0.7,
         verbose: bool = True, require: tuple = REQUIRED) -> dict:
    """Run the gate; returns ``{"forced": counter, "coverage": counter,
    "decisions": n}``; raises :class:`Forced` on a moved digest and
    ``AssertionError`` when the coverage is vacuous."""
    import tempfile
    forced = collections.Counter()      # per hand option name / aim / arg
    cover = collections.Counter()       # per (verb, state)
    n_dec = 0
    tmp = tempfile.TemporaryDirectory()
    for level, seed, decisions in scenarios:
        ld = None
        if level == "zbox":
            ld = Path(tmp.name) / f"lv{seed}"
            ld.mkdir()
            zbox(ld)
        p = Path(tmp.name) / "run.toml"
        p.write_text(run_text(seed, level, levels_dir=ld), encoding="utf-8")
        run = rl_env.load_run(p)
        levels = rl_env.load_levels(run)
        layout = rl_env.make_layout(run, levels)
        S = layout.n_slots
        names = A.hand_names(_hand_rows())
        pair = _Pair(run, levels, layout)
        pair.reset()
        rng = np.random.default_rng(seed)
        for k in range(decisions):
            oa, ob = pair.out
            ra = pair.r[0]
            sim_a = ra.session.sim
            T = int(sim_a.tick)
            units_now = ra.row_units()
            rows_a = np.array([
                _draw(rng, oa, i, bias,
                      None if units_now[i] is None
                      else _near_enemy(sim_a, units_now[i], oa, i))
                for i in range(layout.n_rows)])
            rows_b = rows_a.copy()
            notes = []
            units_a = ra.row_units()
            roster = ra.roster_units()
            # -- every invalid row: anything in B, the default in A --
            for i, u in enumerate(units_a):
                if u is not None:
                    continue
                rows_a[i] = (A.FEET_STAND, A.AIM_WALK, 0, A.HAND_NOTHING, 0)
                sizes = layout.head_sizes
                rows_b[i] = [int(rng.integers(0, n)) for n in sizes]
                if rows_b[i][3] == A.HAND_NOTHING:
                    rows_b[i][3] = int(rng.integers(1, sizes[3]))
                ru = roster[i]
                if ru is not None:
                    state = "dead" if not ru.alive else "turned"
                    cover[("invalid", state)] += 1
                    notes.append((i, "invalid", state, tuple(rows_b[i])))
            # -- one living row: one masked option, the least-covered --
            cands = []
            for i, u in enumerate(units_a):
                if u is None:
                    continue
                hm = oa["mask_hand"][i]
                for h in np.flatnonzero(hm == 0):
                    sts = _states(sim_a, u, int(h))
                    score = min([cover[(_verb(int(h)), s)] for s in sts]
                                + [forced[names[h]]])
                    if h == A.HAND_MELEE and _near_enemy(sim_a, u, oa, i)                             is not None:
                        # a masked blow with someone in reach: the case a
                        # wrong MOMENT would let land -- forced first
                        score = -1
                    cands.append((score, forced[names[h]], i, "hand", int(h),
                                  sts))
                for label, a in _aim_options(oa, i, S):
                    cands.append((forced[label], forced[label], i, "aim", a,
                                  (label,)))
                cands.append((forced["arg"], forced["arg"], i, "arg", 4,
                              ("arg",)))
            if cands:
                order = rng.permutation(len(cands))
                pick = min(order, key=lambda j: cands[j][:2])
                _, _, i, head, opt, sts = cands[pick]
                u = units_a[i]
                if head == "hand":
                    near = (_near_enemy(sim_a, u, oa, i)
                            if opt == A.HAND_MELEE else None)
                    if near is not None and oa["mask_aim"][i][near]:
                        rows_a[i, 1] = rows_b[i, 1] = near
                        cover[("melee", "in_reach")] += 1
                    rows_b[i, 3] = opt
                    rows_b[i, 4] = (int(rng.integers(0, A.N_FRACTIONS))
                                    if opt >= A.HAND_THROW0
                                    or opt == A.HAND_PLANT else 0)
                    rows_a[i, 3], rows_a[i, 4] = A.HAND_NOTHING, 0
                    forced[names[opt]] += 1
                    for s in sts:
                        cover[(_verb(opt), s)] += 1
                elif head == "aim":
                    rows_b[i, 1] = opt
                    rows_a[i, 1] = A.AIM_WALK
                    forced[sts[0]] += 1
                else:
                    # a masked argument: "onto target" with a token aim on a
                    # throw (a plant when no throw is legal)
                    hm = oa["mask_hand"][i]
                    throws = [h for h in range(A.HAND_THROW0, len(hm))
                              if hm[h]]
                    hand = (throws[0] if throws else A.HAND_PLANT
                            if hm[A.HAND_PLANT] else A.HAND_THROW0)
                    rows_b[i, 3], rows_b[i, 4] = hand, A.ARG_ONTO
                    if rows_b[i, 1] >= A.AIM_TOKENS:
                        rows_b[i, 1] = A.AIM_WALK
                        rows_a[i, 1] = A.AIM_WALK
                    rows_a[i, 3], rows_a[i, 4] = A.HAND_NOTHING, 0
                    forced["arg"] += 1
                notes.append((i, head, opt, sts, _gate_inputs(sim_a, u)))
            # -- the decision, tick by tick --
            for r, rows in zip(pair.r, (rows_a, rows_b)):
                lu = r._list_uid
                r._list_uid = None
                for team, pol in r.policies.items():
                    pol.set_decisions(rows[team * S:(team + 1) * S],
                                      lu[team * S:(team + 1) * S])
            ended = False
            for t in range(run.decimation):
                for r in pair.r:
                    r.session.tick()
                da, db = pair.digests()
                if da != db:
                    raise Forced(
                        f"seed {seed} decision {k} (start tick {T}), tick "
                        f"{int(sim_a.tick)}: digests differ {da} != {db}; "
                        f"forced: {notes}")
                if pair.r[0].session.is_terminal():
                    ended = True
                    break
            # the dead / turned rows' latches: never set, never hashed
            for i, ru in enumerate(pair.r[1].roster_units()):
                if ru is not None and (not ru.alive
                                       or int(ru.team) != i // S):
                    for name in _PENDING:
                        v = getattr(ru, name)
                        if v not in (None, False):
                            raise Forced(f"seed {seed} decision {k}: row {i} "
                                         f"(dead/turned) holds {name}={v!r}")
            n_dec += 1
            if ended:
                pair.reset()
            else:
                for r, o in zip(pair.r, pair.out):
                    r.observe(o)
    out = {"forced": forced, "coverage": cover, "decisions": n_dec}
    if verbose:
        print(f"decisions: {n_dec}")
        print("forced per option:")
        for name in sorted(forced):
            print(f"  {name:28s} {forced[name]}")
        print("coverage per (verb, state):")
        for key in sorted(cover):
            print(f"  {key[0]:12s} {key[1]:12s} {cover[key]}")
    hand_masked = [n for n in names if n != "nothing"]
    short = [n for n in hand_masked if forced[n] < floor]
    missing = [rq for rq in require if cover[rq] < 1]
    assert not short and not missing, (
        f"vacuous: hand entries forced < {floor} times: "
        f"{[(n, forced[n]) for n in short]}; states never hit: {missing}")
    return out


def _hand_rows() -> tuple:
    from simulation.weapons import get_tables
    return tuple(get_tables().hand_grenade_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--floor", type=int, default=20)
    ap.add_argument("--level", default="ship_demo")
    a = ap.parse_args()
    sc = SCENARIOS if a.seed is None else tuple(
        (a.level, s, a.decisions) for s in a.seed)
    main(sc, a.floor)
    print("EQUAL: every forced masked choice did nothing")
