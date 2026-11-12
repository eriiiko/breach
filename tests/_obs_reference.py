"""tests/_obs_reference.py -- observation spec v1 written a SECOND time
(#195 1g, ``docs/rl_agent_obs_spec_v1_2026-11-12.md`` section 7, the
reconstruction gate).

A plainer implementation of the spec's sections 1 and 2 from THE DOCUMENT,
reading the sim's state directly (loops over units, records and tiles; the
sight oracle per line; the bots' own ``_teammate_on_line``; the horde's
``flood_reference``), never importing ``simulation.obs_spec_v1``. The test
holds the live builders equal to it column for column. A CORRECTNESS gate:
a leak written into the document would pass both; honesty is the
counterfactual family's.
"""
from __future__ import annotations

import math

import numpy as np

from config import CFG
from control_bots import BotTeam
from simulation import (atmosphere_fixed, combat, fire_fixed, gas_fixed,
                        horde, optics_fixed, room_graph, vision)
from simulation import team_picture as tp
from simulation import unit_impulse as ui
from simulation.damage import KINETIC, POISON, mitigation_for
from simulation.entities.door import DOOR_CLOSED, DOOR_OPEN
from simulation.events import CAUSE_NAMES
from simulation.gases import INERT_N2, O2
from simulation.gases import POISON as G_POISON
from simulation.gases import SMOKE as G_SMOKE
from simulation.gases import TEARGAS as G_TEARGAS
from simulation.exposure import get_settings as exposure_settings
from simulation.loadouts import get_tables as loadout_tables
from simulation.movement import THROTTLE_ONE
from simulation.ruleset import MatchRules
from simulation.status import (BLINDED, IMMOBILIZED, KNOCKED_DOWN,
                               N_STATUS_KINDS, PARALYZED, STUNNED,
                               composed_flags)
from simulation.turning import PI_Q16, law_q16, turn_rate_q16, wrap_q16
from simulation.unit_fixed import (FP_ONE, atan2_q16, cos_q16,
                                   quantize_scalar, sin_q16)
from simulation.unit_impulse import pending_dw
from simulation.unit_orders import (NO_LIMIT, NO_PLACE, NO_TARGET,
                                    ORDER_OPENING_BASE, graph_of)
from simulation.unit_schema import THROW_NO_REACH, table_row
from simulation.weapons import FRAG_ROW, PLANT_WEAPON, fuse_ticks_from_seconds

V, F = 97, 67
OBJ_BASE, DOOR_BASE = 1 << 20, 1 << 22


def squash(x):
    x = float(x)
    if abs(x) <= 1.0:
        return x
    v = min(4.0, 1.0 + math.log(abs(x)))
    return v if x > 0 else -v


def over(x, d):
    return 0.0 if float(d) == 0.0 else float(x) / float(d)


def dc(d, L):
    return squash(math.log(1.0 + max(0.0, d)) / math.log(1.0 + L))


def constants(sim, spec, n_slots):
    """Every table value the spec names (0.1, 1.7), read here on its own."""
    tables = sim.weapons_tables
    lt = loadout_tables()
    ref_lo = lt.loadouts.by_name[str(CFG.marine.loadout)]
    ref = tables.weapons.by_name[ref_lo.weapon]
    hand = tables.weapons.by_name["hand_grenade"]
    c4 = tables.weapons.by_name[PLANT_WEAPON]
    tps = int(CFG.clock.ticks_per_second)
    it = sim.impulse_table
    scale = [1] * N_STATUS_KINDS
    scale[KNOCKED_DOWN] = it.getup_ticks
    scale[BLINDED] = max([int(CFG.exchange.teargas_blind_ticks)]
                         + [p.blind_ticks for p in tables.payloads.by_name.values()])
    for kind in (STUNNED, IMMOBILIZED, PARALYZED):
        t = [w.status_ticks for w in tables.weapons.by_name.values()
             if w.status_kind_id == kind]
        scale[kind] = max(t) if t else 1
    w = turn_rate_q16(CFG)
    return dict(
        S=int(n_slots), L=int(ref.range_tiles),
        v_ref=(FP_ONE * FP_ONE) // sim.speed_table.full_cost_q,
        ref_rof=ref.rof_interval_ticks, ref_spread=ref.spread_deg,
        melee_arc=tables.weapons.by_name[ref_lo.melee].melee_arc_half_q,
        limit=MatchRules.from_spec(spec).limit_ticks(sim),
        memory=tp.memory_ticks(tp.picture_settings(CFG), float(tps)),
        tps=tps, hand=tuple(tables.hand_grenade_rows),
        objs=tuple(tables.hand_grenade_rows) + tuple(tables.charge_rows),
        frag_r=tables.payload_for_ammo(FRAG_ROW).radius,
        throw_r=hand.max_throw_range, windup=hand.throw_windup_ticks,
        plant=c4.plant_ticks,
        c4_fuse=fuse_ticks_from_seconds(c4.fuse_max_seconds, tps),
        hand_fuse=fuse_ticks_from_seconds(hand.fuse_max_seconds, tps),
        scale=[max(1, s) for s in scale], half_turn=-(-PI_Q16 // w),
        clear=int(CFG.bots.line_clear_tiles),
        znear=int(CFG.bots.zombie_near_tiles),
        x_max=exposure_settings().x_max_q, v_fall=it.v_fall_q[0],
        react=it.react_ticks, shove_cd=it.shove_cooldown_ticks,
        zombie_class=len(lt.loadouts.by_name), spec=spec,
    )


# -- helpers over the sim ---------------------------------------------------
def mates_of(sim, team):
    return [u for u in sim.units if u.alive and not u.is_zombie and u.team == team]


def centre(u):
    return int(u.center_tile_x()), int(u.center_tile_y())


def loadout(u):
    return loadout_tables().loadouts.by_name.get(u.loadout_name) \
        if u.loadout_name else None


def weapon(sim, u):
    return sim.weapons_tables.weapons.by_name.get(u.weapon_id or "")


def row(table, name):
    try:
        return table_row(table, name) if name else -1
    except ValueError:
        return -1


def cone_of(sim, K, u):
    if u.is_zombie:
        return K["melee_arc"]
    w = weapon(sim, u)
    if w is not None and w.range_tiles > 0:
        return quantize_scalar(math.radians(w.spread_deg))
    m = sim.weapons_tables.weapons.by_name.get(u.melee_id or "")
    return m.melee_arc_half_q if m is not None else K["melee_arc"]


def aim_of(sim, K, team, u, me_id):
    """Spec 1.7 by hand: the nearest known unit (a mate live, a record at
    its recorded centre) in u's cone with a clear ray; (category, uid)."""
    recs = sim.team_picture.records(team)
    mates = mates_of(sim, team)
    is_mate = any(m is u for m in mates)
    ray = u.facing_target_q if is_mate else u.facing_q
    cone = cone_of(sim, K, u)
    ux, uy = centre(u)
    cands = [(centre(m), m.id) for m in mates if m.id != u.id]
    cands += [(r.centre, uid) for uid, r in recs.items() if uid != u.id]
    inside = []
    for (cx, cy), uid in cands:
        dx, dy = cx - ux, cy - uy
        if abs(wrap_q16(atan2_q16((-dy) << 16, dx << 16) - ray)) <= cone:
            inside.append((dx * dx + dy * dy, uid, cx, cy))
    for _, uid, cx, cy in sorted(inside):
        if vision.ray_clear(sim, uy, ux, cy, cx):
            if uid == me_id:
                return 1, uid
            return (2 if any(m.id == uid for m in mates) else 3), uid
    return 0, -1


def place_centre(graph, where):
    if graph is None or where == NO_PLACE:
        return None
    if where < ORDER_OPENING_BASE:
        for r in graph.rooms:
            if r.id == where:
                return (r.x + r.w // 2, r.y + r.h // 2)
        return None
    for o in graph.openings:
        if ORDER_OPENING_BASE + o.id == where:
            return (o.x + o.w // 2, o.y + o.h // 2)
    return None


def route_field(sim, graph, where, fp):
    """Spec 1.7's route by hand: ``flood_reference`` over the walkable
    tiles plus the openable openings, seeded where the block centre's
    ``place_at`` is the place."""
    gmap = sim.gmap
    ok = horde.walkable_tiles(gmap).copy()
    states = room_graph.opening_states(graph, gmap, sim._doors, sim.level.wires)
    for s in states:
        if s.openable:
            o = next(o for o in graph.openings if o.id == s.id)
            ok[o.y:o.y + o.h, o.x:o.x + o.w] = True
    passable = horde.block_passable(ok, fp)
    h, w = ok.shape
    seeds = []
    for y in range(h):
        for x in range(w):
            if passable[y, x] and room_graph.place_at(
                    graph, y + fp // 2, x + fp // 2) == where:
                seeds.append((y, x))
    return horde.flood_reference(ok, fp, seeds)


def status_left(u, k):
    return max([s.remaining_ticks for s in u.statuses if s.kind == k] + [0])


# -- section 1 -----------------------------------------------------------------
def vector(sim, K, u):
    team = int(u.team)
    pic = sim.team_picture
    tick = int(sim.tick)
    v = np.zeros(V, np.float32)
    L = K["L"]
    w = weapon(sim, u)
    lo = loadout(u)
    melee = sim.weapons_tables.weapons.by_name.get(u.melee_id or "")
    v[0] = squash(over(u.current_hp, pic.bars[u.id]))
    v[1], v[2] = cos_q16(u.facing_q) / FP_ONE, sin_q16(u.facing_q) / FP_ONE
    v[3] = squash(over(u.omega_q, sim.unit_turn_rate_q16(u)))
    v[4] = wrap_q16(u.facing_target_q - u.facing_q) / PI_Q16
    m = pic.motion.get(u.id)
    vel = (m[1][0] - m[0][0], m[1][1] - m[0][1]) if m else (0, 0)
    v[5], v[6] = squash(over(vel[0], K["v_ref"])), squash(over(vel[1], K["v_ref"]))
    v[7], v[8] = squash(over(u.pvx, K["v_fall"])), squash(over(u.pvy, K["v_fall"]))
    v[9] = squash(over(u.effort, FP_ONE))
    v[10] = over(u.react_ticks, K["react"])
    fl = composed_flags(u)
    v[11:15] = [fl.is_prone, fl.can_move, fl.can_act, fl.can_aim]
    for k in range(N_STATUS_KINDS):
        v[15 + k] = squash(over(status_left(u, k), K["scale"][k]))
    st = combat.mag_status(u, w, tick) if w is not None else None
    if st is None:
        v[24:27] = 1.0
    else:
        rounds, spares, reloading = st
        v[24] = over(rounds, w.mag_size)
        if spares is None:
            v[25] = v[26] = 1.0
        else:
            v[25] = squash(over(spares, max(1, lo.magazines if lo else 0)))
        v[27] = reloading
        if reloading:
            v[28] = over(u.reload_done_tick - tick, w.reload_ticks)
    for k, name in enumerate(K["hand"]):
        n = lo.grenades.get(name, 0) if lo else 0
        if n > 0:
            v[29 + 2 * k] = over(u.grenade_counts[k], n)
            v[30 + 2 * k] = 1.0
    if u.throw_release_tick >= 0:
        v[41] = 1.0
        v[42] = over(u.throw_release_tick - tick, K["windup"])
        v[43] = u.throw_kind
    else:
        v[43] = -1.0
    v[44] = over(u.has_explosive, lo.explosives if lo else 0)
    v[45] = bool(lo and lo.charge)
    if u.plant_done_tick >= 0:
        v[46] = 1.0
        v[47] = over(u.plant_done_tick - tick, K["plant"])
        v[48] = over(u.plant_fuse_ticks, K["c4_fuse"])
    if melee is not None and u.last_blow_tick >= 0:
        r = melee.rof_interval_ticks
        v[49] = over(max(0, u.last_blow_tick + r - tick), r)
    v[50] = bool(u.melee_id)
    if u.last_shove_tick >= 0:
        cd = K["shove_cd"]
        v[51] = over(max(0, u.last_shove_tick + cd - tick), cd)
    v[52] = row("loadouts", u.loadout_name)
    v[53] = row("weapons", u.weapon_id)
    if w is not None:
        v[54] = squash(over(w.range_tiles, L))
        v[55] = squash(over(w.rof_interval_ticks, K["ref_rof"]))
        v[56] = squash(over(w.spread_deg, K["ref_spread"]))
    v[57] = bool(u.live_trigger)
    cx0, cy0 = centre(u)
    v[58:66] = [squash(x) for x in air_at(sim, u, cy0, cx0)]
    v[96] = squash(over(felt_temperature(sim, u), u.environment.temperature_max))
    hit = pic.last_hit.get(u.id)
    if hit is None:
        v[67], v[70] = 1.0, -1.0
    else:
        age = tick - hit[0]
        v[66] = age <= K["memory"]
        v[67] = min(1.0, over(age, K["memory"]))
        n = math.hypot(hit[1], hit[2])
        if n:
            v[68], v[69] = -hit[1] / n, hit[2] / n
        v[70] = CAUSE_NAMES.index(hit[3])
    v[71:77] = -1.0
    v[77] = over(K["limit"] - tick, K["limit"])
    spec = K["spec"]
    n_t = spec.teams[team].n
    mates = mates_of(sim, team)
    v[78] = over(len(mates), n_t)
    tlo = loadout_tables().loadouts.by_name[spec.teams[team].loadout]
    tw = sim.weapons_tables.weapons.by_name[tlo.weapon]
    ammo = 0
    for mm in mates:
        mw = weapon(sim, mm)
        mst = combat.mag_status(mm, mw, tick) if mw else None
        if mst:
            r_, s_, _ = mst
            if s_ is None:
                ml = loadout(mm)
                s_ = ml.magazines if ml else 0
            ammo += r_ + s_ * mw.mag_size
    v[79] = squash(over(ammo, n_t * tlo.magazines * tw.mag_size))
    recs = pic.records(team)
    seen = set(vision.state_for(sim).visible_by_team.get(team, ()))
    others = sum(t.n for i, t in enumerate(spec.teams) if i != team)
    marines = [r for r in recs.values() if not r.zombie]
    zombies = [r for r in recs.values() if r.zombie]
    v[80] = over(len(marines), others)
    v[81] = over(sum(1 for r in marines if r.uid in seen), others)
    v[82] = squash(over(len(zombies), spec.zombies))
    near = 0
    for z in zombies:
        if any((z.centre[0] - centre(mm)[0]) ** 2
               + (z.centre[1] - centre(mm)[1]) ** 2 <= K["znear"] ** 2
               for mm in mates):
            near += 1
    v[83] = squash(over(near, spec.zombies))
    v[84] = u.order_what
    v[94] = -1.0
    if u.order_what:
        v[85] = u.order_until != NO_LIMIT
        if u.order_until != NO_LIMIT:
            v[86] = over(u.order_until - tick, K["limit"])
        graph = graph_of(sim)
        cx, cy = centre(u)
        if graph is not None:
            v[87] = room_graph.place_at(graph, cy, cx) == u.order_where
            v[88] = 1 if u.order_where < ORDER_OPENING_BASE else 2
            pc = place_centre(graph, u.order_where)
            if pc:
                v[89] = squash(over(pc[0] - cx, L))
                v[90] = squash(over(pc[1] - cy, L))
                v[91] = dc(math.hypot(pc[0] - cx, pc[1] - cy), L)
            field = route_field(sim, graph, u.order_where, u.footprint)
            ay, ax = u.tile_y, u.tile_x
            c = horde.candidates(field, ay, ax)
            if c:
                v[92], v[93] = c[0][3] - ax, c[0][2] - ay
            here = horde.field_at(field, ay, ax)
            if here != horde.UNREACHED:
                v[94] = squash(over(here, L))
        v[95] = u.order_target != NO_TARGET
    return v


# -- section 2 -----------------------------------------------------------------
def unit_cols(e, it, sim, K, tick):
    e[11] = over(it.exposure[KINETIC], K["x_max"])
    e[12] = bool(it.live_trigger)
    w = weapon(sim, it)
    st = combat.mag_status(it, w, tick) if w else None
    if st and st[2]:
        e[13] = 1.0
        e[14] = over(it.reload_done_tick - tick, w.reload_ticks)
    if it.throw_release_tick >= 0:
        e[15] = 1.0
        e[16] = over(it.throw_release_tick - tick, K["windup"])
        e[17] = it.throw_kind
    else:
        e[17] = -1.0
    if it.plant_done_tick >= 0:
        e[18] = 1.0
        e[19] = over(it.plant_done_tick - tick, K["plant"])
        e[20] = over(it.plant_fuse_ticks, K["c4_fuse"])
    fl = composed_flags(it)
    e[21], e[22], e[23] = not fl.can_act, not fl.can_aim, fl.is_prone


def entry_list(sim, K, u, bots=None):
    """``(entries (E, F), mask (E,), uid (E,))`` for the living mate ``u``;
    ``bots`` a ``BotTeam`` of its team (its ``_teammate_on_line``)."""
    team = int(u.team)
    pic = sim.team_picture
    tick = int(sim.tick)
    L = K["L"]
    S = K["S"]
    caps = (S - 1, 16, 8, 6, 4)
    E = S + 33
    out = np.zeros((E, F), np.float32)
    mask = np.zeros(E, np.uint8)
    uid = np.full(E, -1, np.int32)
    me = int(u.id)
    cx, cy = centre(u)
    mates = mates_of(sim, team)
    others = [m for m in mates if m.id != me]
    recs = pic.records(team)
    vis = vision.state_for(sim)
    seen = set(vis.visible_by_team.get(team, ()))
    by_id = {x.id: x for x in sim.units}
    graph = graph_of(sim)
    if bots is None:
        bots = BotTeam(team)
    w = weapon(sim, u)
    spread_q = quantize_scalar(math.radians(w.spread_deg)) if w else 0
    my_where = u.order_where if u.order_what else NO_PLACE
    n_t = K["spec"].teams[team].n
    a, wq = sim.unit_turn_accel_q16(u), sim.unit_turn_rate_q16(u)
    dw = pending_dw(sim.impulse_table, u)

    def common(e, t, ex, ey):
        e[0] = t
        e[1], e[2] = squash(over(ex - cx, L)), squash(over(ey - cy, L))
        e[3] = dc(math.hypot(ex - cx, ey - cy), L)

    def d2(p):
        return (p[0] - cx) ** 2 + (p[1] - cy) ** 2

    i = 0
    for m in sorted(others, key=lambda m: (d2(centre(m)), m.id))[:caps[0]]:
        e = out[i]
        mx, my = centre(m)
        common(e, 0, mx, my)
        e[4], e[5] = cos_q16(m.facing_q) / FP_ONE, sin_q16(m.facing_q) / FP_ONE
        mo = pic.motion.get(m.id)
        vel = (mo[1][0] - mo[0][0], mo[1][1] - mo[0][1]) if mo else (0, 0)
        e[6], e[7] = squash(over(vel[0], K["v_ref"])), squash(over(vel[1], K["v_ref"]))
        e[8], e[9] = m.team, row("loadouts", m.loadout_name)
        e[10] = squash(over(m.current_hp, pic.bars[m.id]))
        unit_cols(e, m, sim, K, tick)
        ft = m.facing_target_q
        e[39], e[40] = cos_q16(ft) / FP_ONE, sin_q16(ft) / FP_ONE
        e[41] = aim_of(sim, K, team, m, me)[0] != 0
        if m.live_move_dir is not None:
            e[42] = m.live_move_dir.dx_q / FP_ONE
            e[43] = m.live_move_dir.dy_q / FP_ONE
            e[44] = over(m.live_move_dir.throttle_q, THROTTLE_ONE)
        if m.throw_release_tick >= 0:
            reach = 6.0 if m.throw_reach_q == THROW_NO_REACH else m.throw_reach_q / FP_ONE
            lx = mx + reach * cos_q16(ft) / FP_ONE - cx
            ly = my - reach * sin_q16(ft) / FP_ONE - cy
            e[45], e[46] = squash(over(lx, L)), squash(over(ly, L))
        e[47] = m.order_what
        if m.order_what:
            pc = place_centre(graph, m.order_where)
            if pc:
                e[48], e[49] = squash(over(pc[0] - cx, L)), squash(over(pc[1] - cy, L))
            e[50] = my_where != NO_PLACE and m.order_where == my_where
            if m.order_until != NO_LIMIT:
                e[51] = over(m.order_until - tick, K["limit"])
        # the lane pair (spec 2.2 cols 52, 53): each owner's own lane
        cell, hw = int(CFG.bots.lane_cell), int(CFG.bots.lane_halfwidth)
        its = tp.fire_lanes_of(sim.gmap, sim.weapons_tables, [m], team, cell, hw)
        mine = tp.fire_lanes_of(sim.gmap, sim.weapons_tables, [u], team, cell, hw)
        e[52] = its[cy // cell, cx // cell] == m.id
        e[53] = mine[my // cell, mx // cell] == me
        mask[i], uid[i] = 1, m.id
        i += 1
    i = caps[0]
    seen_recs = sorted([r for r in recs.values() if r.uid in seen],
                       key=lambda r: (d2(r.centre), r.uid))
    for r in seen_recs[:caps[1]]:
        it = by_id[r.uid]
        e = out[i]
        ex, ey = r.centre
        common(e, 1, ex, ey)
        e[4], e[5] = cos_q16(it.facing_q) / FP_ONE, sin_q16(it.facing_q) / FP_ONE
        e[6], e[7] = squash(over(r.vel_q[0], K["v_ref"])), squash(over(r.vel_q[1], K["v_ref"]))
        e[8] = r.team
        e[9] = K["zombie_class"] if r.zombie else row("loadouts", r.loadout)
        e[10] = squash(over(it.current_hp, pic.bars[r.uid]))
        unit_cols(e, it, sim, K, tick)
        e[24] = me in vis.observers_of(r.uid)
        solid = not sim.gmap.has_los(cy, cx, ey, ex, physical=True) or any(
            c.alive and c.blocks_segment(cx, cy, ex, ey)
            for c in (sim.cover or ()))
        e[25] = solid
        e[26] = bots._teammate_on_line(u, it, mates)
        dx, dy = ex - cx, ey - cy
        want = atan2_q16((-dy) << 16, dx << 16)
        facing, omega, n = u.facing_q, u.omega_q + dw, 0
        cap = 4 * K["half_turn"]
        while n < cap and abs(wrap_q16(want - facing)) > spread_q:
            facing, omega = law_q16(facing, omega, want, a, wq)
            n += 1
        e[27] = squash(over(n, K["half_turn"]))
        dd = dx * dx + dy * dy
        e[29] = w is not None and dd <= w.range_tiles ** 2
        e[30] = dd <= K["throw_r"] ** 2
        cat, _ = aim_of(sim, K, team, it, me)
        e[31], e[32] = cat == 1, cat
        if it.is_zombie:
            e[33] = 2 if it.zombie_activated else 3 if it.zombie_noise_ticks > 0 else 1
        aiming = [m for m in others if aim_of(sim, K, team, m, me)[1] == r.uid]
        e[34] = over(len(aiming), n_t)
        e[35] = over(sum(1 for m in aiming if m.live_trigger), n_t)
        e[36] = squash(over(pic.damage_last_s(team, r.uid), pic.bars[r.uid]))
        e[37] = u.order_target == r.uid
        e[38] = over(sum(1 for m in mates if (centre(m)[0] - ex) ** 2
                         + (centre(m)[1] - ey) ** 2 <= K["frag_r"] ** 2), n_t)
        mask[i], uid[i] = 1, r.uid
        i += 1
    i = caps[0] + caps[1]
    old = sorted([r for r in recs.values() if r.uid not in seen],
                 key=lambda r: (d2(r.centre), r.uid))
    for r in old[:caps[2]]:
        e = out[i]
        common(e, 2, r.centre[0], r.centre[1])
        e[4], e[5] = cos_q16(r.facing_q) / FP_ONE, sin_q16(r.facing_q) / FP_ONE
        e[6], e[7] = squash(over(r.vel_q[0], K["v_ref"])), squash(over(r.vel_q[1], K["v_ref"]))
        e[8] = r.team
        e[9] = K["zombie_class"] if r.zombie else row("loadouts", r.loadout)
        e[54] = over(tick - r.tick, K["memory"])
        mask[i], uid[i] = 1, r.uid
        i += 1
    i = caps[0] + caps[1] + caps[2]
    objs = []
    for o, ouid in pic.seen_now[team]:
        oy, ox = tp.object_tile(o)
        objs.append(((ox - cx) ** 2 + (oy - cy) ** 2, ouid, o, ox, oy))
    for _, ouid, o, ox, oy in sorted(objs, key=lambda t: t[:2])[:caps[3]]:
        e = out[i]
        common(e, 3, ox, oy)
        charge = hasattr(o, "det_tick")
        if not charge:
            ddx, ddy = o.target_fx - o.start_fx, o.target_fy - o.start_fy
            dist = math.hypot(ddx, ddy)
            if dist >= 0.1 and not (o.fx == o.target_fx and o.fy == o.target_fy):
                vx = ddx / dist * o.travel_speed / K["tps"]
                vy = ddy / dist * o.travel_speed / K["tps"]
                e[6] = squash(over(vx, K["v_ref"] / FP_ONE))
                e[7] = squash(over(vy, K["v_ref"] / FP_ONE))
        e[8] = e[9] = -1.0
        e[55] = K["objs"].index(o.ammo_name) if o.ammo_name in K["objs"] else -1
        roster = {x.id for x in sim.roster(team) if x is not None}
        e[56] = (0 if o.owner_id == me else 1 if o.owner_id in roster
                 else 2 if o.owner_id >= 0 else 3)
        if charge:
            if o.det_tick is None:
                e[58] = 1.0
            else:
                e[57] = squash(over(o.det_tick - tick, K["c4_fuse"]))
            e[61] = 1.0
        else:
            e[57] = squash(over(o.get_detonate_tick() - tick, K["hand_fuse"]))
            e[61] = o.fx == o.target_fx and o.fy == o.target_fy
        try:
            r = sim.weapons_tables.payload_for_ammo(o.ammo_name).radius
        except KeyError:
            r = 0
        e[59] = squash(over(r, L))
        e[60] = (ox - cx) ** 2 + (oy - cy) ** 2 <= r * r
        mask[i], uid[i] = 1, ouid
        i += 1
    i = caps[0] + caps[1] + caps[2] + caps[3]
    wired = room_graph.wired_door_ordinals(sim.level.wires)
    doors = []
    for d in sim._doors:
        if d.alive:
            t = d.span[len(d.span) // 2]
            doors.append((d2((t[1], t[0])), d.ordinal, d, t[1], t[0]))
    for _, ordinal, d, dx_, dy_ in sorted(doors, key=lambda t: t[:2])[:caps[4]]:
        e = out[i]
        common(e, 4, dx_, dy_)
        e[8] = e[9] = -1.0
        e[62] = d.state == DOOR_OPEN
        x0, y0, fp = u.tile_x, u.tile_y, u.footprint
        e[63] = any(y0 - 1 <= ty <= y0 + fp and x0 - 1 <= tx <= x0 + fp
                    for ty, tx in d.span)
        e[64] = horde.is_hull_door(sim.gmap, d.span)
        e[65] = ordinal in wired
        mask[i], uid[i] = 1, DOOR_BASE + ordinal
        i += 1
    return out, mask, uid


# -- section 3: the two windows, tile by tile ------------------------------
WIN, HALF, BLOCK = 25, 12, 4
PHYSICS = ("temperature", "fire", "poison", "teargas", "smoke", "pressure",
           "wind_x", "wind_y")


def felt_temperature(sim, u):
    """Spec 1.3 col 96 by hand: the heat row's chain over the hottest
    footprint tile of max(heat, the radiation sensor)."""
    g = sim.gmap
    h, w = g.heat.shape
    rad = getattr(g, "rad_flux_sweep", None)
    bits = optics_fixed.sweep_fine_bits(g)
    peak = 0
    for tx, ty in u.occupied_tiles():
        if 0 <= ty < h and 0 <= tx < w:
            v = int(g.heat[ty, tx])
            if rad is not None:
                v = max(v, optics_fixed.fine_heat_shr(int(rad[ty, tx]), 0, bits))
            peak = max(peak, v)
    c = CFG.combat
    if peak <= 0:
        return float(c.heat_ambient_ref)
    phi = peak / 65536 * float(c.unit_absorption) * (1.0 - float(c.unit_reflectivity))
    return float(c.heat_ambient_ref) + float(c.heat_flux_to_temp) * phi


def raw_tile(sim, name, y, x):
    """One physics plane's real-unit value on tile (y, x); outside 0."""
    g = sim.gmap
    h, w = g.solid.shape
    if not (0 <= y < h and 0 <= x < w):
        return 0.0
    if name == "temperature":
        return float(gas_fixed.dequantize_q16(g.temperature[y, x]))
    if name == "fire":
        return float(fire_fixed.dequantize(g.fire[y, x]))
    if name == "poison":
        return float(gas_fixed.dequantize_trace(g.gas[G_POISON][y, x]))
    if name == "teargas":
        return float(gas_fixed.dequantize_trace(g.gas[G_TEARGAS][y, x]))
    if name == "smoke":
        return float(gas_fixed.dequantize_trace(g.gas[G_SMOKE][y, x]))
    if name == "pressure":
        return float(atmosphere_fixed.dequantize(g.atmosphere[y, x]))
    raise KeyError(name)


def wind_tile(sim, y, x):
    """``(W_i, vx, vy)`` on tile (y, x) in ints: the footprint read's own
    term (0, 0, 0 on a solid or outside tile)."""
    g = sim.gmap
    h, w = g.solid.shape
    if not (0 <= y < h and 0 <= x < w) or g.solid[y, x]:
        return 0, 0, 0
    vx, vy = int(g.wind_x[y, x]), int(g.wind_y[y, x])
    n = int(g.gas[O2][y, x]) + int(g.gas[INERT_N2][y, x])
    return n * ((vx * vx + vy * vy) >> 16), vx, vy


def wind_value(sim, u, w_i, vx, vy):
    """The wind pair's ratio (spec 3.2 rows 6, 7): one tick of impulse on
    the standing body over its mass, along the wind, over the hold."""
    t = sim.impulse_table
    j = ui.wind_impulse_q(t, w_i)
    n = math.hypot(vx, vy)
    if n == 0 or j == 0:
        return 0.0, 0.0
    d = float(u.mass_g) * float(t.cap_q)
    return j * vx / n / d, j * vy / n / d


def physics_ratio(sim, u, name, raw):
    env = u.environment
    if name == "temperature":
        return over(raw, env.temperature_max)
    if name == "fire":
        return raw
    if name == "poison":
        return over(raw, CFG.exchange.poison_min_density) \
            * mitigation_for(u).resist_mult[POISON]
    if name == "teargas":
        return over(raw, CFG.exchange.teargas_blind_density)
    if name == "smoke":
        return over(raw, CFG.bots.smoke_dense)
    if name == "pressure":
        return over(raw - env.pressure_min, env.pressure_max - env.pressure_min)
    raise KeyError(name)


def air_at(sim, u, y, x):
    """The eight physics ratios on one tile (before the squash)."""
    out = [physics_ratio(sim, u, n, raw_tile(sim, n, y, x)) for n in PHYSICS[:6]]
    out += list(wind_value(sim, u, *wind_tile(sim, y, x)))
    return out


def door_code(sim, y, x):
    for d in sim._doors:
        if d.alive and any((int(t[0]), int(t[1])) == (y, x) for t in d.span):
            if d.state == DOOR_OPEN:
                return 1
            if d.state == DOOR_CLOSED:
                return 2
    return 0


def lane_cell(sim, team, me, y, x):
    g = sim.gmap
    h, w = g.solid.shape
    if not (0 <= y < h and 0 <= x < w):
        return 0
    cell, hw = int(CFG.bots.lane_cell), int(CFG.bots.lane_halfwidth)
    o = tp.fire_lanes(sim, team, cell, hw)[y // cell, x // cell]
    return int(o == -2 or (o >= 0 and o != me))


def near_planes(sim, K, u):
    """``(near_f (8, 25, 25) float64, near_u (5, 25, 25) uint8)``, cell by
    cell from the document: the physics ratios squashed (0 outside the
    map), wall (1 outside), door, watched = OR over the team's records of
    ``has_los`` from the recorded centre, sees_target 0, lanes."""
    g = sim.gmap
    h, w = g.solid.shape
    team = int(u.team)
    cx, cy = centre(u)
    recs = sim.team_picture.records(team)
    nf = np.zeros((8, WIN, WIN))
    nu = np.zeros((5, WIN, WIN), np.uint8)
    for i in range(WIN):
        for j in range(WIN):
            y, x = cy - HALF + i, cx - HALF + j
            inside = 0 <= y < h and 0 <= x < w
            if inside:
                for k, r in enumerate(air_at(sim, u, y, x)):
                    nf[k, i, j] = squash(r)
                nu[0, i, j] = g.solid[y, x]
                nu[1, i, j] = door_code(sim, y, x)
                nu[4, i, j] = lane_cell(sim, team, u.id, y, x)
            else:
                nu[0, i, j] = 1
            nu[2, i, j] = any(g.has_los(r.centre[1], r.centre[0], y, x)
                              for r in recs.values())
    return nf, nu


def far_planes(sim, K, u):
    """``(far_f (10, 25, 25) float64, far_u (5, 25, 25) uint8)`` block by
    block from the document: the wall share (outside tiles solid), the
    physics maxima over the block's tiles (padded with 0; the wind by the
    block's largest load, first in row-major), the route's block minimum
    over the field (-1 unreached or no order), the door's largest code,
    the mate / seen / record counts by centre tile, the place."""
    g = sim.gmap
    h, w = g.solid.shape
    hb, wb = -(-h // BLOCK), -(-w // BLOCK)
    team = int(u.team)
    cx, cy = centre(u)
    by0, bx0 = cy // BLOCK - HALF, cx // BLOCK - HALF
    recs = sim.team_picture.records(team)
    seen = set(vision.state_for(sim).visible_by_team.get(team, ()))
    ff = np.zeros((10, WIN, WIN))
    fu = np.zeros((5, WIN, WIN), np.uint8)
    ff[9] = -1.0
    field = place = None
    graph = graph_of(sim)
    if u.order_what and graph is not None:
        field = route_field(sim, graph, u.order_where, u.footprint)
        place = np.zeros((h, w), bool)
        if u.order_where < ORDER_OPENING_BASE:
            place = np.asarray(graph.grid) == u.order_where
        else:
            for o in graph.openings:
                if ORDER_OPENING_BASE + o.id == u.order_where:
                    place[o.y:o.y + o.h, o.x:o.x + o.w] = True
    for i in range(WIN):
        for j in range(WIN):
            by, bx = by0 + i, bx0 + j
            if not (0 <= by < hb and 0 <= bx < wb):
                ff[0, i, j] = 1.0
                continue
            tiles = [(by * BLOCK + a, bx * BLOCK + b)
                     for a in range(BLOCK) for b in range(BLOCK)]
            inb = [(y, x) for y, x in tiles if 0 <= y < h and 0 <= x < w]
            ff[0, i, j] = (16 - len(inb) + sum(1 for y, x in inb
                                              if g.solid[y, x])) / 16.0
            for k, name in enumerate(PHYSICS[:6]):
                # the map's padding is 0; a block wholly inside takes its
                # own maximum (negative on a cold deck)
                vals = [raw_tile(sim, name, y, x) for y, x in inb]
                raw = max(vals + ([0.0] if len(inb) < 16 else []))
                ff[1 + k, i, j] = squash(physics_ratio(sim, u, name, raw))
            best = (0, 0, 0)
            for y, x in tiles:
                wt = wind_tile(sim, y, x)
                if wt[0] > best[0]:
                    best = wt
            wx, wy = wind_value(sim, u, *best)
            ff[7, i, j], ff[8, i, j] = squash(wx), squash(wy)
            if field is not None:
                m = min(horde.field_at(field, y, x) for y, x in tiles)
                if m != horde.UNREACHED:
                    ff[9, i, j] = squash(over(m, K["L"]))
                fu[4, i, j] = any(0 <= y < h and 0 <= x < w and place[y, x]
                                  for y, x in tiles)
            fu[0, i, j] = max(door_code(sim, y, x) for y, x in tiles)
            for m in mates_of(sim, team):
                if m.id != u.id:
                    mx, my = centre(m)
                    fu[1, i, j] += (my // BLOCK, mx // BLOCK) == (by, bx)
            for uid, r in recs.items():
                if (r.centre[1] // BLOCK, r.centre[0] // BLOCK) == (by, bx):
                    fu[2 if uid in seen else 3, i, j] += 1
    return ff, fu
