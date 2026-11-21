"""Zombie-conversion diagnostic for ship_demo bots (measurement only).

usage: diag.py ROOT TAG SEED [--off flag,flag,...] [--max N]
ROOT = the checkout (HEAD worktree or main worktree), its cpp/build/Release module.
--off turns EOSSolver test seams off at runtime (subtick_loop, subtick_ceiling,
cap_heat, kset_skip_partial) -- HEAD build only.
Writes OUT/{TAG}_s{SEED}.json and prints one summary line.
"""
import sys, json, os, math, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(sys.argv[1]); TAG = sys.argv[2]; SEED = int(sys.argv[3])
OFF = []; MAXT = 100000
a = sys.argv[4:]
while a:
    if a[0] == "--off":
        OFF = [s for s in a[1].split(",") if s]; a = a[2:]
    elif a[0] == "--max":
        MAXT = int(a[1]); a = a[2:]
    else:
        raise SystemExit(f"unknown arg {a[0]}")
os.chdir(ROOT)
for p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools", ROOT / "tests"):
    sys.path.insert(0, str(p))
import breach_physics as bp, level_loader
from simulation.match_session import MatchSession
from simulation import ai_zombie, corpse, events as ev
from simulation.status import KNOCKED_DOWN

OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
level = level_loader.load("ship_demo", str(ROOT / "levels"))
s = MatchSession(level, seed=SEED, breach_physics=bp)
sim = s.sim
eos = sim.physics_runner.eos
for f in OFF:
    assert hasattr(eos, f), f
    setattr(eos, f, False)
seams = {f: getattr(eos, f) for f in ("subtick_loop", "subtick_ceiling", "cap_heat", "kset_skip_partial") if hasattr(eos, f)}

orig_zombie = {u.id: bool(u.is_zombie) for u in sim.units}
team0 = {u.id: u.team for u in sim.units}
kind0 = lambda uid: "zombie" if orig_zombie.get(uid) else "marine"
prev_pos = {u.id: (u.x_q, u.y_q) for u in sim.units}

kills = []          # (tick, victim, victim_kind_now, cause, attacker, attacker_kind_now)
bites = 0; bite_dmg = 0; zhits = Counter()
min_d2 = None; min_d2_tick = None; first_reach = None; first_reach_pair = None
dist_series = []    # every 24 ticks: (tick, min zombie-marine dist tiles, n_z_alive, n_z_awake, n_m_alive)
first_awake = {}; first_hearing = {}
zsteps = defaultdict(int); zticks_awake = defaultdict(int); zdist = defaultdict(float)
z_kd_ticks = Counter(); m_kd_ticks = Counter()
z_pv_max = 0.0; m_pv_max = 0.0; z_pv_ticks = 0
conversions = []
destroyed_seen = {}
max_infest = 0
zombie_kill_victims = set()
bite_log = []; zdeath_pos = {}
rises_true_seen = 0; eligible_eval = Counter()

t0 = time.time()
while not s.is_terminal() and sim.tick < MAXT:
    was_z = {u.id: bool(u.is_zombie) for u in sim.units}
    s.tick()
    T = int(sim.tick)
    for e in sim.tick_events:
        if isinstance(e, ev.UnitKilledEvent):
            vic = sim.unit_by_id(e.unit_id) if hasattr(sim, "unit_by_id") else next(u for u in sim.units if u.id == e.unit_id)
            att = None
            if e.attacker_id >= 0:
                att = next((u for u in sim.units if u.id == e.attacker_id), None)
            vk = "zombie" if was_z.get(vic.id, vic.is_zombie) else "marine"
            az = None if att is None else was_z.get(att.id, att.is_zombie)
            ak = None if att is None else ("zombie" if az else f"marine_t{att.team}")
            same_team = (att is not None and not az and vk == "marine" and att.team == vic.team)
            kills.append(dict(tick=T, victim=e.unit_id, vk=vk, vteam=vic.team, cause=e.killed_by,
                              attacker=e.attacker_id, ak=ak, friendly=same_team,
                              hp_after=int(vic.hp_q) / 65536.0))
            if vk == "zombie":
                zdeath_pos[e.unit_id] = (round(vic.x, 1), round(vic.y, 1))
            if ak == "zombie" and vk == "marine":
                zombie_kill_victims.add(e.unit_id)
        elif isinstance(e, ev.UnitHitEvent):
            att = next((u for u in sim.units if u.id == e.attacker_id), None) if e.attacker_id >= 0 else None
            if att is not None and was_z.get(att.id, att.is_zombie) and e.source == "melee":
                bites += 1; bite_dmg += e.damage
                _v = next(u for u in sim.units if u.id == e.unit_id)
                bite_log.append((T, att.id, e.unit_id, round(int(_v.hp_q) / 65536.0, 1)))
            vic = next(u for u in sim.units if u.id == e.unit_id)
            if was_z.get(vic.id, vic.is_zombie):
                zhits[e.source] += 1
    zs = [u for u in sim.units if u.is_zombie and u.alive]
    ms = [u for u in sim.units if not u.is_zombie and u.alive]
    md2 = None
    for z in zs:
        zx, zy = z.center_tile_x(), z.center_tile_y()
        reach = z.footprint + 1
        for m in ms:
            d2 = ai_zombie._d2(zx, zy, m.center_tile_x(), m.center_tile_y())
            if md2 is None or d2 < md2:
                md2 = d2
            if d2 <= reach * reach and first_reach is None:
                first_reach = T; first_reach_pair = (z.id, m.id)
        if z.zombie_activated and z.id not in first_awake:
            first_awake[z.id] = T
        if z.zombie_noise_ticks and z.id not in first_hearing:
            first_hearing[z.id] = T
        px, py = prev_pos.get(z.id, (z.x_q, z.y_q))
        d = math.hypot(z.x_q - px, z.y_q - py) / 65536.0
        if z.zombie_activated or z.zombie_noise_ticks:
            zticks_awake[z.id] += 1; zdist[z.id] += d
        pv = math.hypot(z.pvx, z.pvy) / 65536.0
        z_pv_max = max(z_pv_max, pv)
        if pv > 0: z_pv_ticks += 1
        if any(st.kind == KNOCKED_DOWN for st in (z.statuses or [])):
            z_kd_ticks[z.id] += 1
    for m in ms:
        m_pv_max = max(m_pv_max, math.hypot(m.pvx, m.pvy) / 65536.0)
        if any(st.kind == KNOCKED_DOWN for st in (m.statuses or [])):
            m_kd_ticks[m.id] += 1
    if md2 is not None and (min_d2 is None or md2 < min_d2):
        min_d2 = md2; min_d2_tick = T
    for u in sim.units:
        prev_pos[u.id] = (u.x_q, u.y_q)
        if u.is_zombie and u.loadout_name is not None and u.id not in [c["id"] for c in conversions]:
            conversions.append(dict(id=u.id, tick=T))
        if not u.alive:
            if corpse.destroyed(u) and u.id not in destroyed_seen:
                destroyed_seen[u.id] = T
            max_infest = max(max_infest, int(getattr(u, "infest_q", 0)))
            r = ai_zombie.rises_as_zombie(u)
            if r: rises_true_seen += 1
            eligible_eval["dead_ticks"] += 1
            if u.killed_by_zombie: eligible_eval["kbz_flag_post_step"] += 1
    if T % 24 == 0:
        dist_series.append((T, None if md2 is None else round(math.sqrt(md2), 1), len(zs),
                            sum(1 for z in zs if z.zombie_activated), len(ms)))

res = s.result()
dead = [u for u in sim.units if not u.alive]
corpses = [dict(id=u.id, kind0=kind0(u.id), zombie_now=bool(u.is_zombie), hp=int(u.hp_q) / 65536.0,
                destroyed=bool(corpse.destroyed(u)), body_floor=-corpse.body_q(u) / 65536.0,
                infest=int(u.infest_q) / 65536.0, kbz=bool(u.killed_by_zombie)) for u in dead]
deaths_m = Counter(); deaths_z = Counter()
for k in kills:
    key = k["cause"] + "/" + (k["ak"] or "env").split("_")[0] + ("(ff)" if k["friendly"] else "")
    (deaths_z if k["vk"] == "zombie" else deaths_m)[key] += 1
zspeed = {zid: (zdist[zid] / zticks_awake[zid]) for zid in zticks_awake if zticks_awake[zid]}
rec = dict(ver=3, bite_log=bite_log, zdeath_pos=zdeath_pos, tag=TAG, seed=SEED, seams=seams, off=OFF, ticks=int(sim.tick), terminal=s.is_terminal(),
           result=None if res is None else dict(outcome=res.outcome, team=res.team, ticks=res.ticks),
           kills=kills, deaths_marine=dict(deaths_m), deaths_zombie=dict(deaths_z),
           bites=bites, bite_dmg=bite_dmg, zombie_hits_by_source=dict(zhits),
           min_dist=None if min_d2 is None else math.sqrt(min_d2), min_dist_tick=min_d2_tick,
           first_reach=first_reach, first_reach_pair=first_reach_pair,
           first_awake=first_awake, first_hearing=first_hearing,
           zspeed_tiles_per_tick_awake=zspeed, z_kd_ticks=dict(z_kd_ticks), m_kd_ticks=dict(m_kd_ticks),
           z_pv_max=z_pv_max, z_pv_ticks=z_pv_ticks, m_pv_max=m_pv_max,
           conversions=conversions, destroyed_tick=destroyed_seen, max_infest=max_infest / 65536.0,
           rises_true_post_step=rises_true_seen, eligible=dict(eligible_eval),
           zombie_kill_victims=sorted(zombie_kill_victims), corpses=corpses, dist_series=dist_series,
           wall_s=round(time.time() - t0, 1))
(OUT / f"{TAG}_s{SEED}.json").write_text(json.dumps(rec, indent=1))
print(f"{TAG} s{SEED} off={OFF}: ticks {rec['ticks']} {rec['result']} marine deaths {dict(deaths_m)} "
      f"zombie deaths {dict(deaths_z)} bites {bites} first_awake {min(first_awake.values()) if first_awake else None} "
      f"first_reach {first_reach} min_dist {rec['min_dist'] and round(rec['min_dist'],1)}@{min_d2_tick} "
      f"conv {len(conversions)} zkills {len(zombie_kill_victims)} kd_z {sum(z_kd_ticks.values())} "
      f"zpvmax {z_pv_max:.3f} ({rec['wall_s']} s)", flush=True)
