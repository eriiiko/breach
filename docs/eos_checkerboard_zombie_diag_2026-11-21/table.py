import json, sys
from pathlib import Path
from collections import Counter
OUT = Path(__file__).parent / "out"
tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["head", "main"]
rows = []
for f in sorted(OUT.glob("*.json")):
    r = json.loads(f.read_text())
    if r["tag"] not in tags:
        continue
    rows.append(r)
rows.sort(key=lambda r: (r["seed"], tags.index(r["tag"])))
def short(d):
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))
tot = {t: Counter() for t in tags}
for r in rows:
    res = r["result"]
    endr = f"{res['outcome']}{'' if res['team'] is None else ' t'+str(res['team'])}" if res else "cap"
    fa = min(r["first_awake"].values()) if r["first_awake"] else None
    nawake = len(r["first_awake"])
    zs = r["zspeed_tiles_per_tick_awake"]
    zsp = sum(zs.values()) / len(zs) if zs else 0
    print(f"s{r['seed']:>2} {r['tag']:<5} len {r['ticks']:>5} end {endr:<7} | M: {short(r['deaths_marine'])} | Z: {short(r['deaths_zombie'])} "
          f"| awake {nawake} first {fa} | mind {r['min_dist'] and round(r['min_dist'],1)}@{r['min_dist_tick']} reach {r['first_reach']} "
          f"| bites {r['bites']} zkills {len(r['zombie_kill_victims'])} conv {len(r['conversions'])} | zspd {zsp:.3f} kd_z {sum(r['z_kd_ticks'].values())} zpv {r['z_pv_max']:.2f} "
          f"| corpses {len(r['corpses'])} destr {sum(c['destroyed'] for c in r['corpses'])}")
    t = tot[r["tag"]]
    t["n"] += 1; t["ticks"] += r["ticks"]; t["reach"] += r["first_reach"] is not None
    t["bites"] += r["bites"]; t["zkills"] += len(r["zombie_kill_victims"]); t["conv"] += len(r["conversions"])
    t["awake_any"] += nawake > 0
    for k, v in r["deaths_marine"].items(): t["M:" + k] += v
    for k, v in r["deaths_zombie"].items(): t["Z:" + k] += v
for tg in tags:
    print(tg, dict(tot[tg]))
