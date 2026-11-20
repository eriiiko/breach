# Zombie bite balance — 2026-11-12 (#183)

## The rule (Erik)

The zombie's bite kills a 100 HP marine in three bites. A lone zombie that
reaches a marine at contact must be able to kill the marine before the
marine, firing full auto from that moment, kills it — unless the marine
shoves and steps back and fires again; a marine that starts firing from range
wins. "Even a single zombie that you don't deal with with respect CAN kill
you." Frag numbers stay as they are.

## The ruling (Erik, 2026-11-12)

| key | was | now |
|---|---|---|
| `[zombie] melee_damage` | 60 | 34 (3 x 34 = 102 kills, 2 x 34 = 68 does not; the marine's kinetic mitigation is neutral, so the bite lands whole) |
| `[zombie] attack_cooldown_seconds` | 1.0 | 2.2 (53 ticks at 24 tps) |
| `[zombie] hp` (a TURNED marine's hp) | 400 | 100 (a spawned zombie already had the human vitality, 100) |
| `[exchange] knockdown_getup_seconds` (the shove's knockdown, shared by every knockdown cause) | 1.5 | 1.5, unchanged |

Side effect: the door bash strikes at the bite's cadence, so a closed door
now holds one zombie about 42 s (20 strikes, 19 x 53 ticks), was 20 s.

## The measurement

`tools/bench_zombie_melee.py`: one marine (the default rifleman loadout:
the carbine, 30-round magazines, 2 s reload) against one zombie on an open
arena at the 0.333 m tile, the real sim (physics on, the conductor's zombie
slot, the marine through the intent facade). Cases:

- **idle**: zombie at contact, marine does nothing.
- **fire_vs_standing**: zombie at contact but not driven; marine fires full auto from tick 0.
- **duel**: both act from contact.
- **shove**: from contact the marine shoves once, walks two tiles straight back (trigger released), resumes full auto.
- **shove_repeat**: as shove, and again each time the zombie is back at contact.
- **range**: the zombie 10 m away, awake, walking in; the marine fires from tick 0.

At the final values (main `37142802` merged in, re-run 2026-11-13). With main
`321babcf` the table was the same except range (5.12 s and 39 rounds, both
zombies); main has since moved the shot cone's random numbers (#29 P3b keyed
Philox draws), and the ranged kill now takes 43 rounds (cause not separately
proven), winner unchanged. Every contact row held to the digit:

| zombie | case | winner | seconds | bites | zombie HP left | rounds |
|---|---|---|---|---|---|---|
| spawned | idle | zombie | 4.46 | 3 | 100 | 0 |
| spawned | fire_vs_standing | marine | 4.71 | 0 | 0 | 34 |
| spawned | duel | zombie | 4.46 | 3 | 5.5 | 31 |
| spawned | shove | **zombie** | 4.67 | 3 | 7.5 | 30 |
| spawned | shove_repeat | **zombie** | 4.88 | 3 | 10.6 | 30 |
| spawned | range | marine | 5.46 | 0 | 0 | 43 |
| turned | idle | zombie | 4.46 | 3 | 100 | 0 |
| turned | fire_vs_standing | marine | 4.71 | 0 | 0 | 34 |
| turned | duel | zombie | 4.46 | 3 | 5.5 | 31 |
| turned | shove | **zombie** | 4.58 | 3 | 7.5 | 30 |
| turned | shove_repeat | **zombie** | 4.71 | 3 | 10.7 | 30 |
| turned | range | marine | 5.46 | 0 | 0 | 43 |

Before (60 / 1.0 s / 400): the zombie killed an idle marine in two bites in
1.04 s, won every contact case, and a turned zombie (400 HP) could not be
killed by one marine at all, even standing (three magazines left it on
122 HP); it won from range too.

## Against the rule

- Contact, marine only fires back: **the zombie wins** (holds, by 5.5 HP).
- From range: **the marine wins** (holds).
- Shove + step back + fire: **the zombie wins** — the rule does NOT hold at
  these values. The first bite lands on the tick of contact, before the
  shove takes effect; the shove knocks the zombie down for about 1.8 s and
  the marine backs off and fires, but the zombie needs ~33 rounds and the
  magazine holds 30: the gun runs dry with the zombie on 7.5 HP, and the 2 s
  reload lets the third bite in. Shoving again each time it comes back does
  not help (10.6 HP left).

The sweep that went to Erik before the ruling (spawned zombie, damage 34)
showed the shove case won only with a longer knockdown: at cooldown
2.1 / 2.2 / 2.3 s with the knockdown at 3.0 s the whole rule held (duel to
the zombie, shove and range to the marine); at the 1.5 s knockdown no
cooldown from 2.0 to 3.0 s satisfies it (from 2.4 s the marine wins the
duel). Erik kept the knockdown at 1.5 s and judges the shove case at the
HUMAN-TEST.
