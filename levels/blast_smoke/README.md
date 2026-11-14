# Blast smoke

The #31 smoke amount, side by side. All four rooms fire `div_40`, the same blast,
at 2 s and then every 3 s. Only the soot changes, through `explosive_kg`. The NW
room is the chosen amount: `[blast_air] smoke_scale = 0.25` (Erik, 2026-11-06).
The other rooms show what turning it down would look like.

| Room | Row | Soot |
|---|---|---|
| NW | `div_40` | 3.5 g, a grenade today |
| NE | `div_40_s50` | 1.8 g |
| SW | `div_40_s25` | 0.9 g |
| SE | `div_40_s10` | 0.4 g |

## Launch

    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level blast_smoke --control gamepad

Run it from this worktree: the prototype needs this branch's own CPU build.

Each room is ~200 m3 with a 1.33 m gap to open sky. The gap is why a strong blast
loses smoke here: it is a property of this test room, not of a sealed ship.
