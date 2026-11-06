# Blast smoke

The #31 smoke amount, side by side. All four rooms fire `div_40`, the same blast,
at 2 s and then every 3 s. Only the soot changes, through `explosive_kg`.

| Room | Row | Soot |
|---|---|---|
| NW | `div_40` | 100 %, today's Comp B grenade, 14 g |
| NE | `div_40_s50` | 50 % |
| SW | `div_40_s25` | 25 % |
| SE | `div_40_s10` | 10 % |

## Launch

    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level blast_smoke --control gamepad

Run it from this worktree: the prototype needs this branch's own CPU build.

Each room is ~200 m3 with a 1.33 m gap to open sky. The gap is why a strong blast
loses smoke here: it is a property of this test room, not of a sealed ship.
