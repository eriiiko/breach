# Blast quad

The same 40 m3 source blast delivered four ways for #31, firing together at 2 s
and then every 3 s. Each has the same total heat, smoke and expansion.

| Room | Row | Delivery |
|---|---|---|
| NW | `quad_src_tile` | one tile, two ticks |
| NE | `div_40` | disc of radius 3 (1 m), two ticks |
| SW | `quad_src_r5` | disc of radius 5 (1.67 m), two ticks |
| SE | `quad_src_slow` | radius 3 over 6 ticks (six charges one tick apart) |

The SE room shows about 20 % less smoke than the others, because each charge's
blast clears the smoke at its own centre. The NW room hits the temperature rail.

Each room has a 1.33 m gap to open sky in its outer wall, so pressure bleeds off
between shots. The room layout is `tools/gen_blast_quad.py`, generated through
`level_lib`.

## Launch

    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level blast_quad --control gamepad

`--control gamepad` runs continuous time with no rounds. Camera and zoom work
from keyboard and mouse, and F6 shows the tile inspector.

## Changing a version

Edit the `[payloads.quad_src_*]` or `[payloads.div_40]` row in `config.toml` and
restart. Total heat is `heat_amount` times the disc's weight: 1.0 for radius 1,
9.38 for radius 3, 26.05 for radius 5. Expansion and smoke are already totals.
