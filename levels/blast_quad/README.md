# Blast quad

Four candidate blast deliveries side by side for #31, firing together at 2 s and
then every 3 s. Each delivers the same total mass and heat.

| Room | Row | Delivery |
|---|---|---|
| NW | `quad_tile` | one tile, one tick |
| NE | `quad_r3` | disc of radius 3, one tick |
| SW | `quad_r5` | disc of radius 5, one tick |
| SE | `quad_slow` | radius 3, spread over 6 ticks (six charges one tick apart) |

Each room has a 1.33 m gap to open sky in its outer wall, so pressure bleeds off
between shots. The room layout is `tools/gen_blast_quad.py`, generated through
`level_lib`.

## Launch

    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level blast_quad --control gamepad

`--control gamepad` runs continuous time with no rounds. Camera and zoom work
from keyboard and mouse, and F6 shows the tile inspector.

## Changing a version

Edit the `[payloads.quad_*]` rows in `config.toml` and restart. The total mass
is `pressure` times the disc's weight, and the total heat is `heat_amount` times
the same weight: 1.0 for radius 1, 9.38 for radius 3, 26.05 for radius 5.
