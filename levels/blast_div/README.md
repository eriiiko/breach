# Blast div

The #31 divergence-source prototype against hot mass, firing together at 2 s and
then every 3 s. Rooms and timing are the same as `levels/blast_quad`.

| Room | Row | What it does |
|---|---|---|
| NW | `quad_r3` | hot mass, the control |
| NE | `div_40` | pushes out 40 m3 of air over 2 ticks, no mass |
| SW | `div_80` | pushes out 80 m3 over 2 ticks, no mass |
| SE | `div_40_puff` | `div_40` plus a small puff of mass |

## Launch

    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level blast_div --control gamepad

Run it from this worktree: the prototype needs this branch's own CPU build.
Change a version in the `[payloads.div_*]` rows of `config.toml` and restart.
