# P3e play test — the match path to integer (2026-11-12)

Branch `29-p3e-integer-match` (issue #29, patch P3e). HUMAN-TEST: it merges
only after you have played it.

## What changed, in plain words

Everything that decides a fight now runs on whole numbers, so the GPU can
later run exactly the same game. You should not feel a difference; if you do,
that is what this test is for.

- **Shots.** The rifle's spread is the same ±3°, but the random number for
  each round is drawn differently, so every individual shot lands somewhere
  else than it did before. Rounds through soft cover (crates) still connect
  55 % of the time, by a new dice roll.
- **Damage.** Bullets, bites and blows do exactly what they did. Fire and
  poison damage are computed in whole steps now; at most 1 part in 10 000
  differs from before.
- **Blasts.** Grenades, the C4, the cook-off: the same walls break, the same
  marines go down and die; the damage per marine is the same to within a
  quarter of a hit point (mostly a hundredth). The smoke and heat a blast
  leaves are the same to within 0.03 %.
- **Grenade throws** land on the same tile (bar a 1-in-130 000 edge case).

## The numbers that matter (`docs/p3e_ab_2026-11-12.md`)

| | before | after |
|---|---|---|
| bots v horde, win rate (48 matches) | 85 % | 88 % |
| bots v horde, rounds per kill | 11.8 | 12.4 |
| bots v bots, rounds per kill (72 matches) | 9.3 | 8.7 |
| bots v bots, match length | 124 s | 126 s |
| focus fire, rounds to kill a zombie (1 / 4 shooters) | 34 / 24 | 34 / 24 |
| C4 at 12 tiles, hp lost | 78.5 | 78.6 |

All inside seed noise. One number to watch: in bots-v-bots firefights more
marines died of HEAT after the change (114 → 138 over 72 matches, about 1.6
standard errors — probably which fights set the furniture alight, since the
blast tests show the same heat damage).

## How to launch

From the worktree `C:\Users\steen\projects\breach\.claude\worktrees\29-p3e-integer-match`
(its CPU build is current):

    # 1. A fight you play yourself (direct control: RDFG move, mouse aim,
    #    left button fire, W throw, X plant the C4)
    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level ship_demo --control direct

    # 2. A watched bots match (the follow camera; C cycles it)
    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level ship_demo --control match

    # 3. Blasts: frag, door charge, cook-off, C4, all at 2 s and every 8 s
    C:/Users/steen/anaconda3/envs/data/python.exe main.py --level blast_final --control direct

Add `--cuda` to any of them for the GPU physics (the CUDA build is current at
the branch's head).

## What to look for

1. Does the carbine's spread feel the same — no clumping, no lean to one side?
2. Firing through a crate: do some rounds still get through?
3. Does a frag still knock down and kill what it used to at the same range?
4. Fire after blasts: does the furniture catch and burn the way it did, and
   does standing near it hurt about as fast?
5. Anything that looks stuck, jittery or wrong in a grenade's flight.

If something feels off, say which of the five; the A/B doc has the
before/after numbers for each.
