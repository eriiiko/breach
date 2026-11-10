"""#122 -- the horde through the control seam (``control_zombies.ZombieControl``,
match controller ``"zombies"``).

Properties: the shipped zombie law is WRAPPED, not rewritten (a match plays
bit for bit as before #122 -- the one golden-style pin below, deliberately);
every zombie has one driver per tick; a bite's kill is credited to melee and
the converted marine is then driven by the zombie controller; an idle zombie
controller leaves the horde standing; the registry builds ``zombies``; the
game path (no match) still runs the conductor's zombie slot.

Levels are written through ``level_lib`` (never a hand-written toml) and
matches run through ``run_match`` (the one match loop).
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import simulation.simulation as sim_mod  # noqa: E402
from control_zombies import ZombieControl  # noqa: E402
from field_ab_harness import _capture_unit_state  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.events import UnitKilledEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.unit import Unit  # noqa: E402

IDLE = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="idle")
# Two marine rooms far apart, the zombie room right next to team 0 (the
# test_match.py box): with idle marines the horde eats team 0 first.
ROOMS = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4),
]
BOX_SPEC = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=3,
                     time_limit_s=20.0)


def _box(tmp_path, name, spec=BOX_SPEC, *, w=40, h=24):
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


class _UnitStream:
    """on_tick hook: a hash over every tick's synced unit records (positions,
    hp, life, team, ...) -- what the zombie law writes; events excluded, so
    the #122 melee events (which change no state) do not enter it."""

    def __init__(self):
        self.h = hashlib.sha256()

    def __call__(self, sim):
        self.h.update(repr(_capture_unit_state(sim)["units"]).encode())


# ---------------------------------------------------------------------------
# (a) the shipped law, wrapped -- pinned against the pre-#122 runner
# ---------------------------------------------------------------------------
# Captured at the start of #122 (main 5ee2712, the conductor driving the
# horde inside Simulation.step) with tools/run_match.py. This pin IS the
# property "wrap, don't rewrite": the zombie controller must reproduce the
# conductor's law tick for tick. A change of the zombie LAW (or of anything
# else these matches touch) moves it -- then re-pin deliberately, with the
# rationale, never to make the wrapping pass.
_PRE_122_BOX = dict(
    # #183 G1p-c re-pin (2026-11-10, speeds in m/s): the box is a 1 m tile,
    # so the zombies' cadence, the old [zombie] 14 ticks per tile (1.7 m/s
    # here), is now the [speeds] shamble, 0.6 m/s = 40 ticks per tile; and a
    # bitten marine that turns now shambles too (it kept the marine's 18-tick
    # attack dial before). Outcome and team UNCHANGED (win 1); ticks 82 ->
    # 130; field_digest and the unit stream move. SOLE CAUSES, verified on
    # the same tree: with the speed table at the old cadences on every map
    # (walk 18, full 8, zombie 14 ticks per tile, the direction factors 1.0)
    # AND a converting marine handed its old 18, this match gives exactly
    # the pin below; with the old cadences alone it ends at tick 78 -- the
    # converted zombies' shamble is the second cause, here only.
    # (was ticks 82; field 87e20008aff3541d2d0b1d64113108d5c09f85dd40ae1b58
    #  a0a6f58f12f83268; units 6dab6957a82fdd55786f7ec375e39761bdced4c61aa69a
    #  4cc6eb66a8cffb580a)
    # #183 G5p-a1 re-pin 1 (2026-11-10, the turning law's representation and
    # routing at UNLIMITED acceleration, a >= 2w: the facing is the int
    # facing_q spawned at quantize(pi/2), hashed as that int, with turn /
    # omega keys while non-zero; every facing writer hands slot 1a a target,
    # so the zombie walk's facing turns at the 15 deg cap, a tick later).
    # Outcome, team, ticks and field_digest UNCHANGED; the unit stream moves.
    # SOLE CAUSES, verified on the same
    # tree (a scratch proof over the five pinned matches): with (1) every
    # non-aim facing writer (travel, zombie walk, attack snap, WEGO,
    # onephase) placing the facing at once in its own slot again AND (2) the
    # old representation (Unit.facing the float pi/2 until first written, the
    # record hashing q(float facing), no turn / omega keys), this tree gives
    # EXACTLY the previous pin; with (2) alone undone, outcome, ticks,
    # field_digest and tick_digest are the previous ones and only the unit
    # stream moves (the zombie walk's facing lands one tick later: its target
    # is set in slot 5 and turned in the next tick's slot 1a).
    # (was units 61ec546723a826944ed2fa0f369f4054346609cca30756b37089b944b262
    #  db23)
    # #183 G5p-a1 re-pin 2 (2026-11-10, the shipped turning acceleration,
    # 30 rad/s^2, PLACEHOLDER). Outcome, team, ticks and field_digest
    # UNCHANGED; the unit stream moves (the zombies' facings now speed up and
    # brake through their turns). SOLE CAUSE, verified on the same tree: with
    # `[marine] turn_accel_rad_per_s2` >= 2w (1000, unlimited) this match
    # gives exactly re-pin 1's values -- tests/test_turning_law.py::
    # test_g3_at_unlimited_acceleration_the_pins_are_re_pin_1s holds that.
    # (was units a8f571251bbb5cc7572f7c4711dc1a3186b67dcdfd4c42837dacfcaeff1f
    #  efa0)
    outcome="win", team=1, ticks=130,
    field_digest="a672c7249e18d0eebbb8e398b696bd9c"
                 "3d34417445f66c3cab498ec7cd783ab1",
    unit_stream="f2ba375dd16868470f118df2bbbcab6c"
                "151ce28d4e2e108ab5f4433360406497")
_PRE_122_SHIP_DEMO = dict(
    # #140 re-pin (2026-10-31, the wind law retires the pressure-gradient
    # push). SOLE CAUSE, verified on the same build: the #140 P1a tree
    # (78d3d78, which still reproduces the old pin: win 0 at 1528, bdc5ad3e /
    # 4d32170a / a5ddf313) with k_push = 0 and knockdown_dv_threshold = 1e30
    # gives exactly the values below, and so does the P1b tree with the wind
    # thresholds at 1e15 / 2e15 Pa -- the push WAS the whole change, the
    # wind law never fires in this match. The match is now won by team 1 at
    # tick 706 (no squad is shoved around a doorway any more).
    # (was win 0 at 1528; field bdc5ad3e6e6a7f797199a2783efc13276f6f515df8e7
    #  2ce0a50ae9e16d06a898; tick 4d32170a7a7ff465ba410f935c8912ae594999c903
    #  169aa66582ae2e7bcb2ab8; units a5ddf313af731dd1b7e983e68ebe448dffbe59a9
    #  fcf238eadcf262576c97f47a)
    # #31 re-pin (2026-11-06, the standard explosion, design section 4): this
    # match throws frags, and `frag_standard` is a generic explosion now: a
    # divergence source in place of no pressure, 10 MJ over its walked tiles in
    # place of a 3200-unit disc, smoke over the same tiles at [blast_air]
    # smoke_scale. A wanted change of the whole match, so all four values move.
    # (was win 1 at 706; field b38c2d29ffeb91d3ea5fb772fa24d59786a53bc1c90004f7
    #  321bffbf442d3ca4; tick ce979185366be487732e587a851ab6e3589d0b0ea8671ab0
    #  6ef7c5f41ba25c37; units b55cc54e4e12bd7002a70cecd01d137703964eb73fa63c15
    #  da59f0b415cedb4e)
    # #183 G2p re-pin (2026-11-10, the carbine's rate: Erik's ruling, one
    # round per pull at 12 a second -- shots_per_trigger 1, rof 1/12 s = 2
    # ticks -- from 5 rounds every 4 ticks, 30 a second). Outcome and team
    # UNCHANGED (win 1); ticks 722 -> 700; field_digest, tick_digest and the
    # unit stream move. SOLE CAUSE, verified on the same tree: with the
    # carbine's row and the rifleman at their pre-G2p values (5 per pull, 4
    # ticks, mag_size 0, no `magazines`) this match gives exactly the G1p pin
    # below (win 1 at 722, 6aa5b17a / 74ea61e8 / 07ee472c); with ONLY the new
    # rate (no magazine) it gives the new values, bit for bit the same as with
    # the magazines -- in this match no marine fires more than 12 rounds in
    # all (tools/match_pin_check.py, a counting probe), so no gun empties and
    # no reload runs: the magazines, the verb and the spares move nothing
    # here; the whole move is the rate. The bots now put one round at a time
    # on a target instead of a five-round burst.
    # (was ticks 722; field 6aa5b17aa32e5ec77ddd56519458cd9b1c115346fabc70dcf
    #  284aaf2bbc96e39; tick 74ea61e8bb5e83a7fe55ee8272f0228a6c333aff2db2fdf3
    #  571f100c1de0e437; units 07ee472c29f86f18e64f2b01141ac8bb0ad64ec97828df
    #  8509830bd8fac1dbc)
    # #183 G1p-c re-pin (2026-11-10, speeds in m/s, Erik's ruling: the
    # marine walks 1.4 m/s at throttle 0 -- was 18 ticks per tile, 0.44 m/s
    # -- slower sideways and backward; the zombies shamble 0.6 m/s, 13 ticks
    # per tile, was 14). The bots walk at throttle 0, so the map plays about
    # three times faster: outcome and team UNCHANGED (win 1), ticks 700 ->
    # 240, team 1 now ends with slots [0, 1, 3] alive (was [0, 1]); all
    # three digests and the unit stream move. SOLE CAUSE, verified on the
    # same tree: with the speed table at the old cadences (walk 18, full 8,
    # zombie 14 ticks per tile, sideways_frac = backward_frac = 1.0) this
    # match gives exactly the G2p pin below (win 1 at 700, 1b9392ab /
    # 230d50a1 / a345834d) -- the direction rule's factors at 1.0 and the
    # bots' throttle 0 leave the forward-only term nothing to do, the
    # converted zombies' new cadence moves nothing here, and the ReloadEvent
    # moves no digest. Measured on the CPU, CUDA and resident backends alike
    # (tools/match_pin_check.py).
    # (was ticks 700; field 1b9392ab2c184a4684a28a6d7489684827ef10968c098fea
    #  2405945a37fc5e83; tick 230d50a18db03437b34b0600d87d795a57af4c93d969f31
    #  474e038db1a87f891; units a345834dfe6685918d0e0887667b978de462e09d0bfa6
    #  fcc26e9b9a68babcb9d)
    # #183 G3p re-pin (2026-11-10, the throw wind-up: the grenade leaves the
    # hand 7 ticks (0.3 s, PLACEHOLDER) after the verb, along the facing the
    # unit has then). Outcome and team UNCHANGED (win 1); ticks 240 -> 246;
    # field_digest, tick_digest and the unit stream move. SOLE CAUSE,
    # verified on the same tree: with `throw_windup_seconds = 0` this match
    # gives exactly the G1p-c pin below (win 1 at 240, 4a6ae8a3 / 6eb3105b /
    # 816192b5) -- tests/test_grenade_kinds.py::test_with_no_wind_up_the_
    # pinned_match_is_the_pre_g3p_match holds that -- and so does the new
    # loadout table alone (the bots throw frags only: smoke and the
    # flashbang are carried, never thrown), the integer fuse, the kind
    # column and the ThrowEvent (render-only). The bots' lobs now land a
    # wind-up later, along the aim they hold through it. Measured on the CPU,
    # CUDA and resident backends alike (tools/match_pin_check.py).
    # (was ticks 240; field 4a6ae8a34011fc3cc832c20326df631cd174b27bf64fc359
    #  d6249717ae6c9d89; tick 6eb3105b24e4debf47cdcb6366caea9df2d8717bbc2ca6
    #  387aba968a3002939e; units 816192b5309273533dff345e4300080b980e5323c43f
    #  b4c46241c92af38ae4ad)
    # #183 G4p re-pin (2026-11-10, the focus-fire law: a kinetic round's
    # `expose` raises the target's exposure, which decays with a 3-tick
    # half-life and multiplies the damage of its type by 1 + level;
    # [damage.exposure] half_life_seconds 0.125, k_expose 1, x_max 2.5, the
    # carbine round's expose 0.35, all PLACEHOLDERS). Outcome, team, ticks and
    # the living slots UNCHANGED (win 1 at 246, team 1 [1, 2, 3]); field_digest,
    # tick_digest and the unit stream move. SOLE CAUSE, verified on the same
    # tree: with `k_expose = 0` (the law dormant) this match gives exactly the
    # G3p pin below (5175fd4c / e4e42ce9 / 6e30ef3f) --
    # tests/test_exposure.py::test_with_the_law_dormant_the_pinned_match_is_
    # the_pre_g4p_match holds that. Per-tick trace (dormant vs shipped): the
    # unit state is identical through tick 197; on tick 198 zombie 11 carries
    # a kinetic exposure from a bot's round (the record's new key), the 22
    # bullet hits of the match then hit it harder, and it dies (to the fire,
    # as before) on tick 233 instead of 236; every other kill is on the same
    # tick by the same cause. Measured on the CPU, CUDA and resident backends
    # alike (tools/match_pin_check.py).
    # (was field 5175fd4cddd17c018e84bc9548b1b81b5a92ebac0d1c25ba06c3b14528
    #  bd6175; tick e4e42ce9320d57c1782bc625289e3429937300e1fbebd767a0440c90
    #  99e53076; units 6e30ef3f7df49f556d8d0b1e43bc55d2499303302095935a26b522
    #  385e7d1cfe)
    outcome="win", team=0, ticks=265,
    # #31 x #169 re-pin (2026-11-06, the merge of the two): each re-pinned this
    # match alone; together outcome, team and ticks stay win 1 at 722 and the
    # three digests move. SOLE CAUSE, verified by a per-tick trace of the #31
    # tip 4b31915 against the merged tree: unit states and events identical
    # through tick 717; on tick 718 a killing burst's next round lands on the
    # corpse of unit 7 instead of flying on into unit 0 (the #169 rule, in the
    # frag-shaped match #31 made), and units 4 and 7 trade shots in the slot.
    # (was field b6ab1595011cfe96ff4297f12b2d7c2e062a43e58bd7e9fe09b7bdf71c740824;
    #  tick d017f087185330c123d5bc81ccc06b6bb7077d28105f3c884404ec834bada3ac,
    #  units 9abe08854b5ea8bebc02cdfb3f4f56c5804f165f9f9364248183dab833dd36cc)
    # #183 G1p re-pin (2026-11-10, the turn-rate cap, Erik's ruling: aim is
    # not immediate; [marine] turn_rate_deg_per_s = 360, 15 deg a tick).
    # Outcome, team and ticks UNCHANGED (win 1 at 722); field_digest,
    # tick_digest and the unit stream move. SOLE CAUSE, verified on the same
    # tree: with the cap at half a turn per tick (4320 deg/s, the instant
    # facing) this match gives exactly the pre-G1p values below --
    # tests/test_turn_rate.py::test_at_half_a_turn_per_tick_the_pinned_match_
    # is_the_pre_g1p_match holds that -- so the throttle (bots walk at 0),
    # the bots' trigger discipline and the format bump move nothing; the bots
    # now turn toward a target over several ticks and fire once within their
    # 3 deg spread, so rounds leave later and along other facings.
    # (was field da05242a99122428c903ad8a96e9373a0e12006c03abdbec1e849f4aca7dc4b0;
    #  tick f35ed2beadb77141ccea20d944a70a66e93c70818ae4c44bbf6ba1edc01024b3;
    #  units 4408c8740425593b0d47d35e7d0658c4efbd62caed87f5fd69a9e8f894ac02dd)
    field_digest="ab499f298a66d2dfe4b0b3134b67ebaa663e4c95d7b7a5791954eb90eb8df573",
    # #123 re-pin (credited events): the last tick holds the winning bullet's
    # UnitHit + UnitKilled, whose hashed record gained `attacker_id` (2, the
    # shooter). SOLE CAUSE, verified on the same build: hashing the record
    # without that key reproduces c69925fa... exactly; outcome, ticks,
    # field_digest and the per-tick unit stream are unchanged.
    # (was c69925fa68e8ba10fbb724eaedbcb53d56058f3b4efaf91a2601d1366be0f170)
    # #169 re-pin (2026-11-06, simultaneous shot resolution): outcome, team,
    # ticks and field_digest UNCHANGED. SOLE CAUSE, verified by a per-tick
    # trace against main 0ef8966b: ticks 0-685 identical; from tick 686 the
    # rounds of a killing k5 burst stop on the body they killed this tick
    # (they used to fly on into the unit behind it) -- tick 686's 5th round
    # lands on unit 2's corpse instead of unit 1, so unit 1 dies on tick 690
    # one round later and the rest of that burst lands on its corpse.
    # Regenerate: <breach py> -m pytest tests/test_zombie_control.py -k
    # pre_122_ship_demo -vv (the diff prints both values).
    # (was tick ce979185366be487732e587a851ab6e3589d0b0ea8671ab06ef7c5f41ba25c37,
    #  units b55cc54e4e12bd7002a70cecd01d137703964eb73fa63c15da59f0b415cedb4e)
    # #183 G5p-a1 re-pin 1 (2026-11-10, the turning law's representation and
    # routing at UNLIMITED acceleration, a >= 2w; see the box pin above).
    # Outcome, team, ticks (win 1 at 246) and field_digest UNCHANGED;
    # tick_digest (the last tick's unit records: the int facing, the turn
    # key) and the unit stream move. SOLE CAUSES, verified on the same
    # tree (a scratch proof over the five pinned matches): with (1) every
    # non-aim facing writer (travel, zombie walk, attack snap, WEGO,
    # onephase) placing the facing at once in its own slot again AND (2) the
    # old representation (Unit.facing the float pi/2 until first written, the
    # record hashing q(float facing), no turn / omega keys), this tree gives
    # EXACTLY the previous pin; with (2) alone undone, outcome, ticks,
    # field_digest and tick_digest are the previous ones and only the unit
    # stream moves (the zombie walk's facing lands one tick later: its target
    # is set in slot 5 and turned in the next tick's slot 1a).
    # Measured on the CPU, CUDA and resident backends alike.
    # (was tick 948fde5ffe4c509fa0d971a145304e5867c6f8b3feeaf12d3c52bbec754f2ccf,
    #  units 15438ec250cd193b2bfa7edd184ce46c109428ba4669891e0cc28a0a81f05015)
    # #183 G5p-a1 re-pin 2 (2026-11-10, the shipped turning acceleration,
    # 30 rad/s^2, PLACEHOLDER: a turn now speeds up and brakes, 90 deg in 11
    # ticks, 180 in 17). The WINNER CHANGES: win 0 at 265 (was win 1 at 246);
    # all four digests move. SOLE CAUSE, verified on the same tree: with
    # `[marine] turn_accel_rad_per_s2` >= 2w (1000, unlimited) this match
    # gives exactly re-pin 1's values -- tests/test_turning_law.py::
    # test_g3_at_unlimited_acceleration_the_pins_are_re_pin_1s holds that.
    # Per-tick trace (re-pin 1 vs 2): positions differ from tick 0 -- the
    # bots aim where they walk, and from the spawn facing (north) their
    # facing now reaches the walk later, so the Speed table's direction rule
    # holds them at the sideways factor a few ticks longer; the zombies'
    # first bites land on other ticks (re-pin 1 bit units 3 and 6 on tick
    # 193); the frag kills four on tick 220 (units 1, 2, 8, 9; re-pin 1 three
    # on 218), and team 0's survivors then kill 4, 5, 6 and 7 (ticks 244 ..
    # 264). Measured on the CPU, CUDA and resident backends alike.
    # (was win 1 at 246; field 4a30c340a0982a80080a1d29f85cdc377bb579e9ff85c6
    #  f685699c86fe725440; tick 6e408f1955d11e567974b29252a93ecb080cff000e6d
    #  38a7d9653476d095267a; units c792732fc27000f1adf5d04c223c598022016368c9
    #  dbd81c79ffba31bac3e5b5)
    # #183 G5p-a2 re-pin 1 (2026-11-10, the unit impulse law built with every input at zero -- [impulse]
    # bullet_push_mult = recoil_mult = 0, no shove -- and the MOVE_DIR walk
    # on the Q16 grid (the step is the speed law's rational rounded ONCE to
    # Q16 tiles per tick, critique B M6)). Outcome,
    # team, ticks (win 0 at 265) and field_digest UNCHANGED; tick_digest and
    # the unit stream move (the walking bots' sub-tile positions). SOLE CAUSE, verified on the same tree: with the pre-G5p-a2 float walk
    # step reinstated (a scratch monkeypatch of Simulation._step_move_dir)
    # this run gives EXACTLY the previous values -- slot 1a's integration,
    # slot 3's displacement loop, the effort member, the digest keys, the
    # events and the door change nothing while every input is off.
    # tests/test_unit_impulse.py::test_g1_with_every_input_off_nothing_is_
    # pushed_and_the_pins_are_re_pin_1s holds that the law is dormant.
    # Measured on the CPU, CUDA and resident backends alike.
    # (was tick f0f3a30ae5757a5833f00ce6f21fbdd4dcc8fad1fb20bf121ebfe3e1b44e7952,
    #  units 1b3b013ff5a33f0802ec853f1c123d1ab6aa2b3a72a4ecdccdf33904277a131a)
    # #183 G5p-a2 re-pin 2 (2026-11-10, hits push and guns recoil: [impulse] bullet_push_mult =
    # recoil_mult = 1.0, the RULED real momentum -- a carbine round 3.6 N s on
    # the target, its recoil on the shooter).
    # Outcome, team, ticks (win 0 at 265) and every kill (tick, unit, cause)
    # UNCHANGED; field_digest, tick_digest and the unit stream move. SOLE CAUSE, verified on the same tree: with both
    # multipliers at 0 this run gives exactly re-pin 1's values
    # (2ca1d895 / ec0dad03) --
    # tests/test_unit_impulse.py::test_g1_with_every_input_off_nothing_is_
    # pushed_and_the_pins_are_re_pin_1s holds that. Per-tick trace (dormant
    # vs shipped): identical through tick 204; on tick 205 the first round
    # lands (unit 6 recoils, zombie 11 is struck); no unit is ever displaced
    # (pv stays 0: a carbine or two is soft input), the effort (peak 0.32 of
    # the hold) slows the shooters' and the struck zombies' walks by a few
    # per cent, so positions and the field drift. Measured on the CPU, CUDA
    # and resident backends alike.
    # (was field 65fd96904dfa3cc25edc0fc344027876be1b6c11b824c92c6b5930bb3c9c3499,
    #  tick 2ca1d895b11749e6b16f281bdf976379e52077e1d29929284683648f8369c6a4,
    #  units ec0dad03ce0b0ef6389589f50b24728dc074a5f81a9340549b44d86e3338fd35;
    #  the box is UNCHANGED: its idle marines never fire)
    tick_digest="2620c39692177c6fe900df314650de1bf9b932bd1667224e5145c9a73068992b",
    unit_stream="badb4a7984486a23519cdbbc7b85b72b8b8fb2953e3a02811ec8b7083fb1c922")


def test_the_wrapped_law_plays_the_pre_122_match_on_a_small_box(tmp_path):
    """Property: idle marines + 3 zombies under the "zombies" controller give
    the pre-#122 result, final fields and per-tick unit stream. Breaks if
    the controller changes the law's unit list, its order or its slot in the
    tick, drives a zombie twice, or skips a tick."""
    stream = _UnitStream()
    o = run_match(_box(tmp_path, "box"), on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               unit_stream=stream.h.hexdigest())
    assert got == _PRE_122_BOX


def test_the_wrapped_law_plays_the_pre_122_ship_demo_match():
    """Property: ship_demo seed 1 (bots vs bots + 4 zombies) ends exactly as
    the conductor-driven law plays it (re-pinned at #140, then at #31 and its merge with #169, then
    at #183 G1p's turn-rate cap, then at G2p's carbine rate, then at
    G1p-c's speeds, then at G3p's throw wind-up and G4p's exposure, then at
    G5p-a1's turning law: win team 0 at tick 265) with the same
    tick_digest. Breaks like the box test, on
    the real level."""
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    stream = _UnitStream()
    o = run_match(level, on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest, unit_stream=stream.h.hexdigest())
    assert got == _PRE_122_SHIP_DEMO


def test_bots_and_zombies_one_seed_one_digest(tmp_path):
    """Property (the seed gate with the horde in it): two runs of one seed
    give one digest; the zombie controller draws nothing of its own. Breaks
    if ZombieControl reads anything unsynced or iterates an unordered set
    into the law."""
    bots = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                     controller="bots")
    lvl = _box(tmp_path, "seeded", replace(BOX_SPEC, teams=(bots, bots)))
    a, b = run_match(lvl), run_match(lvl)
    assert (a.tick_digest, a.result) == (b.tick_digest, b.result)


# ---------------------------------------------------------------------------
# (b) one driver per zombie per tick
# ---------------------------------------------------------------------------
def test_each_zombie_is_driven_once_per_tick_in_a_match(tmp_path,
                                                        monkeypatch):
    """Property: under the match path the zombie law is entered at most ONCE
    per tick (never by the conductor AND the controller), and every living
    zombie is in that call exactly once. Breaks if the conductor's slot keeps
    running for a match, if ZombieControl calls the law itself, or if a
    request names a zombie twice into the law."""
    calls = Counter()
    seen = []
    real = sim_mod.update_zombies_tick

    def counting(gmap, units, tick, events=None, **kw):
        calls[tick] += 1
        ids = [u.id for u in units if u.is_zombie and u.alive]
        seen.append((tick, ids))
        return real(gmap, units, tick, events=events, **kw)

    monkeypatch.setattr(sim_mod, "update_zombies_tick", counting)

    class _Twice(ZombieControl):
        """Asks twice a tick: the second request must not drive again."""
        def handle_frame(self, sim, renderer=None):
            super().handle_frame(sim, renderer)
            super().handle_frame(sim, renderer)

    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "zombies_twice",
                        _Twice)
    living = {}

    def watch(sim):
        living[sim.tick - 1] = sorted(u.id for u in sim.units
                                      if u.is_zombie and u.alive)

    run_match(_box(tmp_path, "once",
                   replace(BOX_SPEC, zombie_controller="zombies_twice")),
              on_tick=watch)
    assert calls and max(calls.values()) == 1
    for tick, ids in seen:
        # The law saw each zombie living at the tick's start exactly once.
        if tick - 1 in living:
            assert sorted(ids) == living[tick - 1], (tick, ids)
    # ... and no tick with a living zombie went undriven.
    for t, ids in living.items():
        if ids and t + 1 in living:
            assert calls[t + 1] == 1, t + 1


def test_drive_zombies_is_refused_on_a_game_path(tmp_path):
    """Property: without a match's hand-over the conductor drives the horde,
    and the ZOMBIE intent is refused -- a second driver is never silently
    added. Breaks if drive_zombies is accepted on a game path."""
    sim = Simulation(_box(tmp_path, "game"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    with pytest.raises(RuntimeError):
        sim.drive_zombies([0])


# ---------------------------------------------------------------------------
# (c) credited bites; a converted marine joins the controller's horde
# ---------------------------------------------------------------------------
def test_zombie_kills_are_credited_and_the_dead_join_the_horde(tmp_path):
    """Property (scope §5 item 4's gate): in a match with idle marines and
    the zombie controller, every death is a UnitKilledEvent credited to
    melee, the dead marines convert (ZOMBIE_TEAM, slot kept in their old
    roster) and are thereafter MOVED by the zombie controller while their
    old team's controller emits nothing. Breaks if the bite leaves the
    event list, a converted marine is left undriven, or its slot moves."""
    kills = []
    track = {}

    def watch(sim):
        for ev in sim.tick_events:
            if isinstance(ev, UnitKilledEvent):
                kills.append((ev.unit_id, ev.killed_by))
        for uid in {k for k, _ in kills}:
            u = sim.get_unit(uid)
            if u.alive and u.is_zombie:
                track.setdefault(uid, set()).add((u.x, u.y))

    lvl = _box(tmp_path, "credit", replace(BOX_SPEC, time_limit_s=30.0))
    o = run_match(lvl, on_tick=watch)
    assert kills, "the horde killed nobody"
    assert all(src == "melee" for _, src in kills), kills
    assert o.converted_alive >= 1
    # Each converted marine walked after it turned (team 0's controller is
    # idle: only the zombie controller can have moved it).
    moved = [uid for uid, pos in track.items() if len(pos) > 1]
    assert moved, track


def test_converted_marine_keeps_its_roster_slot(tmp_path):
    """Property: conversion (where MatchRules runs it) moves the unit to
    ZOMBIE_TEAM but its slot stays in the old team's roster. Breaks if the
    hand-over re-rosters converted units."""
    lvl = _box(tmp_path, "slot")
    rules_sim = {}
    run_match(lvl, on_tick=lambda s: rules_sim.setdefault("sim", s))
    sim = rules_sim["sim"]
    roster0 = sim.roster(0)
    assert [u.slot for u in roster0] == list(range(len(roster0)))
    converted = [u for u in roster0 if u.is_zombie]
    assert converted and all(u.team == ZOMBIE_TEAM for u in converted)


# ---------------------------------------------------------------------------
# (d) an idle horde stands
# ---------------------------------------------------------------------------
def test_an_idle_zombie_controller_leaves_the_horde_motionless(tmp_path):
    """Property: ``zombie_controller = "idle"`` (written through the writer,
    read through the loader door) gives the horde to IdleControl, so no
    zombie moves, activates or bites. Breaks if the conductor still drives
    the horde in a match, or the key is dropped at the door."""
    spec = replace(BOX_SPEC, zombie_controller="idle", time_limit_s=5.0)
    lvl = _box(tmp_path, "passive", spec)
    assert lvl.match.zombie_controller == "idle"
    text = (tmp_path / "passive" / level_loader.MATCH_FILENAME).read_text()
    assert 'zombie_controller = "idle"' in text
    start = {}
    end = {}

    def watch(sim):
        for u in sim.units:
            if u.is_zombie:
                start.setdefault(u.id, (u.x, u.y))
                end[u.id] = (u.x, u.y, u.zombie_activated)
    o = run_match(lvl, on_tick=watch)
    assert start and all(end[i][:2] == start[i] and not end[i][2]
                         for i in start)
    assert o.result.ticks == round(spec.time_limit_s * 24)
    assert o.converted_alive == 0


def test_the_default_zombie_controller_leaves_match_files_unchanged():
    """Property: the default (``zombies``) is not written, so the checked-in
    match files are byte-identical to the writer's text. Breaks if the
    writer starts emitting the default key (every generated level would
    change) or the loader's default drifts from the writer's."""
    for name in ("ship_demo", "playground"):
        level = level_loader.load(name, str(ROOT / "levels"))
        assert level.match.zombie_controller == "zombies"
        path = ROOT / "levels" / name / level_loader.MATCH_FILENAME
        assert level_lib.format_match_text(level.match).encode() == \
            path.read_bytes().replace(b"\r\n", b"\n")


# ---------------------------------------------------------------------------
# (e) the registry
# ---------------------------------------------------------------------------
def test_the_registry_builds_the_zombie_controller(tmp_path):
    """Property: ``zombies`` is a registry row building a ZombieControl for
    ZOMBIE_TEAM; an unregistered zombie_controller is refused at the door.
    Breaks if the row is dropped or the door stops checking the key."""
    assert "zombies" in control_source.registered_controllers()
    c = control_source.create_controller("zombies", ZOMBIE_TEAM)
    assert isinstance(c, ZombieControl)
    with pytest.raises(ValueError):
        control_source.create_controller("zombies", 0)
    with pytest.raises(ValueError, match="zombie_controller"):
        _box(tmp_path, "bad", replace(BOX_SPEC, zombie_controller="nobody"))


# ---------------------------------------------------------------------------
# (f) the game path keeps the conductor's slot
# ---------------------------------------------------------------------------
def test_the_game_path_still_runs_the_conductor_s_zombie_slot(tmp_path):
    """Property: a WEGO sim (the shipped default ruleset) with a zombie and a
    marine and NO controller: the zombie still activates and walks -- the
    conductor drives it. Breaks if the slot's guard ever leaves the horde
    to a controller outside a match."""
    sim = Simulation(_box(tmp_path, "wego"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=4, y=10, team=0, footprint=3)
    z = Unit("Z", x=16, y=10, team=ZOMBIE_TEAM, footprint=3)
    sim.add_unit(m)
    sim.add_unit(z)
    sim.set_paused(False)
    x0 = z.x
    for _ in range(48):
        sim.set_paused(False)
        sim.step()
    assert z.zombie_activated
    assert z.x < x0
