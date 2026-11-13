"""#122 -- the horde through the control seam (``control_zombies.ZombieControl``,
match controller ``"zombies"``).

Properties: the controller drives the conductor's law unchanged (a match
plays bit for bit as the conductor-driven law plays it -- the golden-style
pins below, deliberately); every zombie has one driver per tick; a bite's
kill is credited to melee and the converted marine is then driven by the
zombie controller; an idle zombie controller leaves the horde standing; the
registry builds ``zombies``; the game path (no match) still runs the
conductor's zombie slot.

#183 G9p REWROTE the law (the approved rewrite, residency P3d: the horde walks
the integer distance field of ``simulation.horde``; Erik's ruling of
2026-11-10). The pre-G9p "wrapped, never rewritten" property -- the controller
reproduces the pre-#122 A* law bit for bit -- is superseded: there is no
pre-#122 law left to reproduce. What survives is kept here: one driver per
tick, the controller names every living zombie, a bite is credited ``melee``
to the biter, the converted marine joins the controller's horde. The two
pins are the field law's matches (re-pinned with the rationale beside them;
their names are historical).

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
# (a) the conductor's law through the controller -- the pinned matches
# ---------------------------------------------------------------------------
# Captured at the start of #122 (main 5ee2712, the conductor driving the
# horde inside Simulation.step) with tools/run_match.py; the law itself was
# rewritten at #183 G9p (the horde field) and re-pinned there. A change of
# the zombie LAW (or of anything else these matches touch) moves these --
# then re-pin deliberately, with the rationale, never to make a test pass.
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
    # #183 G5p-b (2026-11-10, the wind and the blast onto the unit impulse
    # law): UNCHANGED, the calm match -- no wind at any unit here is a force
    # the law answers (no frag, no vent), and the pre-G5p-b tree with #140's
    # wind law off gives the same values too.
    # #183 G9p STAGE 1 re-pin (2026-11-11, the horde field law, residency
    # P3d: each awake zombie descends one integer BFS field flooded from
    # every living non-zombie over the footprint-passable mask, refreshed
    # every [zombie] field_refresh_ticks = 2 and on a topology change; no
    # A*; ties by the cell's parity order). A REWRITE: there is no "old =
    # new with the input off" proof. What was proved instead (a scratch
    # per-tick trace of the pre-G9p law, cb884493's ai_zombie.py, against
    # this tree): outcome, team, ticks and every kill (tick, unit, cause)
    # UNCHANGED (win 1 at 130; melee kills of units 1 and 0 on ticks 65 and
    # 130); every activation tick UNCHANGED (zombies 4, 5, 6 on tick 1, the
    # converted 1 on 65 and 0 on 130); the unit states identical through
    # tick 39 -- on tick 40 zombie 6 steps south where A*'s 8-connected path
    # held its row (the field is 4-connected). So field_digest and the unit
    # stream move, from the routes alone.
    # (was field a672c7249e18d0eebbb8e398b696bd9c3d34417445f66c3cab498ec7c
    #  d783ab1; units f2ba375dd16868470f118df2bbbcab6c151ce28d4e2e108ab5f443
    #  3360406497)
    # #183 G9p STAGE 2 re-pin (2026-11-11, the behaviours on the field:
    # [zombie] wander, hearing, fear_fire, bash_doors and sprint, all
    # PLACEHOLDERS). Outcome and team UNCHANGED (win 1); ticks 130 -> 92;
    # the kills (both melee, units 1 and 0) come on ticks 46 and 92 (were 65
    # and 130). SOLE CAUSE: the SPRINT -- with only it on the box gives
    # exactly these ticks and kills, with each of the other four alone it
    # is the stage-1 match (a scratch attribution run); with all five off
    # it is the stage-1 pin bit for bit (tests/test_horde_behaviours.py::
    # test_with_every_behaviour_off_the_horde_plays_the_stage_1_law). The
    # three zombies see the idle marines from tick 1 within 8 m and close at
    # 1.15 m/s (21 ticks per 1 m tile) instead of the 0.6 m/s shamble; their
    # positions first differ on tick 21.
    # (was ticks 130; field 0fdc79f228cc2e64ca6936cf5a2ec41c52642adfca545cbe
    #  51324cfad2e21631; units 05bdbac7c76250b949ef35aef81fdd403bd89af88232e
    #  ebe061bf753e47b1e3d)
    # Erik's zombie bite balance re-pin (2026-11-12: [zombie] melee_damage
    # 60 -> 34, attack_cooldown_seconds 1.0 -> 2.2, hp 400 -> 100; three
    # bites kill a 100 HP marine, docs/zombie_bite_balance_2026-11-12.md).
    # Outcome and team UNCHANGED (win 1); ticks 92 -> 231: the two melee
    # kills (units 1 and 0) come on ticks 106 and 231 (were 46 and 92), each
    # marine now taking three bites. SOLE CAUSE: this tree with the three
    # old values set back in-process gives the old pin bit for bit (outcome,
    # ticks, field digest, unit stream).
    # (was ticks 92; field 2a9339a53e67d28c29ff513f6942dab91586f5e85c6eb552
    #  885554c16ebe4c3b; units 85d051afa1acda1b9bc1d3a680e6373632bfdb9f735b
    #  65bf51ec1a9bbf834238)
    outcome="win", team=1, ticks=231,
    field_digest="2cf3e8481a3792bce622f72c0b5cc0a2"
                 "68f19f856903a7313440ed21573faf5f",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was e082f7dfe103e2fdd594a952da81143ca4202e996238969b22c85eea05b0fd3d)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was ac9bda6c239304078bef33858e2582f883e0684eddd14d541ec7f269e8a88c16)
    # Erik's zombie bite balance re-pin (2026-11-12, see the box's ticks).
    unit_stream="3a6b230941d73d3b6f325af378ffe67b"
                "44a86953bbb4d46736245fbf4caa31ee")
_PRE_122_SHIP_DEMO = dict(
    # Merge of main 37142802 into #183 (2026-11-13; main's #29 P3b keyed
    # Philox draws had moved this pin): re-pinned on the merged tree with
    # Erik's bite values -- outcome and team unmoved (win 0); ticks 2708 on main (2881 on
    # the pre-merge branch) -> 2570; the digests and the unit stream
    # move (each marked below). SOLE CAUSE: on the merged tree with the
    # three old [zombie] values set back in-process main's version of this
    # test file passes whole, i.e. main's pin comes back bit for bit. The
    # "now" values in the 2026-11-12 bite comment are the pre-merge branch's.
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
    # #183 G9p STAGE 1 re-pin (2026-11-11, the horde field law; see the box
    # pin above for the law). The WINNER CHANGES: win 1 at 264 (was win 0
    # at 266); all three digests and the unit stream move. A REWRITE, so
    # proved by what stays (a scratch per-tick trace of the pre-G9p law,
    # cb884493's ai_zombie.py, against this tree): the unit states are
    # identical through tick 194 and every spawned zombie's activation tick
    # is UNCHANGED (8, 9, 10, 11 on tick 183, by sight and the chain); on
    # tick 195 zombies 9 and 10 take their first different steps (9 east to
    # x = 46 where A* held 45, 10 north to y = 18), the bots' rounds then
    # land on other bodies, and the frag of ticks 218-221 kills units 1, 2
    # and 8 as before but not 9; zombie 10 never bites unit 4 (the old law's
    # one melee kill, tick 222), so nobody converts, unit 4 dies to a bullet
    # on tick 227, units 5 and 6 to bullets on 244-245, and team 1's frag
    # kills unit 0 on tick 264. Measured on the CPU, CUDA and resident
    # backends alike (tools/match_pin_check.py).
    # (was win 0 at 266; field 1879b8965a72968467a9cfdb721d92982ab641cf1a86
    #  d5611fe02a733aa13cac; tick 3bd9fb40d7a71327c4e8b00217862c3c119d110d
    #  23e98db53f0be02e74975d8c; units 3262ff82f251fe64443ab17c59be0ad6680f
    #  30f6308e3e0ad34ff32c20631cd9)
    # #183 G9p STAGE 2 re-pin (2026-11-11, the behaviours on the field: see
    # the box pin). The WINNER CHANGES: win 0 at 264 (was win 1 at 264);
    # all three digests and the unit stream move. With all five behaviours
    # off this match is the stage-1 pin bit for bit (tests/test_horde_
    # behaviours.py), so the behaviours are the whole move. Per-tick trace
    # (all off vs shipped): the records differ from tick 1 (each dormant
    # zombie's wander anchor, an absence-transparent key) and the positions
    # from tick 28 (zombie 9's first wander step); activation is UNCHANGED
    # for the spawned horde (8-11 on tick 183). Then the sprinting horde
    # reaches the bots: three melee kills on ticks 204, 211, 214 (units 6, 3,
    # 4, all converted), the frag of ticks 220-223 kills 1, 2, 8, 9, 10, 11,
    # and the converted 3 / 4 / 6 bite units 5 and 7 on ticks 263-264 --
    # team 0 (unit 0) is the last marine team standing. Attribution (each
    # behaviour alone, others off): the sprint alone ends win 1 at 250 with
    # two melee kills; the wander alone win 1 at 246; the fear of fire alone
    # moves three kills (7 dies to the fire on 229); hearing alone and door
    # bashing alone leave the outcome, the end tick and every kill of the
    # stage-1 match (ship_demo has no door entity).
    # Measured on the CPU, CUDA and resident backends alike
    # (tools/match_pin_check.py).
    # (was win 1 at 264; field d9c4c5cef6c04f9d80eaa2b368d88782a8fa28132f7cd
    #  39a8011c230b88cecde; tick f6b702b5550299dcebfe2db8794a1e310b228648319
    #  b7c668b3a4c8ac1920f33; units d83ee4d35c56f376d47daec6179ac3c37dfef360c
    #  d149d7ef6d48c6efa3a9f3f)
    # #183 G8p re-pin (2026-11-11, the bot pass: `bots` is the layered bots,
    # control_bots.BotTeam; the first bots live on unchanged as `bots_v1`).
    # The WINNER CHANGES: win 1 at 2721 (was win 0 at 264); team 1 ends with
    # slots [2, 3] alive, team 0 and the horde with none; all three digests
    # and the unit stream move. A NEW CONTROLLER, so proved by what stays:
    # with every team on `bots_v1` this match is the pre-G8p pin below bit
    # for bit -- outcome, ticks, field digest, tick digest and unit stream
    # (tests/test_bots_v1.py::test_bots_v1_play_the_pre_g8p_pinned_match,
    # permanent) -- so the controller is the whole move. The new match: the
    # horde bites units 4 (team 1) and 0 (team 0) on ticks 423 and 454 (both
    # convert); the bots' frags (12 thrown, a lone zombie too) kill zombies
    # 10 and 11 (586), the converted 4 (683), zombies 9 and 8 (714) and the
    # converted 0 (920); then the squads fight it out: unit 1 to a bullet
    # (1903), unit 5 to a frag's heat (1925), units 3 and 2, team 0's last,
    # to unit 6's bullets (2638, 2721). 93 rounds, 2 shoves, no flashbang
    # (no enemy was known out of sight while the team bounded).
    # Measured on the CPU, CUDA and resident backends alike
    # (tools/match_pin_check.py).
    # (was win 0 at 264; field a7f9fd78c8a13c2bf9fc38657b05eae042993affcd85b
    #  c777493724808fe2b8b; tick 8df76cf7f0a3681c583c99e10f31624d8803f33c4ec
    #  d3adb3fdd0535e5a4cf46; units 1082d3b490f52183e4f047ed136219ecf2e5367
    #  0e946b20a63ed989d46065a76)
    # #195 1d step (ii) re-pin (2026-11-12, the honest team picture and the
    # fire lanes from state, docs/rl_agent_rung1_note_1d_2026-11-12.md 4: a
    # record keeps the class, side and place the team saw and outlives a
    # death the team did not see for its memory time; the layered bots'
    # lanes are every teammate holding its trigger, along its facing to its
    # weapon's range). Outcome and team UNCHANGED (win 1); ticks 2721 ->
    # 2902; team 1 ends with slot [3] alive (was [2, 3]); all three digests
    # and the unit stream move. SOLE CAUSE, verified on the same build:
    # step (i)'s last commit (main 38b5a54b) reproduces the old pin bit for
    # bit, and THIS tree with only the record rules put back to step (i)'s
    # (class and side refreshed from the live unit, a dead unit's record
    # dropped at once; a scratch monkeypatch) does too -- so the state lanes
    # move nothing in this match and the records are the whole move.
    # Per-tick trace (old vs new): unit states identical through tick 920
    # and every kill up to then the same (bites of 4 and 0 on 423 / 454, the
    # frags' kills of 10, 11, 4, 9, 8 and of the converted 0 on 920); on
    # tick 921 they part -- team 0 never saw the converted 0 die to team 1's
    # frag (last seen on 888), so it keeps that record, and its bots keep
    # reading an enemy there. The squads' fight then runs later: units
    # 2 (2500), 5 (2560), 1 (2610), 6 (2896) and 3 (2902) to bullets; 109
    # rounds (was 93), 16 throws (was 12). CPU only (no CUDA build on this
    # checkout).
    # (was ticks 2721)
    # #29 P3e re-pin: the integer match path moved the game -- team 1 at 2902
    # ticks becomes team 0 at 2966 (one seed; the A/B over 24-72 seeds is in
    # docs/p3e_ab_2026-11-12.md and within seed noise).
    # Erik's zombie bite balance re-pin (2026-11-12: [zombie] melee_damage
    # 60 -> 34, attack_cooldown_seconds 1.0 -> 2.2, hp 400 -> 100 -- the hp a
    # TURNED marine comes back with; docs/zombie_bite_balance_2026-11-12.md).
    # Outcome and team UNCHANGED (win 0); ticks 2966 -> 2881; all three
    # digests and the unit stream move. SOLE CAUSE: this tree (main 321babcf
    # merged in) with the three old values set back in-process gives the
    # P3e pin bit for bit (tools/match_pin_check.py: EQUAL; field, tick
    # digests and unit stream too). The new match: the horde's first bites
    # no longer kill on ticks 423 / 454 (two bites at 34 do not); one melee
    # kill (unit 4 by zombie 11, tick 673) where there were two; the frags'
    # heat kills zombies 8 and 9 (457), 10 and 11 (719) and the converted 4
    # (1019); then the squads: units 0 (1160), 6 (1576), 7 (2615) and 5
    # (2881) to bullets -- team 0 (slots 1, 2, 3) stands.
    # (was ticks 2966)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites: every draw
    # keyed (seed; tick, index, salt, id) in place of one numpy stream).
    # Outcome and team UNCHANGED (win 0); ticks 2966 -> 2708. SOLE CAUSE,
    # verified on the same tree: tests/_p3b_sole_cause.py (main's stream
    # words through the keyed sites) gives the P3e pin back bit for bit --
    # outcome, ticks and all three digests; A/B over 24-72 seeds in
    # docs/p3b_ab_2026-11-12.md.
    # (was ticks 2966)
    outcome="win", team=0, ticks=2570,
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
    # #195 1d step (ii) re-pin (see above).
    # (was 58a67f4473cc3fe4ffb37123259f5fa0a8c0ca820be9179218404499b9749c04)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 0afdec8bfef8943048652975d92945c2fafdf0a6751a900ccbf8a1c8702cc617)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 5f8cdd8bbfdb797e58dd163e35a368fabcaa541e106402703fa4893cc7f3e136)
    # Erik's zombie bite balance re-pin, main 37142802 merged in (2026-11-13).
    # (was 13168aaad034fb1f8365ac184244453d7dc8ef6168b2438b48101b492484b89d)
    field_digest="3f82af18a8669a5dcf99ec75a8fd38f7eb7cc5f8f7d49c7fe1a42b5888b334bc",
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
    # #183 G5p-b re-pin (2026-11-10, the wind and the blast onto the unit impulse law: the dynamic
    # pressure at each footprint is a force q * drag_area (0.9 m2
    # standing, 0.15 prone) through unit_impulse.add_wind; #140's
    # cadence scale and bespoke knockdown DELETED in the same commit).
    # Outcome and team UNCHANGED (win 0); ticks 265 -> 266; all three
    # digests and the unit stream move.
    # SOLE CAUSE, verified on the same build: the pre-G5p-b tree
    # (0b57fc6e) with #140's wind law switched off (wind_q_slow_pa /
    # wind_q_fall_pa = 1e15 / 2e15) and this tree with the wind's
    # input off ([impulse] wind_push_mult = 0) give exactly the same
    # values (win 0 at 266, b16e9f79 / 14a6c85a / 79cad822) -- replacing the law is the whole change.
    # Per-tick trace (shipped vs wind off): identical through tick 217;
    # every push lands within 13 ticks of the frags of ticks 218-221 (pv
    # non-zero on units 3, 4, 6, 7, 10, 11 in ticks 222-233, at most 0.39
    # tile, 13 cm), and the law floors one unit (3, ticks 222-230) where
    # #140's law floored six (3, 4, 6, 7, 10, 11) on q alone: a blast's
    # one-tick wind is a small impulse. With zombie 10 left standing, it
    # bites unit 4 on tick 222 (the old law had it down); the match is then
    # won by team 0 one tick later. Measured on the CPU, CUDA and resident
    # backends alike (tools/match_pin_check.py).
    # (was ticks 265; field ab499f298a66d2dfe4b0b3134b67ebaa663e4c95d7b7a5
    #  791954eb90eb8df573; tick 2620c39692177c6fe900df314650de1bf9b932bd16
    #  67224e5145c9a73068992b; units badb4a7984486a23519cdbbc7b85b72b8b8fb2
    #  953e3a02811ec8b7083fb1c922)
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 3344bc96528bb08756456338112f9f25163bdacc56352be453f4debc5a172bf4)
    # #195 1d step (ii) re-pin (see above).
    # (was 0b093a26a96d2c39775240fbc0514fe9f869600026cc5c977e50f5d899294ec1)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 1030e119469b9e9abeaebd504ab5b59c9ae0fac090fe4d04c566bb6488633d54)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 402970f74f8e71af449d04cd8c245de8bd4a3069771ca31529506fe08a1b6dad)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 4e6bc98db44f5be0548ca91969e822294bacd7ac45a360d594f793ffac205509)
    # Erik's zombie bite balance re-pin, main 37142802 merged in (2026-11-13).
    # (was dd6fa61812db62e6dfea7c4e8993370e5ca59789f78edbd7b7f2411bb3b127b0)
    tick_digest="e1a9413a03e8ab185ea66652f3d467549533a3a34bfd68ad93ee1392fb76a0cb",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was bf34f1bbf43f0cb555dd65c57c73d72cd3c56edbacd0a8753e4a6844ee308ab8)
    # #195 1d step (ii) re-pin (see above).
    # (was 869f75cbb8da51b410991c77dbbdeaed0566e7a6dbeb444318c0b0664416bd6d)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 16b425f124d6fcc66b409ff2f0433fb59540b7dbab273f0fc1dfe69c7f1706d7)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 537ec052b34d2e9b52887299ddf38162ae47f4efb3acc51f7df3ffb3175742ca)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 8c7b51885acc094fa06c4fdcc2f6ee315828d1970a75a2efe14848b3d83999ba)
    # Erik's zombie bite balance re-pin, main 37142802 merged in (2026-11-13).
    # (was d30d0069d9a3468be072ce0799cbf1e5fcdee98b774668c9fe4eda227343be64)
    unit_stream="8548eb1e2f94f9a501953615e24ad7bdb23d2922f7a11d8c03d5389540595eae")


def test_the_controller_plays_the_pinned_match_on_a_small_box(tmp_path):
    """Property: idle marines + 3 zombies under the "zombies" controller give
    the pinned result, final fields and per-tick unit stream (the field
    law's, G9p). Breaks if the controller changes the law's unit list, its
    order or its slot in the tick, drives a zombie twice, or skips a tick --
    or if the law changes (then re-pin with the rationale)."""
    stream = _UnitStream()
    o = run_match(_box(tmp_path, "box"), on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               unit_stream=stream.h.hexdigest())
    assert got == _PRE_122_BOX


def test_the_controller_plays_the_pinned_ship_demo_match():
    """Property: ship_demo seed 1 (bots vs bots + 4 zombies) ends exactly as
    the conductor-driven law plays it (re-pinned at #140, then at #31 and its merge with #169, then
    at #183 G1p's turn-rate cap, then at G2p's carbine rate, then at
    G1p-c's speeds, then at G3p's throw wind-up and G4p's exposure, then at
    G5p-a1's turning law, then at G5p-b's wind, then at G9p's horde field
    and its behaviours, then at G8p's layered bots, then at #195 1d's honest
    team picture, then at #29 P3e's integer match path, then at Erik's
    zombie bite balance, then at #29 P3b's keyed Philox draws: win team 0
    at tick 2570) with the same tick_digest. Breaks like the box test, on the real
    level -- and on any change of the bots (then re-pin, the bots_v1 pin of
    tests/test_bots_v1.py staying)."""
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
