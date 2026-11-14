"""Combat: projectiles, shots, shooting bursts, line-of-sight, blast damage.

Lifted from ``game.py``:

- :class:`Projectile`        (game.py:1195-1233)
- :class:`Shot`              (game.py:1239-1244)
- :func:`process_shooting`   (game.py:1749-1791 — was ``Game._process_shooting``)
- :func:`auto_fire`          (game.py:1793-1818 — was ``Game._auto_fire``)
- :func:`fire_burst`         (game.py:1820-1865 — was ``Game._fire_burst``)
- :func:`process_door_explosives`  (game.py:1641-1658)

The two shipped physics->unit coupling responses that used to live here —
``apply_blast_damage`` (game.py:1981-1996, was ``Game._apply_blast_damage``)
and ``apply_environmental_damage`` (+ its ``HEAT_SCALE``) — moved VERBATIM
to :mod:`simulation.exchange`, the coupling-table module (mechanics/05, P1
refactor). They are re-imported below for compatibility (tests and legacy
imports resolve unchanged).

All shooting / burst / LOS code mutates the unit list and emits ``Shot``
tracer events into a caller-owned list. Detonations (the door-explosive
det slots here, the grenade fuse-out in simulation.py, a 40 mm round's
stop) go through the payload EXECUTOR
(:func:`simulation.payloads.execute_payload`, weapons W3) — which itself
sequences the physics entry points (``apply_explosion``,
``add_explosion_smoke``) byte-identically to the pre-W3 inline sites.

Determinism (#29 P3b):
    every ``rng`` parameter below is the sim's KEYED draws
    (:class:`simulation.draws.MatchDraws`, ``sim.draws``): each draw names
    its site (``rng_salts.Salt``), its unit and its local index -- the k-th
    cone draw a shooter makes this tick (``rng.next``), an in-flight round's
    ``round_index(age, ordinal)`` -- so no draw depends on any other. An
    ad-hoc caller (a test) passes ``draws.MatchDraws(seed)``; a numpy
    Generator is refused at the first draw.
"""
from __future__ import annotations

import math

import numpy as np

from config import CFG
from simulation import attack_resolver
from simulation import wall_fixed
# Physical cover (onephase_wego design §7) — continuous-space rectangles the
# march stops on. cover_system imports only the entities package + wall_fixed,
# so this adds no weight and no cycle.
from simulation.cover_system import cover_at
from simulation.damage import (KINETIC, CREDIT_HEAT, DamagePacket, apply_packet,
                               tick_credit)
from simulation.events import (
    LaserFiredEvent, ProjectileGlowEvent, ReloadEvent, ShotFiredEvent,
    SprayJetEvent,
)
from simulation.gases import N_TRACE_GASES
from simulation import gas_fixed
from simulation.field_edit import EditMode, FieldEdit, Region
from simulation.orders import (
    ORDER_EXPLOSIVE, ORDER_MOVE_ATTACK,
    MOVE_ORDER_TYPES,
)
from simulation import unit_fixed
from simulation import unit_traits
from simulation import draws as _draws          # #29 P3e: the match draws
from simulation.rng_salts import Salt            # #29 P3b: the draw sites
from simulation.unit_schema import MAG_UNBOUND
from simulation import noise as _noise          # G9p: shots are noises
# Physics event entry points: re-exported for legacy imports; the detonation
# sites themselves now go through the payload executor (W3). Bare-name import
# of execute_payload on purpose — instrumentation/replica tests rebind
# simulation.combat.execute_payload (the apply_environmental_damage pattern).
from simulation.physics import apply_explosion, add_explosion_smoke  # noqa: F401
from simulation.payloads import execute_payload
from simulation.status import apply_status, composed_flags  # noqa: F401
from simulation.materials import MAT_AIR
# Simultaneous resolution (#169): an attack slot decides against its start
# state and applies its effects together (phase A / phase B).
from simulation.simultaneous import SimultaneousEffects, effects_or_immediate
# The two shipped coupling responses live in simulation.exchange now
# (mechanics/05 coupling table, P1). Re-imported for compatibility — legacy
# imports (`from simulation.combat import apply_environmental_damage, ...`)
# keep resolving, and process_door_explosives calls apply_blast_damage
# unchanged.
from simulation.exchange import (                                # noqa: F401
    HEAT_SCALE, apply_blast_damage, apply_environmental_damage,
)
# The weapon/ammo/payload data tables (mechanics/03 §4, W1). The three shipped
# weapons are re-homed onto rows — same numbers, looked up by literal name at
# the sites that used to read CFG.weapons.rifle/.grenade/.door_explosive.
from simulation.weapons import get_tables as weapon_tables


_CONE_Q_CACHE = {}

#: A grenade thrown less than this far (Q16 tiles: 0.1) lands at once.
_FLIGHT_EPS_Q = unit_fixed.quantize_scalar(0.1)                 # p3e-door


def _bullet_mult_q() -> int:
    """``[zombie] bullet_damage_multiplier`` as Q16 (#29 P3e; the door of a
    config constant, read per call so an F5 reload is seen)."""
    return unit_fixed.quantize_scalar(
        float(CFG.zombie.bullet_damage_multiplier))           # p3e-door


def cone_q16_of_deg(spread_deg) -> int:
    """A cone half-angle in degrees (a weapon row's ``spread_deg`` /
    ``spread_snap_deg``) as Q16.16 radians -- THE door of the shot cone
    (#29 P3e): the config constant quantized once, cached per value
    (round half away, ``quantize_scalar``; ``math.radians`` of a constant)."""
    key = float(spread_deg)
    q = _CONE_Q_CACHE.get(key)
    if q is None:
        q = unit_fixed.quantize_scalar(math.radians(key))   # p3e-door
        if q < 0:
            raise ValueError(f"cone half-angle {spread_deg!r} < 0")
        _CONE_Q_CACHE[key] = q
    return q


# ---------------------------------------------------------------------------
# Projectile: in-flight grenade
# ---------------------------------------------------------------------------
class Projectile:
    """In-flight grenade (LOBBED — ignores unit collision). Travels in a
    straight line from start to target; detonates when
    ``current_tick >= get_detonate_tick()``.

    W3: carries its ROUND (``ammo_name`` — a ``[ammo.*]`` row of family
    ``hand_grenade``); the detonation site resolves the round's payload row
    through the executor. The default is the shipped ``grenade_frag`` (same
    ``travel_speed_tiles_per_second`` on every gas round, so the
    ``update_position()`` arithmetic is unchanged for all of them)."""

    def __init__(self, proj_type, start_fx, start_fy, target_fx, target_fy,
                 fuse_seconds, thrown_tick, ammo_name="grenade_frag",
                 owner_id=-1, fuse_ticks=None):
        self.proj_type = proj_type
        # The thrower (#123): the detonation's blast, heat and gas are
        # credited to it. -1 = nobody (a scripted / test lob).
        self.owner_id = -1 if owner_id is None else int(owner_id)
        # #29 P3e: the flight on the Q16 grid -- start and target quantized
        # ONCE here (the door: exact for the continuous throw, whose target
        # is already a Q16 point), the live position written by
        # update_position in integers; fx / fy / start_* / target_* are the
        # float views the renderer and the observation read.
        q = unit_fixed.quantize_scalar
        self.start_x_q = q(float(start_fx))                      # p3e-door
        self.start_y_q = q(float(start_fy))                      # p3e-door
        self.target_x_q = q(float(target_fx))                    # p3e-door
        self.target_y_q = q(float(target_fy))                    # p3e-door
        self.pos_x_q = self.start_x_q
        self.pos_y_q = self.start_y_q
        self.fuse_seconds = fuse_seconds
        #: The fuse in INTEGER ticks (the direct throw verb's, G3p); None =
        #: derive it from ``fuse_seconds`` (the WEGO order path).
        self.fuse_ticks = None if fuse_ticks is None else int(fuse_ticks)
        self.thrown_tick = thrown_tick
        self.detonated = False
        self.ammo_name = ammo_name
        # Tiles-per-SECOND, from the round's row (W1 re-home: the old
        # CFG.weapons.grenade.travel_speed = 30.0, same float — the
        # update_position() arithmetic below is bit-identical). The round's
        # speed_tiles_per_tick twin is the W2 unified-march data-of-record.
        self.travel_speed = weapon_tables().ammo.by_name[
            ammo_name].travel_speed_tiles_per_second
        self.travel_speed_q = unit_fixed.quantize_scalar(
            float(self.travel_speed))                           # p3e-door

    # Float views of the integer flight (exact n/65536).
    fx = property(lambda self: unit_fixed.dequantize_scalar(self.pos_x_q))
    fy = property(lambda self: unit_fixed.dequantize_scalar(self.pos_y_q))
    start_fx = property(
        lambda self: unit_fixed.dequantize_scalar(self.start_x_q))
    start_fy = property(
        lambda self: unit_fixed.dequantize_scalar(self.start_y_q))
    target_fx = property(
        lambda self: unit_fixed.dequantize_scalar(self.target_x_q))
    target_fy = property(
        lambda self: unit_fixed.dequantize_scalar(self.target_y_q))

    def target_tile(self):
        """The detonation tile ``(y, x)``: the target truncated toward zero
        (what ``int()`` of the float target was), in integers."""
        return (unit_fixed.shr_trunc(self.target_y_q),
                unit_fixed.shr_trunc(self.target_x_q))

    def get_detonate_tick(self):
        """Tick at which this projectile detonates."""
        if self.fuse_ticks is not None:
            return self.thrown_tick + self.fuse_ticks
        tps = CFG.clock.ticks_per_second
        return self.thrown_tick + int(self.fuse_seconds * tps)

    def update_position(self, current_tick):
        """Move the projectile toward the target based on travel time so far
        -- in integers since #29 P3e: the distance is ``isqrt`` of the Q32
        square (Q16 tiles), "arrived" is ``elapsed * speed >= dist * tps``,
        and the position the start plus the delta times ``elapsed * speed /
        (tps * dist)``, ONE rational rounded once (half away). (The
        detonation reads the target, not this position: the flight is what
        the renderer and the observation see.)"""
        tps = int(CFG.clock.ticks_per_second)
        elapsed = int(current_tick) - int(self.thrown_tick)
        dx_q = self.target_x_q - self.start_x_q
        dy_q = self.target_y_q - self.start_y_q
        dist_q = unit_fixed.isqrt(dx_q * dx_q + dy_q * dy_q)
        if dist_q < _FLIGHT_EPS_Q or elapsed * self.travel_speed_q >= \
                dist_q * tps:
            self.pos_x_q = self.target_x_q
            self.pos_y_q = self.target_y_q
            return
        num = elapsed * self.travel_speed_q
        den = tps * dist_q
        self.pos_x_q = self.start_x_q + unit_fixed.div_round(dx_q * num, den)
        self.pos_y_q = self.start_y_q + unit_fixed.div_round(dy_q * num, den)


# ---------------------------------------------------------------------------
# Shot: visual tracer event
# ---------------------------------------------------------------------------
class Shot:
    """A single rifle tracer for visual feedback. Lives on a short-lived
    effects list owned by the caller; the renderer expires entries past
    ``self.duration``. Not gameplay state — purely visual."""

    def __init__(self, fx1, fy1, fx2, fy2, time):
        self.fx1, self.fy1 = fx1, fy1
        self.fx2, self.fy2 = fx2, fy2
        self.time = time
        self.duration = CFG.combat.shot_tracer_duration


# ---------------------------------------------------------------------------
# Wall chew (mechanics/03 §3, W2): the SAME structural wall-HP path
# apply_explosion uses — quantized Q16.16 decrement + destroy at <= 0.
# ---------------------------------------------------------------------------
def chew_wall(gmap, iy, ix, wall_damage):
    """Deposit a round's ``wall_damage`` into the tile at (iy, ix).

    Missed shots are not deleted (mechanics/03 §3): a bullet stopping on a
    solid tile chews it, and a cover tile that absorbs a failed exposure
    roll eats the same damage — until the crate stops *being* cover. The
    arithmetic is exactly the apply_explosion structural path (physics.py):
    ``wall_hp -= wall_fixed.quantize_scalar(damage)`` (door 2 — the int
    config damage quantizes to an exact count), then ``destroy_wall`` at
    <= 0 (which handles solid walls AND non-solid destructibles like
    furniture — see its W2 note). ``wall_damage <= 0`` is a no-op so
    chew-less rounds cost nothing, and so is a tile already gone (air):
    two rounds of one attack slot can both stop on a wall that stood at the
    slot's start (#169), and the one applied second finds it destroyed.
    """
    if wall_damage <= 0:
        return
    if gmap.material[iy, ix] == MAT_AIR:
        return
    gmap.wall_hp[iy, ix] -= wall_fixed.quantize_scalar(float(wall_damage))
    if gmap.wall_hp[iy, ix] <= 0:
        gmap.destroy_wall(iy, ix)


# ---------------------------------------------------------------------------
# The UNIFIED MARCH (mechanics/03 §2, W2): everything ranged is one
# tile-marcher; speed is DATA (ammo.speed_tiles_per_tick).
# ---------------------------------------------------------------------------
class BulletInFlight:
    """One kinetic round on the unified march (mechanics/03 §2).

    Created by :func:`fire_burst` (one per bullet, cone already drawn) and
    advanced once immediately — a round whose ``speed_tiles_per_tick`` covers
    its range resolves in the firing tick, bit-compatible with the shipped
    same-tick march (the k5's 96 t/t ≥ range 90). A slower round persists on
    ``Simulation.bullets`` and continues in tick slot 2 (before movement),
    same slot as the grenade ``Projectile``.

    March arithmetic (all synced-state-feeding, engine/14):

    - **Step budget** — pure integer Q16.16 (door 1): each tick
      ``budget += ammo.speed_q16`` (the door-2 quantized speed); whole tiles
      ``budget >> 16`` are marched (capped by the remaining range), the
      fraction carries. A 2.5 t/t round marches 2, 3, 2, 3, … — exact.
    - **Steps** — the shipped kit-trig step vectors + plain float
      accumulation (exact n/65536 additions, door 3), UNCHANGED from the
      pre-W2 inner loop: same tile-stepping ``int()`` arithmetic, same
      bounds/solid/unit stop rules, in the same order.
    - **Rolls** — exposure on cover entry, crit on connect: both LAZY
      (attack_resolver; mechanics/03 §3) — a k5 burst across open floor
      consumes exactly its per-bullet cone draws, nothing else.
    """

    def __init__(self, shooter, shooter_id, weapon, ammo,
                 origin_x_q, origin_y_q, angle_q, step_x_q, step_y_q):
        if int(ammo.speed_q16) <= 0:
            raise ValueError(
                f"ammo.{ammo.name}.speed_tiles_per_tick must be > 0 for a "
                f"marching round (weapons.{weapon.name} fired it)")
        self.shooter = shooter          # excluded from hits (identity)
        self.shooter_id = shooter_id    # packet attribution
        self.weapon = weapon            # WeaponDef (crit columns, range)
        self.ammo = ammo                # AmmoDef (damage, dtype, ap, chew)
        # #29 P3e: the march in INTEGERS -- the Q16.16 march angle (screen
        # convention), the live position and the kit step vector on the Q16
        # grid (the float march was exact on that grid: bit-identical).
        self.angle_q = int(angle_q)
        self.rx_q = int(origin_x_q)
        self.ry_q = int(origin_y_q)
        self.prev_ix = unit_fixed.shr_trunc(self.rx_q)   # last tile CROSSED
        self.prev_iy = unit_fixed.shr_trunc(self.ry_q)   # (cover inspection)
        self.remaining_steps = int(weapon.range_tiles)
        self.speed_q16 = int(ammo.speed_q16)
        self.budget_q16 = 0             # fractional-tile carry (Q16.16)
        # W3 (mechanics/03 §4): the round's payload row — "" = plain kinetic
        # (every small-arm), else resolved ONCE at spawn through the shared
        # tables. A payload round (the GL-6 40 mm) detonates at its stop and
        # applies NO direct-hit packet — the blast does the work.
        self.payload = None
        if getattr(ammo, "payload", ""):
            self.payload = weapon_tables().payloads.by_name[ammo.payload]
        # The march rules this round was fired under (onephase_wego §7/§9),
        # carried ON the round so a slow projectile still obeys them on the
        # ticks it continues flying — the sim's per-tick advance does not have
        # to know which ruleset launched it.
        self.march_cover = None
        self.march_facing_defense = False
        # G5p-a2 (the unit impulse law): the round's line in integers -- its
        # origin and its kit step vector on the Q16 grid -- for the lever arm
        # of the push it gives the unit it stops on.
        self.ox_q = self.rx_q
        self.oy_q = self.ry_q
        self.sx_q = int(step_x_q)
        self.sy_q = int(step_y_q)
        # #29 P3b: the round's identity for its later draws -- the draws'
        # tick it was fired on and its ordinal among its shooter's rounds of
        # that tick (set by fire_burst from its cone draw; None = never fired
        # through it, and such a round refuses to roll).
        self.spawn_tick = None
        self.ordinal = 0

    def _draw(self, rng, salt):
        """This round's keyed draw of ``salt`` on this tick: id = its
        shooter, index = ``round_index(age, ordinal)`` (#29 P3b)."""
        if self.spawn_tick is None:
            raise ValueError("BulletInFlight: a round rolls only once fired "
                             "through fire_burst (it has no spawn tick)")
        return rng.at(salt, self.shooter_id, _draws.round_index(
            rng.tick - self.spawn_tick, self.ordinal))

    # Float VIEWS of the integer march (exact n/65536): the tracer, the
    # render-only events and the legacy onephase cover read them.
    @property
    def rx(self) -> float:
        return self.rx_q / unit_fixed.FP_ONE_F

    @property
    def ry(self) -> float:
        return self.ry_q / unit_fixed.FP_ONE_F

    @property
    def angle(self) -> float:
        return self.angle_q / unit_fixed.FP_ONE_F

    def push_on(self, gmap, unit, fx) -> None:
        """Record this round's PUSH on ``unit``, the unit it stopped on
        (G5p-a2, design §4 / §13): every projectile and pellet that stops on
        a unit pushes with its own row's momentum (``[ammo.*] bullet_g``,
        ``muzzle_mps``; a payload round with its body's), along its flight,
        times ``[impulse] bullet_push_mult``; its lever arm is the
        perpendicular distance from the unit's centre to the round's line
        (integers, no draw), its spin times ``bullet_twist_mult``. Recorded
        through the slot's ``push`` effect, right after the hit."""
        from simulation import unit_impulse
        table = unit_impulse.table_for(gmap)
        j_q = table.ammo_push_q.get(self.ammo.name, 0)
        if not j_q:
            return
        jx, jy = unit_impulse.along_q(j_q, self.sx_q, self.sy_q)
        cx, cy = unit_impulse.centre_q(unit)
        lever = unit_impulse.lever_q(cx, cy, self.ox_q, self.oy_q,
                                     self.sx_q, self.sy_q)
        fx.push(table, unit, jx, jy, lever, table.twist_q16)

    def advance(self, gmap, units, shots, real_time, rng, events=None,
                queue=None, cover=None, facing_defense=False, effects=None):
        """March one tick's budget. Returns True while still in flight.

        PHYSICAL COVER (onephase_wego design §7). ``cover`` is the sim's
        :class:`~simulation.cover_system.CoverRuntime` list, or ``None`` for
        the shipped statistical model. Passing a list — even an EMPTY one —
        switches the round onto physical cover, because "there is no
        statistical to-hit model" is a property of the RULESET, not of whether
        a particular level happens to have crates in it:

        - a round whose marched position enters a live cover rectangle STOPS
          there and chews it, so a marine hugging a crate is protected exactly
          as much as geometry says;
        - the ``cover_exposure_at`` / ``roll_exposure`` absorption roll is not
          consulted at all.

        ``facing_defense`` applies §9's "facing determines defense": a hit
        arriving from outside the target's facing arc scales up, and a unit on
        overwatch takes a harsher rear penalty — the cost of the side blindness
        its target control buys.

        Both default to the pre-existing behaviour, so every shipped caller is
        byte-identical.

        Emits one tracer ``Shot`` (+ ``ShotFiredEvent``) for the segment
        travelled this tick; applies the packet on a connecting hit; chews
        walls/cover per mechanics/03 §3. The inner loop preserves the
        shipped fire_burst march body verbatim (Q2-lift invariants: kit
        steps, ``int()`` tiling, stop order) with the W2 seams added at the
        stop sites only.

        W3: ``queue`` is the sim's :class:`EditQueue` — required only when
        the round carries a payload (the GL-6 40 mm): the payload executes
        at the round's STOP tile (first solid / first unit footprint /
        cover absorption / max range or grid exit) through
        :func:`simulation.payloads.execute_payload`. Payload rounds apply
        NO direct-hit packet — the blast does the work.

        #169: ``effects`` is the attack slot's
        :class:`~simulation.simultaneous.SimultaneousEffects` -- the march
        DECIDES here (its stop, its target, its rolls, against the slot-start
        state) and RECORDS what it does (packet, wall / crate chew,
        detonation) for the slot's phase B. ``None`` applies at once (the
        pre-#169 behaviour, for direct callers).
        """
        fx = effects_or_immediate(effects)
        # A round carries the rules it was fired under, so continuing flight on
        # a later tick behaves like the tick it launched on.
        if cover is None:
            cover = self.march_cover
        facing_defense = facing_defense or self.march_facing_defense

        h, w = gmap.material.shape
        seg_x, seg_y = self.rx, self.ry          # tracer segment start (view)
        # No step marched yet: this segment leaves the muzzle (the renderer's
        # muzzle flash; read-only, no new round state).
        launch = self.remaining_steps == int(self.weapon.range_tiles)

        # Integer step budget (door 1): whole tiles this tick, fraction carries.
        self.budget_q16 += self.speed_q16
        n_move = self.budget_q16 >> unit_fixed.FP_SHIFT
        self.budget_q16 -= n_move << unit_fixed.FP_SHIFT
        n_steps = n_move if n_move < self.remaining_steps else self.remaining_steps

        hit_unit = None
        stopped = False
        approach_x, approach_y = seg_x, seg_y     # where the hit came FROM
        for _step in range(n_steps):
            px, py = self.rx_q, self.ry_q        # pre-step (absorption rollback)
            self.rx_q += self.sx_q
            self.ry_q += self.sy_q
            # int() of the float march truncated toward zero: shr_trunc
            ix = unit_fixed.shr_trunc(self.rx_q)
            iy = unit_fixed.shr_trunc(self.ry_q)

            if 0 <= iy < h and 0 <= ix < w:
                if gmap.solid[iy, ix]:
                    # W2: missed shots chew walls — the apply_explosion path.
                    fx.world(chew_wall, gmap, iy, ix, self.ammo.wall_damage)
                    stopped = True
                    break
            else:
                stopped = True
                break

            # PHYSICAL COVER (§7): tested BEFORE bodies, so a crate standing
            # between the muzzle and a marine stops the round short of him —
            # which IS the protection, with no exposure roll anywhere. The
            # rectangle eats the round's wall damage and breaks when spent.
            if cover:
                hit_cover = cover_at(cover, self.rx, self.ry)
                if hit_cover is not None:
                    fx.world(hit_cover.chew, self.ammo.wall_damage)
                    self.rx_q, self.ry_q = px, py     # stop AT its face
                    stopped = True
                    break

            for e in units:
                if e is self.shooter or not e.alive:
                    continue
                if (e.tile_x <= ix < e.tile_x + e.footprint
                        and e.tile_y <= iy < e.tile_y + e.footprint):
                    hit_unit = e
                    break
            if hit_unit:
                # (the legacy onephase facing defense reads float views)
                approach_x = unit_fixed.dequantize_scalar(px)
                approach_y = unit_fixed.dequantize_scalar(py)
                # Exposure vs cover (mechanics/06 §5): inspect the tile
                # crossed immediately BEFORE entering the footprint. LAZY:
                # exposure 1.0 (no concealment — the overwhelming case)
                # draws NOTHING; directional cover is geometric by
                # construction (a flanking approach never sees the crate).
                #
                # SKIPPED ENTIRELY under physical cover (§7: "there is no
                # statistical to-hit model and no XCOM-style modifier
                # stack"). The rectangles above already decided this, in
                # geometry, several steps ago — rolling here too would price
                # the same crate twice.
                if cover is None:
                    exposure = attack_resolver.cover_exposure_at(
                        gmap, self.prev_iy, self.prev_ix)
                    if (exposure < unit_fixed.FP_ONE
                            and not attack_resolver.roll_exposure(
                                exposure, self._draw(rng, Salt.EXPOSURE))):
                        # Absorbed by the cover tile: it eats the round's wall
                        # damage, the tracer ends there, no unit packet.
                        fx.world(chew_wall, gmap, self.prev_iy, self.prev_ix,
                                 self.ammo.wall_damage)
                        self.rx_q, self.ry_q = px, py
                        hit_unit = None
                        stopped = True
                break
            self.prev_ix, self.prev_iy = ix, iy

        self.remaining_steps -= n_steps

        # Direct-hit packet — gated on the ROUND'S authored damage (W6):
        # every kinetic small-arm authors damage > 0 and no payload (the
        # shipped branch, bit-identical); the 40 mm authors damage = 0 (the
        # W3 rule — the blast does the work, no packet); a PLASMA bolt
        # authors BOTH (damage 40 HEAT + a splash payload — the armory §6
        # row), so it hits like a slug AND detonates at its stop below.
        if hit_unit is not None and int(self.ammo.damage) > 0:
            actual_dmg = int(self.ammo.damage)
            if hit_unit.is_zombie:
                # SITE-SIDE pre-mitigation amount rule, deliberately NOT a
                # resistance: the int() truncation (and its position before
                # the packet) is part of the shipped numbers; the zombie's
                # KINETIC resist stays neutral (mechanics/06 — only the heat
                # ×4 dissolved into the mitigation tables).
                # #29 P3e: the multiplier in Q16 (the door), the product
                # truncated as int() truncated (a non-negative value).
                actual_dmg = (actual_dmg * _bullet_mult_q()) >> unit_fixed.FP_SHIFT
            # Crit vs facing (mechanics/06 §5), LAZY: crit_chance == 0 (every
            # shipped weapon) draws nothing. On a crit the amount scales by
            # crit_mult in exact ints (round half away from zero), AFTER the
            # site rule, BEFORE mitigation.
            crit_chance = self.weapon.crit_chance       # quantized by the roll
            if crit_chance > 0:
                mult = attack_resolver.arc_multiplier(self.angle_q, hit_unit)
                if attack_resolver.roll_crit(
                        crit_chance, mult, self._draw(rng, Salt.SHOT_CRIT)):
                    actual_dmg = attack_resolver.scale_half_away(
                        actual_dmg, self.weapon.crit_mult)
            # FACING DETERMINES DEFENSE (onephase_wego design §9): a round
            # arriving from outside the target's arc hits harder, and a unit
            # on overwatch pays a steeper rear penalty — the side blindness
            # its target control bought. Applied AFTER crit, on the same
            # exact-integer scaling rule. Off by default (the shipped path is
            # untouched); the direction is the round's own approach point, so
            # it is honest even if the shooter has since moved.
            if facing_defense:
                from simulation.vision import defense_multiplier
                dmult = defense_multiplier(hit_unit, approach_x, approach_y)
                if dmult != 1.0:
                    actual_dmg = attack_resolver.scale_half_away(
                        actual_dmg, dmult)
            # Packet through the pipeline (mechanics/06 §2) — dtype/ap off the
            # ammo row (the k5's kinetic/0 reproduces the shipped packet
            # exactly); bullet deaths never convert. G4p: the round's
            # `expose` rides the packet (the focus-fire law raises the
            # target's exposure at the slot's landing).
            fx.hit(hit_unit,
                   DamagePacket(amount=actual_dmg,
                                dtype=self.ammo.dtype_id,
                                attacker_id=self.shooter_id,
                                ap=self.ammo.ap,
                                expose=getattr(self.ammo, "expose_q", 0),
                                # #185: the hit travels along the flight
                                # (the march's own Q16 step, as the push)
                                dir_x_q=self.sx_q, dir_y_q=self.sy_q),
                   events, source="bullet")
        # G5p-a2: the round that stopped on a unit pushes it -- whatever its
        # damage (the 40 mm's body too); recorded right after the hit.
        if hit_unit is not None:
            self.push_on(gmap, hit_unit, fx)

        if n_steps > 0:   # a budget-starved tick (<1 tile) has nothing to draw
            shots.append(Shot(seg_x, seg_y, self.rx, self.ry, real_time))
            if events is not None:
                hit_id = getattr(hit_unit, "id", -1) if hit_unit else None
                events.append(ShotFiredEvent(
                    unit_id=self.shooter_id,
                    from_tile=(seg_x, seg_y),
                    to_tile=(self.rx, self.ry),
                    hit_target_id=hit_id,
                    launch=launch,
                ))
                # W6 glow rounds (ammo.glow — the plasma bolt): one
                # RENDER-ONLY position ping per advanced tick, the
                # LaserFiredEvent precedent for slow projectiles. Not part
                # of the synced digest (only UnitHit/UnitKilled hash).
                if getattr(self.ammo, "glow", ""):
                    events.append(ProjectileGlowEvent(
                        pos=(self.rx, self.ry), kind=self.ammo.glow))

        still_flying = ((not stopped) and hit_unit is None
                        and self.remaining_steps > 0)

        # W3 (mechanics/03 §2/§4): a payload round DETONATES AT ITS STOP —
        # first solid (the round stops ON the wall tile, so the blast
        # centres there like a door charge on a door), first unit footprint
        # (the entry tile), cover absorption (the round physically stopped
        # short), or max range / grid exit (the round's final tile; off-grid
        # falls back to the last in-bounds tile crossed). Runs AFTER the
        # tracer append, so a detonation tick's event order is fixed:
        # [ShotFired, blast UnitHit/UnitKilled..., Explosion].
        if self.payload is not None and not still_flying:
            if queue is None:
                raise ValueError(
                    f"ammo.{self.ammo.name} carries payload "
                    f"{self.payload.name!r} but advance() got no EditQueue "
                    f"— pass queue= (the sim's edit_queue) so the "
                    f"detonation can deposit")
            det_y = unit_fixed.shr_trunc(self.ry_q)
            det_x = unit_fixed.shr_trunc(self.rx_q)
            if not (0 <= det_y < h and 0 <= det_x < w):
                det_y, det_x = self.prev_iy, self.prev_ix
            fx.world(execute_payload, gmap, queue, units, det_y, det_x,
                     self.payload, rng, events=events, kind="shell",
                     owner_id=self.shooter_id,
                     approach=(self.prev_iy, self.prev_ix))

        return still_flying


# ---------------------------------------------------------------------------
# Ignition from temperature — engine/06 §4 ("Ignition"), proposal §6 step 4b
# ---------------------------------------------------------------------------
def apply_temperature_ignition(gmap, o2_frac_ext, ignition_seed):
    """Ignite flammable tiles whose `temperature` has crossed `ignition_temp`
    AND that have oxygen (engine/06 §4, proposal §6 step 4b).

    This is the READ side of the temperature substrate: the C++
    ``TemperatureSolver`` (convert -> conduction -> cooling, inside
    ``PhysicsRunner.step``) has already filled ``gmap.temperature`` for this
    tick; here we gather it and start fires. For each FLAMMABLE tile::

        if armed[y,x]                                        # edge-trigger (below)
           and temperature[y,x] >= ignition_temp_q16[material]   # Q16.16 threshold
           and X > o2_frac_ext                               # O2 mole fraction
           and wall_hp[y,x] > 0                              # fuel remaining
           and fire[y,x] == 0:                               # never touch a lit tile
               fire[y,x] = ignition_seed                     # seed ONCE, then disarm

    EDGE-TRIGGERED (Fable ruling 2026-07-24, the zombie-smolder fix). The old
    level-triggered ``fire = max(fire, ignition_seed)`` re-floored a tile to
    ``I_seed`` on EVERY hot tick, so a fire the logistic was trying to kill was
    re-seeded forever (an eternal 0.1 smolder). Now a per-tile ``ignition_armed``
    bool gates the seed: a tile ARMS eligible, ignites ONCE (``armed -> False``),
    and re-arms ONLY when it genuinely COOLS (``T < ignition_temp``). The
    ``fire = max(...)`` floor is REMOVED entirely — temperature ignition NEVER
    modifies a tile with ``I > 0`` (the ``fire == 0`` gate), so a declining fire
    is never re-floored and dies clean. ``ignition_armed`` is SYNCED state
    (digest/save/undo), host-only (this function runs on the numpy mirror).

    where ``X = Σn_o2 / Σn_total`` over the OPEN (non-solid, non-vacuum)
    4-neighbours — the continuous-O2 law (docs/continuous_o2_law_design_
    2026-07-24.md §2.4): ignition and sustain read ONE law, so a tile can only
    ignite where a flame could be sustained (X above the extinction limit
    ``o2_frac_ext``). REPLACES the old absolute-density gate (mean n_o2 >=
    o2_threshold), which conflated HOT/thin with VITIATED.

    Determinism / coexistence:

    - **Q16.16 threshold compare.** ``temperature`` is Q16.16 fixed-point;
      ``ignition_temp_q16`` is the per-material threshold quantized into the
      SAME domain ONCE at load (``MaterialTable.ignition_temp_q16``). The test
      is a plain integer ``>=`` — no per-tick rescale, no float on the threshold
      path, bit-identical cross-machine.
    - **Flammable gate.** Non-flammable materials carry ``ignition_temp = 0`` (so
      ``q16 == 0``); gating on ``gmap.flammable`` is what stops a red-hot hull or
      glass tile (also threshold 0) from ever catching. Only fuel ignites.
    - **O2 check reuses the existing fire semantics.** The C++ ``FireSimulation``
      kills a burning tile when the mean REAL ``gas[O2]`` of its 4-connected
      AIR-SIDE (non-solid) neighbours drops below ``o2_threshold`` (EOS refactor
      P4 — was the ``atmosphere``/P proxy); ignition uses the SAME predicate
      (same threshold, same neighbourhood, same field) so a tile cannot be
      ignited into a state the fire step would immediately suffocate. A flammable
      tile is itself solid (wood/door), so its O2 comes from the adjacent air.
    - **Edge-triggered arm/disarm/re-arm (never touches ``I > 0``).** The seed
      is written ONLY on tiles that are ``armed & hot & has_o2 & fuel & fire==0``;
      those tiles then DISARM. Any tile burning while hot (``hot & fire > 0`` —
      whether temperature-seeded or lit by a weapon/external seed) disarms too, so
      it is never re-seeded as it dies. A cooled flammable tile (``T < ignition_
      temp``) RE-ARMS. Because the write is gated on ``fire == 0``, this path can
      never lower an existing, larger fire (the old ``max`` guarantee, kept) and
      never re-floors a declining one (the smolder fix).
    - **Second, additive ignition path.** This runs ALONGSIDE the existing
      cellular fire spread/ignition — it does not replace it. With no sim heat
      sources wired yet, ``temperature`` is ~0, so this path is DORMANT in-game
      (no behaviour change) until fire/beams emit heat — the intended safe seam.
    - **Deterministic.** Fixed row-major traversal (vectorised, order-independent
      anyway — a pure gather that writes each tile from frozen inputs), no RNG.

    Parameters
    ----------
    gmap
        The :class:`GameMap`. Reads ``temperature`` (Q16.16), ``material``,
        ``flammable``, ``solid``, ``is_vacuum``, ``gas`` (the real O2 plane,
        EOS refactor P4) and the material table's ``ignition_temp_q16``;
        writes ``fire`` in place.
    o2_threshold : float
        Minimum air-side-neighbour REAL N_O2 for ignition (reuse the fire's
        ``o2_threshold`` — EOS refactor P4 recalibrated it against the new
        N_O2 scale, ambient ~0.21; see config.toml [physics.fire]).
    ignition_seed : float
        Seed intensity a freshly-ignited tile gets (``I_seed`` ~ 0.1).

    Must run AFTER physics fills ``temperature`` (so the threshold sees this
    tick's heat) and ALONGSIDE the other temperature consumers (unit damage).
    """
    flammable = gmap.flammable
    if not bool(flammable.any()):
        return  # no fuel on this map -> nothing can ever ignite

    temperature = gmap.temperature
    material = gmap.material
    # Per-material Q16.16 ignition threshold (quantized once at load). Project
    # onto the grid so the compare is element-wise against `temperature`.
    thresh_q16 = gmap.materials.ignition_temp_q16[material]   # (h, w) int64
    t_q16 = temperature.astype(np.int64)

    # RE-ARM (edge-trigger hysteresis, Fable ruling 2026-07-24): a flammable tile
    # that has genuinely COOLED below its ignition_temp becomes eligible to seed
    # again. Runs BEFORE the `hot` early-out so a tile that disarmed while burning
    # re-arms the moment it drops below threshold — even on an otherwise dormant
    # tick. `hot` (T >= thresh) and `cooled` (T < thresh) are disjoint, so seed and
    # re-arm never touch the same tile (order-free). Cheap-guarded: in the fully
    # dormant case nothing was ever disarmed, so `armed` is all-True and we skip.
    armed = gmap.ignition_armed
    if not bool(armed.all()):
        armed[flammable & (t_q16 < thresh_q16)] = True

    # Hot enough? (Q16.16 integer compare.) Restrict to flammable tiles that
    # still have FUEL: a burnt-out tile (wall_hp <= 0) can never ignite. (A
    # smoldered wall at the 1-LSB FUEL_FLOOR still has fuel; a fire-destroyed wall
    # at wall_hp <= 0 does not.)
    has_fuel = gmap.wall_hp > 0
    hot = flammable & has_fuel & (t_q16 >= thresh_q16)
    if not bool(hot.any()):
        return  # nothing hot this tick (the dormant case) — re-arm already ran

    # --- O2 MOLE FRACTION over the OPEN (non-wall, non-vacuum) 4-neighbours —
    # the continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md §2.4):
    # X = Σn_o2 / Σn_total, ignite iff X > o2_frac_ext (the extinction limit the
    # C++ fire sustain uses). REPLACES the old absolute-density mean gate. The
    # open-neighbour mask `(~solid) & (~vacuum)` matches the C++ fire/combustion
    # laws exactly (`!is_wall && !is_vacuum`, is_wall == gmap.solid); a vacuum
    # neighbour is dropped from BOTH sums, so a tile venting to vacuum reads
    # X -> 0 and cannot ignite. The fraction is invariant under thermal
    # expansion, so this no longer conflates HOT/thin gas with VITIATED.
    h, w = temperature.shape
    open_nbr = (~gmap.solid) & (~gmap.is_vacuum)   # True == counts toward O2 (== C++)
    o2_idx = gmap.gases.name_to_id["o2"]
    n2_idx = gmap.gases.name_to_id["inert_n2"]
    o2_q = gmap.gas[o2_idx].astype(np.int64)       # int32 Q16.16 -> int64 (exact, order-free)
    # N_total = conservative bulk (O2 + inert_N2), soot EXCLUDED — one source of
    # truth with the EOS/temperature N_total and the C++ fire/combustion laws.
    ntot_q = o2_q + gmap.gas[n2_idx].astype(np.int64)
    sum_o2 = np.zeros((h, w), dtype=np.int64)      # Q16.16 numerator sum
    sum_tot = np.zeros((h, w), dtype=np.int64)     # Q16.16 denominator sum
    count = np.zeros((h, w), dtype=np.int64)       # neighbour count 0..4
    # N, S, E, W (fixed order; integer sum is order-independent regardless).
    for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        ys0, ys1 = max(0, -dy), h - max(0, dy)      # this-tile row span
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        nbr_o2 = o2_q[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        nbr_tot = ntot_q[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        nbr_open = open_nbr[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        sum_o2[ys0:ys1, xs0:xs1] += np.where(nbr_open, nbr_o2, np.int64(0))
        sum_tot[ys0:ys1, xs0:xs1] += np.where(nbr_open, nbr_tot, np.int64(0))
        count[ys0:ys1, xs0:xs1] += nbr_open.astype(np.int64)
    # X = Σn_o2 / max(Σn_total, floor), Q16.16, EXACT integer divide (floor;
    # deterministic cross-machine). The floor (655 == quantize(0.01)) matches the
    # C++ X_N_FLOOR: it guards the divide AND makes a near-vacuum cell read X ~= 0
    # rather than a spurious high fraction on trace gas. gas planes are clamped
    # >= 0 by construction (bulk_transport), so no negative-branch handling is
    # needed. A fully walled-in tile (count == 0) -> both sums 0 -> X = 0 -> never
    # ignites. (The C++ fire sustain uses reciprocal_q16 here; a <=1-ULP mismatch
    # at the exact X_ext boundary only ever means a marginal tile that ignites
    # then extinguishes next tick — deterministic either way, not a bit-parity
    # requirement, since ignition is the seed and the fire logistic is the arbiter.)
    from simulation import fire_fixed as _fire_fx
    X_N_FLOOR = np.int64(_fire_fx.quantize_scalar(0.01))
    den = np.maximum(sum_tot, X_N_FLOOR)
    X_q16 = (sum_o2 << np.int64(16)) // den                # Q16.16 mole fraction
    x_ext_q = np.int64(_fire_fx.quantize_scalar(float(o2_frac_ext)))
    has_o2 = (count > 0) & (X_q16 > x_ext_q)

    # SEED (edge-triggered): ignite ONLY an ARMED, hot, oxygenated, fuelled tile
    # that is currently UNLIT (`fire == 0`). The `fire == 0` gate is what makes
    # this NEVER modify a tile with I > 0 — it can neither lower a bigger existing
    # fire (an explosion's) nor re-floor a declining one (the smolder fix). A
    # freshly-seeded tile gets exactly ignition_seed (== max, since it was 0).
    # ignition_seed quantizes once into Q16.16.
    seed_q = np.int32(_fire_fx.quantize_scalar(float(ignition_seed)))
    seed_mask = armed & hot & has_o2 & (gmap.fire == 0)
    if bool(seed_mask.any()):
        gmap.fire[seed_mask] = seed_q

    # DISARM: any tile now in a hot-fire episode (`hot & fire > 0` — the tiles we
    # just seeded PLUS any lit by a weapon/external seed) is no longer eligible to
    # (re-)seed until it cools. This is the edge that kills the die->reseed smolder
    # loop: once disarmed, a fire the logistic drives toward I_min is never
    # re-floored, so it snaps to 0 and stays out until the tile re-arms on cooling.
    armed[hot & (gmap.fire > 0)] = False


# ---------------------------------------------------------------------------
# Ammo economy (mechanics/03 §4 mag_size / reload_seconds — W3)
# ---------------------------------------------------------------------------
#: ``Unit.spare_mags`` of a unit whose spare magazines are NOT tracked (its
#: loadout row has ``magazines`` 0 or absent): infinite auto-reload, today's
#: law. Any value >= 0 is a counted spare.
SPARES_UNTRACKED = -1


def magazines_carried(u):
    """The TOTAL magazines (the one in the gun included) the unit carries for
    the weapon in its hands, 0 = spares untracked. The loadout row's
    ``magazines`` count belongs to the loadout's OWN weapon
    (``Unit.magazines_weapon``): a weapon swapped in (the debug cycle key,
    OnePhaseWEGO's secondary) is not counted, so it keeps infinite spares
    until a loadout names it -- nothing but the loadout's weapon moves."""
    total = int(getattr(u, "magazines", 0))
    own = getattr(u, "magazines_weapon", None)
    if total > 0 and own is not None and getattr(u, "weapon_id", own) != own:
        return 0
    return total


def mag_bind(u, weapon):
    """First bind of a tracked weapon's magazine (the lazy first-trigger
    bind, and every re-bind after a ``current_mag = MAG_UNBOUND`` reset (it
    was None before #29 P3a): a swapped
    weapon, a WEGO round boundary): a FULL magazine in the gun and, when the
    loadout counts magazines for this weapon (:func:`magazines_carried` > 0),
    ``magazines - 1`` spares; otherwise the spares are untracked."""
    u.current_mag = int(weapon.mag_size)
    total = magazines_carried(u)
    u.spare_mags = SPARES_UNTRACKED if total <= 0 else total - 1
    u.reload_done_tick = -1


def mag_settle(u, weapon, tick):
    """Bring a TRACKED weapon's magazine state up to ``tick``: bind a
    never-bound magazine, and complete a reload whose stall has ended (the
    gun then holds a FULL magazine). Returns False while a reload's stall is
    still running. A reload is PENDING while ``reload_done_tick >= 0``; a dry
    unit (empty magazine, no spare) has none pending. Integer, no RNG."""
    if getattr(u, "current_mag", MAG_UNBOUND) == MAG_UNBOUND:
        mag_bind(u, weapon)
        return True
    done = getattr(u, "reload_done_tick", -1)
    if done >= 0:
        if tick < done:
            return False
        u.current_mag = int(weapon.mag_size)
        u.reload_done_tick = -1
    return True


def mag_gate(u, weapon, tick):
    """The per-TRIGGER magazine gate (W3 ammo economy; gameplay pack G2p).
    True = the trigger may fire this tick; False = the unit is mid-reload
    (the stall) or DRY (an empty magazine and no spare: the trigger does
    nothing, for good).

    ``mag_size <= 0`` = ammo untracked -- an immediate True with ZERO state
    touched (every pre-W3 weapon: shipped scenarios never see the fields, so
    the golden digest cannot move). For tracked weapons the state is
    ``current_mag`` / ``spare_mags`` / ``reload_done_tick`` (see
    :func:`mag_settle`); the reload COMPLETES at the first trigger attempt
    past the stall, and :func:`mag_status` reads it settled without
    mutating. Pure integer state + compares on row constants; no RNG.

    Mag state lives ON THE UNIT as unit-schema columns, hashed in the unit
    digest while set (#29 P3a, ``simulation.unit_schema``).
    """
    if weapon.mag_size <= 0:
        return True                        # untracked -- exactly pre-W3
    if not mag_settle(u, weapon, tick):
        return False                       # mid-reload: the stall
    return u.current_mag > 0               # 0 = dry (no spare was left)


def reload_begin(u, weapon, tick, events=None):
    """Start a reload NOW: the magazine in the gun is DISCARDED with every
    round still in it (``current_mag = 0``; rounds are never moved between
    magazines), one spare is consumed when spares are counted, and the stall
    runs the unit's ``unit_traits.reload_ticks`` (the row's ``reload_ticks``
    times its HANDLING hook, #29 U1) from ``tick``; the full magazine is in the
    gun when it ends (:func:`mag_settle`). With no spare left nothing starts
    and the unit is dry (``current_mag`` 0, none pending). True when a reload
    started. The caller has settled the magazine and checked its own
    preconditions (:func:`mag_spend` on empty, ``Simulation._reload``).

    G1p-c: a started reload appends ONE render-only
    :class:`~simulation.events.ReloadEvent` ``(unit id, stall ticks)`` to
    ``events`` (the tick's events; ``None`` emits nothing) -- the one emission
    site, so the manual and the automatic reload are told alike."""
    spare = getattr(u, "spare_mags", SPARES_UNTRACKED)
    u.current_mag = 0
    if spare == 0:
        u.reload_done_tick = -1
        return False
    if spare > 0:
        u.spare_mags = spare - 1
    # #29 U1: the unit's own stall, the HANDLING hook's accessor
    stall = unit_traits.reload_ticks(u, weapon)
    u.reload_done_tick = tick + stall
    if events is not None:
        events.append(ReloadEvent(unit_id=int(getattr(u, "id", -1)),
                                  ticks=int(stall)))
    return True


def mag_spend(u, weapon, tick, events=None):
    """Spend one round per TRIGGER (a shotgun's 8 pellets = one trigger =
    one round). Emptying the magazine starts the automatic reload NOW
    (:func:`reload_begin`): the stall between the last shot and the next is
    exactly ``reload_seconds``; with no spare left the unit goes dry. No-op
    for untracked weapons. ``events`` receives the reload's
    :class:`~simulation.events.ReloadEvent` (:func:`reload_begin`)."""
    if weapon.mag_size <= 0:
        return
    u.current_mag -= 1
    if u.current_mag <= 0:
        reload_begin(u, weapon, tick, events=events)


def mag_status(u, weapon, tick):
    """What a HUD (or a test) reads, WITHOUT mutating the unit: a tuple
    ``(rounds, spares, reloading)`` for a tracked weapon -- the rounds in the
    gun as :func:`mag_settle` would leave them at ``tick`` (a never-bound
    unit shows a full magazine and its loadout's spares), the counted spare
    magazines (``None`` = untracked spares) and whether a stall is still
    running -- or ``None`` for an untracked weapon (``mag_size <= 0``)."""
    if weapon.mag_size <= 0:
        return None
    full = int(weapon.mag_size)
    if getattr(u, "current_mag", MAG_UNBOUND) == MAG_UNBOUND:
        total = magazines_carried(u)
        return (full, None if total <= 0 else total - 1, False)
    spare = getattr(u, "spare_mags", SPARES_UNTRACKED)
    spares = None if spare < 0 else int(spare)
    done = getattr(u, "reload_done_tick", -1)
    if done >= 0:
        if tick < done:
            return (int(u.current_mag), spares, True)
        return (full, spares, False)
    return (int(u.current_mag), spares, False)


# ---------------------------------------------------------------------------
# Shooting (fire orders + auto-fire) — W2: per-unit weapon, archetype dispatch
# ---------------------------------------------------------------------------
def process_shooting(gmap, units, tick, shots, real_time, rng, events=None,
                     bullets=None, queue=None, effects=None):
    """Run one tick of shooting for every player unit with a fire order
    (or auto-fire during Move & Attack).

    Lifted from ``game.py:_process_shooting``; W2 resolves each shooter's
    weapon via ``unit.weapon_id`` → the WeaponTable row and DISPATCHES BY
    ARCHETYPE — ``projectile`` → :func:`fire_burst`, ``hitscan`` →
    :func:`fire_beam`, ``spray`` → :func:`start_spray_burst` (W4 — the
    burst's per-tick deposits ride :func:`process_sprays` in the same
    shooting slot), ``melee`` → :func:`melee_strike` (W5 — adjacency
    replaces the range/LOS gates; the remaining archetypes have no
    trigger path here: LOBBED/PLACED ride their order flows). The
    deterministic SPREAD MODE RULE (mechanics/03 §3): an explicit stationary
    fire order aims — ``spread_deg``; Move & Attack auto-fire snaps —
    ``spread_snap_deg``. No per-unit aim state.

    ``shots`` is a list the caller owns; new :class:`Shot` tracer events are
    appended to it. ``real_time`` is the wall-clock seconds since the round
    started (tracer fade-out). ``rng`` is the simulation's
    :class:`numpy.random.Generator` (cone/exposure/crit draws — all lazy,
    fixed order). ``bullets`` is the sim's in-flight list
    (``Simulation.bullets``): rounds slower than their range persist there
    and continue next tick (``None`` = same-tick only — every shipped
    small-arm resolves same-tick; tests exercising flight pass a list). If
    ``events`` is a list, :class:`ShotFiredEvent` / :class:`LaserFiredEvent`
    / :class:`UnitHitEvent` / :class:`UnitKilledEvent` are appended for the
    renderer to consume. ``queue`` (W3) is the sim's :class:`EditQueue` —
    payload rounds (the GL-6 40 mm) deposit their detonation through it.

    SIMULTANEOUS (#169, :mod:`simulation.simultaneous`): the walk below is
    PHASE A -- every shooter's eligibility and every shot's geometry are
    decided against the slot-start state, because each effect (packet,
    status, facing, chew, detonation) is RECORDED on ``effects`` instead of
    applied; phase B (``effects.resolve()``) applies them in walk order. So
    a unit killed or stunned this tick by a lower id still fires this tick,
    and two units can kill each other. ``effects=None`` = this call owns
    the slot and resolves before returning; the conductor passes its own so
    :func:`process_sprays` decides in the same phase A, then resolves.
    """
    tpp = CFG.clock.ticks_per_phase
    phase = tick // tpp
    tables = weapon_tables()
    own_slot = effects is None
    fx = SimultaneousEffects() if own_slot else effects

    for u in units:
        if u.is_zombie or not u.alive:
            continue

        # Status gate (mechanics/06 §4): a unit whose composed can_act is
        # suppressed (stunned / paralyzed / knocked down) executes NO attack
        # this tick — neither its fire order nor Move & Attack auto-fire.
        # The order itself stays queued (suppression delays, never cancels).
        flags = composed_flags(u)
        if not flags.can_act:
            continue

        # W2: the unit's weapon row ("" = no ranged weapon — zombie melee
        # stays on its ai_zombie path; NPC weapon rows are future work).
        weapon_id = getattr(u, "weapon_id", "")
        if not weapon_id:
            continue
        weapon = tables.weapons.by_name[weapon_id]

        # A queued WEGO fire order for this phase (the targeted path, below).
        # Under direct control a possessed unit carries NO order — it fires
        # DIRECTIONALLY from a held TRIGGER (the no-order branch just below,
        # free_aim_shooting_design §4b). Byte-identical under WEGO:
        # get_fire_order_in_phase is unchanged and no WEGO unit sets a trigger.
        fire_order = u.get_fire_order_in_phase(phase)
        if not fire_order:
            # FREE-AIM (design §4b): a possessed unit holding TRIGGER under
            # ContinuousRealtime fires directionally along u.facing, BYPASSING
            # the range+LOS pre-gate — the march resolves range and hit (it
            # stops on the first solid/unit or at range_tiles). Dormant under
            # WEGO: no unit sets ``live_trigger``, so this is one falsy attr
            # read and the digest is byte-identical (the P3 dormancy rule).
            if getattr(u, "live_trigger", False):
                _directional_fire(gmap, units, u, tick, shots, real_time, rng,
                                  events, bullets, weapon, flags, queue,
                                  effects=fx)
                continue
            # Move & Attack: auto-fire at nearest visible enemy (snap cone).
            for o in u.orders:
                if o.order_type == ORDER_MOVE_ATTACK and o.phase == phase:
                    auto_fire(gmap, units, u, tick, shots, real_time, rng,
                              events=events, bullets=bullets, weapon=weapon,
                              queue=queue, effects=fx)
                    break
            continue

        # SPRAY (W4, mechanics/03 §5) — the trigger-side gates. A burst in
        # progress owns the trigger (deposits ride process_sprays; no state
        # is touched here until it ends). The STATIONARY RULE (v1, of
        # record): a spray fire order arms only while the unit has no
        # movement order in the same phase — the sprayer stands still.
        if weapon.archetype == "spray":
            if getattr(u, "spray_ticks_left", 0) > 0:
                continue
            if any(o.phase == phase and o.order_type in MOVE_ORDER_TYPES
                   for o in u.orders):
                continue

        # Burst cadence gate.
        if tick - u.last_fire_tick < weapon.rof_interval_ticks:
            continue

        # W3 ammo economy: the magazine gate (reload stall / refill).
        # mag_size == 0 (every pre-W3 weapon) touches nothing — dead path.
        if not mag_gate(u, weapon, tick):
            continue

        target_fx = fire_order.target_fx
        target_fy = fire_order.target_fy

        # MELEE branch of the archetype dispatch (W5, mechanics/03 §5):
        # ADJACENCY REPLACES the range/LOS gates below (touching footprints
        # have no tile between them — the center-distance range check and
        # the ray test are the RANGED marchers' geometry) and the spread
        # cone is meaningless on a blade — no cone draw, ever. A connecting
        # strike charges the rof cadence (and the mag machinery — a no-op
        # at mag_size 0, both shipped rows); a whiff charges NOTHING and
        # retries next tick while the order stands.
        if weapon.archetype == "melee":
            if melee_strike(units, u, target_fx, target_fy, rng, events,
                            weapon, effects=fx):
                mag_spend(u, weapon, tick, events=events)
                u.last_fire_tick = tick
            continue

        uc_fx = u.center_tile_x()
        uc_fy = u.center_tile_y()

        # Range check.
        dist = math.sqrt((uc_fx - target_fx) ** 2 + (uc_fy - target_fy) ** 2)
        if dist > weapon.range_tiles:
            continue

        # LOS check.
        if not gmap.has_los(uc_fy, uc_fx, target_fy, target_fx):
            continue

        # Explicit stationary fire order = AIMED (spread_deg) — unless the
        # unit's composed can_aim is suppressed (teargas BLINDED, STUNNED,
        # PARALYZED — mechanics/06 §4): then even an aimed order fires the
        # SNAP cone. The owed P3 can_aim consumer (W3): ONE clean gate at
        # cone selection; with no statuses it is a dead path (FLAGS_DEFAULT
        # has can_aim True), bit-identical to pre-W3.
        # #29 U1: the class cone's ONE selection (the SPREAD hook enters at
        # the draw, fire_burst / fire_beam -> unit_traits.cone_q16).
        spread = unit_traits.cone_deg_of(weapon, flags.can_aim)
        if weapon.archetype == "spray":
            # SPRAY branch of the archetype dispatch (W4): arm the burst —
            # the deposits themselves ride process_sprays in this same
            # shooting slot. Handled here rather than in _dispatch_trigger
            # because the burst captures the ORDER (interruption consumes
            # it). Spread is meaningless on a cone weapon — no draw, ever.
            start_spray_burst(u, weapon, fire_order, tick, effects=fx)
            _noise.emit(gmap, uc_fy, uc_fx,
                        getattr(weapon, "loudness_tiles", 0))  # G9p
        else:
            _dispatch_trigger(gmap, units, u, uc_fx, uc_fy,
                              target_fx, target_fy, tick, shots, real_time,
                              rng, events, bullets, weapon, spread, queue,
                              effects=fx)
        mag_spend(u, weapon, tick, events=events)
        u.last_fire_tick = tick

    if own_slot:
        fx.resolve()                     # phase B


def _dispatch_trigger(gmap, units, u, fx1, fy1, fx2, fy2, tick, shots,
                      real_time, rng, events, bullets, weapon, spread_deg,
                      queue=None, aim_q=None, cover=None,
                      facing_defense=False, effects=None):
    """Route one trigger pull to its delivery archetype (mechanics/03 §1).
    The archetype set is CLOSED; only the ranged marchers dispatch here.
    ``aim_q`` (free_aim_shooting_design §4a; Q16.16 radians since #29 P3e)
    is forwarded to the resolver: None = tile-derived bearing (WEGO),
    provided = free-aim march bearing.
    ``cover`` / ``facing_defense`` are the OnePhaseWEGO march rules (§7/§9);
    both default to the shipped behaviour. ``effects`` (#169) is the
    attack slot's phase-B record (``None`` = apply at once).

    G9p: every pull is a NOISE at the shooter's centre with the row's
    ``loudness_tiles`` (simulation.noise) -- decided in phase A like the
    shot itself, heard at the zombie slot."""
    _noise.emit(gmap, u.center_tile_y(), u.center_tile_x(),
                getattr(weapon, "loudness_tiles", 0))
    if weapon.archetype == "hitscan":
        fire_beam(gmap, units, u, fx1, fy1, fx2, fy2,
                  tick, shots, real_time, rng, events=events,
                  weapon=weapon, spread_deg=spread_deg, aim_q=aim_q,
                  cover=cover, effects=effects)
    else:  # "projectile" — the default marcher (validated set, mechanics/03)
        fire_burst(gmap, units, u, fx1, fy1, fx2, fy2,
                   tick, shots, real_time, rng, events=events,
                   weapon=weapon, spread_deg=spread_deg, bullets=bullets,
                   queue=queue, aim_q=aim_q, cover=cover,
                   facing_defense=facing_defense, effects=effects)


def _directional_fire(gmap, units, u, tick, shots, real_time, rng,
                      events, bullets, weapon, flags, queue=None,
                      effects=None):
    """Free-aim directional fire for a possessed unit holding TRIGGER
    (free_aim_shooting_design §4b) — the parallel to the targeted block in
    :func:`process_shooting`, but it BYPASSES the range+LOS pre-gate: a
    directional shot always fires (subject to the same cadence / mag / status
    gates), and range + hit are the march's job (it stops on the first
    solid/unit or at ``range_tiles``). Aim comes from ``u.facing`` (set every
    tick by the AIM intent) — no fabricated Order, no tile round-trip.

    The status gate (``can_act``) is already applied by the caller; ``flags``
    is the caller's composed-flags snapshot (for the ``can_aim`` spread
    selection). Melee and lobbed/placed are OUT OF SCOPE here (design §4c/§9):
    melee is the deferred adjacency arc; lobbed/placed ride the THROW / USE
    intent flows, not a trigger ray.
    """
    # Cadence gate (identical to the targeted path).
    if tick - u.last_fire_tick < weapon.rof_interval_ticks:
        return
    # Ammo / magazine gate.
    if not mag_gate(u, weapon, tick):
        return

    # CONVENTION (design §6): ``u.facing`` is math-style (y-up); the march
    # bearing is screen-style (y-down); they are mirror images across the
    # x-axis, so the march bearing is ``-facing``. This is the ONE y-up->y-down
    # flip in the free-aim path — the resolvers take ``aim_q`` already in
    # the march convention and consume it verbatim (see fire_burst §4a).
    # #29 P3e: in integers, the unit's own Q16 facing.
    aim_q = -int(u.facing_q)

    spread = unit_traits.cone_deg_of(weapon, flags.can_aim)     # #29 U1

    if weapon.archetype == "melee":
        return   # out of scope (design §9) — the deferred melee arc owns it
    if weapon.archetype == "spray":
        # A burst in progress owns the trigger until it ends (its deposits ride
        # process_sprays); re-arm only once it has finished (continuous hosing).
        if getattr(u, "spray_ticks_left", 0) > 0:
            return
        start_spray_burst_directional(u, weapon, tick, aim_q)
        _noise.emit(gmap, u.center_tile_y(), u.center_tile_x(),
                    getattr(weapon, "loudness_tiles", 0))      # G9p
    elif weapon.archetype in ("projectile", "hitscan"):
        uc_fx = u.center_tile_x()
        uc_fy = u.center_tile_y()
        # Origin only; the tile target is unused when aim_q is provided.
        _dispatch_trigger(gmap, units, u, uc_fx, uc_fy, uc_fx, uc_fy,
                          tick, shots, real_time, rng, events, bullets,
                          weapon, spread, queue, aim_q=aim_q,
                          effects=effects)
    else:
        return   # lobbed / placed: not a trigger ray (design §4c) — no-op
    mag_spend(u, weapon, tick, events=events)
    u.last_fire_tick = tick


def auto_fire(gmap, units, u, tick, shots, real_time, rng, events=None,
              bullets=None, weapon=None, queue=None, effects=None):
    """Find the nearest visible enemy within weapon range and fire — SNAP
    cone (``spread_snap_deg``): Move & Attack is fire-on-the-move
    (mechanics/03 §3 mode rule).

    Lifted from ``game.py:_auto_fire``; W2 takes the shooter's resolved
    weapon row (or resolves ``unit.weapon_id`` when called directly).
    Skipped if still within the burst cooldown — or mid-reload (W3).
    """
    if weapon is None:
        weapon_id = getattr(u, "weapon_id", "")
        if not weapon_id:
            return
        weapon = weapon_tables().weapons.by_name[weapon_id]
    # SPRAY weapons never auto-fire (W4 v1 rule, mechanics/03 §5): a spray
    # is an explicit stationary commitment, not fire-on-the-move. Bail
    # BEFORE any state is touched (cadence read, mag bind) so a spray-armed
    # unit on Move & Attack is bit-identical to one with no trigger at all.
    # MELEE joins the skip (W5 v1 rule): a strike takes an explicit order
    # naming the target tile — auto-fire's nearest-visible-enemy pick plus
    # _dispatch_trigger's marcher shape don't describe a blade; revisit if
    # Move & Attack should stab-on-contact (mechanics/03 §8).
    if weapon.archetype in ("spray", "melee"):
        return
    if tick - u.last_fire_tick < weapon.rof_interval_ticks:
        return
    # W3 ammo economy — same gate as the fire-order path (dead for mag 0).
    if not mag_gate(u, weapon, tick):
        return

    uc_fx = u.center_tile_x()
    uc_fy = u.center_tile_y()
    best_dist = float('inf')
    best_enemy = None

    for e in units:
        if e.team == u.team or not e.alive:
            continue
        ec_fx = e.center_tile_x()
        ec_fy = e.center_tile_y()
        dist = math.sqrt((uc_fx - ec_fx) ** 2 + (uc_fy - ec_fy) ** 2)
        if dist <= weapon.range_tiles and dist < best_dist:
            if gmap.has_los(uc_fy, uc_fx, ec_fy, ec_fx):
                best_dist = dist
                best_enemy = e

    if best_enemy:
        _dispatch_trigger(gmap, units, u, uc_fx, uc_fy,
                          best_enemy.center_tile_x(),
                          best_enemy.center_tile_y(),
                          tick, shots, real_time, rng, events, bullets,
                          weapon, unit_traits.cone_deg_of(weapon, False),
                          queue, effects=effects)
        mag_spend(u, weapon, tick, events=events)
        u.last_fire_tick = tick


def _record_recoil(gmap, shooter, bullet, effects) -> None:
    """The recoil of ``bullet`` on its ``shooter`` (G5p-a2): minus the
    round's row momentum along its kit step vector, times ``[impulse]
    recoil_mult``, recorded through the attack slot's ``push`` effect (one
    shape for every impulse; :data:`IMMEDIATE` when the caller owns no
    slot)."""
    from simulation import unit_impulse
    table = unit_impulse.table_for(gmap)
    j_q = table.ammo_recoil_q.get(bullet.ammo.name, 0)
    if not j_q or not hasattr(shooter, "impulse_books"):
        return
    jx, jy = unit_impulse.along_q(j_q, bullet.sx_q, bullet.sy_q)
    effects_or_immediate(effects).push(table, shooter, -jx, -jy)


def fire_burst(gmap, units, shooter, fx1, fy1, fx2, fy2,
               tick, shots, real_time, rng, events=None, *,
               weapon=None, ammo=None, spread_deg=None, bullets=None,
               queue=None, aim_q=None, cover=None, facing_defense=False,
               effects=None):
    """Fire a burst of PROJECTILE rounds from (fx1, fy1) toward (fx2, fy2).

    Lifted from ``game.py:_fire_burst``; W2 generalizes the march to the
    weapon/ammo rows (speed as data) without touching its arithmetic: each
    bullet draws one cone angle (``draws.cone_offset_q16``, one integer draw
    since #29 P3e), takes its kit-trig step vector, and marches through
    :class:`BulletInFlight` — which preserves the pre-W2 inner loop verbatim
    (same tile stepping, same stop rules, same packet chain; the k5 resolves
    same-tick because 96 t/t ≥ range 90). A round that outranges its per-tick
    speed persists on ``bullets`` (``None`` = discard residual flight — every
    shipped small-arm resolves same-tick). Zombies take damage scaled by
    ``CFG.zombie.bullet_damage_multiplier``. Every resolved segment appends
    one :class:`Shot` tracer to ``shots``; if ``events`` is a list, matching
    :class:`ShotFiredEvent` / :class:`UnitHitEvent` / :class:`UnitKilledEvent`
    are appended.

    ``weapon``/``ammo``/``spread_deg`` default to the k5 carbine row + its
    standard round + the AIMED cone (the W1-shipped call shape); the
    archetype dispatcher passes them explicitly.
    """
    tables = weapon_tables()
    if weapon is None:
        weapon = tables.weapons.by_name["k5_carbine"]
    if ammo is None:
        ammo = tables.ammo_for_weapon(weapon)
    if spread_deg is None:
        spread_deg = unit_traits.cone_deg_of(weapon)
    # #29 U1: the shooter's cone, THE SPREAD accessor (the class cone's door
    # times its hook)
    cone_q = unit_traits.cone_q16(shooter, spread_deg)
    n_bullets = weapon.shots_per_trigger
    # The bullet march decides hits -> current_hp -> kills, i.e. it FEEDS
    # SYNCED STATE: its trig is the deterministic integer kit (Q2-lift) and,
    # since #29 P3e, its whole arithmetic is integer -- the cone is ONE
    # integer draw per bullet (``draws.cone_offset_q16``, in the order the
    # float ``uniform`` was drawn) added to the Q16.16 bearing.
    #
    # FREE-AIM SEAM (free_aim_shooting_design §4a): ``aim_q`` is an optional
    # Q16.16 bearing in the MARCH convention (screen y-down, exactly this
    # function's base bearing). ``None`` = derive from the tile target (the
    # WEGO/targeted path); provided = march straight from it. When provided,
    # the shooter's facing is the turning law's, left untouched.
    dx_q = unit_fixed.quantize_scalar(fx2 - fx1)     # tile deltas: exact
    dy_q = unit_fixed.quantize_scalar(fy2 - fy1)
    if aim_q is None:
        base_q = unit_fixed.atan2_q16(dy_q, dx_q)
        # W2 (mechanics/06 §5): firing makes the aim bearing (before spread)
        # the shooter's facing TARGET (G5p-a1: the turning law turns it there
        # from the next tick's slot 1a) — through the kit, in the unit facing
        # convention (y up: the dy negation, unit.face_towards).
        effects_or_immediate(effects).face(
            shooter, unit_fixed.atan2_q16(-dy_q, dx_q))
    else:
        base_q = int(aim_q)
    ox_q = unit_fixed.quantize_scalar(fx1)           # the centre tile: exact
    oy_q = unit_fixed.quantize_scalar(fy1)

    shooter_id = getattr(shooter, "id", -1)

    for _ in range(n_bullets):
        # #29 P3b: the shooter's next cone draw this tick (its index is the
        # round's ordinal, carried for the round's later draws)
        d = rng.next(Salt.SHOT_CONE, shooter_id)
        angle_q = base_q + _draws.cone_offset_q16(d, cone_q)
        # Per-bullet step vector through the kit (exact Q16 ints).
        step_x_q = unit_fixed.cos_q16(angle_q)
        step_y_q = unit_fixed.sin_q16(angle_q)
        b = BulletInFlight(shooter, shooter_id, weapon, ammo,
                           ox_q, oy_q, angle_q, step_x_q, step_y_q)
        b.spawn_tick, b.ordinal = d.tick, d.index
        # G5p-a2 (RULED): the shooter recoils with minus the round's
        # momentum along its flight (times [impulse] recoil_mult), through
        # the slot's push effect -- no lever arm.
        _record_recoil(gmap, shooter, b, effects)
        b.march_cover = cover
        b.march_facing_defense = facing_defense
        still_flying = b.advance(gmap, units, shots, real_time, rng,
                                 events=events, queue=queue, cover=cover,
                                 facing_defense=facing_defense,
                                 effects=effects)
        if still_flying and bullets is not None:
            bullets.append(b)


# ---------------------------------------------------------------------------
# HITSCAN (mechanics/03 §5, W2): the beam — full-range march in the firing
# tick, skewers units, integer Beer-Lambert gas attenuation, chews the
# stopping solid. The first laser: Lance-3.
# ---------------------------------------------------------------------------
# Beam-death threshold: energy below this is extinguished (quantized ONCE —
# 0.01 real → 655 counts on the shared Q16.16 grid; door 2). Keeps a fully
# absorbed beam from marching on depositing 0-packets.
BEAM_MIN_ENERGY_Q16 = unit_fixed.quantize_scalar(0.01)


def fire_beam(gmap, units, shooter, fx1, fy1, fx2, fy2,
              tick, shots, real_time, rng, events=None, *,
              weapon, ammo=None, spread_deg=None, aim_q=None, cover=None,
              effects=None):
    """Fire a HITSCAN beam from (fx1, fy1) toward (fx2, fy2) — physically
    instant: the full-range march happens in the firing tick (photons don't
    persist; mechanics/03 §2).

    Beam mechanics (mechanics/03 §5, all fixed evaluation order —
    attenuate-then-interact per tile, units in march order):

    - ONE spread draw per shot (the same door-4 uniform as bullets), kit-trig
      step vectors, the same tile stepping as the unified march.
    - Beam ENERGY starts at ``ammo.damage`` quantized to Q16.16
      (``damage << 16`` — the int config value is exact on the grid; door 2).
    - Per tile crossed the energy multiplies by
      ``max(0, ONE − Σ_g absorb_g · density_g)`` in PURE INTEGER Q16.16
      (door 1): ``density_g`` is the int32 gas slice count at the tile;
      ``absorb_g`` is ``GasTable.beam_absorb_q16`` (derived once at table
      build from the mean of the gas's RGB absorption triple — see gases.py).
      Products truncate ``>> 16`` (all quantities non-negative, so truncation
      is the pinned toward-zero form). The same Beer-Lambert the renderer
      integrates — no ``exp``, no transcendentals. Smoke grenades are laser
      countermeasures and no code knows it.
    - The beam dies below :data:`BEAM_MIN_ENERGY_Q16` (checked after each
      tile's attenuation, before interactions — a dead beam deposits nothing).
    - SKEWER: the beam passes THROUGH units — each unit crossed takes an
      ENERGY :class:`DamagePacket` of the beam's CURRENT attenuated energy at
      its entry tile, rounded half away from zero to an int
      (``(energy + 32768) >> 16`` — exact for non-negative energy); beam
      energy is NOT reduced by unit hits in v1. Each unit is hit once.
    - The beam STOPS at the first solid tile, depositing ``ammo.wall_damage``
      there (the same :func:`chew_wall` path as bullets — a beam bites).
    - HITSCAN does not crit and does not consult cover in v1 —
      skewer + attenuation is its identity (mechanics/03 §5); with
      ``crit_chance = 0`` on every Lance row the lazy-roll rule already
      guarantees zero draws beyond the cone.
    - Visual: one :class:`Shot` tracer (legacy list) + one
      :class:`LaserFiredEvent` carrying the segment + kind for the distinct
      beam draw. Glow-as-light-source is DEFERRED to the explosion-light
      pass (mechanics/03 §8).

    #169: the march decides against the slot-start state and records its
    packets, its chew and the shooter's facing on ``effects`` (``None`` =
    apply at once): a skewered unit another shot kills this tick still takes
    the beam, and the beam stops on a wall another shot destroys this tick.
    """
    fx = effects_or_immediate(effects)
    tables = weapon_tables()
    if ammo is None:
        ammo = tables.ammo_for_weapon(weapon)
    if spread_deg is None:
        spread_deg = unit_traits.cone_deg_of(weapon)
    cone_q = unit_traits.cone_q16(shooter, spread_deg)     # #29 U1: SPREAD
    # FREE-AIM SEAM (free_aim_shooting_design §4a): see fire_burst — aim_q is
    # the march-convention (screen y-down) Q16.16 bearing; None = tile-derived
    # (WEGO), provided = march straight from it (the facing is the turning
    # law's, left untouched). Integer since #29 P3e.
    dx_q = unit_fixed.quantize_scalar(fx2 - fx1)
    dy_q = unit_fixed.quantize_scalar(fy2 - fy1)
    if aim_q is None:
        base_q = unit_fixed.atan2_q16(dy_q, dx_q)
        # Facing = aim bearing at fire, before spread (the fire_burst rule).
        fx.face(shooter, unit_fixed.atan2_q16(-dy_q, dx_q))
    else:
        base_q = int(aim_q)
    ox_q = unit_fixed.quantize_scalar(fx1)
    oy_q = unit_fixed.quantize_scalar(fy1)
    shooter_id = getattr(shooter, "id", -1)
    h, w = gmap.material.shape
    # Per-gas Q16.16 absorption (once-at-build constants; plain ints), Q16 per
    # REAL unit of density. Only the TRACE planes absorb (the gases door refuses
    # an optical coefficient on a bulk row, #88 V10), and each term finishes
    # through the trace currency's one shift, trace_density_shr.
    absorb = getattr(getattr(gmap, "gases", None), "beam_absorb_q16", None)
    gas = getattr(gmap, "gas", None)
    n_gases = N_TRACE_GASES if (absorb is not None and gas is not None) else 0
    one = unit_fixed.FP_ONE

    for _ in range(int(weapon.shots_per_trigger)):
        angle_q = base_q + _draws.cone_offset_q16(
            rng.next(Salt.BEAM_CONE, shooter_id), cone_q)
        # #185: the beam's hit direction, its own kit step (Q16 ints)
        sx_q = unit_fixed.cos_q16(angle_q)
        sy_q = unit_fixed.sin_q16(angle_q)
        rx_q, ry_q = ox_q, oy_q
        energy_q = int(ammo.damage) << unit_fixed.FP_SHIFT   # door 2 (exact)
        skewered = []   # units already hit by THIS beam (identity — hit once)

        for _step in range(int(weapon.range_tiles)):
            rx_q += sx_q
            ry_q += sy_q
            ix = unit_fixed.shr_trunc(rx_q)
            iy = unit_fixed.shr_trunc(ry_q)
            if not (0 <= iy < h and 0 <= ix < w):
                break

            # 1. Attenuate (integer Beer-Lambert over the tile's gas column).
            if n_gases:
                total_absorb = 0
                for g in range(n_gases):
                    density_q = int(gas[g, iy, ix])
                    if density_q > 0:
                        total_absorb += gas_fixed.trace_density_shr(
                            int(absorb[g]) * density_q)
                if total_absorb > 0:
                    trans_q = one - total_absorb
                    if trans_q <= 0:
                        energy_q = 0
                    else:
                        energy_q = (energy_q * trans_q) >> unit_fixed.FP_SHIFT
            if energy_q < BEAM_MIN_ENERGY_Q16:
                break   # extinguished inside the cloud — nothing deposited

            # 2. Interact: first solid stops the beam (and takes the bite).
            if gmap.solid[iy, ix]:
                fx.world(chew_wall, gmap, iy, ix, ammo.wall_damage)
                break

            # 2b. PHYSICAL COVER (onephase_wego design §7): a beam is stopped
            # by a crate exactly as a bullet is — cover eats whatever clips it,
            # and a laser is emphatically something that clips. Placed after
            # the solid check and BEFORE the skewer so a crate protects the
            # bodies behind it. ``cover=None`` (every shipped caller) skips
            # this entirely, so the pre-existing beam is byte-identical.
            if cover:
                hit_cover = cover_at(cover, unit_fixed.dequantize_scalar(rx_q),
                                     unit_fixed.dequantize_scalar(ry_q))
                if hit_cover is not None:
                    fx.world(hit_cover.chew, ammo.wall_damage)
                    break

            # 3. Skewer: every living unit whose footprint holds this tile
            #    takes the CURRENT energy; the beam marches on undiminished.
            for e in units:
                if e is shooter or not e.alive:
                    continue
                if any(e is s for s in skewered):
                    continue
                if (e.tile_x <= ix < e.tile_x + e.footprint
                        and e.tile_y <= iy < e.tile_y + e.footprint):
                    skewered.append(e)
                    amount = (energy_q + (one >> 1)) >> unit_fixed.FP_SHIFT
                    if amount > 0:
                        fx.hit(
                            e,
                            DamagePacket(amount=amount, dtype=ammo.dtype_id,
                                         attacker_id=shooter_id, ap=ammo.ap,
                                         expose=getattr(ammo, "expose_q", 0),
                                         dir_x_q=sx_q, dir_y_q=sy_q),
                            events, source="laser")

        rx = unit_fixed.dequantize_scalar(rx_q)   # render-only views (exact)
        ry = unit_fixed.dequantize_scalar(ry_q)
        shots.append(Shot(fx1, fy1, rx, ry, real_time))
        if events is not None:
            events.append(LaserFiredEvent(
                unit_id=shooter_id,
                from_tile=(fx1, fy1),
                to_tile=(rx, ry),
            ))


# ---------------------------------------------------------------------------
# SPRAY (mechanics/03 §1/§5, W4): a sustained cone of FIELD WRITES — no
# projectile entity, no unit code. The two-terminals invariant, hard: this
# section writes ONLY world fields (heat / gas) through the FieldEdit queue.
# Units standing in the flames/cloud are damaged by the EXISTING exchange
# rows — heat | max (apply_environmental_damage, step 9c) and gas[poison]
# (apply_poison_dose, step 9c3) — ZERO new damage code, and a test asserts
# no W4 path touches unit HP.
#
# v1 burst model (the stationary rule, documented of record):
#   - a burst starts ONLY from an EXPLICIT fire order while the unit has no
#     movement order in the same phase (the sprayer stands still — a braced
#     hose, not a fire-on-the-move weapon);
#   - Move & Attack auto-fire SKIPS spray weapons entirely;
#   - one trigger = one burst = weapon.burst_ticks consecutive ticks of cone
#     deposits (1.5 s -> 36 @ 24 tps), deposited in the shooting slot;
#   - a standing fire order re-triggers the next burst as soon as the last
#     one ends (continuous hosing) until the mag runs dry — mag_size counts
#     BURSTS (W3 machinery: mag_gate / mag_spend per trigger);
#   - interruption: composed can_act going False (stun / knockdown /
#     paralysis) stops the burst THAT tick, the fire order is CONSUMED, and
#     the burst does not resume when the status clears.
#
# Determinism (engine/14): NO RNG anywhere. Aim bearing + cone cosine
# through the deterministic kit (unit_fixed — door 1); cone membership is
# PURE INTEGER (squared Q16.16 compare, below); falloff is an exact IEEE
# divide by an integer distance (door 3), quantized ONCE at the FieldEdit
# combine (door 2); traversal is fixed row-major.
# ---------------------------------------------------------------------------
# source_id namespace for spray-issued edits (engine/13 stable-sort key):
# physics.py owns 1 (_SRC_EXPLOSION) and 2 (_SRC_EXPLOSION_SMOKE),
# payloads.py 3 (gas) and 4 (ignite); the spray continues the sequence.
_SRC_SPRAY_HEAT = 5
_SRC_SPRAY_GAS = 6


def spray_cone_tiles(gmap, ay, ax, target_fx, target_fy, range_tiles,
                     cone_half_angle_degrees, exclude=(), aim_q=None):
    """Yield ``(y, x, falloff_div)`` for every tile of an aimed spray cone,
    in FIXED ROW-MAJOR order (mechanics/03 §5, W4).

    Apex = the shooter's centre tile ``(ay, ax)`` (integers). Membership is
    INTEGER-SAFE — no per-tile atan2, no per-tile float compare:

        tile_dir = (dx, dy) = (x - ax, y - ay)            # plain ints
        aim_q    = (cos_q, sin_q) of the kit aim bearing  # Q16.16 ints
        dot_q    = dx*cos_q + dy*sin_q                    # Q16.16 int, exact
        member  <=>  dot_q >= 0  AND
                     dot_q^2 >= (dx^2 + dy^2) * c_q^2     # Q32.32 int compare

    which is ``dot(tile_dir, aim_dir) >= |tile_dir| * cos(half_angle)`` with
    both sides squared (valid for the non-negative branch; half-angles are
    < 90°, so the ``dot_q >= 0`` gate loses nothing). ``c_q`` is the cone
    cosine through the deterministic kit (``cos_rad`` of the half-angle —
    the exact n/65536 value, computed once per call; door 1). The aim
    bearing is the kit ``atan2`` from apex to the order target.

    Range: ``dx^2 + dy^2 <= range_tiles^2`` (integer). The apex tile itself
    (``dx == dy == 0``) is never a member. ``exclude`` is a set of
    ``(tx, ty)`` tiles skipped ON TOP of membership — the caller passes the
    shooter's own footprint (the NOZZLE RULE: the jet projects beyond the
    operator's body, so the sprayer never hoses itself; without it the
    3x3 footprint's ring tiles sit at distance 1 inside every cone).

    Occlusion: a member tile is yielded only if ``gmap.has_los(ay, ax, y,
    x)`` — flames do not pour through walls. The Bresenham check returns
    True for a SOLID target tile with a clear path (it tests the tiles
    CROSSED, not the endpoint), so the flame lands ON a wall face — which
    is exactly how the wood wall receives its heat — but never beyond it.

    ``falloff_div`` is the integer falloff divisor — the documented
    1/distance form: ``max(1, isqrt(dx^2 + dy^2))`` (``math.isqrt`` — exact
    integer floor square root, door 1). The caller authors each deposit as
    ``amount / falloff_div`` (one correctly-rounded IEEE divide, door 3).
    """
    h, w = gmap.material.shape
    ay = int(ay)
    ax = int(ax)
    r = int(range_tiles)
    r_sq = r * r

    # Aim bearing + unit vector through the kit; the exact n/65536 doubles
    # scale to Q16.16 ints losslessly (quantize of an exact n/65536 is n).
    #
    # FREE-AIM SEAM (free_aim_shooting_design §4c): ``aim_q`` (Q16.16 radians,
    # march convention, screen y-down — the same frame this atan2 produces)
    # overrides the target-tile-derived bearing. ``None`` = tile-derived (the
    # WEGO spray path). Provided = the free-aim shooter's facing bearing; no
    # target tile is consulted for the cone geometry. #29 P3e: integers (the
    # float round trip was exact: bit-identical).
    if aim_q is None:
        angle_q = unit_fixed.atan2_q16(
            unit_fixed.quantize_scalar(target_fy - ay),
            unit_fixed.quantize_scalar(target_fx - ax))
    else:
        angle_q = int(aim_q)
    cos_q = unit_fixed.cos_q16(angle_q)
    sin_q = unit_fixed.sin_q16(angle_q)
    # Cone cosine through the kit, the half-angle at its door.
    # a spray JET's cone, not a shot cone: no SPREAD hook (#29 U1)
    c_q = unit_fixed.cos_q16(cone_q16_of_deg(cone_half_angle_degrees))  # traits-class: a jet
    c_sq = c_q * c_q                                   # Q32.32, exact int

    for y in range(max(0, ay - r), min(h - 1, ay + r) + 1):
        dy = y - ay
        for x in range(max(0, ax - r), min(w - 1, ax + r) + 1):
            dx = x - ax
            dist_sq = dx * dx + dy * dy
            if dist_sq == 0 or dist_sq > r_sq:
                continue                               # apex / out of range
            if (x, y) in exclude:
                continue                               # the nozzle rule
            dot_q = dx * cos_q + dy * sin_q            # Q16.16 int, exact
            if dot_q < 0:
                continue                               # behind the shooter
            if dot_q * dot_q < dist_sq * c_sq:
                continue                               # outside the cone
            if not gmap.has_los(ay, ax, y, x, physical=True):
                continue                               # occluded — no pour
            yield y, x, max(1, math.isqrt(dist_sq))


def deposit_spray_cone(gmap, queue, shooter, weapon, ammo,
                       target_fx, target_fy, aim_q=None, effects=None):
    """Enqueue ONE tick of a spray burst's cone deposits (mechanics/03 §5).

    Per member tile (fixed row-major order from :func:`spray_cone_tiles`),
    falloff-scaled by the documented 1/distance divisor:

    - ``ammo.heat_deposit > 0`` (the Dragon-7's fuel round): a TILE ADD
      FieldEdit into the ``heat`` field — the engine/06 ingress buffer the
      C++ TemperatureSolver converts (heat -> temperature -> ignition) and
      the heat|max exchange row samples for unit damage. Quantized ONCE at
      the FieldEdit heat combine (Q16.16 saturating add); heat lands on
      walls too (no skip-mask) — that is how wood catches.
    - ``ammo.gas_species`` nonempty: a TILE ADD FieldEdit into that gas
      slice (the W3 ``field="gas"`` + ``channel`` path — resolved BY NAME
      via ``gmap.gases.name_to_id``, the emit_gas rule), [0, 1] clamp +
      solid skip-mask from the gas policy. Fuel haze for the Dragon-7,
      the poison cloud itself for the Miasma Vent.

    NO RNG, no unit reads, no unit writes. The queue flush (step 6b)
    applies everything before the physics solvers run, so this tick's
    flame heat converts to temperature THIS tick.

    CREDIT (#123): the cone's tiles are claimed for the sprayer in the
    map's :class:`simulation.damage.TickCredit` (heat and/or the gas slice),
    so a unit the jet's own deposit hurts THIS tick is credited to the
    sprayer; a fire the jet lit, or the cloud left after it, is the
    environment's (the accepted gap — no per-tile provenance).

    #169: the cone (its member tiles, occluded by the walls standing at the
    attack slot's start) is decided here; the enqueue + claims are recorded
    on ``effects`` and run at its turn in phase B (``None`` = at once), so
    the slot's world writes keep one order.
    """
    owner = getattr(shooter, "id", -1)
    # #29 P3e: the row's per-tick amounts in their fields' currencies,
    # quantized once (the doors); each tile's share divided in integers.
    heat_q = unit_fixed.quantize_scalar(
        float(ammo.heat_deposit))                              # p3e-door
    gas_q = gas_fixed.quantize_trace_scalar(float(ammo.gas_amount))  # p3e-door
    gas_id = None
    if ammo.gas_species:
        gas_id = int(gmap.gases.name_to_id[ammo.gas_species])
    own = set(shooter.occupied_tiles())
    cells = list(spray_cone_tiles(
        gmap, shooter.center_tile_y(), shooter.center_tile_x(),
        target_fx, target_fy, weapon.range_tiles,
        weapon.cone_half_angle_degrees, exclude=own, aim_q=aim_q))
    effects_or_immediate(effects).world(
        _enqueue_spray_cells, gmap, queue, owner, heat_q, gas_id, gas_q,
        cells)


def _enqueue_spray_cells(gmap, queue, owner, heat_q, gas_id, gas_q, cells):
    """The world write of one spray tick: per member ``(y, x, div)`` in the
    cone's row-major order, the heat and / or gas FieldEdit and its
    TickCredit claim (:func:`deposit_spray_cone`'s body)."""
    credit = tick_credit(gmap)
    for y, x, div in cells:
        if heat_q > 0:
            a_q = unit_fixed.div_round(heat_q, div)
            queue.enqueue(FieldEdit(
                field="heat", region=Region.TILE, coords=(y, x),
                amount=unit_fixed.dequantize_scalar(a_q), amount_q=a_q,
                mode=EditMode.ADD, source_id=_SRC_SPRAY_HEAT,
            ))
            credit.claim(CREDIT_HEAT, ((y, x),), owner)
        if gas_id is not None and gas_q > 0:
            credit.claim(gas_id, ((y, x),), owner)
            g_q = unit_fixed.div_round(gas_q, div)
            queue.enqueue(FieldEdit(        # p3e-view: amount_q lands
                field="gas", region=Region.TILE, coords=(y, x),
                amount=float(gas_fixed.dequantize_trace(g_q)), amount_q=g_q,
                mode=EditMode.ADD,
                channel=gas_id,   # policy TRACE_CLAMP: additive (D1)
                source_id=_SRC_SPRAY_GAS,
            ))


def start_spray_burst(u, weapon, fire_order, tick, effects=None):
    """Arm a spray burst on ``u`` (the trigger side of the SPRAY archetype).

    Called from :func:`process_shooting` once every gate has passed
    (cadence, mag, range, LOS, the stationary rule). The burst state lives
    ON THE UNIT (``spray_ticks_left`` / ``spray_target`` / ``spray_order``)
    and is deliberately NOT in the synced digest surface — the
    ``last_fire_tick`` / mag-state precedent: a deterministic derivation of
    synced inputs (orders + tick) whose divergence would surface in the
    hashed field/hp stream one tick later. Facing snaps to the aim bearing
    (the fire_burst rule — kit trig, unit y-up convention).
    """
    u.spray_ticks_left = int(weapon.burst_ticks)
    # #29 P3e: the target as a Q16 point (the order's tile, quantized once).
    u.spray_target = (unit_fixed.quantize_scalar(fire_order.target_fx),
                      unit_fixed.quantize_scalar(fire_order.target_fy))
    u.spray_order = fire_order
    u.spray_aim_angle = None   # tile-derived cone bearing (the WEGO path)
    effects_or_immediate(effects).face(u, unit_fixed.atan2_q16_of(
        -(float(fire_order.target_fy) - u.center_tile_y()),
        float(fire_order.target_fx) - u.center_tile_x()))


def start_spray_burst_directional(u, weapon, tick, aim_q):
    """Arm a spray burst aimed along a FREE-AIM bearing (design §4c): the cone
    bearing is ``u.facing`` (passed as the march-convention ``aim_angle``), NOT
    a target tile. Stores the bearing (Q16, #29 P3e) as ``spray_aim_angle`` —
    derived state (the ``last_fire_tick`` precedent, not in the digest surface)
    that :func:`process_sprays` feeds :func:`deposit_spray_cone`. ``spray_target``
    is kept for the RENDER-ONLY :class:`SprayJetEvent` only (a nominal tile one
    range along the bearing); the cone geometry uses ``spray_aim_angle``.
    ``facing`` is left as the AIM intent set it — never re-derived."""
    u.spray_ticks_left = int(weapon.burst_ticks)
    # #29 P3e: the bearing a Q16 angle, the nominal target a Q16 point.
    u.spray_aim_angle = int(aim_q)
    u.spray_order = None
    r_q = unit_fixed.quantize_scalar(float(weapon.range_tiles))  # p3e-door
    u.spray_target = (
        (u.center_tile_x() << unit_fixed.FP_SHIFT)
        + unit_fixed.mul_q16_round(unit_fixed.cos_q16(aim_q), r_q),
        (u.center_tile_y() << unit_fixed.FP_SHIFT)
        + unit_fixed.mul_q16_round(unit_fixed.sin_q16(aim_q), r_q))


def process_sprays(gmap, units, queue, events=None, effects=None):
    """One tick of every active spray burst — the W4 deposit pass, invoked
    from the conductor in the SHOOTING SLOT (directly after
    :func:`process_shooting`; call line only in simulation.py).

    ``events`` (W6 — the flame-jet visual): when a list is passed (the
    sim's ``tick_events``), every tick that actually DEPOSITS also appends
    one RENDER-ONLY :class:`SprayJetEvent` carrying the aimed cone
    (apex / target / range / half-angle / kind), the LaserFiredEvent
    precedent — a pure function of already-synced state. The parameter
    carries NO generator and the jet event can touch no unit: the
    two-terminals invariant is unchanged (the runtime no-HP proof in
    tests/test_spray_weapons.py still stands; only UnitHit/UnitKilled
    are digest-hashed, so the jet moves no digest).

    Fixed stored-unit order (the apply_environmental_damage convention —
    order-free anyway: each burst writes through the stable-sorted edit
    queue). Per unit with ``spray_ticks_left > 0``:

    - dead: the burst dies with the unit (state cleared, nothing deposited);
    - composed ``can_act`` False (mechanics/06 §4): INTERRUPTION — the
      burst stops THIS tick (no deposit), the originating fire order is
      CONSUMED (removed from the queue), no resume when the status clears;
    - otherwise: deposit one cone tick (:func:`deposit_spray_cone`) toward
      the burst's captured target and count the burst down. The weapon/ammo
      rows are re-resolved from ``unit.weapon_id`` each tick (config-static
      within a run — same rows every tick).

    Draws NO randomness (the queue's noise machinery is unused — spray
    edits carry ``noise = 0``), so a spray-free trajectory is bit-identical
    to pre-W4 (the dormancy gate).

    SIMULTANEOUS (#169): this is the second half of the shooting slot's
    phase A. The conductor passes the slot's ``effects`` (the one
    :func:`process_shooting` recorded on), so a burst's alive / ``can_act``
    check reads the slot-start state -- a sprayer another shooter kills or
    stuns this tick still deposits this tick, and stops on the next -- and
    the deposits are recorded after the shots, in the order they ran
    before. ``effects=None`` = this call resolves its own before returning.
    """
    tables = weapon_tables()
    own_slot = effects is None
    fx = SimultaneousEffects() if own_slot else effects
    for u in units:
        ticks_left = getattr(u, "spray_ticks_left", 0)
        if ticks_left <= 0:
            continue
        if not u.alive:
            u.spray_ticks_left = 0
            u.spray_order = None
            u.spray_target = None
            u.spray_aim_angle = None
            continue
        if not composed_flags(u).can_act:
            # Interruption: stop NOW, consume the order, never resume.
            order = getattr(u, "spray_order", None)
            if order is not None and order in u.orders:
                u.orders.remove(order)
            u.spray_ticks_left = 0
            u.spray_order = None
            u.spray_target = None
            u.spray_aim_angle = None
            continue
        weapon = tables.weapons.by_name[u.weapon_id]
        ammo = tables.ammo_for_weapon(weapon)
        tx_q, ty_q = u.spray_target            # Q16 (#29 P3e)
        tx = unit_fixed.dequantize_scalar(tx_q)
        ty = unit_fixed.dequantize_scalar(ty_q)
        # Free-aim bursts carry an exact facing bearing (design §4c); WEGO
        # bursts leave it None -> the tile-derived bearing.
        aim = getattr(u, "spray_aim_angle", None)
        deposit_spray_cone(gmap, queue, u, weapon, ammo, tx, ty, aim_q=aim,
                           effects=fx)
        if events is not None:
            # W6: the jet visual — one event per DEPOSITING tick (an
            # interrupted / dead / finished burst emits nothing). "flame"
            # for heat-carrying rounds (the Dragon family), "miasma" for
            # gas-only sprays (the fainter, sickly renderer variant).
            events.append(SprayJetEvent(
                unit_id=getattr(u, "id", -1),
                from_tile=(u.center_tile_x(), u.center_tile_y()),
                to_tile=(float(tx), float(ty)),
                range_tiles=int(weapon.range_tiles),
                cone_half_angle_degrees=float(weapon.cone_half_angle_degrees),
                kind=("flame" if float(ammo.heat_deposit) > 0 else "miasma"),
            ))
        u.spray_ticks_left = ticks_left - 1
        if u.spray_ticks_left <= 0:
            u.spray_order = None
            u.spray_target = None
            u.spray_aim_angle = None
    if own_slot:
        fx.resolve()                     # phase B


# ---------------------------------------------------------------------------
# MELEE (mechanics/03 §1/§5, W5): adjacency + the §3 resolver. The knife and
# the arc baton — the last live archetype branch of the closed set.
#
# The resolver collapses for melee exactly as the chapter says: TO-HIT IS
# TRIVIALLY 1.0 — a strike happens at touching footprints, so there is no
# intervening tile to be cover and no march to absorb; the exposure roll
# does not exist on this path (NOT "always passes": it is NEVER DRAWN — the
# lazy-roll rule). What remains is the crit-vs-facing roll (the knife's
# assassin fantasy: crit_chance 0.15 x the behind-arc x4) — the SAME
# attack_resolver seams the bullet march uses, drawn LAZILY (crit_chance 0,
# the baton, draws nothing).
#
# Statuses are applied AT THE DELIVERY SITE (the §1 two-terminals wording:
# "a baton applies STUNNED where it connects; packets themselves stay
# damage-only"): melee_strike applies the packet through apply_packet and
# THEN applies the weapon row's status_kind separately through
# simulation.status.apply_status — the W3 TEARGAS->BLINDED pattern (the
# coupling row applies BLINDED at the exposure site; the DamagePacket type
# has no status field to smuggle one through). A strike that KILLS applies
# no status — corpses don't get stunned (statuses freeze on corpses,
# mechanics/06 §4, so a corpse status would be dead digest weight).
#
# Zombie melee is NOT this path: ai_zombie.py keeps its shipped bite
# (center-distance threshold + CFG.zombie.melee_damage + the converting
# kill) untouched; migrating NPC attacks onto weapon rows is future work
# (mechanics/03 §7).
# ---------------------------------------------------------------------------
def melee_adjacent(attacker, target):
    """THE ADJACENCY PREDICATE (of record, W5): two units are melee-adjacent
    iff SOME occupied tile of one is within CHEBYSHEV DISTANCE 1 of SOME
    occupied tile of the other — 8-connected footprint contact: edge contact
    AND diagonal corner contact both count, and overlapping footprints
    (distance 0) count trivially.

    Exact for ANY footprint shape (it walks ``occupied_tiles()`` pairwise —
    the spec §6 occupancy interface — not a bounding box), so a future
    non-square rig gets the right answer for free; for today's square
    footprints it equals "the two anchor rectangles, one dilated by 1,
    intersect". Cost is bounded by the footprints (3x3 vs 3x3 = 81 integer
    compares) and only paid per trigger attempt, never per tick.

    Pure integer arithmetic on synced tile state (door 1); consumes nothing.
    Deliberately NO ``has_los`` term: touching footprints have no tile
    between them to occlude (diagonal corner contact across a wall corner
    therefore CAN stab — accepted v1, documented in mechanics/03 §5).
    """
    a_tiles = attacker.occupied_tiles()
    for (tx, ty) in target.occupied_tiles():
        for (ax, ay) in a_tiles:
            if -1 <= ax - tx <= 1 and -1 <= ay - ty <= 1:
                return True
    return False


def melee_strike(units, attacker, target_fx, target_fy, rng, events, weapon,
                 effects=None):
    """Resolve one melee trigger pull (mechanics/03 §5, W5). Returns True if
    a strike CONNECTED (the caller then charges the rof cadence); False = a
    whiff — no target at the order tile / not adjacent — which costs
    nothing and is retried next tick while the order stands.

    Target resolution: the FIRST living enemy in stored unit order whose
    footprint occupies the order's target tile (``int()`` tiling, the march
    convention) — deterministic, and the order names a TILE, not a unit
    (the shipped fire-order shape; a moved target whiffs honestly).

    The resolved strike, in fixed order (all inputs synced, engine/14):

    1. facing snaps to the strike bearing (the fire_burst rule — kit atan2,
       unit y-up convention), from attacker centre to TARGET centre;
    2. amount = ``weapon.melee_damage`` (door-2 row int). No zombie
       ``bullet_damage_multiplier`` — that is the BULLET site rule
       (mechanics/06: a shipped-numbers artifact of the rifle path, not a
       resistance; melee packets take plain mitigation);
    3. the crit roll (LAZY: only if ``crit_chance > 0``): arc multiplier
       off the target's synced facing vs the strike angle (screen
       convention — the BulletInFlight.advance shape), one door-4 uniform,
       amount scales by ``crit_mult`` in exact ints on success;
    4. the packet through the pipeline (``apply_packet``, source="melee";
       ``mark_killed_by_zombie`` stays False — player melee kills never
       convert, conversion is the ZOMBIE bite's semantics);
    5. the delivery-site status (``weapon.status_kind``, if any) on a
       target still alive: ``apply_status`` with the row's derived
       ``status_ticks`` (magnitude 0 — pure CC; the packet already carried
       the damage). Applied AFTER the packet so "the baton stuns where it
       connects" and a killing blow stuns no corpse.

    #169: target, adjacency and the crit roll are decided here against the
    slot-start state; the facing snap, the packet and the status are
    RECORDED on ``effects`` in that order (``None`` = apply at once), so a
    striker another attack kills or stuns this tick still lands its strike.
    """
    fx = effects_or_immediate(effects)
    tx, ty = int(target_fx), int(target_fy)
    target = None
    for e in units:
        if e is attacker or not e.alive or e.team == attacker.team:
            continue
        if e.occupies((tx, ty)):
            target = e
            break
    if target is None:
        return False
    if not melee_adjacent(attacker, target):
        return False
    melee_blow(attacker, target, rng, events, weapon, fx, snap_facing=True)
    return True


def melee_blow(attacker, target, rng, events, weapon, effects,
               snap_facing=True):
    """The strike itself, shared by the order path (:func:`melee_strike`,
    which found its target by tile) and the melee verb
    (:func:`process_melee_intents`, which found it by facing arc): the
    steps 1-5 of :func:`melee_strike`'s docstring, recorded on ``effects``.
    ``snap_facing`` False (the verb: the arc IS the aim, and the turning law
    owns the facing) records no facing snap. A row with ``crit_chance`` > 0
    draws from ``rng`` (the order path only: a loadout's melee row is
    refused with a crit at the loadouts door)."""
    fx = effects_or_immediate(effects)
    fx1, fy1 = attacker.center_tile_x(), attacker.center_tile_y()
    fx2, fy2 = target.center_tile_x(), target.center_tile_y()
    # Facing = strike bearing (kit trig, unit y-up convention — the
    # fire_burst rule); the strike angle itself stays in SCREEN convention
    # for the arc classifier (the march-angle contract of arc_multiplier).
    if snap_facing:
        fx.face(attacker, unit_fixed.atan2_q16_of(-(fy2 - fy1), fx2 - fx1))
    angle_q = unit_fixed.atan2_q16((fy2 - fy1) << unit_fixed.FP_SHIFT,
                                   (fx2 - fx1) << unit_fixed.FP_SHIFT)

    crit_mult = None
    crit_chance = float(weapon.crit_chance)
    if crit_chance > 0.0:
        mult = attack_resolver.arc_multiplier(angle_q, target)
        d = rng.next(Salt.MELEE_CRIT, getattr(attacker, "id", -1))
        if attack_resolver.roll_crit(crit_chance, mult, d):
            crit_mult = weapon.crit_mult
    # #29 U1: THE STRIKE accessor -- the row's melee_damage (a crit applied
    # on whole points, as before) times the striker's hook, in Q16.
    amount_q = unit_traits.strike_damage_q(attacker, weapon,
                                           crit_mult=crit_mult)

    # #185: the blow travels striker centre -> victim centre (Q16)
    from simulation.unit_impulse import centre_q
    (ax_q, ay_q), (tx_q, ty_q) = centre_q(attacker), centre_q(target)
    fx.hit(target,
           DamagePacket(amount=unit_fixed.dequantize_scalar(amount_q),  # p3e-view
                        amount_q=amount_q, dtype=weapon.melee_dtype_id,
                        attacker_id=getattr(attacker, "id", -1),
                        dir_x_q=tx_q - ax_q, dir_y_q=ty_q - ay_q),
           events, source="melee")

    if weapon.status_kind_id is not None:
        # Applied at its turn only if the target is still alive then.
        fx.status(target, weapon.status_kind_id, magnitude=0,
                  duration_ticks=weapon.status_ticks,
                  source_id=getattr(attacker, "id", -1))


def melee_arc_target(units, u, weapon):
    """The unit ``u``'s melee verb strikes (gameplay pack G6p): among the
    other LIVING units of a DIFFERENT team (a zombie's team is its own, so
    zombies are enemies; a teammate never is) that are
    :func:`melee_adjacent` to ``u`` and whose bearing from ``u``'s centre
    lies within ``weapon.melee_arc_half_q`` of ``u``'s facing, the one
    nearest that facing (smallest ``|wrap(bearing - facing)|``), ties to the
    lower id. ``None`` when there is none. Integer only (``atan2_q16`` of
    the Q16 centre offset against ``Unit.facing_q``,
    ``turning.wrap_q16``), no random number; reads the slot-start state. A
    target whose centre coincides with ``u``'s has no bearing: it counts as
    dead ahead."""
    from simulation import turning, unit_impulse
    cx, cy = unit_impulse.centre_q(u)
    best, best_key = None, None
    for t in units:
        if t is u or not t.alive or t.team == u.team:
            continue
        if not melee_adjacent(u, t):
            continue
        tx, ty = unit_impulse.centre_q(t)
        dx, dy = tx - cx, ty - cy
        off = 0 if (dx == 0 and dy == 0) else abs(turning.wrap_q16(
            unit_fixed.atan2_q16(-dy, dx) - u.facing_q))
        if off > weapon.melee_arc_half_q:
            continue
        key = (off, int(t.id))
        if best_key is None or key < best_key:
            best, best_key = t, key
    return best


def process_melee_intents(units, tick, tables, impulse_table, events=None,
                          effects=None):
    """THE MELEE VERB (gameplay pack G6p, ``Simulation.melee_intent``):
    slot 4's phase A walk over the units' latched ``pending_melee``. Every
    latch is CONSUMED here (a verb that finds nothing to do is not kept).
    A unit strikes only if it is alive, not a zombie (whose bite is its own
    law), has a melee row (``melee_id``), can act (composed ``can_act``),
    is not winding up a throw, its row's cadence has run
    (``rof_interval_ticks`` since ``last_blow_tick``), and
    :func:`melee_arc_target` finds a target; a reload stall does not stop a
    blow (the butt needs no magazine). The blow is :func:`melee_blow`'s
    (a ``DamagePacket`` of the row's damage credited to the striker with the
    ``melee`` cause, the row's delivery-site stun, no facing snap) plus a
    ``push`` effect of the row's ``melee_push_ns`` along striker -> target;
    all RECORDED on ``effects`` (decided against the slot-start state,
    applied together at phase B), and a render-only ``MeleeEvent`` is
    emitted. A whiff (nothing in arc) costs nothing: no cooldown. Integer,
    no draw (the loadouts door refuses a row with a crit)."""
    from simulation import unit_impulse, verb_gates
    fx = effects_or_immediate(effects)
    for u in units:
        if not getattr(u, "pending_melee", False):
            continue
        u.pending_melee = False
        if not verb_gates.can_melee(u, tick, tables, at_tick_start=False):
            continue
        weapon = tables.weapons.by_name[u.melee_id]
        target = melee_arc_target(units, u, weapon)
        if target is None:
            continue
        u.last_blow_tick = int(tick)
        melee_blow(u, target, None, events, weapon, fx, snap_facing=False)
        # #29 U1: THE PUSH accessor (the row's impulse times the striker's
        # hook)
        j_q = unit_traits.push_impulse_q(impulse_table, u, weapon)
        if j_q:
            cx, cy = unit_impulse.centre_q(u)
            tx, ty = unit_impulse.centre_q(target)
            bearing = (unit_fixed.atan2_q16(-(ty - cy), tx - cx)
                       if (tx != cx or ty != cy) else u.facing_q)
            sx, sy = unit_fixed.cos_q16(bearing), -unit_fixed.sin_q16(bearing)
            jx, jy = unit_impulse.along_q(j_q, sx, sy)
            fx.push(impulse_table, target, jx, jy)
        if events is not None:
            from simulation.events import MeleeEvent
            events.append(MeleeEvent(unit_id=int(u.id), target_id=int(target.id),
                                     ticks=int(weapon.rof_interval_ticks)))


# ---------------------------------------------------------------------------
# Door explosives (scheduled detonations at phase boundaries)
# ---------------------------------------------------------------------------
def process_door_explosives(gmap, queue, units, slot, rng, events=None):
    """Detonate every door-explosive order scheduled for ``slot``.

    Lifted from ``game.py:_process_door_explosives``. Called three times
    per round (start P1, between phases, end P2). Skips zombies — only
    player-issued orders detonate. ``rng`` rides through to the executor
    (the smoke noise itself is drawn at the queue flush); ``events``
    (optional) collects :class:`ExplosionEvent` and unit hit / kill events
    for the renderer.

    W3: the PLACED charge is executed through the payload EXECUTOR
    (:func:`simulation.payloads.execute_payload`) via its ROUND: the order's
    ``ammo_name`` (default ``demo_breach`` -> ``payloads.breach_focus`` —
    byte-identical to the pre-W3 inline triple, the replica gate) or
    ``demo_c4`` -> ``payloads.demolition_c4`` (the C4 satchel — same order
    flow, bigger warhead; selection UI is W6, tests name the round).
    """
    tables = weapon_tables()
    for u in units:
        if u.is_zombie:
            continue
        for o in u.orders:
            if o.order_type == ORDER_EXPLOSIVE and o.det_slot == slot:
                fy, fx = o.target_fy, o.target_fx
                payload = tables.payload_for_ammo(
                    getattr(o, "ammo_name", None) or "demo_breach")
                execute_payload(gmap, queue, units, fy, fx, payload, rng,
                                events=events, kind="door_explosive",
                                owner_id=getattr(u, "id", -1),
                                approach=(u.center_tile_y(),
                                          u.center_tile_x()))
