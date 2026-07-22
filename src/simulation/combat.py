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

Determinism — Phase 2:
    ``fire_burst`` and the explosion-smoke helper accept a
    :class:`numpy.random.Generator` (``rng`` parameter) for the bullet
    cone offsets and smoke noise. The Simulation facade owns one
    :class:`numpy.random.Generator` and plumbs it through these calls
    on every tick so AI rollouts can be reproduced bit-for-bit. If you
    invoke these helpers ad-hoc (e.g. from a test), pass an RNG with a
    known seed — the legacy fallback to process-global ``random`` has
    been removed.
"""
from __future__ import annotations

import math

import numpy as np

from config import CFG
from simulation import attack_resolver
from simulation import wall_fixed
from simulation.damage import KINETIC, DamagePacket, apply_packet
from simulation.events import (
    LaserFiredEvent, ProjectileGlowEvent, ShotFiredEvent, SprayJetEvent,
    ExplosionEvent,
)
from simulation.gases import N_GASES
from simulation.field_edit import EditMode, FieldEdit, Region
from simulation.orders import (
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE, ORDER_MOVE_ATTACK,
    MOVE_ORDER_TYPES,
)
from simulation import unit_fixed
# Physics event entry points: re-exported for legacy imports; the detonation
# sites themselves now go through the payload executor (W3). Bare-name import
# of execute_payload on purpose — instrumentation/replica tests rebind
# simulation.combat.execute_payload (the apply_environmental_damage pattern).
from simulation.physics import apply_explosion, add_explosion_smoke  # noqa: F401
from simulation.payloads import execute_payload
from simulation.status import apply_status, composed_flags
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
                 fuse_seconds, thrown_tick, ammo_name="grenade_frag"):
        self.proj_type = proj_type
        self.fx = float(start_fx)
        self.fy = float(start_fy)
        self.start_fx = float(start_fx)
        self.start_fy = float(start_fy)
        self.target_fx = float(target_fx)
        self.target_fy = float(target_fy)
        self.fuse_seconds = fuse_seconds
        self.thrown_tick = thrown_tick
        self.detonated = False
        self.ammo_name = ammo_name
        # Tiles-per-SECOND, from the round's row (W1 re-home: the old
        # CFG.weapons.grenade.travel_speed = 30.0, same float — the
        # update_position() arithmetic below is bit-identical). The round's
        # speed_tiles_per_tick twin is the W2 unified-march data-of-record.
        self.travel_speed = weapon_tables().ammo.by_name[
            ammo_name].travel_speed_tiles_per_second

    def get_detonate_tick(self):
        """Tick at which this projectile detonates."""
        tps = CFG.clock.ticks_per_second
        return self.thrown_tick + int(self.fuse_seconds * tps)

    def update_position(self, current_tick):
        """Move the projectile toward the target based on travel time so far."""
        tps = CFG.clock.ticks_per_second
        elapsed_sec = (current_tick - self.thrown_tick) / tps
        dx = self.target_fx - self.start_fx
        dy = self.target_fy - self.start_fy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            self.fx = self.target_fx
            self.fy = self.target_fy
            return
        travel_time = dist / self.travel_speed
        if elapsed_sec >= travel_time:
            self.fx = self.target_fx
            self.fy = self.target_fy
        else:
            frac = elapsed_sec / travel_time
            self.fx = self.start_fx + dx * frac
            self.fy = self.start_fy + dy * frac


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
    chew-less rounds cost nothing.
    """
    if wall_damage <= 0:
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
                 origin_fx, origin_fy, angle, step_x, step_y):
        if int(ammo.speed_q16) <= 0:
            raise ValueError(
                f"ammo.{ammo.name}.speed_tiles_per_tick must be > 0 for a "
                f"marching round (weapons.{weapon.name} fired it)")
        self.shooter = shooter          # excluded from hits (identity)
        self.shooter_id = shooter_id    # packet attribution
        self.weapon = weapon            # WeaponDef (crit columns, range)
        self.ammo = ammo                # AmmoDef (damage, dtype, ap, chew)
        self.angle = angle              # march angle, screen convention
        self.rx = float(origin_fx)      # live position (tile floats)
        self.ry = float(origin_fy)
        self.step_x = step_x            # kit-trig unit step (exact n/65536)
        self.step_y = step_y
        self.prev_ix = int(origin_fx)   # last tile CROSSED (cover inspection)
        self.prev_iy = int(origin_fy)
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

    def advance(self, gmap, units, shots, real_time, rng, events=None,
                queue=None):
        """March one tick's budget. Returns True while still in flight.

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
        """
        h, w = gmap.material.shape
        seg_x, seg_y = self.rx, self.ry          # tracer segment start

        # Integer step budget (door 1): whole tiles this tick, fraction carries.
        self.budget_q16 += self.speed_q16
        n_move = self.budget_q16 >> unit_fixed.FP_SHIFT
        self.budget_q16 -= n_move << unit_fixed.FP_SHIFT
        n_steps = n_move if n_move < self.remaining_steps else self.remaining_steps

        hit_unit = None
        stopped = False
        for _step in range(n_steps):
            px, py = self.rx, self.ry            # pre-step (absorption rollback)
            self.rx += self.step_x
            self.ry += self.step_y
            ix, iy = int(self.rx), int(self.ry)

            if 0 <= iy < h and 0 <= ix < w:
                if gmap.solid[iy, ix]:
                    # W2: missed shots chew walls — the apply_explosion path.
                    chew_wall(gmap, iy, ix, self.ammo.wall_damage)
                    stopped = True
                    break
            else:
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
                # Exposure vs cover (mechanics/06 §5): inspect the tile
                # crossed immediately BEFORE entering the footprint. LAZY:
                # exposure 1.0 (no concealment — the overwhelming case)
                # draws NOTHING; directional cover is geometric by
                # construction (a flanking approach never sees the crate).
                exposure = attack_resolver.cover_exposure_at(
                    gmap, self.prev_iy, self.prev_ix)
                if (exposure < 1.0
                        and not attack_resolver.roll_exposure(exposure, rng)):
                    # Absorbed by the cover tile: it eats the round's wall
                    # damage, the tracer ends there, no unit packet.
                    chew_wall(gmap, self.prev_iy, self.prev_ix,
                              self.ammo.wall_damage)
                    self.rx, self.ry = px, py
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
            if hit_unit.team == 1:  # zombie
                # SITE-SIDE pre-mitigation amount rule, deliberately NOT a
                # resistance: the int() truncation (and its position before
                # the packet) is part of the shipped numbers; the zombie's
                # KINETIC resist stays neutral (mechanics/06 — only the heat
                # ×4 dissolved into the mitigation tables).
                actual_dmg = int(actual_dmg * CFG.zombie.bullet_damage_multiplier)
            # Crit vs facing (mechanics/06 §5), LAZY: crit_chance == 0 (every
            # shipped weapon) draws nothing. On a crit the amount scales by
            # crit_mult in exact ints (round half away from zero), AFTER the
            # site rule, BEFORE mitigation.
            crit_chance = float(self.weapon.crit_chance)
            if crit_chance > 0.0:
                mult = attack_resolver.arc_multiplier(self.angle, hit_unit)
                if attack_resolver.roll_crit(crit_chance, mult, rng):
                    actual_dmg = attack_resolver.scale_half_away(
                        actual_dmg, self.weapon.crit_mult)
            # Packet through the pipeline (mechanics/06 §2) — dtype/ap off the
            # ammo row (the k5's kinetic/0 reproduces the shipped packet
            # exactly); bullet deaths never convert.
            apply_packet(hit_unit,
                         DamagePacket(amount=actual_dmg,
                                      dtype=self.ammo.dtype_id,
                                      source_id=self.shooter_id,
                                      ap=self.ammo.ap),
                         events, source="bullet")

        if n_steps > 0:   # a budget-starved tick (<1 tile) has nothing to draw
            shots.append(Shot(seg_x, seg_y, self.rx, self.ry, real_time))
            if events is not None:
                hit_id = getattr(hit_unit, "id", -1) if hit_unit else None
                events.append(ShotFiredEvent(
                    unit_id=self.shooter_id,
                    from_tile=(seg_x, seg_y),
                    to_tile=(self.rx, self.ry),
                    hit_target_id=hit_id,
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
            det_y, det_x = int(self.ry), int(self.rx)
            if not (0 <= det_y < h and 0 <= det_x < w):
                det_y, det_x = self.prev_iy, self.prev_ix
            execute_payload(gmap, queue, units, det_y, det_x, self.payload,
                            rng, events=events, kind="shell")

        return still_flying


# ---------------------------------------------------------------------------
# Ignition from temperature — engine/06 §4 ("Ignition"), proposal §6 step 4b
# ---------------------------------------------------------------------------
def apply_temperature_ignition(gmap, o2_threshold, ignition_seed):
    """Ignite flammable tiles whose `temperature` has crossed `ignition_temp`
    AND that have oxygen (engine/06 §4, proposal §6 step 4b).

    This is the READ side of the temperature substrate: the C++
    ``TemperatureSolver`` (convert -> conduction -> cooling, inside
    ``PhysicsRunner.step``) has already filled ``gmap.temperature`` for this
    tick; here we gather it and start fires. For each FLAMMABLE tile::

        if temperature[y,x] >= ignition_temp_q16[material]   # Q16.16 threshold
           and mean(atmosphere of air-side 4-neighbours) >= o2_threshold:   # O2
               fire[y,x] = max(fire[y,x], ignition_seed)      # never lowers a fire

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
    - **``max``, not assign.** ``fire = max(fire, ignition_seed)`` never lowers an
      existing, possibly larger, fire (e.g. one an explosion already set).
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

    # Hot enough? (Q16.16 integer compare.) Restrict to flammable tiles only.
    hot = flammable & (temperature.astype(np.int64) >= thresh_q16)
    if not bool(hot.any()):
        return  # nothing crossed its threshold this tick (the dormant case)

    # --- O2 proxy: mean REAL `gas[O2]` over the OPEN (non-wall, non-vacuum)
    # 4-neighbours — the EXACT predicate the C++ fire O2 check uses
    # (fire_simulation.cpp, mask `!is_wall && !is_vacuum`, with
    # `is_wall == gmap.solid`). EOS refactor P4 (design §6, item 3): this used
    # to read `atmosphere` (P, a pressure proxy) — it now reads the REAL bulk
    # O2 density plane (gmap.gas[O2]), the SAME re-pointing FireSimulation's
    # own O2 gate got in C++, so a tile still cannot ignite into a state the
    # fire step would immediately suffocate (the design invariant, unchanged —
    # only the underlying field did). S3a made this an INTEGER reduction on the
    # int32 Q16.16 field, bit-matching the integer mean the C++ fire adopts
    # (fixed_point.h mean_sum/mean_round): an int64 neighbour-sum + a
    # round-half-away-from-zero mean, then a Q16.16 threshold compare.
    #
    # WHY the mask excludes vacuum: the C++ O2 gate excludes vacuum neighbours
    # from BOTH sum and count; including them would lower the mean below the
    # C++ value (a vacuum tile holds no gas, but the count would still
    # increment). Excluding vacuum (matching the C++ mask) is what makes the
    # two O2 predicates bit-identical.
    h, w = temperature.shape
    open_nbr = (~gmap.solid) & (~gmap.is_vacuum)   # True == counts toward O2 (== C++)
    o2_idx = gmap.gases.name_to_id["o2"]
    o2_q = gmap.gas[o2_idx].astype(np.int64)       # int32 Q16.16 -> int64 (exact, order-free)
    sum_o2 = np.zeros((h, w), dtype=np.int64)      # Q16.16 sum (int64, no overflow)
    count = np.zeros((h, w), dtype=np.int64)       # neighbour count 0..4
    # N, S, E, W (fixed order; integer sum is order-independent regardless).
    for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        ys0, ys1 = max(0, -dy), h - max(0, dy)      # this-tile row span
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        nbr_o2 = o2_q[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        nbr_open = open_nbr[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        sum_o2[ys0:ys1, xs0:xs1] += np.where(nbr_open, nbr_o2, np.int64(0))
        count[ys0:ys1, xs0:xs1] += nbr_open.astype(np.int64)
    # Round-half-away-from-zero mean == fixed_point.h::mean_round (sign-symmetric,
    # no DC bias). A fully walled-in tile (count == 0) averages to 0 -> below
    # threshold -> never ignites (the C++ `count > 0 ? sum/count : 0` guard, here
    # via safe_count).
    #
    # NEGATIVE-BRANCH FIX (S3b, review carry-forward #2): the C++ mean_round divide
    # TRUNCATES TOWARD ZERO (C++ integer `/`), NOT toward -inf. Python `//` FLOORS
    # (toward -inf), so the two diverge on a NEGATIVE neighbour sum. EOS refactor
    # P4: `gas[O2]` is a transported bulk density, clamped >= 0 by construction
    # (bulk_transport.cpp's final clamp) — it can never actually go negative the
    # way the old `atmosphere` proxy could (wave forcing had no hard floor) — but
    # the shared trunc-toward-zero emulation is KEPT so this stays bit-identical
    # to the C++ mean_round on ANY input, not just the ones this field happens to
    # produce today: trunc(a/b) = -((-a)//b) for b>0.
    safe_count = np.where(count < 1, np.int64(1), count)
    half = safe_count // 2
    pos_num = sum_o2 + half            # >= 0 branch: (sum+half) trunc == floor
    neg_num = sum_o2 - half            # <  0 branch: trunc toward 0, NOT floor
    mean_o2 = np.where(sum_o2 >= 0,
                       pos_num // safe_count,
                       -((-neg_num) // safe_count))       # Q16.16 mean (int64), trunc-to-0
    # Threshold quantized ONCE into Q16.16 — a plain integer >= compare, no float.
    from simulation import fire_fixed as _fire_fx
    o2_threshold_q = _fire_fx.quantize_scalar(float(o2_threshold))
    has_o2 = (count > 0) & (mean_o2 >= o2_threshold_q)

    ignite = hot & has_o2
    if not bool(ignite.any()):
        return
    # max(fire, ignition_seed) on the igniting tiles only — an INTEGER max (exact,
    # order-free) that never lowers a bigger existing fire. ignition_seed quantizes
    # once into Q16.16.
    seed_q = np.int32(_fire_fx.quantize_scalar(float(ignition_seed)))
    np.maximum(gmap.fire, np.where(ignite, seed_q, gmap.fire), out=gmap.fire)


# ---------------------------------------------------------------------------
# Ammo economy (mechanics/03 §4 mag_size / reload_seconds — W3)
# ---------------------------------------------------------------------------
def mag_gate(u, weapon, tick):
    """The per-TRIGGER magazine gate (W3 ammo economy). True = the trigger
    may fire this tick; False = the unit is mid-reload (the stall).

    ``mag_size <= 0`` = ammo untracked — an immediate True with ZERO state
    touched (the k5/pistol tier and every pre-W3 weapon: shipped scenarios
    never see the new fields, so the golden digest cannot move). For tracked
    weapons: a unit inside its reload window (``tick < reload_done_tick``)
    is stalled; otherwise an empty (or never-bound: ``current_mag is None``
    — the lazy first-trigger bind) magazine refills to ``mag_size`` here,
    i.e. the auto-reload COMPLETES at the first trigger attempt past the
    stall. Pure integer state + compares (door 1) on door-2 row constants;
    no RNG. Deterministic mirror of the rof cadence gate it sits beside.

    Mag state (``current_mag`` / ``reload_done_tick``) lives ON THE UNIT and
    is deliberately NOT in the synced digest surface
    (field_ab_harness.SYNCED_UNIT_FIELDS) — matching the ``last_fire_tick``
    precedent: combat-cadence state is a deterministic derivation of synced
    inputs (orders + tick), not hashed directly; a divergence would surface
    one tick downstream in the hashed hp/event stream.
    """
    if weapon.mag_size <= 0:
        return True                        # untracked — exactly pre-W3
    if tick < getattr(u, "reload_done_tick", -1):
        return False                       # mid-reload: the stall
    if getattr(u, "current_mag", None) is None or u.current_mag <= 0:
        u.current_mag = int(weapon.mag_size)   # first bind / reload complete
    return True


def mag_spend(u, weapon, tick):
    """Spend one round per TRIGGER (a shotgun's 8 pellets = one trigger =
    one round). Emptying the magazine starts the auto-reload NOW:
    ``reload_done_tick = tick + reload_ticks`` — so the stall between the
    last shot and the next is exactly ``reload_seconds`` (no manual reload
    order in v1). No-op for untracked weapons."""
    if weapon.mag_size <= 0:
        return
    u.current_mag -= 1
    if u.current_mag <= 0:
        u.reload_done_tick = tick + int(weapon.reload_ticks)


# ---------------------------------------------------------------------------
# Shooting (fire orders + auto-fire) — W2: per-unit weapon, archetype dispatch
# ---------------------------------------------------------------------------
def process_shooting(gmap, units, tick, shots, real_time, rng, events=None,
                     bullets=None, queue=None):
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
    """
    tpp = CFG.clock.ticks_per_phase
    phase = tick // tpp
    tables = weapon_tables()

    for u in units:
        if u.team != 0 or not u.alive:
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

        fire_order = u.get_fire_order_in_phase(phase)
        if not fire_order:
            # Move & Attack: auto-fire at nearest visible enemy (snap cone).
            for o in u.orders:
                if o.order_type == ORDER_MOVE_ATTACK and o.phase == phase:
                    auto_fire(gmap, units, u, tick, shots, real_time, rng,
                              events=events, bullets=bullets, weapon=weapon,
                              queue=queue)
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
                            weapon):
                mag_spend(u, weapon, tick)
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
        spread = weapon.spread_deg if flags.can_aim else weapon.spread_snap_deg
        if weapon.archetype == "spray":
            # SPRAY branch of the archetype dispatch (W4): arm the burst —
            # the deposits themselves ride process_sprays in this same
            # shooting slot. Handled here rather than in _dispatch_trigger
            # because the burst captures the ORDER (interruption consumes
            # it). Spread is meaningless on a cone weapon — no draw, ever.
            start_spray_burst(u, weapon, fire_order, tick)
        else:
            _dispatch_trigger(gmap, units, u, uc_fx, uc_fy,
                              target_fx, target_fy, tick, shots, real_time,
                              rng, events, bullets, weapon, spread, queue)
        mag_spend(u, weapon, tick)
        u.last_fire_tick = tick


def _dispatch_trigger(gmap, units, u, fx1, fy1, fx2, fy2, tick, shots,
                      real_time, rng, events, bullets, weapon, spread_deg,
                      queue=None):
    """Route one trigger pull to its delivery archetype (mechanics/03 §1).
    The archetype set is CLOSED; only the ranged marchers dispatch here."""
    if weapon.archetype == "hitscan":
        fire_beam(gmap, units, u, fx1, fy1, fx2, fy2,
                  tick, shots, real_time, rng, events=events,
                  weapon=weapon, spread_deg=spread_deg)
    else:  # "projectile" — the default marcher (validated set, mechanics/03)
        fire_burst(gmap, units, u, fx1, fy1, fx2, fy2,
                   tick, shots, real_time, rng, events=events,
                   weapon=weapon, spread_deg=spread_deg, bullets=bullets,
                   queue=queue)


def auto_fire(gmap, units, u, tick, shots, real_time, rng, events=None,
              bullets=None, weapon=None, queue=None):
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
                          weapon, weapon.spread_snap_deg, queue)
        mag_spend(u, weapon, tick)
        u.last_fire_tick = tick


def fire_burst(gmap, units, shooter, fx1, fy1, fx2, fy2,
               tick, shots, real_time, rng, events=None, *,
               weapon=None, ammo=None, spread_deg=None, bullets=None,
               queue=None):
    """Fire a burst of PROJECTILE rounds from (fx1, fy1) toward (fx2, fy2).

    Lifted from ``game.py:_fire_burst``; W2 generalizes the march to the
    weapon/ammo rows (speed as data) without touching its arithmetic: each
    bullet draws one cone angle (``rng.uniform(-cone, +cone)`` — the shipped
    door-4 pattern), takes its kit-trig step vector, and marches through
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
        spread_deg = weapon.spread_deg
    cone = math.radians(spread_deg)
    n_bullets = weapon.shots_per_trigger
    # Q2-lift: the bullet march decides hits -> current_hp -> kills, i.e. it
    # FEEDS SYNCED STATE — so its trig must be the deterministic integer kit,
    # not libm (math.atan2/cos/sin differ at the last ULP across CRT/Python
    # versions; a ULP can flip a grazing hit cross-machine). math.radians and
    # rng.uniform stay: pure IEEE arithmetic + the seeded Generator's fixed
    # bit-stream, already cross-machine exact.
    base_angle = unit_fixed.atan2_rad(fy2 - fy1, fx2 - fx1)

    # W2 (mechanics/06 §5): firing SETS the shooter's facing to the aim
    # bearing (before spread) — through the kit, in the unit facing
    # convention (y up: the dy negation, unit.face_towards). Facing is
    # already-synced state derived from synced inputs; the digest surface is
    # unchanged.
    shooter.facing = unit_fixed.atan2_rad(-(fy2 - fy1), fx2 - fx1)

    shooter_id = getattr(shooter, "id", -1)

    for _ in range(n_bullets):
        angle = base_angle + float(rng.uniform(-cone, cone))
        # Per-bullet step vector through the kit (angle is constant per
        # bullet — hoisted out of the march loop, bit-identical to
        # re-evaluating inside it). Steps are exact n/65536 doubles; the
        # rx/ry accumulation is plain float + (IEEE-exact, deterministic).
        step_x = unit_fixed.cos_rad(angle)
        step_y = unit_fixed.sin_rad(angle)
        b = BulletInFlight(shooter, shooter_id, weapon, ammo,
                           fx1, fy1, angle, step_x, step_y)
        still_flying = b.advance(gmap, units, shots, real_time, rng,
                                 events=events, queue=queue)
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
              weapon, ammo=None, spread_deg=None):
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
    """
    tables = weapon_tables()
    if ammo is None:
        ammo = tables.ammo_for_weapon(weapon)
    if spread_deg is None:
        spread_deg = weapon.spread_deg
    cone = math.radians(spread_deg)
    base_angle = unit_fixed.atan2_rad(fy2 - fy1, fx2 - fx1)
    # Facing = aim bearing at fire, before spread (the fire_burst rule).
    shooter.facing = unit_fixed.atan2_rad(-(fy2 - fy1), fx2 - fx1)
    shooter_id = getattr(shooter, "id", -1)
    h, w = gmap.material.shape
    # Per-gas Q16.16 absorption (once-at-build constants; plain ints).
    absorb = getattr(getattr(gmap, "gases", None), "beam_absorb_q16", None)
    gas = getattr(gmap, "gas", None)
    n_gases = N_GASES if (absorb is not None and gas is not None) else 0
    one = unit_fixed.FP_ONE

    for _ in range(int(weapon.shots_per_trigger)):
        angle = base_angle + float(rng.uniform(-cone, cone))
        step_x = unit_fixed.cos_rad(angle)
        step_y = unit_fixed.sin_rad(angle)
        rx, ry = float(fx1), float(fy1)
        energy_q = int(ammo.damage) << unit_fixed.FP_SHIFT   # door 2 (exact)
        skewered = []   # units already hit by THIS beam (identity — hit once)

        for _step in range(int(weapon.range_tiles)):
            rx += step_x
            ry += step_y
            ix, iy = int(rx), int(ry)
            if not (0 <= iy < h and 0 <= ix < w):
                break

            # 1. Attenuate (integer Beer-Lambert over the tile's gas column).
            if n_gases:
                total_absorb = 0
                for g in range(n_gases):
                    density_q = int(gas[g, iy, ix])
                    if density_q > 0:
                        total_absorb += (int(absorb[g]) * density_q) \
                            >> unit_fixed.FP_SHIFT
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
                chew_wall(gmap, iy, ix, ammo.wall_damage)
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
                        apply_packet(
                            e,
                            DamagePacket(amount=amount, dtype=ammo.dtype_id,
                                         source_id=shooter_id, ap=ammo.ap),
                            events, source="laser")

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
                     cone_half_angle_degrees, exclude=()):
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
    angle = unit_fixed.atan2_rad(float(target_fy) - ay, float(target_fx) - ax)
    cos_q = unit_fixed.quantize_scalar(unit_fixed.cos_rad(angle))
    sin_q = unit_fixed.quantize_scalar(unit_fixed.sin_rad(angle))
    # Cone cosine through the kit (math.radians is pure arithmetic; the cos
    # itself is the deterministic fixed-point kit — no libm on this path).
    c_q = unit_fixed.quantize_scalar(
        unit_fixed.cos_rad(math.radians(float(cone_half_angle_degrees))))
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
            if not gmap.has_los(ay, ax, y, x):
                continue                               # occluded — no pour
            yield y, x, max(1, math.isqrt(dist_sq))


def deposit_spray_cone(gmap, queue, shooter, weapon, ammo,
                       target_fx, target_fy):
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
    """
    heat = float(ammo.heat_deposit)
    gas_amount = float(ammo.gas_amount)
    gas_id = None
    if ammo.gas_species:
        gas_id = int(gmap.gases.name_to_id[ammo.gas_species])
    own = set(shooter.occupied_tiles())
    for y, x, div in spray_cone_tiles(
            gmap, shooter.center_tile_y(), shooter.center_tile_x(),
            target_fx, target_fy, weapon.range_tiles,
            weapon.cone_half_angle_degrees, exclude=own):
        if heat > 0.0:
            queue.enqueue(FieldEdit(
                field="heat", region=Region.TILE, coords=(y, x),
                amount=heat / div, mode=EditMode.ADD,
                source_id=_SRC_SPRAY_HEAT,
            ))
        if gas_id is not None and gas_amount > 0.0:
            queue.enqueue(FieldEdit(
                field="gas", region=Region.TILE, coords=(y, x),
                amount=gas_amount / div, mode=EditMode.ADD,
                clamp=(0.0, 1.0), channel=gas_id,
                source_id=_SRC_SPRAY_GAS,
            ))


def start_spray_burst(u, weapon, fire_order, tick):
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
    u.spray_target = (float(fire_order.target_fx), float(fire_order.target_fy))
    u.spray_order = fire_order
    u.facing = unit_fixed.atan2_rad(
        -(float(fire_order.target_fy) - u.center_tile_y()),
        float(fire_order.target_fx) - u.center_tile_x())


def process_sprays(gmap, units, queue, events=None):
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
    """
    tables = weapon_tables()
    for u in units:
        ticks_left = getattr(u, "spray_ticks_left", 0)
        if ticks_left <= 0:
            continue
        if not u.alive:
            u.spray_ticks_left = 0
            u.spray_order = None
            u.spray_target = None
            continue
        if not composed_flags(u).can_act:
            # Interruption: stop NOW, consume the order, never resume.
            order = getattr(u, "spray_order", None)
            if order is not None and order in u.orders:
                u.orders.remove(order)
            u.spray_ticks_left = 0
            u.spray_order = None
            u.spray_target = None
            continue
        weapon = tables.weapons.by_name[u.weapon_id]
        ammo = tables.ammo_for_weapon(weapon)
        tx, ty = u.spray_target
        deposit_spray_cone(gmap, queue, u, weapon, ammo, tx, ty)
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


def melee_strike(units, attacker, target_fx, target_fy, rng, events, weapon):
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
    """
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

    fx1, fy1 = attacker.center_tile_x(), attacker.center_tile_y()
    fx2, fy2 = target.center_tile_x(), target.center_tile_y()
    # Facing = strike bearing (kit trig, unit y-up convention — the
    # fire_burst rule); the strike angle itself stays in SCREEN convention
    # for the arc classifier (the march-angle contract of arc_multiplier).
    attacker.facing = unit_fixed.atan2_rad(-(fy2 - fy1), fx2 - fx1)
    angle = unit_fixed.atan2_rad(fy2 - fy1, fx2 - fx1)

    amount = int(weapon.melee_damage)
    crit_chance = float(weapon.crit_chance)
    if crit_chance > 0.0:
        mult = attack_resolver.arc_multiplier(angle, target)
        if attack_resolver.roll_crit(crit_chance, mult, rng):
            amount = attack_resolver.scale_half_away(amount, weapon.crit_mult)

    apply_packet(target,
                 DamagePacket(amount=amount, dtype=weapon.melee_dtype_id,
                              source_id=getattr(attacker, "id", -1)),
                 events, source="melee")

    if weapon.status_kind_id is not None and target.alive:
        apply_status(target, weapon.status_kind_id, magnitude=0,
                     duration_ticks=weapon.status_ticks,
                     source_id=getattr(attacker, "id", -1))
    return True


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
        if u.team != 0:
            continue
        for o in u.orders:
            if o.order_type == ORDER_EXPLOSIVE and o.det_slot == slot:
                fy, fx = o.target_fy, o.target_fx
                payload = tables.payload_for_ammo(
                    getattr(o, "ammo_name", None) or "demo_breach")
                execute_payload(gmap, queue, units, fy, fx, payload, rng,
                                events=events, kind="door_explosive")
