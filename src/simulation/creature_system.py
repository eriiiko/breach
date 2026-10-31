"""Creature bodies (#106 P6d1): the runtime that turns a creature unit into an
articulated animal -- ``CreatureBody`` (one per creature unit), the conductor's
slot 5b (``CreatureSystem.tick``), the ONE serializer of the ``__creature__``
digest section (``serialize_record`` / ``decode_record``) and the spawn check.

Design: docs/octopus_p6_sim_port_design_2026-10-30.md sections 2, 4, 5, 7.
P6d1 shipped a LIMP animal; P6d2 (``creature_primitives``, the ONE module
owning the state machines) drives it: the hub turns the unit's intents into a
goal, the goal into a primitive per arm -- idle (the default: no intent),
crawl, swim, reach, grab -- and the primitives into the solver's per-tick
inputs (the goal table is in that module's docstring). The chain solver
(``arm_chain.tick``, the C++ ``ArmChainSolver``) integrates everything
else: gravity, drag, the floor, the walls, anchors, grabs, the crown's
collision with other units' footprints. ``CreatureBody.set_limp`` puts a body
in the LIMP gait (no primitive: P6d1's animal).

The creature key is DATA: a unit is a creature iff its species row says so
(``species.creature is not None``, :func:`is_creature`) -- nothing here or in
any caller names a creature or a species.

Who writes a creature unit's position (``Unit.x``/``y``/``facing``)
--------------------------------------------------------------------
=====================================  =========================================
writer                                 what it does to a creature unit
=====================================  =========================================
``Simulation.add_unit(position=)``     load-time only: the constructor anchor
``Simulation.spawn_creatures``         sets the anchor + facing FROM the
                                       quantized spawn pose (through
                                       ``creature_fixed``), once
slot 5b, ``CreatureSystem.tick``       THE per-tick writer: anchor + facing
                                       from the integer body pose
                                       (``pose_to_unit_anchor``,
                                       ``heading_rad_to_facing``)
slot 3 / ``OnePhaseWEGO.drive_units``  SKIP it (MOVE_DIR / AIM are recorded as
                                       the hub's goal instead, slot 5b)
slot 1b AIM (``live_aim``)             skips its facing write
slot 1b USE (``pending_use``)          skips it: a creature's USE is the hub's
                                       grab request (consumed at slot 5b)
slot 9c2 wave push                     never displaces it: the coupling row
                                       ``apply_creature_wave_impulse`` adds the
                                       push to the BODY's velocity instead
                                       (``exchange.py``; vetoable -- deleting
                                       that row's call leaves the push ignored)
``Simulation._end_round`` snap         skips it (the pose is not tile-snapped)
zombie conversion                      never (``can_become_zombie = false``)
=====================================  =========================================

State and determinism
---------------------
Everything synced is an integer: the solver's buffers (``ArmChainBuffers``,
numpy, owned here -- the solver is stateless, so a creature pickles) and the
hub's fields. Every timer is an ELAPSED-tick counter, never compared to
``sim.tick`` (TwoPhaseWEGO rewinds it each round). The hub's randomness is a
per-creature xorshift64 seeded from the sim seed and the unit id (never
``sim.rng``, whose draw count would couple an arm to the shots fired that tick);
its state is hashed. Bodies tick in unit-id order; the other units' footprints
are handed to the solver in unit-id order. A body may own a PROP (``add_prop``:
the grab's demo body, solver body 1, not a Unit -- a grabbed UNIT is #107's).

``Unit.x/y/facing`` stay the existing float fields, written ONLY from the
integer pose through the ``creature_fixed`` converters (render-side floats).

Arms never block, stamp, jam a door, sense heat or gas, or count for melee
adjacency: the body's footprint is the unit's footprint, and only it stamps.
They CAN be shot (P6d3, the hit seam): each time the pose is written to the
unit (``write_unit``: spawn and slot 5b) the body rebuilds its ``HitShape`` --
the footprint plus every tile an arm segment crosses, rasterised from the
integer chain (``creature_fixed.segment_tiles``, top-down: z ignored) -- and
hands it to the unit (``Unit.hit_shape``), which the bullet march and the
laser skewer read through ``Unit.in_hit_tiles``; a connecting hit writes the
struck (arm, node) to ``last_hit`` through ``Unit.note_hit``. Derived, never
digested; a dead creature is not ticked, so its shape freezes with its pose.
Arms pass through marines (Erik's ruling of 2026-10-30, a NAMED ACCEPTED GAP);
only the crown collides, inside the solver.

The ``__creature__`` section (design section 5)
-----------------------------------------------
``CREATURE_SECT_V2\\n`` then one record per creature body in unit-id order,
little-endian, each ending in ``\\n`` (so a recorder ``np.bytes_`` ring can
never strip a trailing NUL). V2 (P6d2) adds the primitives' hub state and the
creature-owned bodies to V1's layout, and reads a grab's id as a BODY index:

  header      int64 unit_id, int64 row version, 32 B row_hash (sha256 raw),
               int32 arms A, int32 nodes n
  nodes        int32 (A, n, 3) positions, Q24 m; int32 (A, n, 3) velocities
  body pose    int32 x, y, z (Q24 m), heading (Q24 rad, the pose's: 0 = East,
               CCW on screen), vx, vy, vz (Q24 m/s), yaw rate (Q24 rad/s)
  anchors      uint8 (A, n) mask; int32 (A, n, 3) positions
  grabs        int32 count, then per grab sorted by (arm, node): int32 arm,
               int32 node, int64 the grabbed BODY's index in the creature's
               body array (1 = its prop; v1 grabs no unit -- #107), int32
               local x, y, z (Q24, the grabbed body's frame)
  hub          uint8 (A) per-arm state (``creature_primitives.ARM_*``); int32
               (A) elapsed ticks in state; int32 (A) drawn lengths (ticks);
               int32 stance count; int32 ticks since the last decision; uint8
               travel goal (``GOAL_*``); int32 goal x, y (Q16 direction);
               int32 steering samples held; int32 last hit arm, int32 last
               hit node (-1, -1: none yet, or the last connecting hit
               struck the crown's footprint; P6d3 writes it)
  hub (V2)     uint8 gait, uint8 swim phase, uint8 grab request, int8 grab
               wrap side; int32 swim ticks, swim root angle (Q16 rad), swim
               rig (Q16: 0 = the rest roots, ONE = the swimming ring); int32
               reach goal x, y (Q16 direction); int32 grab pops; int32 (A)
               per-arm root aim (Q16 rad); int32 (A) per-arm stretch (Q16);
               int32 K, then int32 (K, 2) the steering ring (Q24 m crown
               positions, oldest first)
  bodies (V2)  int32 B, then for each creature-owned body 1 .. B-1 its pose
               in the body pose's 8-int encoding
  rng          uint64 xorshift state
  clock        int64 (A) per-arm elapsed activation substeps
  terminator   ``\\n``

Derived, NOT stored: activation, rest scale, bend targets, root frames, the
roots' body-frame places (from the aims, the lift and the swim rig),
``hit_tiles`` (P6d3). ``CREATURE_SECT_V<n>`` bumps only when the ENCODING
changes. The carrier ``CREATURE_DIGEST_KEY`` is ALWAYS written by capture
paths (``n_creatures == 0`` for a creature-free sim); ``tick_digest`` folds the
section only when ``n_creatures > 0``, so every creature-free digest is
byte-identical to before P6d1.
"""
from __future__ import annotations

import struct

import numpy as np

from simulation import arm_chain as AC
from simulation import creature_fixed as CF
from simulation import creature_primitives as P
from simulation import fixedpoint_twins as K
from simulation.species import get_species

#: Snapshot key of the creature presence carrier (the ``__entity__`` twin).
CREATURE_DIGEST_KEY = "__creature__"
CREATURE_SECT_PREAMBLE = b"CREATURE_SECT_V2\n"
RECORD_TERMINATOR = b"\n"

#: The hub's per-arm states and goals live with the state machines
#: (``creature_primitives``); the two P6d1 names are re-exported.
ARM_LIMP = P.ARM_LIMP
GOAL_NONE, GOAL_MOVE_DIR = P.GOAL_NONE, P.GOAL_MOVE_DIR

_U64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# The creature key and the engine
# ---------------------------------------------------------------------------
def is_creature(unit) -> bool:
    """True iff ``unit``'s species is a creature (``species.creature is not
    None`` -- the ONLY creature key)."""
    return get_species(getattr(unit, "species_id", "human")).creature is not None


def require_engine(engine=None):
    """The compiled chain solver, or a clear refusal: a creature needs the
    engine binding (the integer reference is not engine code; there is no
    Python fallback chain)."""
    if engine is None:
        try:
            import breach_physics as engine
        except ImportError:
            engine = None
    if engine is None or not hasattr(engine, "arm_chain_tick"):
        raise RuntimeError(
            "a creature needs the compiled arm chain solver "
            "(breach_physics.arm_chain_tick, #106 P6c) and this build has none: "
            "rebuild cpp/ (docs/dev_setup.md)")
    return engine


# ---------------------------------------------------------------------------
# The xorshift (the hub's own randomness)
# ---------------------------------------------------------------------------
def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _U64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return z ^ (z >> 31)


def seed_xorshift(sim_seed, unit_id: int) -> int:
    """The per-creature xorshift64 state from the sim seed (``None`` -> 0) and
    the unit id: one splitmix64 of the pair, never 0."""
    s = 0 if sim_seed is None else int(sim_seed) & _U64
    st = _splitmix64((s * 0x100000001B3 + int(unit_id)) & _U64)
    return st if st else 0x2545F4914F6CDD1D


def xorshift64(state: int) -> tuple[int, int]:
    """One xorshift64 step: ``(new_state, value)`` (Marsaglia 2003, 13/7/17)."""
    x = int(state) & _U64
    x ^= (x << 13) & _U64
    x ^= x >> 7
    x ^= (x << 17) & _U64
    return x, x


# ---------------------------------------------------------------------------
# The rest pose (spawn)
# ---------------------------------------------------------------------------
def _rot(th_q24: int, lx: int, ly: int) -> tuple[int, int]:
    """Rotate a body-frame point by the body yaw exactly as the solver's root
    attachment does (``chain_ref_q.World._attach``): Q16 kit trig at the Q24
    yaw narrowed by 8, symmetric rounding."""
    th16 = K.shr_round_away(int(th_q24), CF.CHAIN_SHIFT - 16)
    c, s = K.cos_q16(th16), K.sin_q16(th16)
    return (K.shr_round_away(c * lx - s * ly, 16),
            K.shr_round_away(s * lx + c * ly, 16))


def rest_pose(cdef, centre_q24, heading_q24: int):
    """The limp spawn pose: every arm straight and radial from the crown RIM
    at its row angle (CCW on screen from the heading), at rest length, on the
    floor plane (z = 0). Pure integer (the kit's trig), so the loader's check
    and the spawn see the same nodes.

    Returns ``(p, root_local, th_q24)``: ``p`` (A, n, 3) int32 Q24 m world
    positions, ``root_local`` (A, 2, 3) int32 Q24 m body-frame roots (nodes 0
    and 1, the solver's attachments), ``th_q24`` the solver's body yaw."""
    b = cdef.body
    A, n = int(b.arms), int(b.nodes_per_arm)
    th = CF.wrap_heading_q24(CF.heading_q24_to_body_th(heading_q24))
    rim = CF.chain_from_m(b.body_radius_m)
    seg = CF.chain_from_m(b.arm_length_m / (n - 1))
    cx, cy = int(centre_q24[0]), int(centre_q24[1])
    p = np.zeros((A, n, 3), np.int32)
    root = np.zeros((A, 2, 3), np.int32)
    for a, deg in enumerate(b.arm_angles_deg):
        # Body-frame angle: the body frame is y-down like the grid, so a
        # screen-CCW angle is its negative (creature_fixed, the heading note).
        phi16 = CF.q16_from_scalar(-float(deg) * (np.pi / 180.0))
        c, s = K.cos_q16(phi16), K.sin_q16(phi16)
        for i in range(n):
            r = rim + i * seg
            lx, ly = K.shr_round_away(r * c, 16), K.shr_round_away(r * s, 16)
            if i < 2:
                root[a, i] = (lx, ly, 0)
            rx, ry = _rot(th, lx, ly)
            p[a, i] = (cx + rx, cy + ry, 0)
    return p, root, th


def check_creature_spawn(spawn, open_mask, tile_size_m: float, where: str):
    """The grid door of a ``[[creature]]`` row (``simulation.match.
    CreatureSpawn``): the species' footprint (anchored at ``floor(x),
    floor(y)``) inside the map and on open tiles, and every node of the rest
    pose (:func:`rest_pose`) on an open tile inside the map -- an arm laid
    into a wall would be pushed out explosively on the first tick (P6c's
    lesson). ``open_mask`` is (h, w) bool, True where a node may lie (the
    loader: open interior floor; the sim: ``~gmap.solid``). Returns the
    creature row (``CreatureDef``)."""
    from simulation.creatures import get_creature
    sdef = get_species(spawn.species)
    cdef = get_creature(sdef.creature)
    om = np.asarray(open_mask, dtype=bool)
    h, w = om.shape
    F = int(cdef.body.footprint)
    ax, ay = int(np.floor(spawn.x)), int(np.floor(spawn.y))
    if ax < 0 or ay < 0 or ax + F > w or ay + F > h:
        raise ValueError(f"{where}: the {F}x{F} footprint at ({spawn.x}, "
                         f"{spawn.y}) leaves the {w}x{h} map")
    if not om[ay:ay + F, ax:ax + F].all():
        raise ValueError(f"{where}: the footprint at ({spawn.x}, {spawn.y}) "
                         f"is not on open floor")
    cx, cy = CF.unit_anchor_to_pose(spawn.x, spawn.y, F, tile_size_m)
    p, _, _ = rest_pose(cdef, (CF.chain_from_m(cx), CF.chain_from_m(cy)),
                        CF.heading_q24_from_deg(spawn.heading))
    tile_q = CF.chain_from_m(tile_size_m)
    tx = p[..., 0].astype(np.int64) // tile_q
    ty = p[..., 1].astype(np.int64) // tile_q
    inside = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    ok = inside.copy()
    ok[inside] = om[ty[inside], tx[inside]]
    if not ok.all():
        a, i = (int(v) for v in np.argwhere(~ok)[0])
        raise ValueError(
            f"{where}: arm {a} node {i} of the rest pose (arms straight and "
            f"radial at heading {spawn.heading} deg) lies on a blocked or "
            f"off-map tile ({int(tx[a, i])}, {int(ty[a, i])}) -- a creature "
            f"needs open floor {cdef.body.body_radius_m + cdef.body.arm_length_m:.2f} m "
            f"around its centre")
    return cdef


# ---------------------------------------------------------------------------
# One creature
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The hit seam's shape (P6d3)
# ---------------------------------------------------------------------------
class HitShape:
    """What a bullet or a beam can strike on one creature unit this tick
    (design 4.2): ``tiles`` -- sorted ``(tx, ty)`` -- is the body's footprint
    plus every in-map tile an arm segment crosses; ``node_at`` maps each arm
    tile to the (arm, node) a hit there strikes. DERIVED from the integer
    chain by :func:`build_hit_shape`, never digested; holds its body so a hit
    can write ``last_hit`` (the one hashed trace of a hit's place)."""

    __slots__ = ("body", "tiles", "tile_set", "body_tiles", "node_at")

    def __init__(self, body, body_tiles, node_at):
        self.body = body
        self.body_tiles = frozenset(body_tiles)
        self.node_at = node_at
        self.tile_set = frozenset(self.body_tiles | node_at.keys())
        self.tiles = tuple(sorted(self.tile_set))

    def struck(self, ix: int, iy: int) -> tuple[int, int]:
        """The (arm, node) a hit on tile ``(ix, iy)`` strikes: (-1, -1) on the
        crown's footprint (the body wins a tile it shares with an arm)."""
        t = (int(ix), int(iy))
        if t in self.body_tiles:
            return (-1, -1)
        return self.node_at.get(t, (-1, -1))

    def note_hit(self, ix: int, iy: int) -> None:
        self.body.last_hit = self.struck(ix, iy)


def build_hit_shape(body, body_tiles, h: int, w: int) -> HitShape:
    """``body``'s hit shape from its synced chain (``bufs.p``, Q24 m) and the
    unit's footprint ``body_tiles``: each arm's segments node k -> k+1 are
    rasterised by ``creature_fixed.segment_tiles`` in the solver's tile law
    (the topology's quantized tile side and origin, ``WD_TILE_Q24``), z
    ignored. A tile is struck at the FIRST (arm, node) to claim it in (arm,
    node) order, where node k claims its own tile and the tiles its segment
    to k + 1 clips before reaching node k + 1's. Tiles off the (h, w) map
    are dropped."""
    b = body.bufs
    T = int(b.world[0, AC.WD_TILE_Q24])
    ox = int(b.world[0, AC.WD_ORIGIN_X_Q24])
    oy = int(b.world[0, AC.WD_ORIGIN_Y_Q24])
    p = b.p[0, :, :, :2].tolist()             # (A, n, 2) Q24 m, python ints
    node_at: dict = {}
    for a, arm in enumerate(p):
        n = len(arm)
        for k in range(n):
            x0, y0 = arm[k][0] - ox, arm[k][1] - oy
            node_at.setdefault((x0 // T, y0 // T), (a, k))
            if k + 1 < n:
                seg = CF.segment_tiles(x0, y0, arm[k + 1][0] - ox, arm[k + 1][1] - oy, T)
                for t in seg[1:-1]:
                    node_at.setdefault(t, (a, k))
    node_at = {t: an for t, an in node_at.items()
               if 0 <= t[0] < w and 0 <= t[1] < h}
    return HitShape(body, body_tiles, node_at)


class CreatureBody:
    """One creature unit's articulated body: the solver's buffers (owned
    here; the solver is stateless) and the hub's integer state. Holds the
    unit's ID, never the unit (the sim's unit list is the one owner)."""

    def __init__(self, unit, cdef, *, ticks_per_second: int, tile_size_m: float,
                 solid, topology_serial: int, sim_seed, heading_q24: int):
        b, t = cdef.body, cdef.tissue
        A, n = int(b.arms), int(b.nodes_per_arm)
        h, w = np.asarray(solid).shape
        self.unit_id = int(unit.id)
        self.creature = cdef.name
        self.row_version = int(cdef.version)
        self.row_hash = bytes.fromhex(cdef.row_hash)
        self.arms, self.nodes = A, n
        self.footprint = int(b.footprint)
        self.tile_size_m = float(tile_size_m)
        self.row, self.row_node = AC.row_arrays(cdef, ticks_per_second)
        inv_dt = int(ticks_per_second) * int(cdef.solver.substeps)

        self.inv_dt = inv_dt
        self.dials = P.PrimDials(cdef)
        # The swim's arms carry the web: the row's broadside drag x web_factor
        # (swim_sim3d), a second row_node quantized once here and handed to
        # the solver while the body swims (one call drives all eight arms).
        self.row_node_swim = self.row_node.copy()
        self.row_node_swim[:, AC.RN_K_PER_Q16] = [
            CF._round_half_away(float(k) * float(cdef.swim.web_factor))
            for k in self.row_node[:, AC.RN_K_PER_Q16]]
        self.solver_row_node = self.row_node

        bufs = AC.allocate(1, A, n, bodies=1, grabs=0, unit_slots=0, h=h, w=w)
        for a in range(A):
            # The anchors' grip limit is the crawl's (P2; the idle's seated
            # patches are the same suckers); a grab's grip is the hub's
            # release threshold on the grab force (creature_primitives).
            bufs.arm_cfg[0, a] = AC.arm_cfg_row(inv_dt=inv_dt, mu=t.mu_floor, body=0,
                                                grip_n=cdef.crawl.grip_limit)
        self.body_cfg_crawl = AC.body_cfg_row(
            inv_dt=inv_dt, water_density=b.water_density, mass_kg=b.body_mass_kg,
            radius_m=b.body_radius_m, c_drag=b.body_c_drag, area_m2=b.body_crawl_area_m2,
            rot_damp=b.body_rot_damp, mu=t.mu_floor, coll_radius_m=b.body_radius_m)
        # mantle first (swim_sim3d): the swim's drag on the crown
        self.body_cfg_swim = AC.body_cfg_row(
            inv_dt=inv_dt, water_density=b.water_density, mass_kg=b.body_mass_kg,
            radius_m=b.body_radius_m, c_drag=cdef.swim.body_c_drag,
            area_m2=cdef.swim.body_area_m2, rot_damp=b.body_rot_damp, mu=t.mu_floor,
            coll_radius_m=b.body_radius_m)
        bufs.body_cfg[0, 0] = self.body_cfg_crawl
        AC.set_topology(bufs, 0, solid, tile_size_m)
        self.topology_serial = int(topology_serial)
        self._cdef = cdef

        # The spawn pose from the unit's anchor (the one load-time writer),
        # quantized once; the unit is then re-anchored FROM it.
        cx, cy = CF.unit_anchor_to_pose(unit.x, unit.y, self.footprint, tile_size_m)
        centre = (CF.chain_from_m(cx), CF.chain_from_m(cy))
        p, root, th = rest_pose(cdef, centre, heading_q24)
        bufs.p[0] = p
        bufs.root_local[0] = root
        bufs.body[0, 0, AC.BS_X], bufs.body[0, 0, AC.BS_Y] = centre
        bufs.body[0, 0, AC.BS_TH] = th
        self.bufs = bufs
        self.rest_root = root.copy()                 # the roots' rest places (derived)

        # The hub (creature_primitives drives it; every field is hashed).
        self.arm_state = np.full(A, P.ARM_STILL, np.uint8)
        self.arm_ticks = np.zeros(A, np.int32)       # elapsed ticks in state
        self.arm_drawn = np.zeros(A, np.int32)       # drawn durations, ticks
        self.arm_aim = np.zeros(A, np.int32)         # root yaw in the body frame, Q16 rad
        self.arm_stretch = np.full(A, P.ONE, np.int32)   # a drawn / released scale, Q16
        self.stance = 0
        self.decision_ticks = 0                      # since the last decision
        self.goal_kind = GOAL_NONE                   # the travel goal (P.GOAL_*)
        self.goal_q16 = (0, 0)                       # Q16 direction
        self.reach_q16 = (0, 0)                      # the reach goal, Q16 direction
        self.grab_req = 0                            # a USE latched the grab
        self.grab_side = 1                           # the wrap's side (+-1)
        self.grab_pops = 0                           # seals let go on the grab force
        self.gait = P.GAIT_IDLE
        self.swim_phase = P.SWIM_OPEN
        self.swim_ticks = 0
        self.swim_phi = self.dials.open_rad          # Q16 rad from the trailing axis
        self.swim_rig = 0                            # Q16: rest roots .. swimming ring
        self.steer_ring = np.zeros((self.dials.steer_k, 2), np.int32)
        self.steer_ticks = 0                         # ring samples held
        self.last_hit = (-1, -1)                     # (arm, node): P6d3 writes
        self.rng = seed_xorshift(sim_seed, self.unit_id)
        # idle is the default: every arm seated (its mid-arm patch on the deck)
        for a in range(A):
            P._anchor(self, a, self.dials.anchor_nodes)
        self.write_unit(unit)

    # -- set-up: the limp gait and the prop -------------------------------------
    def set_limp(self, limp: bool = True) -> None:
        """The LIMP gait (no primitive: P6d1's animal -- activation 0, rest
        1, no anchors, no grabs) on ``True``; back to the hub's goal routing
        (idle first) on ``False``. For tests, tools and a future stun (#118)."""
        if limp:
            P._enter_gait(self, P.GAIT_LIMP)
        elif self.gait == P.GAIT_LIMP:
            P._enter_gait(self, P.GAIT_IDLE)

    @property
    def has_prop(self) -> bool:
        return self.bufs.body.shape[1] > 1

    def add_prop(self, at_body_m=None) -> None:
        """Give the creature its grab's demo PROP (P4/P5z): solver body 1, a
        disc of the row's ``prop_mass`` / ``prop_radius`` resting on the deck
        at ``at_body_m`` (body frame, metres, x along the heading, y to its
        LEFT on screen; default the row's ``prop_at_m``), with one grab slot
        per node of the row's ``grab_nodes`` range. Not a Unit: a grabbed
        unit is #107's. Set-up only (load time / a scenario)."""
        if self.has_prop:
            raise ValueError(f"creature {self.unit_id} already has a prop")
        cdef, d = self._cdef, self.dials
        g = cdef.grab
        px, py = d.prop_at_q24 if at_body_m is None else (
            CF.chain_from_m(at_body_m[0]), CF.chain_from_m(at_body_m[1]))
        b = self.bufs
        bd = b.body[0, 0]
        # left on screen is -y in the y-down body frame (rest_pose's rule)
        rx, ry = _rot(int(bd[AC.BS_TH]), px, -py)
        pose = np.zeros((1, 1, AC.BS_N), np.int32)
        pose[0, 0, AC.BS_X] = int(bd[AC.BS_X]) + rx
        pose[0, 0, AC.BS_Y] = int(bd[AC.BS_Y]) + ry
        cfg = AC.body_cfg_row(
            inv_dt=self.inv_dt, water_density=cdef.body.water_density, mass_kg=g.prop_mass,
            radius_m=g.prop_radius, c_drag=g.prop_c_drag, area_m2=g.prop_area_m2,
            rot_damp=cdef.body.body_rot_damp, mu=cdef.tissue.mu_floor,
            coll_radius_m=g.prop_radius)
        b.body = np.ascontiguousarray(np.concatenate([b.body, pose], axis=1))
        b.body_cfg = np.ascontiguousarray(np.concatenate([b.body_cfg, cfg[None, None]], axis=1))
        b.body_contact = np.zeros((1, 2), np.uint8)
        G = d.grab_k1 - d.grab_k0 + 1
        b.grabs = np.zeros((1, G, AC.GR_N), np.int64)
        b.grabs[..., AC.GR_ARM] = -1
        b.grab_force = np.zeros((1, G), np.int64)

    # -- the pose -> the unit ---------------------------------------------------
    @property
    def heading_q24(self) -> int:
        return CF.body_th_to_heading_q24(int(self.bufs.body[0, 0, AC.BS_TH]))

    def write_unit(self, unit) -> None:
        """``Unit.x/y/facing`` FROM the integer pose (the converters; floats
        are render-side, never read back), and the unit's hit shape from the
        integer chain (P6d3: ``build_hit_shape``, derived)."""
        bd = self.bufs.body[0, 0]
        x, y = CF.pose_to_unit_anchor(float(CF.chain_to_m(int(bd[AC.BS_X]))),
                                      float(CF.chain_to_m(int(bd[AC.BS_Y]))),
                                      self.footprint, self.tile_size_m)
        unit.x, unit.y = x, y
        unit.facing = CF.heading_rad_to_facing(CF.heading_q24_to_rad(self.heading_q24))
        # the hit seam (P6d3): rebuilt from the chain with every pose write
        h, w = self.bufs.blocked.shape[1:]
        self.hit_shape = build_hit_shape(self, unit.occupied_tiles(), h, w)
        unit.hit_shape = self.hit_shape

    # -- intents -> the hub's goal ----------------------------------------------
    def route_intents(self, unit) -> None:
        """The unit's intents -> the hub's goal (``creature_primitives.
        route_intents``: the travel goal from MOVE_DIR and its speed mode,
        the reach goal from AIM, the grab request from a USE latch)."""
        P.route_intents(self, unit)

    # -- the per-tick world inputs ----------------------------------------------
    def write_inputs(self, gmap, unit_rects) -> None:
        """The world's per-tick solver inputs: the other units' footprints,
        and the static topology again whenever ``on_tile_changed`` fired.
        (Each arm's inputs are the primitives', ``creature_primitives.
        hub_step``.)"""
        b = self.bufs
        b.unit_rects = np.asarray(unit_rects, dtype=np.int32).reshape(1, -1, 4)
        serial = int(gmap.topology_serial)
        if serial != self.topology_serial:
            AC.set_topology(b, 0, gmap.solid, self.tile_size_m)
            self.topology_serial = serial

    def add_velocity_q24(self, dvx_q24: int, dvy_q24: int) -> None:
        """Add a Q24 m/s velocity to the crown (the swim jet, the carry
        jets), saturated to the int32 chain range. Never a position."""
        bd = self.bufs.body[0, 0]
        lim = CF.CHAIN_INT32_MAX
        for k, dv in ((AC.BS_VX, int(dvx_q24)), (AC.BS_VY, int(dvy_q24))):
            bd[k] = max(-lim, min(lim, int(bd[k]) + dv))

    def add_velocity_m_s(self, dvx: float, dvy: float) -> None:
        """A body impulse (the wave-push coupling row, ``exchange.py``): add
        ``(dvx, dvy)`` m/s to the crown's velocity, quantized once, saturated
        to the int32 chain range."""
        self.add_velocity_q24(CF.chain_from_m(dvx), CF.chain_from_m(dvy))

    # -- the record -------------------------------------------------------------
    def contact_report(self):
        """This tick's per-node contact bits (A, n) and the crown's (a copy;
        P6d2's primitives read it, never re-derive it)."""
        return self.bufs.contact[0].copy(), int(self.bufs.body_contact[0, 0])


# ---------------------------------------------------------------------------
# The serializer (ONE: digest + recorder) and its decoder twin
# ---------------------------------------------------------------------------
_I32, _U8, _I64 = np.dtype("<i4"), np.dtype("<u1"), np.dtype("<i8")


def serialize_record(body: CreatureBody) -> bytes:
    """One creature's ``__creature__`` record (module docstring layout)."""
    b = body.bufs
    A, n = body.arms, body.nodes
    bd = b.body[0, 0]
    out = [struct.pack("<qq32sii", body.unit_id, body.row_version, body.row_hash, A, n),
           np.ascontiguousarray(b.p[0], _I32).tobytes(),
           np.ascontiguousarray(b.v[0], _I32).tobytes(),
           struct.pack("<8i", int(bd[AC.BS_X]), int(bd[AC.BS_Y]), int(bd[AC.BS_Z]),
                       CF.body_th_to_heading_q24(int(bd[AC.BS_TH])),
                       int(bd[AC.BS_VX]), int(bd[AC.BS_VY]), int(bd[AC.BS_VZ]),
                       CF.body_th_to_heading_q24(int(bd[AC.BS_OM]))),
           np.ascontiguousarray(b.anchored[0], _U8).tobytes(),
           np.ascontiguousarray(b.anchor_pos[0], _I32).tobytes()]
    grabs = sorted((int(g[AC.GR_ARM]), int(g[AC.GR_NODE]), int(g[AC.GR_BODY]),
                    int(g[AC.GR_LX]), int(g[AC.GR_LY]), int(g[AC.GR_LZ]))
                   for g in b.grabs[0] if int(g[AC.GR_ARM]) >= 0)
    out.append(struct.pack("<i", len(grabs)))
    for arm, node, unit_id, lx, ly, lz in grabs:
        out.append(struct.pack("<iiqiii", arm, node, unit_id, lx, ly, lz))
    out += [np.ascontiguousarray(body.arm_state, _U8).tobytes(),
            np.ascontiguousarray(body.arm_ticks, _I32).tobytes(),
            np.ascontiguousarray(body.arm_drawn, _I32).tobytes(),
            struct.pack("<iiBiiiii", body.stance, body.decision_ticks, body.goal_kind,
                        body.goal_q16[0], body.goal_q16[1], body.steer_ticks,
                        body.last_hit[0], body.last_hit[1]),
            struct.pack("<BBBbiiiiii", body.gait, body.swim_phase, body.grab_req,
                        body.grab_side, body.swim_ticks, body.swim_phi, body.swim_rig,
                        body.reach_q16[0], body.reach_q16[1], body.grab_pops),
            np.ascontiguousarray(body.arm_aim, _I32).tobytes(),
            np.ascontiguousarray(body.arm_stretch, _I32).tobytes(),
            struct.pack("<i", body.steer_ring.shape[0]),
            np.ascontiguousarray(body.steer_ring, _I32).tobytes(),
            struct.pack("<i", b.body.shape[1])]
    for bi in range(1, b.body.shape[1]):
        pb = b.body[0, bi]
        out.append(struct.pack("<8i", int(pb[AC.BS_X]), int(pb[AC.BS_Y]), int(pb[AC.BS_Z]),
                               CF.body_th_to_heading_q24(int(pb[AC.BS_TH])),
                               int(pb[AC.BS_VX]), int(pb[AC.BS_VY]), int(pb[AC.BS_VZ]),
                               CF.body_th_to_heading_q24(int(pb[AC.BS_OM]))))
    out += [struct.pack("<Q", body.rng & _U64),
            np.ascontiguousarray(b.wave_clock[0], _I64).tobytes(),
            RECORD_TERMINATOR]
    return b"".join(out)


def decode_record(rec: bytes) -> dict:
    """The decoder twin of :func:`serialize_record` (the A/B harness's per
    arm / node locator). Refuses a record whose length or terminator
    disagrees with its own header."""
    pos = 0

    def take(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, rec, pos)
        pos += size
        return vals

    def arr(dtype, shape):
        nonlocal pos
        count = int(np.prod(shape))
        a = np.frombuffer(rec, dtype=dtype, count=count, offset=pos).reshape(shape)
        pos += count * dtype.itemsize
        return a.copy()

    unit_id, version, row_hash, A, n = take("<qq32sii")
    d = {"unit_id": unit_id, "row_version": version, "row_hash": row_hash.hex(),
         "arms": A, "nodes": n}
    d["p"] = arr(_I32, (A, n, 3))
    d["v"] = arr(_I32, (A, n, 3))
    d["body"] = np.array(take("<8i"), np.int64)   # x y z heading vx vy vz yaw_rate
    d["anchored"] = arr(_U8, (A, n))
    d["anchor_pos"] = arr(_I32, (A, n, 3))
    (ng,) = take("<i")
    d["grabs"] = [take("<iiqiii") for _ in range(ng)]
    d["arm_state"] = arr(_U8, (A,))
    d["arm_ticks"] = arr(_I32, (A,))
    d["arm_drawn"] = arr(_I32, (A,))
    (d["stance"], d["decision_ticks"], d["goal_kind"], gx, gy, d["steer_ticks"],
     ha, hn) = take("<iiBiiiii")
    d["goal_q16"], d["last_hit"] = (gx, gy), (ha, hn)
    (d["gait"], d["swim_phase"], d["grab_req"], d["grab_side"], d["swim_ticks"],
     d["swim_phi"], d["swim_rig"], rx, ry, d["grab_pops"]) = take("<BBBbiiiiii")
    d["reach_q16"] = (rx, ry)
    d["arm_aim"] = arr(_I32, (A,))
    d["arm_stretch"] = arr(_I32, (A,))
    (k,) = take("<i")
    d["steer_ring"] = arr(_I32, (k, 2))
    (nb,) = take("<i")
    d["bodies"] = [np.array(take("<8i"), np.int64) for _ in range(nb - 1)]
    (d["rng"],) = take("<Q")
    d["wave_clock"] = arr(_I64, (A,))
    if rec[pos:] != RECORD_TERMINATOR:
        raise ValueError(f"decode_record: {len(rec) - pos} trailing byte(s) after "
                         f"the layout (expected the terminator only)")
    return d


#: The body pose's fields in record order (for locators).
BODY_FIELDS = ("x", "y", "z", "heading", "vx", "vy", "vz", "yaw_rate")


def section_bytes(carrier: dict) -> bytes:
    """The hashed ``__creature__`` bytes for a carrier."""
    return CREATURE_SECT_PREAMBLE + b"".join(carrier["records"])


def creature_carrier(system) -> dict:
    """The ``__creature__`` snapshot value -- the strict presence carrier,
    ALWAYS written by capture paths (``n_creatures == 0`` without creatures)."""
    bodies = [] if system is None else system.ordered_bodies()
    return {"n_creatures": len(bodies),
            "records": tuple(serialize_record(b) for b in bodies)}


def require_creature_carrier(system, snapshot: dict) -> None:
    """Strictness (the entity rule's twin): a sim WITH creature bodies may
    never meet a snapshot lacking the carrier, or a capture path could
    silently compute the creature-free digest for a creature run."""
    n = 0 if system is None else len(system.bodies)
    if n and CREATURE_DIGEST_KEY not in snapshot:
        raise KeyError(
            f"snapshot is missing the '{CREATURE_DIGEST_KEY}' presence carrier but "
            f"the sim has {n} creature bodies -- a creature run must never hash "
            f"creature-free")


def diff_creature_records(ra: bytes, rb: bytes) -> list[str]:
    """Locate where two records of one creature differ: per field, the first
    (arm, node, axis) for the node arrays. Empty == identical."""
    if ra == rb:
        return []
    da, db = decode_record(ra), decode_record(rb)
    out = []
    for key in ("unit_id", "row_version", "row_hash", "arms", "nodes"):
        if da[key] != db[key]:
            out.append(f"{key} {da[key]!r} != {db[key]!r}")
    if out:
        return out
    for key in ("p", "v", "anchor_pos", "anchored"):
        a, b = da[key], db[key]
        if not np.array_equal(a, b):
            idx = tuple(int(i) for i in np.argwhere(a != b)[0])
            where = f"arm {idx[0]} node {idx[1]}" + (f" axis {'xyz'[idx[2]]}" if len(idx) > 2 else "")
            out.append(f"{key} differs at {where} (a={int(a[idx])} b={int(b[idx])}; "
                       f"{int((a != b).sum())} entries differ)")
    for i, name in enumerate(BODY_FIELDS):
        if da["body"][i] != db["body"][i]:
            out.append(f"body {name} {int(da['body'][i])} != {int(db['body'][i])}")
    for key in ("arm_state", "arm_ticks", "arm_drawn", "arm_aim", "arm_stretch", "wave_clock"):
        if not np.array_equal(da[key], db[key]):
            a = int(np.argwhere(da[key] != db[key])[0][0])
            out.append(f"{key} differs at arm {a} ({int(da[key][a])} != {int(db[key][a])})")
    if not np.array_equal(da["steer_ring"], db["steer_ring"]):
        out.append("steer_ring differs")
    if len(da["bodies"]) != len(db["bodies"]) or any(
            not np.array_equal(x, y) for x, y in zip(da["bodies"], db["bodies"])):
        out.append("a creature-owned body (prop) differs")
    for key in ("grabs", "stance", "decision_ticks", "goal_kind", "goal_q16",
                "steer_ticks", "last_hit", "gait", "swim_phase", "grab_req", "grab_side",
                "swim_ticks", "swim_phi", "swim_rig", "reach_q16", "grab_pops", "rng"):
        if da[key] != db[key]:
            out.append(f"{key} {da[key]!r} != {db[key]!r}")
    if not out:
        out.append("records differ but no field located -- serialization drift?")
    return out


# ---------------------------------------------------------------------------
# Every creature: slot 5b
# ---------------------------------------------------------------------------
def _footprint_rect(u) -> tuple[int, int, int, int]:
    """A unit's footprint as a half-open tile rect (x0, y0, x1, y1)."""
    tiles = u.occupied_tiles()
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


class CreatureSystem:
    """Every creature body of one sim, keyed by unit id. Dormant (no body,
    no solver call, no section bytes, no recorder entries) when the sim has no
    creature unit. Holds no reference to the sim, its units or the engine, so
    it pickles."""

    def __init__(self):
        self.bodies: dict[int, CreatureBody] = {}

    def ordered_bodies(self) -> list:
        return [self.bodies[k] for k in sorted(self.bodies)]

    def add(self, body: CreatureBody) -> None:
        if body.unit_id in self.bodies:
            raise ValueError(f"unit {body.unit_id} already has a creature body")
        self.bodies[body.unit_id] = body

    def tick(self, sim) -> None:
        """Slot 5b: for every LIVING creature unit in unit-id order -- route
        its intents into the hub's goal, write the world's inputs, the hub's
        tick (the primitives write every arm's inputs), ONE solver call, the
        hub reads the solver's report, write the pose back to the unit
        (anchor + facing). A dead creature is skipped: frozen."""
        if not self.bodies:
            return
        engine = require_engine(sim._bp)
        by_id = {u.id: u for u in sim.units}
        rects = [(u.id, _footprint_rect(u)) for u in sorted(sim.units, key=lambda u: u.id)
                 if u.alive]
        gmap = sim.gmap
        for body in self.ordered_bodies():
            u = by_id.get(body.unit_id)
            if u is None or not u.alive:
                continue
            body.route_intents(u)
            body.write_inputs(gmap, [r for uid, r in rects if uid != body.unit_id])
            P.hub_step(body)
            AC.tick(body.row, body.solver_row_node, body.bufs, engine)
            P.after_solver(body)
            body.write_unit(u)

    def section_bytes(self) -> bytes:
        """The ``__creature__`` section of the current state."""
        return section_bytes(creature_carrier(self))


def spawn_creature_units(sim, rows) -> list:
    """``Simulation.spawn_creatures``'s body: refuse without the engine,
    check each row against the live map, build its Unit (``C-<slot>``, the
    species' footprint from the creature row) and its CreatureBody. Returns
    the units in row order."""
    from simulation.unit import Unit
    require_engine(sim._bp)
    from config import CFG
    gmap = sim.gmap
    tps = int(CFG.clock.ticks_per_second)
    out = []
    for i, row in enumerate(rows):
        where = f"creature #{i} ({row.species} at {row.x}, {row.y})"
        cdef = check_creature_spawn(row, ~np.asarray(gmap.solid, dtype=bool),
                                    gmap.tile_size_m, where)
        slot = len(sim.roster(row.team))
        u = Unit(f"C-{slot}", x=row.x, y=row.y, team=row.team,
                 footprint=int(cdef.body.footprint), species_id=row.species)
        sim.add_unit(u)
        sim._creature_system.add(CreatureBody(
            u, cdef, ticks_per_second=tps, tile_size_m=gmap.tile_size_m, solid=gmap.solid,
            topology_serial=gmap.topology_serial, sim_seed=sim._seed,
            heading_q24=CF.heading_q24_from_deg(row.heading)))
        out.append(u)
    return out


__all__ = [
    "CREATURE_DIGEST_KEY", "CREATURE_SECT_PREAMBLE", "ARM_LIMP",
    "GOAL_NONE", "GOAL_MOVE_DIR", "BODY_FIELDS",
    "is_creature", "require_engine", "seed_xorshift", "xorshift64",
    "rest_pose", "check_creature_spawn", "HitShape", "build_hit_shape",
    "CreatureBody", "CreatureSystem",
    "serialize_record", "decode_record", "section_bytes", "creature_carrier",
    "require_creature_carrier", "diff_creature_records", "spawn_creature_units",
]
