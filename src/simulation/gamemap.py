"""GameMap — world state container (tiles + physics fields).

Lifted from ``game.py:GameMap`` (lines 292-542 in the legacy file) and merged
with the level-driven constructor that previously lived as a shim in
``main.py``. Canonical signature is ``GameMap(level_data)`` — the no-arg
form and ``_build_ship`` fallback from the legacy implementation are gone
(CSV loading via :mod:`level_loader` is the only path now).

Owns the cached arrays the physics systems read and write:

    material, wall_hp, solid, is_vacuum, flammable,
    atmosphere, wave_p, wave_v, wave_source, wind_x, wind_y,
    smoke, fire, obstacles, light_map, heat, smoke_glow

Plus ``self.level`` (the :class:`level_loader.LevelData` instance) and the
methods needed by combat / pathfinding / physics (``stamp_units``,
``is_passable``, ``is_passable_block``, ``has_los``, ``destroy_wall``).

No pygame, no pyray — pure numpy + config.
"""
from __future__ import annotations

import numpy as np

from config import CFG
from level_loader import materials_from_tilemap

# Material IDs are defined once in :mod:`simulation.materials` (the single
# source of truth) and re-exported here so existing
# ``from simulation.gamemap import MAT_*`` imports keep working.
from simulation.materials import (  # noqa: F401  (re-exported)
    MAT_AIR,
    MAT_HULL,
    MAT_WOOD,
    MAT_DOOR,
    MAT_STEEL,
    MAT_GLASS,
    MAT_FURNITURE,
    MaterialTable,
)

# Multi-gas system (engine/05 §6.2, M1): the gas-property table + slice ids.
# Re-exported so ``from simulation.gamemap import BLACK_SMOKE`` keeps working.
from simulation.gases import (  # noqa: F401  (re-exported)
    GasTable,
    N_GASES,
    WHITE_SMOKE,
    BLACK_SMOKE,
    POISON,
    TEARGAS,
    FUEL_GAS,
)


class GameMap:
    """2D grid map at fine-tile resolution, sized from the loaded level."""

    def __init__(self, level_data):
        """Build a GameMap from a :class:`level_loader.LevelData`.

        Grid dimensions come from ``level_data.tilemap`` — no longer from
        ``CFG.display.fine_w/fine_h``. The CSV decides the world size.
        """
        self.level = level_data
        h, w = int(level_data.tilemap.shape[0]), int(level_data.tilemap.shape[1])
        self._h, self._w = h, w

        # Material-property table (ch.02): single source of every per-material
        # constant. Derived caches below are projections of this table indexed
        # by the ``material`` grid. Rebuilt on config hot-reload via
        # :meth:`reload_material_table`.
        self.materials = MaterialTable.from_config(CFG)

        # Gas-property table (engine/05 §6.2, M1): the multi-gas analogue of the
        # material table — one row per gas (white_smoke / black_smoke / poison /
        # teargas / fuel_gas), loaded into per-gas absorption/scatter/diffusion/
        # decay/flag arrays + a name->index map. Drives the per-gas transport
        # loop (PhysicsRunner.step). Allocated once; rebuilt on hot-reload.
        self.gases = GasTable.from_config(CFG)

        # Field grids (allocate up front; populate from level + caches below)
        self.material     = np.zeros((h, w), dtype=np.int8)
        self.wall_hp      = np.zeros((h, w), dtype=np.float32)
        self.is_vacuum    = np.zeros((h, w), dtype=bool)
        self.flammable    = np.zeros((h, w), dtype=bool)
        self.atmosphere   = np.ones((h, w), dtype=np.float32)
        # S2a: the explicit WAVE state is int32 Q16.16 (scale 2^16, shared with
        # water/heat) — integer transport is bit-identical cross-machine (the
        # determinism the float path lacked). wave_p (acoustic anomaly, signed),
        # wave_v (velocity, signed), wave_source (injected energy, >= 0). The
        # Q-S2-2 measurement pinned wave_v to Q16.16 (peak ~2674 << 32768). Field
        # edits author wave_source in real units and quantize at the boundary
        # (field_edit.py "wave" dtype); the renderer/recorder dequantize. See
        # simulation.wave_fixed for the metres<->Q16.16 helpers.
        self.wave_p       = np.zeros((h, w), dtype=np.int32)
        self.wave_v       = np.zeros((h, w), dtype=np.int32)
        self.wave_source  = np.zeros((h, w), dtype=np.int32)
        self.wind_x       = np.zeros((h, w), dtype=np.float32)
        self.wind_y       = np.zeros((h, w), dtype=np.float32)
        # Multi-gas density fields (engine/05 §6.2, M1): a dense (N, h, w) array,
        # one (h, w) slice per gas type (slice order == the GAS_* ids). S2b: now
        # int32 Q16.16 (scale 2^16, shared with water/heat/wave) — the smoke + 5
        # gas planes are the integer-SL transport's synced state (deterministic,
        # non-conservative; docs/s2_fixed_point_plan.md §S2b). ``gas`` is
        # C-contiguous, so each ``gas[i]`` is itself a CONTIGUOUS (h, w) view —
        # the smoke C++ solver holds a raw int32 pointer to the buffer it is
        # handed, and a contiguous slice's pointer stays valid for in-place writes
        # (project gotcha: in-place vs reassignment). The per-gas transport loop
        # (PhysicsRunner.step) steps each non-empty slice. Boundary helpers in
        # simulation.gas_fixed quantize/dequantize (field edits, render, recorder).
        self.gas          = np.zeros((N_GASES, h, w), dtype=np.int32)
        # ``smoke`` is the canonical name for the BLACK_SMOKE slice (combustion
        # soot — what fire/explosions emit; its diffusion 0.10 matches today's
        # d_smoke=0.1). It is a VIEW into ``gas[BLACK_SMOKE]``: every reader and
        # in-place writer of ``gmap.smoke`` (recorder, renderer, raycaster, fire,
        # sink-pull, the FieldEdit deposit path) sees the same buffer, and writing
        # one is visible in the other. Behaviour-preserving: with only black_smoke
        # populated the result matches the pre-multigas single smoke field. NEVER
        # reassign ``smoke`` (do ``smoke[:] = ...``) — a reassignment would break
        # the aliasing and orphan any C++ view of the slice.
        self.smoke        = self.gas[BLACK_SMOKE]
        self.fire         = np.zeros((h, w), dtype=np.float32)
        self.obstacles    = np.zeros((h, w), dtype=bool)
        # Smoke sink-direction field (ch.05 smoke v2): a per-cell unit-ish
        # vector pointing, through air only, toward the NEAREST exposed-vacuum
        # breach; (0, 0) where there is no path to a breach (and everywhere when
        # the map is unbreached). The smoke solver adds ``sink_strength`` times
        # this to its advecting velocity, so smoke is gently pulled out of a
        # breached room even after the interior wind has died (the lingering-haze
        # fix). Built lazily from a BFS over air cells whenever topology changes;
        # ``_sink_dirty`` marks it stale. Allocated once, filled IN-PLACE by
        # :meth:`_rebuild_sink_field` (never reassigned → any C++ view stays
        # valid). Read through :meth:`sink_fields`.
        self.sink_x       = np.zeros((h, w), dtype=np.float32)
        self.sink_y       = np.zeros((h, w), dtype=np.float32)
        self._sink_dirty  = True
        # Scalar light field (legacy: fire raycaster output + render unit/smoke
        # tinting). Kept alongside light_rgb during the RGB migration.
        self.light_map    = np.zeros((h, w), dtype=np.float32)
        # RGB light field (ch.03 render byproduct): total light colour reaching
        # each tile, summed over all sources. Shape (h, w, 3), f32 accumulator
        # down-converted to the RGBA16F render textures at pack time (ch.05).
        self.light_rgb    = np.zeros((h, w, 3), dtype=np.float32)
        # Per-tile STATIC light attenuation (ch.02/03): the material table's
        # ``light_atten`` projected onto the grid, shape (h, w, 3) f32. This is
        # the static half of ``total_atten = material(static) × dynamic(live)``
        # — a structural-change cache (rebuilt in _update_caches, patched per
        # tile in on_tile_changed), NOT recomputed each tick. The directional
        # ray march reads it per channel: opaque [1,1,1] kills the ray (== old
        # wall hard-stop), air [0,0,0] passes untouched, glass [0.1,..] dims.
        self.light_atten  = np.zeros((h, w, 3), dtype=np.float32)
        # Per-tile STATIC heat attenuation (ch.02/03, engine/06 §1): the heat
        # analogue of ``light_atten`` — the material table's scalar ``heat_atten``
        # column projected onto the grid, shape (h, w) f32 (air 0.0, walls 1.0,
        # glass 0.3). The directional ray march reads it as the INDEPENDENT 4th
        # channel: heat survival attenuates by ``(1 - heat_atten)`` exactly as
        # each RGB channel attenuates by ``(1 - light_atten[c])``, so heat and
        # light occlusion can diverge (a heat-shield is light-clear/heat-opaque;
        # smoked glass is the converse). A structural-change cache, NOT recomputed
        # each tick: built in ``_update_caches`` and patched per tile in
        # ``on_tile_changed`` — the SAME seam as ``light_atten`` / ``conductivity``
        # / ``face_shift``. Static material heat only; units blocking heat is a
        # later dynamic refinement (no ``dyn_heat_atten`` yet). Allocated once,
        # filled IN-PLACE (never reassigned) so a C++ view stays valid.
        self.heat_atten   = np.zeros((h, w), dtype=np.float32)
        # Per-tile DYNAMIC light attenuation (ch.02 §static×dynamic, ch.03
        # §units): the live per-channel field the ray march actually reads.
        # Rebuilt every tick in ``stamp_units`` = static ``light_atten`` (copy)
        # combined per-channel via MAX with each living unit's opacity stamped
        # over its footprint (default [1,1,1] = opaque → unit shadow, restoring
        # pre-S2 behaviour). An occluder can only ADD opacity, never remove it.
        # Allocated once here and filled IN-PLACE each tick (never reassigned)
        # so a C++ view of the buffer never goes stale (project gotcha:
        # in-place writes vs reassignment). Away from units it equals the
        # static field, so behaviour matches S2 in unoccupied regions.
        self.dyn_light_atten = np.zeros((h, w, 3), dtype=np.float32)
        # Per-tile thermal conductivity (table-derived). Allocated + populated
        # now; consumed later by the temperature/conduction pass (ch.04).
        self.conductivity = np.zeros((h, w), dtype=np.float32)
        # Per-tile gas/smoke PERMEABILITY (table-derived): 0 = sealed wall,
        # 1 = open air, partial = porous. The physics-solid boundary derives
        # from this (a tile is solid to flow iff permeability == 0), replacing
        # the old occlusion-flag (is_wall) as the gas/wave boundary source. For
        # the current materials it equals the is_wall set, so behaviour is
        # unchanged; the gas/smoke solver consuming it as a *continuous* field
        # (partial units/grills) lands in a later step (ch.04).
        self.permeability = np.ones((h, w), dtype=np.float32)
        # Per-tile DYNAMIC gas/smoke permeability (ch.04 §3a): the live field
        # the C++ flux gather actually reads, = static ``permeability`` (copy)
        # with each living unit's footprint stamped to 0 (a unit fully blocks
        # flow this step — identical to today's obstacle stamp). Rebuilt IN-PLACE
        # each tick in ``stamp_units`` (never reassigned, so a C++ view of the
        # buffer never goes stale). Away from units it equals the static field,
        # so behaviour matches the pre-3a obstacle-mirror in unoccupied regions.
        self.dyn_permeability = np.ones((h, w), dtype=np.float32)
        # Per-tile STATIC wave-absorption (ch.04 §4a): the material table's
        # ``wave_absorb`` projected onto the grid (air 0, hull/steel/glass 0.1,
        # wood/door 0.4). Fraction of shockwave energy a tile damps. A
        # structural-change cache (rebuilt in _update_caches, patched per tile in
        # on_tile_changed), NOT recomputed each tick.
        self.wave_absorb = np.zeros((h, w), dtype=np.float32)
        # Per-tile DYNAMIC wave-absorption (ch.04 §4a): the live field the C++
        # wave update reads = static ``wave_absorb`` (copy) combined via MAX with
        # each living unit's footprint absorption (default
        # ``CFG.physics.unit_wave_absorb``, high — a body soaks blast). Rebuilt
        # IN-PLACE each tick in ``stamp_units`` (never reassigned, so a C++ view
        # of the buffer never goes stale). Away from units it equals the static
        # field; air is 0 there, so OPEN-AIR WAVE BEHAVIOUR IS UNCHANGED.
        self.dyn_wave_absorb = np.zeros((h, w), dtype=np.float32)
        # Heat deposit buffer (ch.03 output / ch.04 §Fixed-point format): the
        # only SIM-affecting ray output. Q16.16 FIXED-POINT int32 — 16 integer
        # bits, 16 fractional bits, so 1.0 energy == 65536 raw counts (the C++
        # HEAT_SCALE constant). The ray march SATURATING-adds into it (clamp at
        # INT32_MAX, never wrap). Integer += is order-independent -> determinism
        # (cross-machine / future lockstep multiplayer). Nothing READS it this
        # slice — the temperature pass (ch.04) consumes it non-destructively and
        # the per-tick deposit is cleared at cleanup. Allocated once, written
        # IN-PLACE (never reassigned) so any C++ view stays valid.
        self.heat = np.zeros((h, w), dtype=np.int32)
        # Temperature field (engine/06 §1, proposal §1 / §3.1): the persistent
        # consumer of the `heat` deposit. Q16.16 FIXED-POINT int32, SAME format
        # and scale as `heat` (TEMP_SCALE == HEAT_SCALE == 65536). Allocated to
        # 0 == AMBIENT: we store ΔT above a 20°C reference (T_ambient == 0,
        # proposal §3.1), so a freshly-allocated field is "cold" by construction.
        # Lives on SOLIDS only — the conversion/conduction passes skip air, so an
        # air tile starting at 0 stays bit-exactly 0. Written IN-PLACE by the C++
        # TemperatureSolver (never reassigned) so any C++ view stays valid.
        # STEP A: only the heat -> temperature conversion consumes it; conduction
        # (§2) and cooling (§3) land in later steps.
        self.temperature = np.zeros((h, w), dtype=np.int32)
        # Per-tile inverse-thermal-mass SHIFT cache (engine/06 §1.2): the
        # precomputed log2(thermal_mass) per tile, so the conversion is a pure
        # arithmetic right shift `temperature += heat >> heat_inv_shift` (no
        # divide, bit-identical cross-machine). Table-derived, built in
        # _update_caches and patched per tile in on_tile_changed — the SAME seam
        # as the `conductivity` cache. int32 to cross to C++ as a plain (h, w).
        self.heat_inv_shift = np.zeros((h, w), dtype=np.int32)
        # Per-tile CONDUCTION face-shift cache (engine/06 §2.5): for each tile
        # the shift for its 4 faces in fixed dir order N,S,E,W, looked up from
        # the material table's harmonic-mean `face_shift_table[mat_i][mat_n]`.
        # A face is NO_FACE (== materials.no_face) at a grid edge or when either
        # side has kappa==0 (air) — so the runtime conduction pass skips it and
        # air is a structural no-op. Baked in _update_caches and patched per tile
        # (plus its 4 neighbours) in on_tile_changed — the SAME structural-edit
        # seam as conductivity/heat_inv_shift, so a breached wall's faces update
        # the instant the tile changes. (h, w, 4) int32, C-contiguous for C++.
        self.face_shift = np.zeros((h, w, 4), dtype=np.int32)
        # Smoke-glow buffer (ch.03 C16 / ch.05 §God-rays): RENDER-ONLY god-ray
        # glow. The light each tile's smoke ABSORBS is deposited here per
        # channel by the march (energy-conserving). Shape (h, w, 3) f32 ->
        # packed into render Texture B at pack time (ch.05). Supersedes the old
        # surface-tint light_modulation path (no double-count). float (no
        # downstream sim threshold). Allocated once, written IN-PLACE.
        self.smoke_glow = np.zeros((h, w, 3), dtype=np.float32)
        # --- Water layer (engine/07 §2, water plan W2) --------------------
        # ``water_depth`` — metres of standing water on the floor — is THE
        # shared field of the water<->fire interface: the C++ WaterSolver pipe
        # model advances it each tick, and the fire side will read it as a
        # heat sink (boil-off emits white_smoke; that consumer is the fire
        # side's lane). Written IN-PLACE by the solver (never reassigned) so
        # any C++ view of the buffer stays valid.
        # S1 (docs/s1_water_fixed_point_plan.md): the SYNCED water state is int32
        # Q16.16 (metres, scale 2^16 == 65536) — the first fixed-point field
        # migration. Integer transport is bit-identical cross-machine (the
        # determinism the float path could not give). water_depth is CONSERVED
        # (Σ bit-conserved in a sealed flood). Dequantize (/65536) only at the
        # renderer + the float bridges (atmosphere/smoke, until S2). See
        # WATER_FP_ONE on the C++ module; mirrored here as WATER_FP_ONE.
        self.water_depth  = np.zeros((h, w), dtype=np.int32)
        # Cell-centred pipe-model flow velocity (Q16.16 m/s) — PERSISTENT solver
        # state, not a per-tick scratch: the damped velocity kick integrates
        # across ticks (water keeps sloshing between calls).
        self.flow_vx      = np.zeros((h, w), dtype=np.int32)
        self.flow_vy      = np.zeros((h, w), dtype=np.int32)
        # W6a ripple — the VISUAL-ONLY surface wave (canon §6, plan W6a): a
        # damped kick-drift displacement (m) riding ON TOP of water_depth,
        # splash-fed by wave_p, clamped to k_amp*depth, zeroed on dry/solid.
        # It NEVER feeds back into transport (the locked canon rule) — the
        # renderer is its only consumer (W6b). PERSISTENT solver state
        # (ripple_v is its m/s velocity auxiliary), written IN-PLACE.
        self.ripple       = np.zeros((h, w), dtype=np.float32)
        self.ripple_v     = np.zeros((h, w), dtype=np.float32)
        # OPTIONAL terrain height under the water (canon §2.1/§3): raises the
        # surface potential so water pools in low spots. S1: Q16.16 int32 metres
        # (it is added to water_depth in the surface potential, so it shares the
        # integer domain). Flat zero until a level paints it; a painter must
        # quantize metres -> Q16.16 (water_quantize) before writing here.
        self.floor_height = np.zeros((h, w), dtype=np.int32)
        # Ship tilt (radians, about the grid centre) — gameplay writes these;
        # the solver adds the tilt plane to the surface potential so water
        # slides low-side (the Titanic). Sane range |tilt| < ~30 deg.
        self.tilt_x       = 0.0
        self.tilt_y       = 0.0
        # Physical tile size in metres, from the level (a REQUIRED LevelData
        # field — the loader supplies the 0.333 default; do NOT add a second
        # default here). The water solver is the first consumer needing real
        # SI lengths: its CFL bound and gradients are in metres, unlike the
        # tile-unit shockwave.
        self.tile_size_m  = float(level_data.tile_size_m)
        # Continuous water sources [(y, x, level_m)]: per-tick HOLDS applied
        # in the runner (depth = max(depth, level_m)) — the pipe/breach
        # analogue of wave_source feeding. Event-shaped dumps (tank rupture,
        # scripted flood) go through the FieldEdit queue instead.
        self.water_sources = []

        # --- stamp_units C++ seam --------------------------------------------
        # The per-tick dynamic-field rebuild (``stamp_units``) can run either in
        # Python (the reference path) or in the C++ ``PhysicsEngine`` (the live
        # path). ``Simulation`` injects the engine via :meth:`bind_physics_engine`
        # once its ``PhysicsRunner`` is built; a bare ``GameMap`` (e.g. a unit
        # test that calls ``stamp_units`` directly) has no engine and falls back
        # to the Python path automatically. ``use_cpp_stamp`` is the A/B toggle:
        # the C++ path is the DEFAULT, but the field-level harness flips it to
        # False to capture the Python reference trajectory for the 0-ULP diff.
        # Both paths are byte-for-byte identical (the C++ port is a pure-structure
        # move — copies + a boolean compare + per-cell min/max, no float math).
        self._physics_engine = None
        self.use_cpp_stamp = True

        # Populate material + vacuum from the level's CSV (vocabulary is
        # format-version dependent — v1 generator codes vs v2 canon ids).
        mat, vac = materials_from_tilemap(level_data.tilemap, level_data.version)
        self.material[:] = mat
        self.is_vacuum[vac] = True

        self._update_caches()

    # ------------------------------------------------------------------
    # Cache rebuild
    # ------------------------------------------------------------------
    def _update_caches(self):
        """Rebuild all table-derived caches from the material grid.

        Every cache is a projection of the material-property table
        (``self.materials``) indexed by ``material`` — no hardcoded material
        lists. The two distinct masks are preserved (ch.02 §two masks):

        - ``solid`` — the physics/light/smoke/vision boundary mask. Derived from
          ``permeability`` (a tile is solid iff ``permeability == 0``), so it
          includes doors but not air — exactly the old ``{HULL, WOOD, DOOR}``
          set for the current materials.
        - ``is_passable`` (the walkability predicate) lives in the query
          methods and is the derived view ``mobility > 0`` over the material
          table's ``mobility`` column (mobility design §2/§8) — a terrain-only
          accessor; callers compose it with the live occupancy re-check.

        ``flammable`` and ``wall_hp`` come from the table; ``conductivity`` is
        populated for the later thermal pass. Atmosphere starts at 1.0 in
        interior air, 0.0 at walls and vacuum.
        """
        m = self.material
        tbl = self.materials

        # Static per-channel light attenuation: table column projected onto the
        # grid (ch.03 march input). C-contiguous f32 so it crosses to C++ as a
        # plain (h, w, 3) buffer with no copy.
        self.light_atten = np.ascontiguousarray(tbl.light_atten[m], dtype=np.float32)
        # Static scalar heat attenuation: the heat analogue of light_atten
        # (engine/06 §1), the material table's ``heat_atten`` column projected
        # onto the grid. C-contiguous f32 so it crosses to C++ as a plain (h, w)
        # buffer with no copy. The ray march reads it as the independent 4th
        # channel; built/patched through the same seam as light_atten.
        self.heat_atten = np.ascontiguousarray(tbl.heat_atten[m], dtype=np.float32)
        self.flammable = tbl.flammable[m]
        self.wall_hp = tbl.hp[m].astype(np.float32, copy=True)
        self.conductivity = tbl.conductivity[m].astype(np.float32, copy=True)
        # Per-tile inverse-thermal-mass shift = log2(thermal_mass), parallel to
        # the conductivity cache (engine/06 §1.2). Drives the heat -> temperature
        # conversion `temperature += heat >> heat_inv_shift`. int32 for C++.
        self.heat_inv_shift = tbl.heat_inv_shift[m].astype(np.int32, copy=True)
        # Per-tile conduction face-shift cache (engine/06 §2.5): baked from the
        # material grid via the harmonic-mean face table. NO_FACE at grid edges
        # and on any kappa==0 (air) face -> structural air no-op (built IN-PLACE
        # so a C++ view stays valid).
        self._rebuild_face_shift()
        # Gas/smoke permeability projected onto the grid (0 sealed, 1 open).
        self.permeability = tbl.permeability[m].astype(np.float32, copy=True)
        # Shockwave absorption projected onto the grid (ch.04 §4a).
        self.wave_absorb = tbl.wave_absorb[m].astype(np.float32, copy=True)

        # Solid mask (the physics solid boundary): a tile is solid iff it is
        # impermeable to gas (permeability == 0). For the current materials this
        # is exactly the old occlusion set ({HULL, WOOD, DOOR}), so behaviour is
        # unchanged; it replaces the retired ``is_wall`` flag as the
        # physics/light/smoke/vision boundary source. Always boolean-typed.
        self.solid = self.permeability <= 0.0

        # Atmosphere: 1.0 in interior air, 0.0 at solid tiles and vacuum.
        self.atmosphere = np.where(
            self.solid | self.is_vacuum, 0.0, 1.0
        ).astype(np.float32)

        # Obstacles (the physics solid boundary) == solid tiles (permeability
        # == 0) until stamp_units paints unit footprints over it. Sourced from
        # permeability, not the occlusion flag, so flow and optics can diverge.
        self.obstacles = self.solid

    # Conduction face directions, fixed order N,S,E,W (MUST match the C++
    # TemperatureSolver DIR_* / DY,DX and the binding's (h,w,4) layout).
    _FACE_DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

    def _rebuild_face_shift(self):
        """Bake the per-tile ``face_shift[y][x][dir]`` cache from the material
        grid (engine/06 §2.5), IN-PLACE so any C++ view of the buffer stays
        valid.

        For each tile ``i`` and each of its 4 faces (dir order N,S,E,W) the cache
        holds ``materials.face_shift_table[mat_i][mat_n]`` — the harmonic-mean
        face shift between this tile's material and the neighbour's. NO_FACE
        (``materials.no_face``) is written where the neighbour is OUT OF BOUNDS
        (grid edge) or on any kappa==0 (air) face; the face table already encodes
        the kappa==0 case as NO_FACE, so the only edge-specific work here is the
        grid boundary. Vectorised per direction (no Python per-tile loop).
        """
        m = self.material
        h, w = self._h, self._w
        face_tbl = self.materials.face_shift_table     # (N, N) int32
        no_face = int(self.materials.no_face)

        # Default every face to NO_FACE, then fill the in-bounds slabs per dir.
        self.face_shift[:] = no_face
        for d, (dy, dx) in enumerate(self._FACE_DIRS):
            # Slices of the (tile, neighbour) overlap region for this direction.
            ty0, ty1 = max(0, -dy), h - max(0, dy)
            tx0, tx1 = max(0, -dx), w - max(0, dx)
            mi = m[ty0:ty1, tx0:tx1]                   # this tile's material
            mn = m[ty0 + dy:ty1 + dy, tx0 + dx:tx1 + dx]  # neighbour's material
            self.face_shift[ty0:ty1, tx0:tx1, d] = face_tbl[mi, mn]

    def _patch_face_shift(self, fy, fx):
        """Patch the face_shift cache for tile (fy, fx) AND the facing entry of
        each of its 4 neighbours, after a structural edit to ``material``.

        A face is shared: changing tile i's material flips both ``face_shift[i]``
        (its 4 faces) and the ONE entry of each neighbour that points back at i.
        Symmetric table -> ``face(a,b) == face(b,a)``, so the neighbour's facing
        face gets the same value. O(1) (a handful of cells) — never an O(grid)
        rebuild. NO_FACE at the grid edge.
        """
        m = self.material
        h, w = self._h, self._w
        face_tbl = self.materials.face_shift_table
        no_face = int(self.materials.no_face)
        mi = int(m[fy, fx])
        for d, (dy, dx) in enumerate(self._FACE_DIRS):
            ny, nx = fy + dy, fx + dx
            if 0 <= ny < h and 0 <= nx < w:
                mn = int(m[ny, nx])
                self.face_shift[fy, fx, d] = int(face_tbl[mi, mn])
                # The neighbour's face that points BACK at (fy, fx) is the
                # opposite direction: N<->S (0<->1), E<->W (2<->3).
                opp = d ^ 1
                self.face_shift[ny, nx, opp] = int(face_tbl[mn, mi])
            else:
                self.face_shift[fy, fx, d] = no_face

    # ------------------------------------------------------------------
    # Incremental cache patch (single structural-edit seam — ch.02 review #10)
    # ------------------------------------------------------------------
    def on_tile_changed(self, fy, fx):
        """Patch ALL table-derived static caches for one tile after a
        structural edit to ``material[fy, fx]``.

        Centralizes cache invalidation so callers (``destroy_wall``, the future
        laser pre-phase) never patch caches inline. O(1) per tile — never an
        O(grid) ``_update_caches`` rebuild (which won't scale when a firestorm
        melts many walls per tick). Does NOT touch atmosphere/vacuum — those
        carry edit-specific semantics owned by the caller (see
        ``destroy_wall``).
        """
        if not (0 <= fy < self._h and 0 <= fx < self._w):
            return
        mat_id = int(self.material[fy, fx])
        tbl = self.materials
        self.light_atten[fy, fx] = tbl.light_atten[mat_id]
        # Heat attenuation — patched through the SAME seam as light_atten so a
        # breached wall's heat occlusion updates the instant the tile changes.
        self.heat_atten[fy, fx] = float(tbl.heat_atten[mat_id])
        self.flammable[fy, fx] = bool(tbl.flammable[mat_id])
        self.wall_hp[fy, fx] = float(tbl.hp[mat_id])
        self.conductivity[fy, fx] = float(tbl.conductivity[mat_id])
        # Inverse-thermal-mass shift cache — patched through the SAME seam as
        # conductivity so a breached wall's thermal coupling updates instantly.
        self.heat_inv_shift[fy, fx] = int(tbl.heat_inv_shift[mat_id])
        # Conduction face-shift cache — patch this tile's 4 faces AND the facing
        # entry of each neighbour (a shared face), so a breached wall's thermal
        # coupling to its neighbours updates the instant it changes.
        self._patch_face_shift(fy, fx)
        self.permeability[fy, fx] = float(tbl.permeability[mat_id])
        self.wave_absorb[fy, fx] = float(tbl.wave_absorb[mat_id])
        # Solid mask follows permeability (sealed iff permeability == 0).
        self.solid[fy, fx] = bool(self.permeability[fy, fx] <= 0.0)

    # ------------------------------------------------------------------
    # Config hot-reload: rebuild the table + static caches (ch.02 §14)
    # ------------------------------------------------------------------
    def reload_material_table(self):
        """Re-read the material table from config and rebuild static caches.

        Call after ``CFG.reload()``. Preserves the live ``material``/vacuum
        grids; only table-derived caches change. (A GPU material-mirror re-sync
        wires in here when CUDA lands — ch.02 §14.)
        """
        self.materials = MaterialTable.from_config(CFG)
        # Gas table is data-only (no per-tile cache projection in M1), so rebuild
        # it straight from config — the per-gas transport loop reads the fresh
        # diffusion/decay/flags next tick. Does NOT touch the ``gas`` array.
        self.gases = GasTable.from_config(CFG)
        # Rebuild only the table-derived caches; keep atmosphere/obstacles as
        # the running sim left them by snapshotting and restoring them.
        atmosphere = self.atmosphere
        obstacles = self.obstacles
        self._update_caches()
        self.atmosphere = atmosphere
        self.obstacles = obstacles

    # ------------------------------------------------------------------
    # Per-tick rebuild: units act as walls for all physics
    # ------------------------------------------------------------------
    def bind_physics_engine(self, engine):
        """Wire the C++ ``PhysicsEngine`` for the C++ ``stamp_units`` path.

        Called by :class:`Simulation` once its :class:`PhysicsRunner` is built
        (the runner owns the engine). A bare ``GameMap`` with no engine bound
        always uses the Python reference path. Idempotent."""
        self._physics_engine = engine

    def stamp_units(self, units):
        """Per-tick dynamic-field rebuild — dispatches to C++ or Python.

        The field rebuild (``obstacles`` + ``dyn_permeability`` +
        ``dyn_wave_absorb`` + ``dyn_light_atten``) runs in the C++
        ``PhysicsEngine`` when one is bound (:meth:`bind_physics_engine`) AND
        ``use_cpp_stamp`` is True (the default); otherwise the Python reference
        path (:meth:`_stamp_units_python`). The two are byte-for-byte identical
        (the C++ port is a pure-structure move: copies + a boolean compare +
        per-cell min/max — no float arithmetic). The atmosphere-refill bit
        (wall->free transitions) ALWAYS runs in Python — it is not unit-driven
        and is intentionally unchanged (design intent: units do NOT push
        atmosphere as they walk; they only block shockwaves via ``wave_absorb``).
        """
        if self._physics_engine is not None and self.use_cpp_stamp:
            self._stamp_units_cpp(units)
        else:
            self._stamp_units_python(units)

    def _stamp_units_cpp(self, units):
        """C++ path: flatten living units' footprints, call the engine, then run
        the Python-only atmosphere refill.

        The unit iteration + ``occupied_tiles()`` + the ``u.alive`` filter + the
        per-tile bounds check + the per-unit getattr-or-default all stay in
        Python (CPU actors own that). We build one row per stamped footprint
        tile — ``ys/xs`` (int32) and ``perm/wabsorb/atten_{r,g,b}`` (float32) —
        and hand them to :meth:`PhysicsEngine.stamp_units`, which does the
        in-place reset (``obstacles`` + the three ``dyn_*`` copies) and the
        min/max stamp loop. ``prev_obstacles`` is captured HERE, before the C++
        reset overwrites ``obstacles`` in place, so the atmosphere-refill diff
        below sees the pre-tick walls (exactly as the Python path did)."""
        # Capture the previous walls BEFORE the C++ reset writes obstacles in
        # place (the Python path snapshots self.obstacles, then reassigns; here
        # the engine writes the SAME buffer in place, so copy first).
        prev_obstacles = self.obstacles.copy()

        default_atten = (1.0, 1.0, 1.0)
        default_perm = float(getattr(CFG.physics, "unit_permeability", 0.5))
        default_wabsorb = float(getattr(CFG.physics, "unit_wave_absorb", 0.5))
        h, w = self._h, self._w

        # Build the flat stamp rows: one per (living unit, in-bounds footprint
        # tile). Plain Python lists — the unit count and footprints are tiny.
        ys, xs = [], []
        perm, wabsorb = [], []
        atten_r, atten_g, atten_b = [], [], []
        for u in units:
            if not u.alive:
                continue
            u_atten = getattr(u, "light_atten", default_atten)
            u_perm = float(getattr(u, "permeability", default_perm))
            u_wabsorb = float(getattr(u, "wave_absorb", default_wabsorb))
            ar, ag, ab = float(u_atten[0]), float(u_atten[1]), float(u_atten[2])
            for (tx, ty) in u.occupied_tiles():
                if 0 <= ty < h and 0 <= tx < w:
                    ys.append(ty)
                    xs.append(tx)
                    perm.append(u_perm)
                    wabsorb.append(u_wabsorb)
                    atten_r.append(ar)
                    atten_g.append(ag)
                    atten_b.append(ab)

        ys_a = np.asarray(ys, dtype=np.int32)
        xs_a = np.asarray(xs, dtype=np.int32)
        perm_a = np.asarray(perm, dtype=np.float32)
        wabsorb_a = np.asarray(wabsorb, dtype=np.float32)
        atten_r_a = np.asarray(atten_r, dtype=np.float32)
        atten_g_a = np.asarray(atten_g, dtype=np.float32)
        atten_b_a = np.asarray(atten_b, dtype=np.float32)

        # C++ reset + obstacles + min/max stamp loop (all IN-PLACE).
        self._physics_engine.stamp_units(
            self.permeability, self.wave_absorb, self.light_atten,
            self.dyn_permeability, self.dyn_wave_absorb, self.dyn_light_atten,
            self.obstacles,
            ys_a, xs_a, perm_a, wabsorb_a, atten_r_a, atten_g_a, atten_b_a,
        )

        # Atmosphere refill (Python-only, UNCHANGED — gamemap.py contract §c).
        # `freed` = walls that became free this tick (wall destroyed). Units are
        # not in `obstacles`, so a moving/dying unit triggers no fill.
        freed = prev_obstacles & ~self.obstacles
        if freed.any():
            for fy, fx in zip(*np.where(freed)):
                if not self.is_vacuum[fy, fx]:
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)

    def _stamp_units_python(self, units):
        """Rebuild ``obstacles`` = static walls (units are NO LONGER stamped
        here), and in the SAME pass rebuild the dynamic per-channel
        light-attenuation field ``dyn_light_atten`` and the dynamic gas/smoke
        permeability field ``dyn_permeability`` from each living unit's
        footprint (ch.04 §3b, ch.03 §units, ch.02 §static×dynamic).

        Three outputs of one pass:

        * ``obstacles`` = solid set (``permeability == 0``), i.e. WALLS ONLY.
          The C++ hard-zeroing BCs (zero pressure / Neumann skip) key off
          ``obstacles``/``is_wall``, so walls keep their hard wall behaviour
          and a unit is no longer force-zeroed — gas/pressure may exist in a
          unit's cell.
        * ``dyn_permeability`` = static ``permeability`` with each living unit's
          footprint set to a PARTIAL value ``unit_perm`` (ch.04 §3b). A unit is
          a *soft, porous body*: smoke/air seep past it (slowed by the
          ``face = min(perm)`` flux weighting), not perfectly blocked. The value
          comes from an optional per-unit ``unit.permeability`` hook, defaulting
          to ``CFG.physics.unit_permeability`` (0.5 = "slows flow, doesn't
          seal"). 0 would restore the old hard wall; 1 would be invisible.
          The stamp takes MIN with the static permeability: a body can make an
          open tile porous but never RAISE a sealed tile's permeability (a
          closed door under a unit stays flow-sealed — stamping it open made
          the solvers destroy mass at its faces, the door-stamp leak).
        * ``dyn_light_atten`` = static material attenuation combined per-channel
          via MAX with each living unit's opacity (UNCHANGED — units still cast
          solid shadows). Because the field is RGB a unit can occlude *per
          colour* via an optional ``unit.light_atten`` (default ``[1,1,1]`` =
          full block → a shadow). An occluder can only ADD opacity, never remove
          it.

        Uses ``unit.occupied_tiles()`` so the footprint contract (spec §6)
        is the only dependency — no assumption about storage representation.

        When *wall* tiles transition from blocked to free (wall destroyed),
        fill them with the neighbor mean of ``atmosphere`` to avoid a spurious
        vacuum pulse. ``freed`` keys off ``prev_obstacles & ~obstacles``, which
        now only changes on wall destruction (units are no longer in
        ``obstacles``), so a moving/dying unit triggers no fill — correct, since
        a unit no longer zeros its cell's atmosphere.
        """
        h, w = self._h, self._w
        prev_obstacles = self.obstacles
        # Base = solid tiles (permeability == 0) = WALLS ONLY. Units are no
        # longer painted into ``obstacles`` (3b): they are soft bodies, not
        # hard walls, so the C++ hard-zeroing BCs must not fire on them.
        self.obstacles = self.permeability <= 0.0
        # Reset the dynamic attenuation field to the static material baseline
        # IN-PLACE (no reassignment — keeps any C++ view valid). Units then
        # raise opacity per-channel below.
        self.dyn_light_atten[:] = self.light_atten
        # Reset the dynamic permeability field to the static material baseline
        # IN-PLACE (no reassignment — keeps any C++ view valid). Units then
        # lower their footprint to a PARTIAL value below (3b: porous body).
        self.dyn_permeability[:] = self.permeability
        # Reset the dynamic wave-absorption field to the static material baseline
        # IN-PLACE (no reassignment — keeps any C++ view valid). Units then raise
        # their footprint via MAX below (4a: a body soaks the blast).
        self.dyn_wave_absorb[:] = self.wave_absorb
        default_atten = (1.0, 1.0, 1.0)
        default_perm = float(getattr(CFG.physics, "unit_permeability", 0.5))
        default_wabsorb = float(getattr(CFG.physics, "unit_wave_absorb", 0.5))
        for u in units:
            if not u.alive:
                continue
            # Per-unit opacity hook: a unit may declare ``light_atten`` (RGB)
            # to occlude per colour; default is fully opaque (a shadow).
            u_atten = getattr(u, "light_atten", default_atten)
            # Per-unit permeability hook (mirrors light_atten): a unit may
            # declare ``permeability`` (e.g. a denser/looser body); default is
            # the config value (porous, slows flow but does not seal).
            u_perm = float(getattr(u, "permeability", default_perm))
            # Per-unit wave-absorption hook (mirrors the others): a unit may
            # declare ``wave_absorb``; default is the config value (high — a
            # body soaks the blast).
            u_wabsorb = float(getattr(u, "wave_absorb", default_wabsorb))
            for (tx, ty) in u.occupied_tiles():
                if 0 <= ty < h and 0 <= tx < w:
                    # Unit is a soft body: partial permeability, NOT an obstacle.
                    # MIN vs the static material: a body makes an OPEN tile
                    # porous, but must never RAISE a sealed tile's permeability.
                    # A closed DOOR is passable to movement yet solid to flow;
                    # stamping u_perm over it opened a flow face into a cell the
                    # solvers exclude and hold at zero — a mass sink that
                    # drained the sealed ship (the door-stamp leak).
                    sp = self.permeability[ty, tx]
                    self.dyn_permeability[ty, tx] = u_perm if u_perm < sp else sp
                    # Wave absorption: MAX so a unit can only ADD damping, never
                    # remove a lossy material's absorption underneath it.
                    cur = self.dyn_wave_absorb[ty, tx]
                    self.dyn_wave_absorb[ty, tx] = cur if cur >= u_wabsorb else u_wabsorb
                    # Per-channel MAX: opacity can only increase.
                    cell = self.dyn_light_atten[ty, tx]
                    cell[0] = cell[0] if cell[0] >= u_atten[0] else u_atten[0]
                    cell[1] = cell[1] if cell[1] >= u_atten[1] else u_atten[1]
                    cell[2] = cell[2] if cell[2] >= u_atten[2] else u_atten[2]

        freed = prev_obstacles & ~self.obstacles
        if freed.any():
            for fy, fx in zip(*np.where(freed)):
                if not self.is_vacuum[fy, fx]:
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)

    # ------------------------------------------------------------------
    # Pure queries (used by AI, combat, pathfinding)
    # ------------------------------------------------------------------
    def is_passable(self, fy, fx):
        """True if (fy, fx) is in-bounds and terrain-enterable.

        The walkability predicate is the derived view ``mobility > 0`` over the
        material table (mobility design §2/§8): a tile is enterable iff its
        material has positive mobility. ``mobility <= 0`` is the impassable
        sentinel (a wall), mirroring ``solid = permeability <= 0``. Terrain
        only — the caller composes this with the live occupancy re-check.
        """
        if fy < 0 or fy >= self._h or fx < 0 or fx >= self._w:
            return False
        return bool(self.materials.mobility[self.material[fy, fx]] > 0)

    def is_passable_block(self, fy, fx, footprint: int = 3):
        """True if every tile of a footprint-sized block at (fy, fx) is enterable.

        Enterability is geometry: a unit cannot overlap a wall, so *any* single
        ``mobility <= 0`` tile blocks the placement (mobility design §4 — the
        "best tile wins" intuition must NOT reach enterability). Projects the
        material block through the table's ``mobility`` column and requires all
        positive. Terrain only; speed (the area-average) is a separate axis.
        """
        if fy < 0 or fx < 0 or fy + footprint > self._h or fx + footprint > self._w:
            return False
        block = self.material[fy:fy + footprint, fx:fx + footprint]
        return bool(np.all(self.materials.mobility[block] > 0))

    def footprint_mobility(self, fy, fx, footprint: int = 3):
        """Per-tile ``mobility`` (milli-units) under a footprint at (fy, fx).

        The static-terrain input to the movement-cadence speed reduction
        (mobility design §4 / §4.1): the ``mobility`` column projected through
        the material grid for every tile of the footprint, as a flat list of
        Python ints. Out-of-bounds is clamped to the in-bounds overlap (the
        caller has already passed ``is_passable_block`` for a real step, so the
        footprint is in-bounds; the clamp is purely defensive). Pure read.
        """
        y0 = max(0, fy)
        x0 = max(0, fx)
        y1 = min(self._h, fy + footprint)
        x1 = min(self._w, fx + footprint)
        block = self.material[y0:y1, x0:x1]
        return self.materials.mobility[block].reshape(-1).tolist()

    def has_los(self, fy1, fx1, fy2, fx2):
        """Bresenham line-of-sight check. Stops on ``solid``."""
        h, w = self._h, self._w
        dx = abs(fx2 - fx1)
        dy = abs(fy2 - fy1)
        sx = 1 if fx1 < fx2 else -1
        sy = 1 if fy1 < fy2 else -1
        err = dx - dy
        x, y = fx1, fy1
        while True:
            if x == fx2 and y == fy2:
                return True
            if 0 <= y < h and 0 <= x < w and self.solid[y, x]:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _neighbor_mean(self, field, fy, fx):
        """Mean of field values from passable (non-solid, non-vacuum) 4-neighbors."""
        h, w = field.shape
        total = 0.0
        count = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = fy + dy, fx + dx
            if (0 <= ny < h and 0 <= nx < w
                    and not self.solid[ny, nx]
                    and not self.is_vacuum[ny, nx]):
                total += field[ny, nx]
                count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Smoke sink-direction field — toward the nearest breach (ch.05 smoke v2)
    # ------------------------------------------------------------------
    def sink_fields(self):
        """Return the (``sink_x``, ``sink_y``) sink-direction arrays, rebuilding
        them first if the map topology has changed since the last build.

        This is the read seam the physics runner uses each tick. The rebuild is
        lazy and gated by ``_sink_dirty`` (set at init and wherever ``solid`` /
        ``is_vacuum`` change), so the O(h·w) BFS runs only on the rare ticks a
        wall is destroyed, not every tick.
        """
        if self._sink_dirty:
            self._rebuild_sink_field()
        return self.sink_x, self.sink_y

    def _rebuild_sink_field(self):
        """Rebuild the smoke sink-direction field by a BFS over air cells.

        The field is a per-cell unit-ish vector pointing toward the nearest
        exposed-vacuum breach, propagated **through air only** (never through a
        solid / impermeable tile). It is what biases smoke advection toward a
        breach so a vented room actually clears (ch.05 smoke v2).

        Algorithm:

        1. An **air** cell is non-solid and non-vacuum (``~solid & ~is_vacuum``).
        2. **Sources** are air cells 4-adjacent to an exposed-vacuum tile (a
           breach: ``is_vacuum`` that is not solid). These start the BFS at
           distance 0.
        3. 4-connected BFS over air cells assigns each a hop-distance to the
           nearest breach. The BFS never steps onto a solid / impermeable /
           vacuum tile, so the distance respects walls — a sealed neighbouring
           room is unreachable and stays at "no path".
        4. Each reached air cell's **direction** = unit vector toward the
           in-bounds air neighbour with the SMALLEST distance (descending the
           distance field, i.e. the next hop along a shortest path to a breach).
        5. Cells with no path to a breach — and the whole field when the map has
           no breach at all — are (0, 0). Safe by construction: no breach ⇒ no
           pull ⇒ a sealed room is bit-identical to the no-sink solver.

        Written IN-PLACE into ``self.sink_x`` / ``self.sink_y`` (never
        reassigned), then clears ``_sink_dirty``.
        """
        from collections import deque

        h, w = self._h, self._w
        self.sink_x[:] = 0.0
        self.sink_y[:] = 0.0
        self._sink_dirty = False

        solid = self.solid
        is_vacuum = self.is_vacuum
        # Air = traversable by gas/smoke: not a wall, not vacuum.
        air = (~solid) & (~is_vacuum)

        INF = np.iinfo(np.int32).max
        dist = np.full((h, w), INF, dtype=np.int32)
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

        # Sources: air cells adjacent to an EXPOSED-VACUUM tile (a breach is a
        # vacuum tile that is NOT solid; an intact hull is vacuum AND solid).
        breach = is_vacuum & (~solid)
        q = deque()
        ys, xs = np.where(air)
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and breach[ny, nx]:
                    dist[y, x] = 0
                    q.append((y, x))
                    break

        if not q:
            # No breach reachable from any air cell → field stays all-zero.
            return

        # BFS through air only.
        while q:
            y, x = q.popleft()
            d = dist[y, x]
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and air[ny, nx]
                        and dist[ny, nx] == INF):
                    dist[ny, nx] = d + 1
                    q.append((ny, nx))

        # Direction = toward the in-bounds air neighbour of smallest distance
        # (the next hop down the shortest path to a breach), then normalised.
        reached_ys, reached_xs = np.where((dist < INF) & air)
        for y, x in zip(reached_ys.tolist(), reached_xs.tolist()):
            best_d = dist[y, x]
            best_dy = best_dx = 0
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and air[ny, nx]
                        and dist[ny, nx] < best_d):
                    best_d = dist[ny, nx]
                    best_dy, best_dx = dy, dx
            # A breach-adjacent source (best_d == its own 0) still has a smaller
            # neighbour only if one exists; otherwise it points at the breach via
            # the vacuum step it can't descend to — fall back to the breach dir.
            if best_dy == 0 and best_dx == 0:
                # No air neighbour is strictly closer (this is a source cell, or
                # a local min). Point directly at the adjacent breach tile.
                for dy, dx in dirs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and breach[ny, nx]:
                        best_dy, best_dx = dy, dx
                        break
            # (best_dy, best_dx) is one of the 4 unit steps → already unit length
            # for cardinals; store as float (sink_x is +x = column, sink_y = row).
            self.sink_x[y, x] = float(best_dx)
            self.sink_y[y, x] = float(best_dy)

    # ------------------------------------------------------------------
    # Over-pressure wall failure — the emergent pressure-relief valve (ch.04 §5)
    # ------------------------------------------------------------------
    def find_burst_walls(self, max_pops: int | None = None):
        """Find wall tiles holding a pressure differential above their material's
        ``burst_threshold``. Pure scan — does NOT mutate state.

        A sealed room that keeps absorbing grenades builds pressure without
        limit; this is the emergent relief valve (ch.04 §5). For each wall tile,
        the differential it holds is the **spread across its opposing sides**:
        ``max(neighbour atmosphere) - min(neighbour atmosphere)`` over its
        in-bounds 4-neighbours, where a *solid or sealed-vacuum* neighbour
        contributes 0 (so a hull between a pressurised room and outside-vacuum
        sees ``p_room - 0``). A wall between two equal-pressure rooms has ~0
        spread and never pops — correct.

        A material with ``burst_threshold <= 0`` is treated as never-bursting
        (air, or any material omitting the column).

        Parameters
        ----------
        max_pops
            Optional cap. When set, only the ``max_pops`` worst-differential
            tiles are returned (sorted descending), so a mistuned threshold
            cannot nuke the whole ship in one tick.

        Returns
        -------
        list of (int, int)
            ``(fy, fx)`` wall tiles that should fail this tick. Caller runs
            :meth:`destroy_wall` on each (mirrors fire burn-through plumbing).
        """
        h, w = self._h, self._w
        atm = self.atmosphere
        solid = self.solid
        is_vacuum = self.is_vacuum
        thresh = self.materials.burst_threshold

        failing = []  # (differential, fy, fx)
        ys, xs = np.where(solid)
        for fy, fx in zip(ys.tolist(), xs.tolist()):
            mat_id = int(self.material[fy, fx])
            t = float(thresh[mat_id])
            if t <= 0.0:
                continue  # n/a material (e.g. air) never bursts
            lo = None
            hi = None
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = fy + dy, fx + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                # A solid neighbour (wall, incl. sealed-hull which is also
                # solid) or an exposed-vacuum breach holds no air → 0; an
                # air tile contributes its atmosphere.
                if solid[ny, nx] or is_vacuum[ny, nx]:
                    p = 0.0
                else:
                    p = float(atm[ny, nx])
                lo = p if lo is None or p < lo else lo
                hi = p if hi is None or p > hi else hi
            if lo is None:
                continue
            spread = hi - lo
            if spread > t:
                failing.append((spread, fy, fx))

        if not failing:
            return []
        # Worst differentials first; apply the per-tick cap.
        failing.sort(key=lambda r: r[0], reverse=True)
        if max_pops is not None:
            failing = failing[:max_pops]
        return [(fy, fx) for _, fy, fx in failing]

    # ------------------------------------------------------------------
    # Mutators (used by explosions, fire wall burn-through)
    # ------------------------------------------------------------------
    def destroy_wall(self, fy, fx):
        """Convert (fy, fx) to air. Handles hull breach (edge => vacuum).

        Interior walls and non-edge hulls are refilled with the neighbor
        mean of ``atmosphere`` so we don't open with an artificial vacuum
        pulse. Edge hull tiles become vacuum and rely on relaxation BCs
        to drain smoothly.
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            return
        was_hull = (self.material[fy, fx] == MAT_HULL)
        # A wall is anything the solid mask marks (hull/wood/door today,
        # plus steel/glass when placed) — replaces the hardcoded id list.
        if self.solid[fy, fx]:
            self.material[fy, fx] = MAT_AIR
            # Topology changed → the smoke sink-direction field is stale; the
            # next ``sink_fields()`` read rebuilds it (cheap, breaches are rare).
            self._sink_dirty = True
            # Patch ALL table-derived caches for this tile through the single
            # incremental seam (solid, flammable, wall_hp, conductivity) —
            # no inline cache fixups, no O(grid) rebuild.
            self.on_tile_changed(fy, fx)
            if was_hull:
                if (fy < 1 or fy >= h - 1
                        or fx < 1 or fx >= w - 1):
                    # True hull breach — wall tile is on the map edge.
                    self.is_vacuum[fy, fx] = True
                    # Don't hard-zero — let relaxation BC drain smoothly.
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)
                else:
                    # Interior hull: fill with neighbor mean.
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)
            else:
                # Interior wall: fill with neighbor mean.
                self.atmosphere[fy, fx] = self._neighbor_mean(
                    self.atmosphere, fy, fx)
