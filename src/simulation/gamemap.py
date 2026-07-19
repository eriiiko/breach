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
    O2,
    INERT_N2,
)


class SealBlocked(ValueError):
    """A seal precondition failed on live state. State is untouched (atomic).

    Raised by :meth:`GameMap.seal_tiles` for the state-dependent refusals —
    standing water on the span, a gas-holding sealed pocket — the cases a
    caller resolves in play (drain, vent), not by fixing code. Caller bugs
    (bounds, duplicates, already-solid, non-solid material) raise plain
    ``ValueError`` instead. Subclasses ``ValueError`` so a coarse caller can
    catch both. Design: docs/a5_evacuation_impl_2026-07-18.md §2.
    """


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
        # wall_hp — int32 Q16.16 (S3b): structural HP, the fire's fuel source
        # (F = clamp01(wall_hp/fuel_ref)). PHYSICAL >1 quantity, but the burn-through
        # depletion (wall_damage*dt*I ≪ 1 HP/tick) needs the Q16.16 fraction. Boundary
        # helpers in simulation.wall_fixed; populated from the table in _update_caches.
        self.wall_hp      = np.zeros((h, w), dtype=np.int32)
        self.is_vacuum    = np.zeros((h, w), dtype=bool)
        self.flammable    = np.zeros((h, w), dtype=bool)
        # S2c: the atmosphere (bulk pressure) is int32 Q16.16 (scale 2^16, shared
        # with water/heat/wave/gas) — the CLOSER of the S2 group: with atmosphere
        # + wind integer the whole atmosphere/wave/wind/smoke/gas group is
        # cross-machine bit-identical (only the downstream FIRE coupling stays a
        # float bridge, S3). 1.0 atm == FP_ONE (65536) counts. atmosphere is the
        # CONSERVED field (the wave transfer is a conservative ±-pair); the
        # vacuum/sponge + W3 compression are the deliberate-sink exceptions.
        # simulation.atmosphere_fixed has the real<->Q16.16 helpers (field edits,
        # render/recorder dequantize, the fire bridge). NEVER reassign (write
        # ``atmosphere[:] = ...``) — the C++ solvers hold a pointer to this buffer.
        from simulation import atmosphere_fixed as _atm_fx
        self.atmosphere   = np.full((h, w), _atm_fx.FP_ONE, dtype=np.int32)
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
        # S2c: wind is int32 Q16.16 (= -grad(atmosphere + wave_p), same 2^16
        # scale) — the smoke advection + the n_smoke CFL cliff read it integer,
        # the renderer/fire bridge dequantize it (atmosphere_fixed helpers). A
        # signed derived field (NOT conserved). Filled IN-PLACE by the C++
        # diffuse_solve (never reassigned) so a C++ view stays valid.
        self.wind_x       = np.zeros((h, w), dtype=np.int32)
        self.wind_y       = np.zeros((h, w), dtype=np.int32)
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
        # Fire intensity I — int32 Q16.16 (S3a, the THIRD/final field migration).
        # [0,1]-clamped tracer (0 == unlit, FP_ONE == fully ablaze). Boundary
        # helpers in simulation.fire_fixed quantize/dequantize (debug seeds, the
        # renderer/recorder, the C++ float bridge in physics_engine.step_tail —
        # which keeps dequantizing fire for the still-float C++ logistic until
        # S3b — and the heat-ray range/intensity params). The Python ignition
        # twin (combat.apply_temperature_ignition) writes it as an integer max.
        self.fire         = np.zeros((h, w), dtype=np.int32)
        self.obstacles    = np.zeros((h, w), dtype=bool)
        # (The smoke sink-direction field — sink_x/sink_y/_sink_dirty + the
        # BFS rebuild — is DELETED, EOS refactor P3 / decisions.md #3: venting
        # is native to the compressible solver; smoke rides the real venting
        # wind out of a breach instead of a scripted BFS pull.)
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

        # --- [water] initial state seed (engine/15 §2.3, P5) --------------
        # The seed lives HERE in __init__, right after _update_caches — and
        # NEVER inside _update_caches itself, despite the atmosphere t=0
        # precedent living there: _update_caches re-runs on config hot-reload
        # (reload_material_table), and a literal mirror of that precedent
        # would RE-FLOOD A DRAINED TANK on Ctrl+R. __init__ runs exactly once
        # per map; Simulation.reset() builds a fresh GameMap, so the seed
        # reapplies there by construction (and the runner's
        # _water_depth_before snapshot re-arms with it — level-seeded water
        # is "pre-existing", no tick-1 compression spike).
        #
        # Mask: only interior air gets water — the solver zeroes depth on
        # solid every step (a mass sink) and vacuum flash-boils it, so a
        # seed there would silently destroy mass. The editor masks at save;
        # this warn is the hand-authored-file backstop (count once, in-place
        # write, water_depth is never reassigned).
        water_seed_q = getattr(level_data, "water_depth_q", None)
        if water_seed_q is not None:
            seed = np.asarray(water_seed_q)
            mask = (~self.solid) & (~self.is_vacuum)
            self.water_depth[mask] = seed[mask]
            dropped = int(np.count_nonzero(seed[~mask]))
            if dropped:
                import warnings
                warnings.warn(
                    f"[water] depth_map for level "
                    f"'{getattr(level_data, 'name', '?')}': {dropped} "
                    f"cell(s) carry depth on solid/vacuum tiles — ignored "
                    f"(the solver zeroes depth on solid; the editor masks "
                    f"at save, so this file was likely hand-edited)",
                    RuntimeWarning, stacklevel=2)

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
        # wall_hp -> int32 Q16.16 (S3b): quantize the per-material HP table once at
        # cache build (round-to-nearest; integer HP values are exact at Q16.16).
        from simulation import wall_fixed as _wall_fx
        self.wall_hp = _wall_fx.quantize(tbl.hp[m])
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
        # S2c: int32 Q16.16 (1.0 atm == FP_ONE counts). _update_caches reassigns
        # the cache fields (the engine re-fetches field pointers each step), and
        # the running atmosphere is snapshotted/restored around this call below,
        # so this fresh allocation only seeds tick 0 / a reset.
        from simulation import atmosphere_fixed as _atm_fx
        self.atmosphere = np.where(
            self.solid | self.is_vacuum, 0, _atm_fx.FP_ONE
        ).astype(np.int32)

        # EOS refactor P1 (docs/eos_refactor_design.md §2.1): ambient bulk-gas
        # split. The two CONSERVATIVE species (O2 / inert_N2) seed the SAME
        # open-air mask atmosphere just used, split 21/79 by mole fraction
        # (Earth-normal air) — 0.21*FP_ONE + 0.79*FP_ONE happens to round back
        # to EXACTLY FP_ONE (13763 + 51773 == 65536), so N_O2+N_N2 at ambient
        # reproduces today's atmosphere==1.0 scale to the LSB (the calibration
        # tests/test_eos_p1_calibration.py pins). 0 on solid/vacuum, exactly
        # like atmosphere. IN-PLACE write (self.gas is never reassigned — a
        # C++ view of the buffer must stay valid); reload_material_table
        # snapshots + restores the running gas array around this call so a
        # hot-reload does not stomp live O2/N2 state.
        from simulation import gas_fixed as _gas_fx
        open_air = ~(self.solid | self.is_vacuum)
        self.gas[O2][:] = np.where(open_air, _gas_fx.quantize_scalar(0.21), 0)
        self.gas[INERT_N2][:] = np.where(open_air, _gas_fx.quantize_scalar(0.79), 0)

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
        # wall_hp -> int32 Q16.16 (S3b): quantize the new material's HP scalar.
        from simulation import wall_fixed as _wall_fx
        self.wall_hp[fy, fx] = _wall_fx.quantize_scalar(float(tbl.hp[mat_id]))
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
        # the running sim left them by snapshotting and restoring them. EOS
        # P1: _update_caches() now ALSO re-seeds ambient O2/N2 in-place
        # (self.gas is never reassigned, so a plain "snapshot the reference"
        # trick like atmosphere's would be a no-op — the mutation already
        # landed in the SAME buffer). Snapshot a COPY of the whole gas array
        # and copy it back in-place after, so a hot-reload does not stomp the
        # running O2/N2 (or any trace gas) state.
        atmosphere = self.atmosphere
        obstacles = self.obstacles
        gas_snapshot = self.gas.copy()
        self._update_caches()
        self.atmosphere = atmosphere
        self.obstacles = obstacles
        self.gas[:] = gas_snapshot

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

    def _seed_bulk_gas_neighbor_mean(self, fy, fx):
        """Seed ``gas[O2]``/``gas[INERT_N2]`` at a newly-opened tile (EOS
        refactor P1, docs/eos_refactor_design.md §2.2's minimal occupancy-
        transition slice) — mirrors the ``atmosphere`` neighbor-mean refill
        right next to every call site of this method in :meth:`destroy_wall`,
        same anti-vacuum-pulse intent, now on the bulk species too. The FULL
        evacuation rule (flooding/door-close) is P3's; this is only the
        cell-JOINS-open-air half (§2.2's last sentence)."""
        self.gas[O2][fy, fx] = self._neighbor_mean(self.gas[O2], fy, fx)
        self.gas[INERT_N2][fy, fx] = self._neighbor_mean(self.gas[INERT_N2], fy, fx)

    # (sink_fields / _rebuild_sink_field DELETED — EOS refactor P3,
    # decisions.md #3: native venting replaces the BFS smoke sink-pull.)

    # ------------------------------------------------------------------
    # Over-pressure wall failure — the emergent pressure-relief valve (ch.04 §5)
    # ------------------------------------------------------------------
    def find_burst_walls(self, max_pops: int | None = None):
        """Find wall tiles holding a pressure differential above their material's
        ``burst_threshold``. Pure scan — does NOT mutate state.

        A sealed room that keeps absorbing grenades builds pressure without
        limit; this is the emergent relief valve (ch.04 §5). For each wall tile,
        the differential it holds is the **spread across its open sides**:
        ``max(neighbour atmosphere) - min(neighbour atmosphere)`` over its
        in-bounds 4-neighbours, where a *solid* neighbour is not a side at all
        (it is more wall — skipped) and an *exposed-vacuum* neighbour is a real
        side holding 0 (so a hull between a pressurised room and outside-vacuum
        sees ``p_room - 0``). A wall between two equal-pressure rooms has ~0
        spread and never pops, even along a straight run whose along-wall
        neighbours are solid.

        Consequence: only 1-tile-deep wall membranes can burst. A tile of a
        >=2-thick slab has at most one open side, so its spread is 0 — thick
        walls hold ANY differential and breach via damage/explosions instead.
        (Deliberate: thickness-as-strength for free, no baked thickness field.)

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
        # S2c: atmosphere is int32 Q16.16 — dequantize to REAL pressure here so
        # the spread (hi-lo) compares against the real-unit burst_threshold `t`.
        from simulation import atmosphere_fixed as _atm_fx
        atm = _atm_fx.dequantize(self.atmosphere)
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
                # A solid neighbour is not a side — it's more wall; skipping
                # it (rather than counting 0) is what makes the spread a true
                # differential: equal pressure on both open sides -> 0, and a
                # tile with fewer than two open/vacuum sides can never burst.
                if solid[ny, nx]:
                    continue
                # An exposed-vacuum breach is a real side holding no air.
                p = 0.0 if is_vacuum[ny, nx] else float(atm[ny, nx])
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

        W2 (mechanics/03 §3): the gate is ``material != MAT_AIR`` — any
        destructible MATERIAL tile converts, solid walls (the shipped set,
        unchanged behaviour) AND non-solid destructibles like furniture:
        bullet chew must be able to break a crate so it stops *being* cover.
        No shipped caller ever reached here with a non-solid tile (the C++
        fire burn-through list is is_wall-gated; find_burst_walls scans
        walls; explosions gate on wall materials), so the widened gate only
        activates for the new W2 chew path.
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            return
        was_hull = (self.material[fy, fx] == MAT_HULL)
        if self.material[fy, fx] != MAT_AIR:
            self.material[fy, fx] = MAT_AIR
            # (sink-field staleness mark DELETED — EOS P3: no BFS sink field.)
            # Patch ALL table-derived caches for this tile through the single
            # incremental seam (solid, flammable, wall_hp, conductivity) —
            # no inline cache fixups, no O(grid) rebuild.
            self.on_tile_changed(fy, fx)
            # EOS refactor P3 (design §2.3): breach→vacuum GENERALIZED beyond
            # the edge-hull-only rule — ANY destroyed tile becomes vacuum if
            # it EXPOSES vacuum (any 4-neighbour is already vacuum: chained
            # breaches, a hole blown next to space), plus the original
            # edge-hull case. A destroyed tile NOT exposing vacuum joins
            # open-air with a neighbor-mean seed (anti-vacuum-pulse, as ever).
            on_edge_hull = was_hull and (
                fy < 1 or fy >= h - 1 or fx < 1 or fx >= w - 1)
            # "Exposing vacuum" == a 4-neighbour that is EXPOSED vacuum
            # (vacuum AND not solid — an intact hull tile is vacuum AND solid
            # and does NOT count; see _rebuild-era `breach` predicate).
            exposes_vacuum = any(
                0 <= fy + dy < h and 0 <= fx + dx < w
                and self.is_vacuum[fy + dy, fx + dx]
                and not self.solid[fy + dy, fx + dx]
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            if on_edge_hull or exposes_vacuum:
                # True breach — the tile joins vacuum; the solver's Dirichlet
                # P=0 + donor-cell venting drain it natively (no hard zero).
                self.is_vacuum[fy, fx] = True
            self.atmosphere[fy, fx] = self._neighbor_mean(
                self.atmosphere, fy, fx)
            self._seed_bulk_gas_neighbor_mean(fy, fx)

    # ------------------------------------------------------------------
    # EOS evacuation rule — seal / unseal (A5)
    #
    # The door-close half of the eos_refactor_design.md §2.2 occupancy-
    # transition rule (only the destroy direction existed before): a tile
    # leaving the open-air mask has its gas EVACUATED conservatively into
    # adjacent open cells before any solver pass sees the new mask — the
    # bulk-flux solver defensively zeroes N on solid every pass, so a seal
    # without evacuation silently deletes mass. The symmetric open half
    # (`unseal_tiles`) withdraws its seed from the donors instead of minting
    # (destroy_wall's neighbor-mean seed stays the rule for DESTRUCTION
    # events only). Both primitives are pure-integer, order-pinned, and
    # atomic. No sim path calls them yet (doors wire in at A6) — dormancy is
    # structural. Full design + critique fold:
    # docs/a5_evacuation_impl_2026-07-18.md (v2).
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_span(tiles):
        """Normalize an iterable of ``(fy, fx)`` into the pinned ROW-MAJOR
        sorted span (list of int tuples). Duplicate tiles are a caller bug →
        ``ValueError``. The caller's ordering can never matter (determinism:
        design §9)."""
        span = [(int(fy), int(fx)) for fy, fx in tiles]
        if len(set(span)) != len(span):
            raise ValueError(f"seal/unseal span contains duplicate tiles: {span}")
        span.sort()
        return span

    def _seal_receivers(self, fy, fx, span_set):
        """Open 4-neighbors of ``(fy, fx)`` eligible to receive evacuated gas,
        in the pinned N,S,E,W order (``_FACE_DIRS``). Span members are
        excluded — the span seals simultaneously, so receivers are defined
        against the POST-span solidity (design §3.1.5). Exposed-vacuum
        neighbors qualify: a breach is an open side; the share pushed there
        vents on the next flux pass through the sanctioned vacuum sink."""
        h, w = self._h, self._w
        out = []
        for dy, dx in self._FACE_DIRS:
            ny, nx = fy + dy, fx + dx
            if (0 <= ny < h and 0 <= nx < w
                    and not self.solid[ny, nx]
                    and (ny, nx) not in span_set):
                out.append((ny, nx))
        return out

    def _seal_blockers(self, span, material_id=None):
        """Shared validation for :meth:`seal_tiles` / :meth:`can_seal_tiles`.

        Runs the design §3.1 checks in their pinned order over the row-major
        span and returns ``(error, receivers)`` — ``error`` is the exception
        instance :meth:`seal_tiles` would raise (``None`` if the seal would
        succeed), ``receivers`` maps each span tile to its receiver list.
        ``material_id`` is checked only when given (``can_seal_tiles`` has no
        material argument — validity of the id is the caller's own argument,
        not state). NO mutation.
        """
        h, w = self._h, self._w
        # 1. bounds — strict (a primitive caller passing OOB is a bug;
        #    destroy_wall's silent OOB return is event-driven leniency).
        for fy, fx in span:
            if not (0 <= fy < h and 0 <= fx < w):
                return ValueError(
                    f"seal_tiles: tile ({fy}, {fx}) out of bounds"), None
        # 2. already solid — catches double-close bugs.
        for fy, fx in span:
            if self.solid[fy, fx]:
                return ValueError(
                    f"seal_tiles: tile ({fy}, {fx}) is already solid"), None
        # 3. material must be solid (permeability <= 0): sealing to a
        #    non-solid material is incoherent (tile stays open to flow while
        #    its gas was evacuated).
        if material_id is not None:
            mid = int(material_id)
            if not (0 <= mid < len(self.materials.permeability)):
                return ValueError(
                    f"seal_tiles: unknown material id {mid}"), None
            if float(self.materials.permeability[mid]) > 0.0:
                return ValueError(
                    f"seal_tiles: material id {mid} is not solid "
                    f"(permeability > 0)"), None
        # 4. water rule v1 — hard invariant guard at the primitive: the
        #    water solver zeroes depth on solid, so sealing over standing
        #    water is silent conserved-mass deletion. Span tiles only; a
        #    flooded RECEIVER is fine (gas parks under the water column,
        #    conserved — design §8).
        for fy, fx in span:
            if int(self.water_depth[fy, fx]) != 0:
                return SealBlocked(
                    f"seal_tiles: tile ({fy}, {fx}) holds standing water "
                    f"(drain before sealing)"), None
        # 5+6. receivers + sealed-pocket rule: a gas-holding tile with no
        #    receiver must be REFUSED, never zeroed (§2.2 canon: "it is
        #    never zeroed"). A gas-free tile seals fine with no receivers.
        span_set = set(span)
        receivers = {}
        for t in span:
            rs = self._seal_receivers(t[0], t[1], span_set)
            receivers[t] = rs
            if not rs and any(int(self.gas[g][t]) != 0 for g in range(N_GASES)):
                return SealBlocked(
                    f"seal_tiles: tile {t} holds gas but has no open "
                    f"receiver (sealed pocket — refusing to delete mass)"
                ), None
        # 7. overflow pre-check — loud, pre-mutation. N is a conserved
        #    field: a saturating store would SILENTLY break conservation,
        #    so a receiver that could exceed int32 must raise instead.
        #    Generous over-bound (assumes each receiver takes every adjacent
        #    span tile's whole load); unreachable at shipped densities.
        rec_order = []
        rec_donors = {}
        for t in span:
            for r in receivers[t]:
                if r not in rec_donors:
                    rec_donors[r] = []
                    rec_order.append(r)
                rec_donors[r].append(t)
        limit = 2 ** 31
        for g in range(N_GASES):
            for r in rec_order:
                bound = int(self.gas[g][r]) + sum(
                    int(self.gas[g][t]) for t in rec_donors[r])
                if bound >= limit:
                    return OverflowError(
                        f"seal_tiles: receiver {r} would overflow int32 on "
                        f"gas slice {g} (bound {bound})"), None
        return None, receivers

    def can_seal_tiles(self, tiles):
        """Policy query: True iff :meth:`seal_tiles` on ``tiles`` would
        succeed for a VALID solid ``material_id`` (material validity is the
        caller's own argument, not state — it is not re-checked here).

        Covers bounds / already-solid / water / receiver availability AND
        the int32 overflow pre-check, so True really means the seal
        completes. Does NOT check unit occupancy — that is caller policy
        (the A6 door composes ``occupancy_clear(span) and
        can_seal_tiles(span)``). Duplicate span tiles still raise
        ``ValueError`` (a caller bug, not a polite refusal). Pure query, no
        mutation. Design: docs/a5_evacuation_impl_2026-07-18.md §2.
        """
        span = self._normalize_span(tiles)
        err, _ = self._seal_blockers(span)
        return err is None

    def seal_tiles(self, tiles, material_id):
        """Seal a span of open tiles to ``material_id`` (a solid material),
        evacuating their gas conservatively to open neighbors.

        The door-close half of the §2.2 occupancy-transition rule: each
        tile's gas (all ``N_GASES`` slices) is split equally over its open
        non-span 4-neighbors — remainder to the first receivers in N,S,E,W
        order — with pure Python-int arithmetic, so grid-total N per slice
        is unchanged to the LSB. Solver-owned fields on the sealed tile are
        set to their solid steady state; ``temperature`` becomes the integer
        mean of the tile's PRE-call solid 4-neighbors' temperatures — the
        door panel belongs to the wall assembly it slides from, so no
        instant "hot door" from post-grenade air — falling back to keeping
        the local air T only when the tile has no pre-existing solid
        neighbor (Erik's ruling 4, 2026-07-19; design §4a). ``is_vacuum``
        is never written (sealing a breach yields the sealed-hull state). The whole
        span seals as ONE simultaneous edit (a 2-tile door closing is one
        call). Atomic: validates everything, then mutates; raises
        ``SealBlocked`` (water, sealed pocket) / ``ValueError`` (caller
        bugs) / ``OverflowError`` (loud conservation guard) with no partial
        mutation. Structural, not a FieldEdit: effects reach the solvers
        next tick via the step-6 restamp, exactly like ``destroy_wall``.
        Design: docs/a5_evacuation_impl_2026-07-18.md §3.
        """
        span = self._normalize_span(tiles)
        err, receivers = self._seal_blockers(span, material_id)
        if err is not None:
            raise err
        mid = int(material_id)

        # Close-T (Erik ruling 4, 2026-07-19; design §4a): each sealed tile's
        # temperature becomes the integer mean (floor) of its PRE-call solid
        # 4-neighbors' temperatures, summed in the pinned N,S,E,W order — the
        # door panel takes the temperature of the pre-existing wall assembly
        # it slides from (no instant "hot door" from post-grenade air;
        # conduction heats the panel honestly over subsequent ticks). Span
        # members never donate: their just-assigned close-T would be
        # circular, so "solid" means solid BEFORE this call — which is why
        # the means are computed HERE, before any mutation (this also keeps
        # the mutation pass below raise-free and span-order-independent).
        # A tile with no pre-existing solid neighbor keeps its air T.
        h, w = self._h, self._w
        close_t = {}
        for fy, fx in span:
            wall_ts = []
            for dy, dx in self._FACE_DIRS:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < h and 0 <= nx < w and self.solid[ny, nx]:
                    wall_ts.append(int(self.temperature[ny, nx]))
            if wall_ts:
                close_t[(fy, fx)] = sum(wall_ts) // len(wall_ts)

        # ATOMICITY PIN (design §3.2): atomicity rests on this mutation pass
        # being RAISE-FREE BY CONSTRUCTION — every precondition was validated
        # above and the pass is pure int loads/stores + on_tile_changed table
        # lookups — NOT on any transaction/rollback machinery. Extensions to
        # this pass must stay raise-free or add real rollback.
        for t in span:
            fy, fx = t
            rs = receivers[t]
            k = len(rs)
            for g in range(N_GASES):
                n = int(self.gas[g][fy, fx])
                if n == 0:
                    continue
                q, r = divmod(n, k)
                for j, (ny, nx) in enumerate(rs):
                    share = q + (1 if j < r else 0)
                    if share:
                        self.gas[g][ny, nx] = int(self.gas[g][ny, nx]) + share
                self.gas[g][fy, fx] = 0

            self.material[fy, fx] = mid
            self.on_tile_changed(fy, fx)

            # Solid steady-state values for the solver-owned fields (design
            # §6 table) — no "haunted door" values for the recorder snapshot.
            self.atmosphere[fy, fx] = 0
            self.wave_p[fy, fx] = 0
            self.wind_x[fy, fx] = 0
            self.wind_y[fy, fx] = 0
            self.flow_vx[fy, fx] = 0
            self.flow_vy[fy, fx] = 0
            self.ripple[fy, fx] = 0.0
            self.ripple_v[fy, fx] = 0.0

            # Close-T write (computed pre-mutation above — design §4a).
            if t in close_t:
                self.temperature[fy, fx] = close_t[t]

    def unseal_tiles(self, tiles):
        """Open a span of solid tiles to ``MAT_AIR``, seeding each from its
        open neighbors CONSERVATIVELY (withdrawn, not minted).

        The joins-open-air rule's shape with an exact conservation story:
        each opened tile is seeded at ``sum(donors) // (k + 1)`` — the
        opened tile joins the donor set as an EQUAL member, so the
        neighborhood relaxes toward its local uniform value (the correct
        anti-vacuum-pulse statement for a withdrawn seed; a single donor is
        halved, never drained to 0). The seed is withdrawn balanced-then-
        greedy from the donors (pinned N,S,E,W order), so grid-total N per
        slice is unchanged to the LSB. Donors come from the PRE-call open
        mask only (a 2-tile door's second tile never seeds from the first's
        fresh gas). A tile that is, or borders, exposed vacuum joins vacuum
        instead — ``is_vacuum`` set, NO seed (zeroing is correct only for
        vacuum); this predicate reads the LIVE solid mask, so the join
        chains down the row-major span order (pinned, deliberate). Unlike
        ``destroy_wall``, which mint-seeds unconditionally, opening never
        creates gas. Atomic like ``seal_tiles`` (``ValueError`` on caller
        bugs, no partial mutation).
        Design: docs/a5_evacuation_impl_2026-07-18.md §7.
        """
        span = self._normalize_span(tiles)
        h, w = self._h, self._w
        for fy, fx in span:
            if not (0 <= fy < h and 0 <= fx < w):
                raise ValueError(
                    f"unseal_tiles: tile ({fy}, {fx}) out of bounds")
        for fy, fx in span:
            if not self.solid[fy, fx]:
                raise ValueError(
                    f"unseal_tiles: tile ({fy}, {fx}) is not solid")
        span_set = set(span)
        # Donor snapshot: pre-existing open air only (design §7). numpy's
        # ``~`` allocates a fresh array, so this is immune to the in-place
        # per-tile solid updates below.
        pre_open = ~self.solid

        # Mutation pass — raise-free by construction, same atomicity story
        # as seal_tiles (design §3.2 pin).
        for t in span:
            fy, fx = t
            self.material[fy, fx] = MAT_AIR
            self.on_tile_changed(fy, fx)

            # Vacuum join (destroy_wall's exposes_vacuum predicate, minus
            # its unconditional mint): LIVE solid mask — chains down-span.
            joins_vacuum = bool(self.is_vacuum[fy, fx])
            if not joins_vacuum:
                for dy, dx in self._FACE_DIRS:
                    ny, nx = fy + dy, fx + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and self.is_vacuum[ny, nx]
                            and not self.solid[ny, nx]):
                        joins_vacuum = True
                        break
            if joins_vacuum:
                self.is_vacuum[fy, fx] = True
                continue

            donors = []
            for dy, dx in self._FACE_DIRS:
                ny, nx = fy + dy, fx + dx
                if (0 <= ny < h and 0 <= nx < w
                        and pre_open[ny, nx]
                        and not self.is_vacuum[ny, nx]
                        and (ny, nx) not in span_set):
                    donors.append((ny, nx))
            if not donors:
                # Opens empty (gas-free pocket) — NEVER mint.
                continue

            k = len(donors)
            for g in range(N_GASES):
                avail = [int(self.gas[g][d]) for d in donors]
                target = sum(avail) // (k + 1)
                if target == 0:
                    continue
                # Balanced two-pass withdrawal: equal shares clamped to each
                # donor's holdings, shortfall cascaded in N,S,E,W order.
                q, r = divmod(target, k)
                take = [min(q + (1 if j < r else 0), avail[j])
                        for j in range(k)]
                short = target - sum(take)
                for j in range(k):
                    if short == 0:
                        break
                    extra = min(short, avail[j] - take[j])
                    take[j] += extra
                    short -= extra
                for j, (ny, nx) in enumerate(donors):
                    self.gas[g][ny, nx] = avail[j] - take[j]
                self.gas[g][fy, fx] = target

            # Display-alias stopgap (design §6): the MEAN of the donors'
            # displayed atmosphere (divisor k, deliberately NOT k+1 — this
            # is the minted display value destroy_wall also provides, not
            # part of the conservation ledger; the solver rematerializes P
            # next tick). wave_p matches so the |P - P_prev| ripple splash
            # sees no phantom spike in the window before step 0.
            self.atmosphere[fy, fx] = (
                sum(int(self.atmosphere[d]) for d in donors) // k)
            self.wave_p[fy, fx] = int(self.atmosphere[fy, fx])
