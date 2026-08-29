"""GameMap — world state container (tiles + physics fields).

Lifted from ``game.py:GameMap`` (lines 292-542 in the legacy file) and merged
with the level-driven constructor that previously lived as a shim in
``main.py``. Canonical signature is ``GameMap(level_data)`` — the no-arg
form and ``_build_ship`` fallback from the legacy implementation are gone
(CSV loading via :mod:`level_loader` is the only path now).

Owns the cached arrays the physics systems read and write:

    material, wall_hp, fuel_recip, solid, thermal_solid, cool_shift,
    is_vacuum, flammable,
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
    MAT_DOOR_CLOSED,
    MaterialTable,
)

# Multi-gas system (engine/05 §6.2, M1): the gas-property table + slice ids.
# Re-exported so ``from simulation.gamemap import SMOKE`` keeps working.
from simulation.gases import (  # noqa: F401  (re-exported)
    GasTable,
    N_GASES,
    STEAM,
    SMOKE,
    POISON,
    TEARGAS,
    FUEL_GAS,
    O2,
    INERT_N2,
)


# P-O2b (docs/fire_realism_design_2026-08-01.md v5.2 "F-O2b") — the draw's two
# authored dials, resolved in ONE place so the dem_acc plane's depth and the
# value the C++ pass is told can never disagree.
COMBUSTION_DRAW_R_MAX = 3          # the baked offset tables' radius ceiling


def combustion_draw_slot_count(draw_r: int) -> int:
    """|{(dy,dx) : 1 <= |dy|+|dx| <= draw_r}| — 4 / 12 / 24 at R = 1 / 2 / 3.

    The twin of ``combustion_draw::slot_count`` in cpp/src/combustion.h.
    """
    return 2 * draw_r * (draw_r + 1)


def combustion_draw_slots() -> int:
    """MAX_CLAIMANTS — the ``dem_acc`` plane's slot depth, from ``config.toml``.

    HARD-CHECKED here as well as in C++: a plane shallower than the radius
    needs would ALIAS two sources' sub-count debts onto one slot, which is a
    silent corruption of synced state, so v5.2 rules a hit cap a violation
    rather than a note.
    """
    comb = getattr(CFG.physics, "combustion", None)
    draw_r = int(getattr(comb, "draw_r", 1))
    if not 1 <= draw_r <= COMBUSTION_DRAW_R_MAX:
        raise ValueError(
            f"[physics.combustion] draw_r = {draw_r} out of range "
            f"(1..{COMBUSTION_DRAW_R_MAX})")
    need = combustion_draw_slot_count(draw_r)
    cap = int(getattr(comb, "max_claimants", need))
    if cap < need:
        raise ValueError(
            f"[physics.combustion] max_claimants = {cap} is smaller than "
            f"draw_r = {draw_r} requires ({need} claimant slots)")
    return cap


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

    # ------------------------------------------------------------------
    # S8a Path B — GPU residency field sets (docs/cuda_s8a_residency_spec §2/§5b)
    # ------------------------------------------------------------------
    # The synced physics fields kept as CuPy device arrays in residency mode.
    # `_RESIDENT_SYNCED` are the OUTPUT fields the resident block writes and the
    # once-per-tick batched D2H (:meth:`to_host`) copies back to the numpy mirror
    # (combat/recorder/render read the mirror unchanged — the Q4 baseline). The
    # `_RESIDENT_MASKS` are the read-only inputs the resident kernels consume;
    # they + the four §5b unit-stamp masks (`dyn_permeability`/`dyn_wave_absorb`/
    # `dyn_light_atten` + `obstacles`) ride the per-tick always-upload set
    # (:meth:`from_host`). Rung 2 must NOT narrow these masks to structural-edit
    # deltas — body-shielding (a unit damping a shockwave for the unit behind it)
    # depends on them being re-uploaded every tick (units move every tick).
    _RESIDENT_SYNCED = (
        "atmosphere", "wave_p", "wind_x", "wind_y", "temperature", "heat",
        "fire", "wall_hp", "water_depth", "flow_vx", "flow_vy", "gas",
        # gas-energy conservation arc #54, design §5 (P-G0): no device kernel
        # reads/writes this field yet (no consumer this patch), but it joins
        # the resident set now so the round-trip is already wired for P-G1a+
        # — PhysicsRunner._step_resident uploads/downloads it explicitly
        # beside `temperature`.
        "gas_energy",
    )
    _RESIDENT_MASKS = (
        "solid", "is_vacuum", "is_ambient", "obstacles", "flammable",
        "floor_height", "heat_inv_shift", "face_shift",
        "dyn_permeability", "dyn_wave_absorb", "conductivity", "dyn_light_atten",
        # THERMAL-MASS AXIS, P2 (docs/thermal_mass_axis_design_2026-07-25.md +
        # build addendum §3): the per-medium THERMAL mask (`thermal_mass > 0`)
        # the temperature pass's six medium tests key on, on BOTH backends. It
        # joins the resident set for three reasons: (1) it gets a device buffer +
        # ONE upload at :meth:`enable_residency` (the static-mask precedent —
        # like `floor_height` / the sponge grids it is omitted from
        # `_step_resident`'s explicit per-tick lists); (2) the __setattr__
        # stale-pointer guard now covers it, exactly as it covers `solid` — both
        # are REASSIGNED by `_update_caches` and patched IN PLACE by
        # `on_tile_changed`; (3) `device_ptrs()["thermal_solid"]` is the pointer
        # a future resident temperature kernel takes.
        # CAVEAT, NOW LIVE (P-EOS, docs/thermal_mass_eos_ruling_2026-07-30.md):
        # unlike the sponge grids this mask is NOT static — `on_tile_changed`
        # patches it whenever a tile's material changes (a crate burning out) —
        # so any DEVICE kernel that reads it needs it re-uploaded every tick.
        # Device kernels DO read it now (the resident EOS: the SL T advection's
        # occluder mask + step-4c's skip), so `thermal_solid` RIDES THE PER-TICK
        # `from_host` LIST in `physics_runner._step_resident` beside
        # `solid`/`is_vacuum`/`is_ambient`. Do not move it back to a one-shot
        # upload (that is the exact staleness bug the is_ambient note records).
        # The resident temperature pass itself is still a host bracket
        # (`step_tail` on the mirror), and the per-call CUDA temperature kernel
        # does its own H2D from `GameMap.thermal_solid`.
        "thermal_solid",
        # COOL-SHIFT AXIS (2026-07-30): the per-tile ambient-decay shift the
        # temperature pass's Pass-3 reads (`T -= T >> cool_shift[i]`). Joins the
        # resident set for the same three reasons `thermal_solid` did: a device
        # buffer + ONE upload at :meth:`enable_residency`, the __setattr__
        # stale-pointer guard (it is REASSIGNED by `_update_caches` and patched
        # IN PLACE by `on_tile_changed`, exactly like `heat_inv_shift`), and
        # `device_ptrs()["cool_shift"]` as the pointer a future resident
        # temperature kernel takes.
        # SAME CAVEAT AS `thermal_solid`, recorded so it is not re-learned: this
        # grid is NOT static — `on_tile_changed` patches it whenever a tile's
        # material changes — so the moment a DEVICE kernel reads the resident
        # pointer it MUST join the per-tick `from_host` list in
        # `physics_runner._step_resident` beside `solid`/`is_vacuum`/
        # `is_ambient`/`thermal_solid`. No device kernel reads it today: the
        # resident tick's temperature pass is a host bracket (`step_tail` on the
        # mirror) and the per-call CUDA temperature kernel does its own H2D from
        # `GameMap.cool_shift`. The EOS (the one resident consumer of
        # `thermal_solid`) does not read this grid at all.
        "cool_shift",
        # FUEL-FRACTION AXIS (2026-07-30): the per-tile `make_recip` reciprocal
        # of the material's own full-health hp, which the fire logistic's fuel
        # term reads (`F = clamp01(wall_hp[i] * fuel_recip[i])`). Joins the
        # resident set for the same three reasons `cool_shift` did: a device
        # buffer + ONE upload at :meth:`enable_residency`, the __setattr__
        # stale-pointer guard (it is REASSIGNED by `_update_caches` and patched
        # IN PLACE by `on_tile_changed`), and `device_ptrs()["fuel_recip"]` as
        # the pointer a future resident fire kernel takes.
        # SAME CAVEAT AS `thermal_solid`/`cool_shift`: NOT static —
        # `on_tile_changed` patches it whenever a tile's material changes (a
        # crate burning out is exactly that), so the moment a DEVICE kernel
        # reads the resident pointer it MUST join the per-tick `from_host` list
        # in `physics_runner._step_resident`. No device kernel reads it today:
        # the resident tick's fire pass is a host bracket (`step_tail` on the
        # mirror) and the per-call CUDA fire kernel does its own H2D from
        # `GameMap.fuel_recip`.
        "fuel_recip",
        # PER-MATERIAL EXTINCTION TEMPERATURE (P-R3, 2026-07-31): the per-tile
        # Q16.16 foot of the fire logistic's `hot` ramp
        # (`hot = clamp01((T[i] - fire_T_ext_plane[i]) / fire_T_span)`). Joins
        # the resident set for the same three reasons `fuel_recip` did: a device
        # buffer + ONE upload at :meth:`enable_residency`, the __setattr__
        # stale-pointer guard (it is REASSIGNED by `_update_caches` and patched
        # IN PLACE by `on_tile_changed`), and `device_ptrs()["fire_T_ext_plane"]`
        # as the pointer a future resident fire kernel takes.
        # SAME CAVEAT AS `fuel_recip`: NOT static — `on_tile_changed` patches it
        # whenever a tile's material changes — so the moment a DEVICE kernel
        # reads the resident pointer it MUST join the per-tick `from_host` list
        # in `physics_runner._step_resident`. No device kernel reads it today:
        # the resident tick's fire pass is a host bracket (`step_tail` on the
        # mirror) and the per-call CUDA fire kernel does its own H2D from
        # `GameMap.fire_T_ext_plane`.
        "fire_T_ext_plane",
        # S8a Path A: the BC sponge grids — static per map (built ONCE in
        # __init__, never recomputed — unlike is_ambient, which destroy_wall's
        # joins-ambient twin mutates and therefore rides the per-tick EOS
        # upload). Device-needed by the resident EOS (σ-fold in the MG build;
        # u-damping band in the kick). Like floor_height they are omitted from
        # _step_resident's explicit per-tick lists; a defaulted from_host()
        # re-uploading them is harmless.
        "sponge_sigma", "sponge_udamp",
    )
    _RESIDENT_FIELD_NAMES = _RESIDENT_SYNCED + _RESIDENT_MASKS

    def __setattr__(self, name, value):
        """Stale-pointer guard (S8a): in residency mode a resident field must be
        written IN PLACE (``field[:] = ...``), never REASSIGNED — the CUDA
        launch cores hold raw device pointers into the CuPy buffers, and a
        reassignment would orphan them. Off (the default) this is a plain set."""
        if (name in GameMap._RESIDENT_FIELD_NAMES
                and self.__dict__.get("_residency_on", False)):
            cur = self.__dict__.get(name)
            if cur is not None and value is not cur:
                raise RuntimeError(
                    f"[residency] resident field '{name}' was REASSIGNED while "
                    f"GPU residency is on -- write it in place (`{name}[:] = ...`) "
                    f"so the device pointer stays valid, or disable residency first.")
        object.__setattr__(self, name, value)

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
        # material table — one row per gas (steam / smoke / poison /
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
        # Planetside reservoir ring (BC build, boundary_conditions_spec §1). On
        # ambient maps the SPACE-code tiles route here instead of is_vacuum (a
        # wholesale reinterpretation — no is_vacuum on a planetside map); on
        # space maps this stays all-False and every branch keying on it is
        # dormant. B3 pins these tiles to P_amb and clamps their gas to N_amb.
        self.is_ambient   = np.zeros((h, w), dtype=bool)
        # sky-exchange (docs/sky_exchange_design_2026-07-24.md §1.1): the
        # sky-connected INTERIOR air tiles — a flood fill from the is_ambient ring
        # through open air, EXCLUDING the ring itself and all solid/vacuum. Built
        # once at the end of __init__ (below _build_sponge_grid) and rebuilt
        # LAZILY (:meth:`ensure_sky_mask`) at a FIXED tick-order point whenever a
        # structural edit dirties it — a wall breach EXPANDS the mask, so a newly
        # opened room starts breathing. HOST-ONLY (never in the resident field
        # set): the per-tick sky pass runs on the mirror after combustion.
        self.sky_mask     = np.zeros((h, w), dtype=bool)
        self._sky_mask_dirty = False
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
        # --- mass-books channel: destruction seeding (P-M3 §3.4) -----------
        # Signed, MEASURED lifetime total of every change to Sum(bulk N) made by
        # :meth:`destroy_wall` — the named channel for the one sanctioned mass
        # SOURCE in the destruction path (and, on furniture/breach tiles, sink).
        # Signed because it is not a formula: writing the constant ambient seed
        # into a non-solid destructible that already held gas (furniture ships
        # permeability 0.5) is `seed - prior`, negative in any room above
        # ambient. Python int, so it never wraps; raw Q16.16 counts, so it can be
        # compared against a bulk-N sum to the LSB. P-M1 folds this into the
        # per-tick `Delta(Sum N) == Sum(named channels)` identity.
        self.n_destruction_seed_sum = 0
        # ``smoke`` is the canonical name for the SMOKE slice (combustion
        # soot — what fire/explosions emit; its diffusion 0.10 matches today's
        # d_smoke=0.1). It is a VIEW into ``gas[SMOKE]``: every reader and
        # in-place writer of ``gmap.smoke`` (recorder, renderer, raycaster, fire,
        # sink-pull, the FieldEdit deposit path) sees the same buffer, and writing
        # one is visible in the other. Behaviour-preserving: with only smoke
        # populated the result matches the pre-multigas single smoke field. NEVER
        # reassign ``smoke`` (do ``smoke[:] = ...``) — a reassignment would break
        # the aliasing and orphan any C++ view of the slice.
        self.smoke        = self.gas[SMOKE]
        # Fire intensity I — int32 Q16.16 (S3a, the THIRD/final field migration).
        # [0,1]-clamped tracer (0 == unlit, FP_ONE == fully ablaze). Boundary
        # helpers in simulation.fire_fixed quantize/dequantize (debug seeds, the
        # renderer/recorder, the C++ float bridge in physics_engine.step_tail —
        # which keeps dequantizing fire for the still-float C++ logistic until
        # S3b — and the heat-ray range/intensity params). The Python ignition
        # twin (combat.apply_temperature_ignition) writes it as an integer max.
        self.fire         = np.zeros((h, w), dtype=np.int32)
        # Edge-triggered ignition arm (Fable ruling 2026-07-24, zombie-smolder
        # fix): a per-tile bool that gates temperature ignition
        # (combat.apply_temperature_ignition). A tile is ARMED (eligible to be
        # seeded) until it ignites while hot, at which point it DISARMS; it
        # re-arms only once it genuinely COOLS (T < ignition_temp). This is the
        # hysteresis that stops a dying fire from being re-seeded to I_seed every
        # tick (the eternal 0.1 smolder). Init True == every tile can catch on its
        # first heating. SYNCED state (digest/save/undo), so it mirrors is_vacuum/
        # obstacles; a mole-fraction bool plane like them. HOST-ONLY, never in the
        # resident field set: apply_temperature_ignition runs on the numpy mirror
        # in Simulation.step (both the CPU and GPU-resident ticks), so no device
        # kernel reads/writes it. Back-compat: absent in any older serialization =>
        # True (the level format never persists it; GameMap always inits it here).
        self.ignition_armed = np.ones((h, w), dtype=bool)
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
        # P-R4 (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1.7):
        # the RADIATION accumulator — the second per-tick deposit plane, with a
        # DIFFERENT contract from ``heat`` above, which is why it is a separate
        # buffer and not another writer into that one:
        #   * ``heat``    — POSITIVE-SATURATING adds. Order-free only because
        #                   positives are monotone under the clamp. Writers:
        #                   combustion (H_fuel gas-side, H_bed fuel-bed side)
        #                   and weapons/payloads.
        #   * ``rad_net`` — PLAIN SIGNED adds (wrapping). The net-T⁴ exchange
        #                   applies the SAME integer + to the receiver and − to
        #                   the emitter, so a tile's total is a sum of signed
        #                   terms; under SATURATION that sum would be order-
        #                   DEPENDENT and the CPU↔CUDA tol-0 gate would break the
        #                   moment two rays reached one cell in a different
        #                   order. Plain signed adds ARE order-free.
        # Same Q16.16 scale and the SAME per-tick lifetime as ``heat`` (cleared
        # together at the very end of Simulation.step, after every consumer).
        # Written IN-PLACE (never reassigned) so any C++ view stays valid.
        self.rad_net = np.zeros((h, w), dtype=np.int32)
        # P-F1a (design v6.1 rule 4 / v7.1 items 8-9) — THE AMBIENT (SKY) LEDGER.
        # The ONLY place energy leaves the tile books. When an emission ray
        # LEAVES THE GRID the emitter is charged the escaping residual
        # ``a_s * tau_end * w * (E[T_s] - E[0])`` and the SAME integer is booked
        # here, keyed by the EMITTER's cell index:
        #     rad_net[s] -= sky ;  rad_amb[s] += sky
        # so ``rad_net.sum() + rad_amb.sum() == 0`` EXACTLY, pre-fold — gate (ii),
        # and the whole content of the conservation claim.
        #
        # WHY A PLANE AND NOT A SCALAR: a single global counter would be a
        # contended atomic on the device and — worse — an ORDER-DEPENDENT one if
        # it ever saturated. A per-tile int32 with PLAIN adds is order-free by
        # the same argument ``rad_net`` uses, and the host reduces it to a
        # uint64 total once per tick (entries are non-negative, since
        # ``E[T_s] >= E[0]`` for every bucket, so the reduction is exact).
        #
        # Same Q16.16 scale and the SAME per-tick lifetime as ``heat`` /
        # ``rad_net``: cleared together at the very end of Simulation.step.
        # Written IN-PLACE (never reassigned) so any C++ view stays valid.
        self.rad_amb = np.zeros((h, w), dtype=np.int32)
        # D3 (ruling amendment 5) — the RADIANT-FLUX SENSOR plane.
        # *** NOT part of the energy ledger. ***  It moves no energy, changes no
        # temperature, and nothing is debited to pay for it; no solver reads it.
        # Its ONE consumer is unit heat damage
        # (:func:`simulation.exchange.apply_environmental_damage`), which used
        # to sample the retired painter's deposit on the AIR tiles a unit
        # stands on. The radiation exchange lands only on SOLIDS (air has
        # ``heat_atten == 0`` so by Kirchhoff it neither absorbs nor emits), so
        # without this plane a fire could not burn a marine standing beside it.
        # Written along the rays at air cells as ``τ·w·a_s·E°[T_s]`` — same
        # occlusion, same 1/r ray-density falloff the painter had — with
        # POSITIVE-SATURATING adds, i.e. ``heat``'s order-free contract (it can
        # never go negative, unlike ``rad_net``). Same per-tick lifetime as
        # ``heat``/``rad_net``: cleared together at the end of Simulation.step.
        self.rad_flux = np.zeros((h, w), dtype=np.int32)
        # D1 (ruling amendment 5) — the COMBUSTION DEMAND ACCUMULATOR.
        # ``(max_claimants, h, w)`` int32, PERSISTENT SYNCED state (not a
        # per-tick buffer). Slot ``[s, y, x]`` is the sub-count oxygen debt the
        # air cell (y, x) owes toward the flammable claimant at draw-slot
        # offset ``s`` — the same offset keying ``alloc_slot`` uses inside the
        # combustion pass.
        #
        # P-O2b (design v5.2 "F-O2b"): the plane WIDENS with the extended
        # oxygen draw, from the 4 faces to the ``2R(R+1)`` offsets within BFS
        # hop-radius ``DRAW_R`` (4 / 12 / 24 at R = 1 / 2 / 3). The slot key is
        # the SOURCE OFFSET, not an enumeration ordinal, so a carried debt
        # still means the same thing next tick as fires appear and die. Ring 1
        # of the offset table is exactly ``D4``'s order, so at DRAW_R = 1 this
        # IS the pre-P-O2b (4, h, w) plane, slot for slot — which is why the
        # R = 1 regression oracle reproduces the old digests byte for byte.
        #
        # WHY IT EXISTS: the per-claimant demand ``burn_rate·dt·I·o2f`` is ~1.06
        # Q16.16 counts at the blessed operating point, and the old chained
        # ``mul_q16(mul_q16(...))`` truncated it to a STAIRCASE — 0 counts below
        # I = 0.200, 1 count above. A fire born at ``ignition_seed`` 0.12 drew
        # no oxygen at all, released no fuel-bed heat, and died at 21 s. The
        # accumulator keeps the wide product's remainder instead of throwing it
        # away, so the draw is exact IN EXPECTATION (1 count every ~1.65 ticks)
        # and the Huggett ``burn_rate`` anchor is untouched.
        #
        # Written IN-PLACE by the combustion pass; a slot is zeroed the moment
        # its source stops being a burning, REACHABLE claimant (see combustion.h
        # for the full reset rule). Single-writer per air cell -> no atomics on
        # CUDA.
        self.dem_acc = np.zeros((combustion_draw_slots(), h, w), dtype=np.int32)
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
        # Gas energy field (gas-energy conservation arc #54, design §2.2,
        # P-G0): int64 per cell, the EXACT unshifted product `N_raw *
        # T_abs_raw` (Q32 raw, no >>16) on the accountable set (!solid &
        # !thermal_solid & !is_vacuum & !is_ambient — bulk_transport.cpp's
        # `e_participates()`, mirrored here rather than re-derived
        # differently); 0 elsewhere, never read there. `temperature` stays
        # the TRUTH for thermal solids and, for now, the per-tick MIRROR for
        # gas cells (P-G0 has NO physics: this field is entirely
        # mirror-derived — :meth:`refresh_gas_energy` recomputes it from
        # (N, T) at the end of every tick; P-G1a+ makes it the live stored
        # truth and the writers in design §2.7 keep it in sync instead).
        # Populated for the level's t=0 state at the end of __init__, after
        # every N/T seed above has run.
        self.gas_energy = np.zeros((h, w), dtype=np.int64)
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
        # heat sink (boil-off emits steam; that consumer is the fire
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
        # Boundary mode (BC build, boundary_conditions_spec §1). On a planetside
        # map the SPACE-code tiles ARE the ambient reservoir ring — route them
        # WHOLESALE to is_ambient and leave is_vacuum empty (interior SPACE tiles
        # become legal "sky shafts" open to the atmosphere). Space maps take the
        # today's-behavior branch, byte-identical (dormancy by branch, not by
        # arithmetic identity).
        self._boundary = getattr(level_data, "boundary", "space")
        self._ambient = getattr(level_data, "ambient", None)
        if self._boundary == "ambient":
            self.is_ambient[vac] = True
            if self._ambient is None:      # hand-built LevelData: Earth defaults
                from simulation.ambient import derive_ambient
                self._ambient = derive_ambient()
        else:
            self.is_vacuum[vac] = True

        # --- A6 door-entity load stamp (a6 doors design §4.1) -------------
        # BETWEEN the tilemap fill and _update_caches, per the entity-doc §7
        # load-order rule: each door entity's runtime span is stamped
        # MAT_DOOR_CLOSED (authored closed) or MAT_AIR (authored open) so
        # the atmosphere/gas/water seeding below runs against the POST-stamp
        # solidity — authored-open ≡ authored-air is FIELD-identity by
        # construction, and conservation at t=0 is trivially exact (no gas
        # exists yet, so no seal_tiles call). Validation (§4.2 — bounds,
        # vacuum, CSV material, overlapping spans) hard-errors here. Zero
        # doors → zero work (dormancy is structural).
        if any(e.class_name == "door"
               for e in (getattr(level_data, "entities", None) or [])):
            from simulation.door_system import stamp_door_tiles
            # is_ambient joins the vacuum check: a door on the reservoir ring
            # is an authoring error exactly like a door on the hull ring.
            stamp_door_tiles(self.material, self.is_vacuum, level_data,
                             is_ambient=self.is_ambient)

        self._update_caches()

        # --- ambient sponge grid (BC build, boundary_conditions_spec §3) ---
        # Static per-tile absorber-mass grid, built ONCE from the final (already
        # upscaled) grid so `--res` keeps the physical band depth. Inert data in
        # this patch — B3's σ-sponge on the level-0 Helmholtz diagonal consumes
        # it. All-zero on space maps / when sponge_width == 0.
        self._build_sponge_grid(level_data)

        # --- sky-connected mask (sky-exchange §1.1) ------------------------
        # Same seeds + passability as the sponge BFS above, but no distance cap
        # and a bool reachability output. Built eagerly here so a bare GameMap
        # (no tick yet) already carries a valid mask; thereafter the dirty flag
        # drives lazy rebuilds. All-false on space maps / ring-free maps → the
        # per-tick pass is dead (gate a byte-identity).
        self._build_sky_mask()

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
            # The ambient ring is a water sink like the vacuum ring (BC spec
            # decision 3: no water BC — oceans are authored reservoirs).
            mask = (~self.solid) & (~self.is_vacuum) & (~self.is_ambient)
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

        # --- air_init.npy atmosphere seed (entity design §10, A9) ---------
        # Same placement rationale as the water seed above: __init__ runs
        # exactly ONCE per map (reset() builds a fresh GameMap, so the seed
        # reapplies by construction), and reload_material_table snapshots +
        # restores the running atmosphere/gas around _update_caches, so a
        # Ctrl+R can never stomp — or re-apply — the seed mid-run.
        #
        # THE PINNED TILE RULE: the override applies to OPEN-AIR tiles only
        # (~solid & ~is_vacuum & ~is_ambient). Values on SOLID tiles are
        # IGNORED — the atmosphere on solid is identically 0 (a boundary, not
        # a gas state; _update_caches pins it). Values on SPACE tiles are
        # IGNORED too, whichever boundary owns them: the vacuum ring IS the
        # boundary condition on space maps, and on planetside maps the
        # reservoir ring (is_ambient) wins over any author override (BC build;
        # the ring is clamped to N_amb every substep by B3). Silent by design,
        # unlike water's warning: a full-coverage grid (np.full(shape, FP_ONE))
        # is the natural authoring output and no conserved mass is lost there.
        #
        # The seed writes atmosphere = P_override AND splits the two
        # conservative bulk species to N_total == P (O2 21% half-up-rounded,
        # inert_N2 the exact remainder). This is what makes the seed REAL
        # under the EOS: pressure is re-derived every tick as
        # p* = C·N_total·T_abs (eos_solver.h §2), so an atmosphere-only
        # seed would evaporate on tick 1. At ambient temperature (T = 0
        # counts) N_total == P sustains the override; at P == FP_ONE the
        # split reproduces the P1 calibration EXACTLY (13763 / 51773 —
        # tests/test_eos_p1_calibration.py), so an explicit all-ambient
        # grid is bit-identical to no grid at all. Pure integer math
        # (int64 intermediate), deterministic cross-machine.
        air_seed_q = getattr(level_data, "air_init_q", None)
        if air_seed_q is not None:
            from simulation import gas_fixed as _gas_fx
            # The override applies to interior open air only — solid, vacuum AND
            # the ambient ring are excluded (ring rules win: the reservoir is
            # not author-overridable). On ambient maps the override's O2 split
            # follows the level's o2_frac; on space maps it stays 0.21.
            open_air = (~self.solid) & (~self.is_vacuum) & (~self.is_ambient)
            p = np.asarray(air_seed_q)[open_air].astype(np.int64)
            o2_frac = self._ambient.o2_frac if self._ambient is not None else 0.21
            o2_frac_q = _gas_fx.quantize_scalar(o2_frac)   # 13763 at 0.21 (P1 calib)
            o2 = (p * o2_frac_q + (1 << 15)) >> 16      # round-half-up
            self.atmosphere[open_air] = p.astype(np.int32)
            self.gas[O2][open_air] = o2.astype(np.int32)
            self.gas[INERT_N2][open_air] = (p - o2).astype(np.int32)

        # Gas energy seed (design §2.2/§5, P-G0): mirrors the level's t=0
        # (N, T) state now that every seed above (water/air_init) has run.
        # `temperature` is all-ambient (0) at this point on every level (no
        # per-level temperature seed exists yet), so this only matters where
        # `air_init_q` set N != the P1 calibration default.
        self.refresh_gas_energy()

    # ------------------------------------------------------------------
    # Gas energy field (gas-energy conservation arc #54, design §2.2/§2.7)
    # ------------------------------------------------------------------
    def _gas_energy_accountable(self):
        """The accountable set for ``gas_energy`` (design §2.2): the one
        canonical skip-set complement, MIRRORING (not re-deriving)
        ``bulk_transport.cpp``'s ``e_participates()`` — ``!solid &
        !thermal_solid & !is_vacuum & !is_ambient``. ``is_ambient`` is
        all-False on a space map, so this is the SAME predicate on both
        boundary modes without an ambient-mode branch (the C++ side's
        ``is_ambient != nullptr`` dormancy-by-branch collapses to the same
        answer here because the array itself is the dormant value)."""
        return (~self.solid) & (~self.thermal_solid) & (~self.is_vacuum) \
            & (~self.is_ambient)

    def _gas_bulk_n_raw(self):
        """Sum of the two CONSERVATIVE bulk gas planes (o2 + inert_n2), int64
        per cell — the Dalton N_total the EOS derives pressure from
        (``gases.conservative`` — reused, not re-derived; design §2.2)."""
        n = np.zeros((self._h, self._w), dtype=np.int64)
        for gi in np.flatnonzero(self.gases.conservative):
            n += self.gas[gi].astype(np.int64)
        return n

    def _gas_energy_t_amb_raw(self):
        """``T_AMB_K``, in raw Q16.16 counts (design §2.1/§2.2) — read
        through the canonical :mod:`temperature_scale` accessor and
        quantized via :mod:`simulation.gas_fixed` (never a hardcoded 290 or
        65536), mirroring the ``std::max<q16>(1, quantize(T_AMB_K))`` fold
        ``EOSSolver::step`` computes every tick (eos_solver.cpp)."""
        import temperature_scale
        from simulation import gas_fixed as _gas_fx
        t_amb_k = temperature_scale.load(CFG).eos_t_amb_k
        return max(1, _gas_fx.quantize_scalar(t_amb_k))

    def refresh_gas_energy(self):
        """Recompute ``gas_energy := N_raw * (temperature_raw + T_AMB_K_raw)``
        on the accountable set, 0 elsewhere (design §2.2/§5).

        P-G0's 'mirror-derived' state: there is NO physics behind this field
        yet (no writer stores an independent truth in it), so it is always
        fully re-derivable from the current ``(N, T)`` mirror and this method
        just does that — cheap, and always consistent by construction.
        :meth:`seed_gas_temperature` and :class:`~simulation.physics_runner.
        PhysicsRunner` (once per tick, after the thermal solver) are its two
        callers. P-G1a+ replaces this whole-grid recompute with the seam-
        written stored truth (design §2.6/§2.7) — this method's call sites
        will change, but the accountable-set/T_AMB_K helpers above stay.
        """
        accountable = self._gas_energy_accountable()
        t_amb_raw = self._gas_energy_t_amb_raw()
        n_bulk = self._gas_bulk_n_raw()
        t_abs = self.temperature.astype(np.int64) + t_amb_raw
        self.gas_energy[:] = np.where(accountable, n_bulk * t_abs, 0)

    def seed_gas_temperature(self, sel, T_q):
        """Write a gas cell's ``temperature`` (the mirror) AND ``gas_energy``
        (design §2.2's stored field) TOGETHER — the one sanctioned way
        tests/tools seed a GAS cell's temperature (design §2.7 last row, arc
        #54; CLAUDE.md rule 'Gas temperature is a mirror').

        ``sel`` is any numpy selection into the ``(h, w)`` grid (boolean
        mask, slice tuple, fancy index — whatever ``gmap.temperature[sel] =
        ...`` already accepted); ``T_q`` is the ABSOLUTE (relative-to-
        ambient) Q16.16 temperature to write, scalar or shaped like the
        selection, exactly like a plain ``temperature[sel] = T_q`` would
        take.

        Wall / thermal_solid tiles are NOT this method's business — those
        stay on ``temperature`` alone, unchanged (thermal solids are their
        own truth; nothing about their heat is booked in ``gas_energy``).
        Calling this on a non-accountable cell is harmless, not silently
        wrong: ``temperature`` is written as asked, and the follow-on
        :meth:`refresh_gas_energy` leaves ``gas_energy`` at its correct
        value there (0, by the accountable-set definition) either way.
        """
        self.temperature[sel] = T_q
        self.refresh_gas_energy()

    # ------------------------------------------------------------------
    # Ambient sponge grid (BC build, boundary_conditions_spec_2026-07-19 §3)
    # ------------------------------------------------------------------
    def _build_sponge_grid(self, level_data):
        """Static per-tile absorber grids for the planetside sponge band.

        Two coefficient grids share ONE 4-neighbour BFS distance ``d`` from the
        ring through open air, both an integer quadratic ramp ``c(d) =
        c_max·(W−d)²//W²`` over the band ``1 ≤ d < W`` (deterministic — BFS
        distance is order-free; only d is used, never which ring tile):

        - ``sponge_sigma`` — the σ pressure-sponge mass (spec §3 rung 1). B3b
          measured this REFLECTS (a soft Dirichlet), so its dial ships at 0; the
          grid stays wired for experimentation.
        - ``sponge_udamp`` — the k(d) VELOCITY-damping coefficient (spec §3 rung
          2, the real absorber, B3c). Q16 damping fraction in [0, FP_ONE); the
          kick multiplies the band's |u| by (1 − k(d)) magnitude-first, so
          outgoing momentum is dissipated in the band instead of reflecting off
          the hard ring.

        Built ONCE from the FINAL (already-upscaled) grid, so ``--res`` preserves
        the physical band depth: ``sponge_width`` is in BASE tiles, the effective
        W scales by ``res_factor`` (the door-span precedent). Both grids all-zero
        on space maps, ring-free maps, ``sponge_width == 0`` (hard-ring escape),
        or when the corresponding c_max is 0.
        """
        h, w = self.material.shape
        self.sponge_sigma = np.zeros((h, w), dtype=np.int32)
        self.sponge_udamp = np.zeros((h, w), dtype=np.int32)
        amb = self._ambient
        if amb is None or not self.is_ambient.any():
            return
        res_factor = int(getattr(level_data, "res_factor", 1) or 1)
        W = int(amb.sponge_width) * res_factor
        sigma_max = int(amb.sponge_strength)
        kmax = int(amb.sponge_u_damp)
        if W <= 0 or (sigma_max <= 0 and kmax <= 0):
            return
        # Multi-source 4-neighbour BFS: distance (in tiles) from the nearest ring
        # tile, propagated through open air only (walls block, so a sealed room
        # behind the ring is never damped — its σ stays 0). Ring tiles are d=0
        # (pinned anyway); the band is interior air at 1 ≤ d < W.
        from collections import deque
        INF = np.iinfo(np.int32).max
        dist = np.full((h, w), INF, dtype=np.int32)
        passable = ~self.solid
        dq = deque()
        ring_ys, ring_xs = np.nonzero(self.is_ambient)
        for fy, fx in zip(ring_ys.tolist(), ring_xs.tolist()):
            dist[fy, fx] = 0
            dq.append((fy, fx))
        while dq:
            fy, fx = dq.popleft()
            d1 = dist[fy, fx] + 1
            if d1 >= W:
                continue          # nothing beyond the band needs a distance
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = fy + dy, fx + dx
                if (0 <= ny < h and 0 <= nx < w
                        and passable[ny, nx] and d1 < dist[ny, nx]):
                    dist[ny, nx] = d1
                    dq.append((ny, nx))
        # Quadratic ramp on the band interior (exclude the ring tiles themselves,
        # d == 0: they are Dirichlet-pinned, so a band coefficient there is moot).
        # Both grids share the ramp numerator (W−d)²; strong near the ring, ~0 at
        # the inner band edge — the taper that lets a front enter without a
        # reflecting impedance jump and be dissipated as it nears the boundary.
        band = (dist >= 1) & (dist < W)
        d = dist[band].astype(np.int64)
        wq = np.int64(W)
        ramp = (wq - d) * (wq - d)          # quadratic taper numerator
        denom = wq * wq
        if sigma_max > 0:
            self.sponge_sigma[band] = (
                (np.int64(sigma_max) * ramp) // denom).astype(np.int32)
        if kmax > 0:
            self.sponge_udamp[band] = (
                (np.int64(kmax) * ramp) // denom).astype(np.int32)

    # ------------------------------------------------------------------
    # Sky-connected mask (sky-exchange, sky_exchange_design_2026-07-24 §1.1)
    # ------------------------------------------------------------------
    def _build_sky_mask(self):
        """(Re)build :attr:`sky_mask` — the sky-connected interior air tiles.

        A multi-source 4-neighbour BFS reachability from the ``is_ambient`` ring
        through open air (``~solid``) — the EXACT sponge-BFS pattern
        (:meth:`_build_sponge_grid`), same seeds and same passability, but with
        NO distance cap and a bool output. The final mask is the reachable set
        MINUS the ring itself and any solid/vacuum tile, i.e. the sky-connected
        *interior* air the per-tick exchange acts on.

        Determinism: reachability is order-free (a visited-set flood fill depends
        only on the graph, never on visit order), so the mask is identical
        cross-machine. Sealed rooms behind walls are never reached → correctly
        excluded (no sky). All-false on space maps (no ring) and ring-free maps.
        """
        h, w = self.material.shape
        mask = np.zeros((h, w), dtype=bool)
        if self._ambient is None or not self.is_ambient.any():
            self.sky_mask = mask
            return
        from collections import deque
        passable = ~self.solid
        visited = np.zeros((h, w), dtype=bool)
        dq = deque()
        ring_ys, ring_xs = np.nonzero(self.is_ambient)
        for fy, fx in zip(ring_ys.tolist(), ring_xs.tolist()):
            visited[fy, fx] = True
            dq.append((fy, fx))
        while dq:
            fy, fx = dq.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = fy + dy, fx + dx
                if (0 <= ny < h and 0 <= nx < w
                        and passable[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    dq.append((ny, nx))
        # Interior sky-connected open air only: drop the reservoir ring itself and
        # any solid/vacuum tile (their gas planes are 0 / reservoir-clamped — a
        # composition relaxation there is meaningless or would fight the ring).
        self.sky_mask = (visited & (~self.is_ambient)
                         & (~self.solid) & (~self.is_vacuum))

    def ensure_sky_mask(self):
        """Return :attr:`sky_mask`, rebuilding it first iff a structural edit
        dirtied it since the last build.

        DETERMINISM (Erik's caveat, 2026-07-24): the BFS is order-free, but WHEN
        it runs relative to the other passes must be pinned or two machines could
        disagree for a tick. The runner calls this at ONE fixed tick-order point —
        the top of the sky pass, immediately after combustion, in both the normal
        and resident ticks — never opportunistically."""
        if self._sky_mask_dirty:
            self._build_sky_mask()
            self._sky_mask_dirty = False
        return self.sky_mask

    # ------------------------------------------------------------------
    # The map's ambient air constant — ONE accessor (P-M3 §3.1.1)
    # ------------------------------------------------------------------
    def ambient_seed(self):
        """This map's ambient air, as EXACT Q16.16 integers:
        ``(n_total_q, o2_q, n2_q, pin_q)``.

        The single source of truth for "what one cell of this map's air is".
        Before P-M3 the same constant was re-derived at several sites by two
        derivations that agree only at default dials; anything that needs it
        (the tick-0/reload fill below, ``destroy_wall``'s destruction seed)
        now routes through here.

        ``self._ambient`` is ``None`` on EVERY space map — it is populated only
        under ``boundary == "ambient"`` (see ``__init__``) — so the two branches
        are load-bearing, not defensive:

        - **ambient map**: the level's own derived constants (``derive_ambient``)
          — ``n_total_q = quantize(p_amb)``, the split from the authored
          ``o2_frac``, and ``pin_q`` = the sim's own ``p*(N_amb, ΔT=0)``.
        - **space map**: Earth-normal ``FP_ONE`` of air, split 21/79 by mole
          fraction — ``quantize_scalar(0.21) == 13763`` and
          ``quantize_scalar(0.79) == 51773``, which sum back to EXACTLY
          ``FP_ONE`` (65536), and ``pin_q = FP_ONE`` to match.

        Note ``n_total_q != pin_q`` in general: on an ambient map the effective
        pin is ``p*(N)``, 65540 raw at Earth defaults, not 65536. Callers that
        mean *mass* must use ``n_total_q``; callers that mean *displayed
        pressure* must use ``pin_q``. Never assert the two are equal.

        Design: docs/mass_books_pm3_destroy_wall_seed_design_2026-08-18.md §3.1.1.
        """
        from simulation import atmosphere_fixed as _atm_fx
        from simulation import gas_fixed as _gas_fx
        amb = self._ambient
        if amb is not None:
            return (int(amb.n_total_q), int(amb.n_o2_q), int(amb.n_n2_q),
                    int(amb.pin_q))
        o2_q = int(_gas_fx.quantize_scalar(0.21))
        n2_q = int(_gas_fx.quantize_scalar(0.79))
        return (o2_q + n2_q, o2_q, n2_q, int(_atm_fx.FP_ONE))

    # ------------------------------------------------------------------
    # Cache rebuild
    # ------------------------------------------------------------------
    def _update_caches(self):
        """Rebuild all table-derived caches from the material grid.

        Every cache is a projection of the material-property table
        (``self.materials``) indexed by ``material`` — no hardcoded material
        lists. The distinct per-AXIS masks are preserved (ch.02 §two masks):

        - ``solid`` — the physics/light/smoke/vision boundary mask. Derived from
          ``permeability`` (a tile is solid iff ``permeability == 0``), so it
          includes doors but not air — exactly the old ``{HULL, WOOD, DOOR}``
          set for the current materials.
        - ``thermal_solid`` — the THERMAL-medium mask (``thermal_mass > 0``,
          thermal-mass axis design 2026-07-25). A THIRD axis, independent of
          flow: furniture is permeable (gas seeps past a crate) yet thermally
          solid (a crate holds an object temperature). The temperature solver's
          per-medium branches read THIS, never ``solid``.
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
        # THERMAL-MEDIUM mask (docs/thermal_mass_axis_design_2026-07-25.md
        # §2.1/§2.2; build addendum 2026-07-30 D3) — the per-tile projection of
        # the material table's `thermal_mass > 0`. THIS, not `solid`, is what the
        # temperature solver's six per-medium branches test: `solid` is a FLOW
        # property (permeability <= 0) and using it as the thermal medium put
        # furniture (permeability 0.5, the deliberate "shield but not seal" soft
        # body) into the GAS regime, so a burning crate's temperature was
        # advected away by the fire's own plume. Built HERE, on the same
        # structural-rebuild seam as `solid` / `heat_inv_shift`, and patched in
        # `on_tile_changed` — ONE seam, so the future movable-furniture version
        # has a single place to become dynamic (design §2.4).
        #
        # furniture is the ONLY material with permeability > 0 AND
        # thermal_mass > 0, so on a furniture-free map `thermal_solid == solid`
        # elementwise and every thermal path is byte-identical (addendum D4).
        self.thermal_solid = tbl.thermal_solid[m].copy()
        # COOL-SHIFT AXIS (2026-07-30) — the per-tile AMBIENT-DECAY shift, the
        # LOSS-side twin of `heat_inv_shift`. The cooling pass on a thermal-solid
        # tile is `T -= T >> cool_shift[i]` (engine/06 §3), e-fold 2^shift/24 s.
        # Built HERE, in the SAME ONE function as `heat_inv_shift` /
        # `thermal_solid`, and patched at the SAME single site in
        # `on_tile_changed` — the addendum's D3 rule: one seam, so the future
        # movable-furniture version has one place to become dynamic.
        # WHY it is per-tile at all: furniture's conductivity is 0, so with the
        # thermal-mass arc routing a crate into the solid thermal regime this
        # decay is its ONE loss channel, and the old single global could not be
        # right for both a thin hull plate and a wooden crate.
        # The VACUUM-exposed rate is derived from this SAME number by the global
        # offset (COOL_SHIFT - COOL_SHIFT_VACUUM) at the cooling site — ONE dial
        # per material, no second grid. See materials.py's `cool_shift` block.
        self.cool_shift = tbl.cool_shift[m].astype(np.int32, copy=True)
        # FUEL-FRACTION AXIS (2026-07-30) — the per-tile FUEL NORMALISER: the
        # `make_recip` reciprocal of THIS tile's material's full-health hp, so
        # the fire logistic's fuel term
        #     F = clamp01(wall_hp[i] * fuel_recip[i])
        # is "the fraction of THIS tile's own fuel still left". It was one
        # global ([physics.fire] fuel_ref = 60.0 — which is WOOD's hp), so a
        # full-health furniture crate (hp 30) permanently read F = 0.5, half
        # burnt out before it was ever lit, and a crate fire could not clear the
        # sustain ceiling at ambient O2 at any intensity or temperature.
        # DERIVED FROM `hp`, not a new dial — see materials.py's `fuel_recip`.
        # Built HERE, in the SAME ONE function as `heat_inv_shift` /
        # `thermal_solid` / `cool_shift`, and patched at the SAME single site in
        # `on_tile_changed` — the thermal-mass addendum's D3 rule: one seam, so
        # the future movable-furniture version has one place to become dynamic.
        # int64 because a Q16.16 reciprocal at RECIP_SHIFT = 32 does not fit
        # int32 (2^32/hp exceeds INT32_MAX for hp <= 2).
        self.fuel_recip = tbl.fuel_recip[m].astype(np.int64, copy=True)
        # PER-MATERIAL EXTINCTION TEMPERATURE (P-R3, 2026-07-31 — docs/
        # radiation_raycaster_extinction_ruling_2026-07-31.md A3 ride-along) —
        # the per-tile foot of the fire logistic's `hot` ramp, in Q16.16:
        #     hot = clamp01((T[i] - fire_T_ext_plane[i]) / fire_T_span)
        # It was one global ([physics.fire] fire_T_ext = 350) — which sits ABOVE
        # both shipped ignition temps (wood 300, furniture 280), so a tile could
        # ignite and be permanently below its own extinction floor. Now derived
        # per material as `ignition_temp[mat] - ignition_to_ext_delta`, which
        # makes `fire_T_ext < ignition_temp` STRUCTURAL. `fire_T_span` stays
        # global (the ramp's WIDTH is not a per-material quantity).
        # DERIVED FROM `ignition_temp`, not a new dial — see materials.py's
        # `fire_T_ext_from_ignition`. Built HERE, in the SAME ONE function as
        # `heat_inv_shift` / `thermal_solid` / `cool_shift` / `fuel_recip`, and
        # patched at the SAME single site in `on_tile_changed` (the
        # thermal-mass addendum's D3 rule: one seam).
        self.fire_T_ext_plane = tbl.fire_T_ext_q16[m].astype(np.int32, copy=True)
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

        # Atmosphere: interior air + (on ambient maps) the reservoir ring seed
        # to the ambient fill; 0.0 at solid tiles and vacuum. On space maps the
        # fill is FP_ONE and is_ambient is empty, so this is byte-identical to
        # before. On ambient maps the fill is the effective pin (65540 raw at
        # defaults — the sim's own p*(N_amb, 0)), so the interior materializes
        # flat against the ring pin (spec §1). S2c: int32 Q16.16. _update_caches
        # reassigns the cache fields (the engine re-fetches field pointers each
        # step), and the running atmosphere is snapshotted/restored around this
        # call below, so this fresh allocation only seeds tick 0 / a reset.
        # ONE accessor (P-M3 §3.1.1) — this was the reference derivation of the
        # constant and is now its single home; `ambient_seed()` reproduces both
        # branches exactly (ambient map -> derive_ambient's values; space map ->
        # FP_ONE split 21/79). Pure refactor, no value moves.
        _n_total_q, o2_fill, n2_fill, fill_p = self.ambient_seed()
        self.atmosphere = np.where(
            self.solid | self.is_vacuum, 0, fill_p
        ).astype(np.int32)

        # EOS refactor P1 (docs/eos_refactor_design.md §2.1): ambient bulk-gas
        # split. The two CONSERVATIVE species (O2 / inert_N2) seed the SAME
        # open-air mask atmosphere just used. On space maps the split is 21/79
        # by mole fraction (Earth-normal) — 0.21*FP_ONE + 0.79*FP_ONE rounds
        # back to EXACTLY FP_ONE (13763 + 51773 == 65536), reproducing today's
        # atmosphere==1.0 scale to the LSB (tests/test_eos_p1_calibration.py).
        # On ambient maps the split is the level's N-primary N_amb (o2_fill /
        # n2_fill), summing to N_total = quantize(p_amb), so p*(N_amb, 0) ==
        # the effective pin used for atmosphere just above. 0 on solid/vacuum,
        # exactly like atmosphere. IN-PLACE write (self.gas is never reassigned
        # — a C++ view of the buffer must stay valid); reload_material_table
        # snapshots + restores the running gas array around this call so a
        # hot-reload does not stomp live O2/N2 state.
        open_air = ~(self.solid | self.is_vacuum)
        self.gas[O2][:] = np.where(open_air, o2_fill, 0)
        self.gas[INERT_N2][:] = np.where(open_air, n2_fill, 0)

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
        # Thermal-medium mask — patched through the SAME seam as heat_inv_shift
        # (thermal-mass axis design §2.1; addendum D3: ONE build + ONE patch
        # site), so a destroyed crate leaves the solid thermal regime the
        # instant its material changes to air.
        self.thermal_solid[fy, fx] = bool(tbl.thermal_solid[mat_id])
        # Ambient-decay shift cache (cool-shift axis) — patched through the SAME
        # seam as heat_inv_shift/thermal_solid (D3: ONE build + ONE patch site),
        # so a burnt-out crate stops cooling like wood the instant it turns to
        # air (a moot value there — the cooling pass is thermal-solid only — but
        # the cache must not go stale).
        self.cool_shift[fy, fx] = int(tbl.cool_shift[mat_id])
        # Fuel-fraction normaliser cache — patched through the SAME seam as
        # wall_hp above (they are the numerator and the denominator of ONE
        # quantity, F = wall_hp/hp, and must never come from different
        # materials): the instant a burnt-out crate becomes air, its fuel
        # reciprocal becomes air's (0 == "no fuel"), matching the fresh
        # `wall_hp` this seam just wrote.
        self.fuel_recip[fy, fx] = int(tbl.fuel_recip[mat_id])
        # Per-material extinction-temperature cache (P-R3) — patched through
        # the SAME seam as fuel_recip/cool_shift above (D3: ONE build + ONE
        # patch site), so a burnt-out crate stops carrying furniture's `hot`
        # floor the instant its material becomes air (a moot value there — the
        # logistic is flammable-only — but the cache must not go stale).
        self.fire_T_ext_plane[fy, fx] = int(tbl.fire_T_ext_q16[mat_id])
        # Conduction face-shift cache — patch this tile's 4 faces AND the facing
        # entry of each neighbour (a shared face), so a breached wall's thermal
        # coupling to its neighbours updates the instant it changes.
        self._patch_face_shift(fy, fx)
        self.permeability[fy, fx] = float(tbl.permeability[mat_id])
        self.wave_absorb[fy, fx] = float(tbl.wave_absorb[mat_id])
        # Solid mask follows permeability (sealed iff permeability == 0).
        self.solid[fy, fx] = bool(self.permeability[fy, fx] <= 0.0)
        # sky-exchange (§1.1): any structural edit can change reachability (this
        # tile's solidness, and — via destroy_wall's joins-ambient twin, which
        # runs AFTER this seam — the ring). Mark the sky mask dirty; it rebuilds
        # once, lazily, at the next tick's fixed sky-pass point. Idempotent, so a
        # firestorm melting many walls per tick still costs at most one rebuild.
        self._sky_mask_dirty = True

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

    # ------------------------------------------------------------------
    # S8a Path B — GPU residency mode (CuPy device mirror of the field set)
    # ------------------------------------------------------------------
    def enable_residency(self):
        """Allocate CuPy device copies of the resident field set and switch the
        map into residency mode (:attr:`_residency_on`).

        Idempotent. Requires CuPy (the resident path is opt-in behind the
        ``--resident`` flag; with residency OFF, ``import cupy`` is never touched
        and the CPU/per-call paths are byte-for-byte unchanged). After this the
        numpy fields are the HOST MIRROR (all host code — ``cast_fire_heat``,
        ``stamp_units``, structural edits, combat/recorder/render — reads/writes
        them unchanged); the CuPy copies in ``self._dev`` are the device-resident
        buffers ``PhysicsEngine.step_resident`` runs the launch cores on, kept in
        sync by :meth:`from_host` / :meth:`to_host`.
        """
        if self.__dict__.get("_residency_on", False):
            return
        import cupy as cp  # opt-in only — never imported on the CPU path
        self._cp = cp
        dev = {}
        for name in self._RESIDENT_FIELD_NAMES:
            arr = getattr(self, name)
            dev[name] = cp.ascontiguousarray(cp.asarray(arr))
        self._dev = dev
        self._residency_on = True

    def residency_on(self):
        """True iff this map is in GPU residency mode (device buffers live)."""
        return self.__dict__.get("_residency_on", False)

    def device_ptrs(self):
        """Raw device addresses (int) of every resident field's CuPy buffer,
        keyed by field name — the ``uintptr_t`` the C++ launch cores take. Stable
        across a tick: :meth:`from_host`/:meth:`to_host` copy IN PLACE (``.set()``/
        ``.get(out=)``), never reassigning a device array."""
        return {name: int(arr.data.ptr) for name, arr in self._dev.items()}

    def from_host(self, names=None):
        """Batched H2D: copy the numpy mirror INTO the persistent CuPy buffers
        (in place, pointer-stable). ``names=None`` uploads the full always-upload
        set (Rung 1, spec §3.2 — covers every host writer: FieldEdits, ``heat``,
        structural edits, the §5b unit-stamp masks)."""
        for name in (names if names is not None else self._RESIDENT_FIELD_NAMES):
            self._dev[name].set(getattr(self, name))

    def to_host(self, names=None):
        """Batched D2H: copy the CuPy buffers back INTO the numpy mirror (in
        place — the C++ views + ``smoke`` alias stay valid). ``names=None``
        downloads the synced OUTPUT set (spec §3.4 — combat/recorder/render read
        the mirror unchanged)."""
        for name in (names if names is not None else self._RESIDENT_SYNCED):
            self._dev[name].get(out=getattr(self, name))

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

    # (``_seed_bulk_gas_neighbor_mean`` DELETED — P-M3: it WAS the mint. It
    # seeded gas[O2]/gas[INERT_N2] at a newly-opened tile with the neighbour
    # mean, withdrawing nothing from the donors, so every destroyed tile
    # created one neighbour-mean cell of air out of nothing — scaling with
    # exactly the pressure that made the wall burst. :meth:`destroy_wall` now
    # writes a CONSTANT ambient total with an inherited composition, and books
    # it. Design: docs/mass_books_pm3_destroy_wall_seed_design_2026-08-18.md.)

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
        # BC (boundary_conditions_spec_2026-07-19 §1, audit (c)): an ambient-side
        # neighbour is a real side holding the effective pin P_amb (not 0 like a
        # vacuum side, and not its possibly-stale materialized `atm` value — a
        # just-breached ring tile is seeded by neighbour-mean, not yet re-pinned).
        # So a wall with ambient on BOTH sides holds ~0 spread and never bursts.
        # On a space map is_ambient is empty -> the vacuum/atm branch is unchanged.
        is_ambient = self.is_ambient
        p_amb_real = (float(self._ambient.pin_q) / _atm_fx.FP_ONE
                      if self._ambient is not None else 0.0)
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
                # An exposed-vacuum breach is a real side holding no air; an
                # ambient breach is a real side holding the effective pin P_amb.
                if is_vacuum[ny, nx]:
                    p = 0.0
                elif is_ambient[ny, nx]:
                    p = p_amb_real
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
    def _bulk_n_sum(self):
        """Exact int64 total of the CONSERVATIVE bulk species (O2 + inert_N2)
        over the whole grid, raw Q16.16 — the quantity the mass books track.

        Integer sum, never a float reduction: the ledger has to close to the
        LSB or it cannot distinguish a mint from rounding."""
        return (int(self.gas[O2].astype(np.int64).sum())
                + int(self.gas[INERT_N2].astype(np.int64).sum()))

    def destroy_wall(self, fy, fx):
        """Convert (fy, fx) to air. Handles hull breach (edge => boundary).

        Interior walls and non-edge hulls are seeded with **one cell of the
        map's ambient air** (:meth:`ambient_seed`) so we don't open with an
        artificial vacuum pulse — a CONSTANT total, whatever the local
        pressure. Edge hull tiles (and any tile that exposes the boundary)
        join the boundary reservoir instead: no seed, and whatever gas the
        tile already held is evacuated here rather than left for the
        transport clamp to wipe unbooked.

        **Why a constant, and why it is not withdrawn from the donors**
        (P-M3, docs/mass_books_pm3_destroy_wall_seed_design_2026-08-18.md §2):
        this used to write the neighbour MEAN, minting one neighbour-mean cell
        of air per destroyed tile. Since ``find_burst_walls`` fires on a
        pressure differential, the mint scaled with exactly the pressure that
        triggered the burst — the emergent relief valve was a pressure
        AMPLIFIER, and carried 87.7% of a measured 2.201x session mass growth.
        Withdrawing the seed from the donors (as ``unseal_tiles`` does) would
        close the books exactly but was rejected on physical grounds: an
        explosion redistributes matter, it does not eliminate it, and the sim
        carries no rubble, so charging the player for the opened volume makes a
        blasted-open cave suffocate you for digging it out. Seeding ambient
        models the reservoir that is actually there. The seed is therefore a
        NAMED, BOOKED source, not a silent one — see
        ``n_destruction_seed_sum`` below; the load-bearing property is the
        CONSTANT TOTAL (it breaks the feedback loop), not the value.

        W2 (mechanics/03 §3): the gate is ``material != MAT_AIR`` — any
        destructible MATERIAL tile converts, solid walls (the shipped set,
        unchanged behaviour) AND non-solid destructibles like furniture:
        bullet chew must be able to break a crate so it stops *being* cover.
        Furniture ships ``permeability = 0.5`` -> NOT solid -> it already holds
        bulk N, so writing the constant into it is ``seed - prior``, which is
        NEGATIVE whenever the room is above ambient. That is why the booking
        channel is measured and signed rather than a formula.
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            return
        was_hull = (self.material[fy, fx] == MAT_HULL)
        if self.material[fy, fx] != MAT_AIR:
            # --- mass-books bracket (P-M3 §3.5) ---------------------------
            # The bracket lives INSIDE destroy_wall, not around a "destruction
            # block", because there is no such block: the six callers are
            # scattered across THREE regions straddling physics_runner.step()
            # — grenade/explosion (slot 2), bullet chew (2) and beam chew (4)
            # run BEFORE it; fire burn-through (9), the burst valve (9b) and
            # door-assembly death (9e) after. A bracket around slots 9/9b would
            # miss 2 and 4; one around the whole tick would enclose the solver,
            # whose own Sum N legitimately moves via the ambient rail and the
            # vacuum sink. Bracketing here is the only form that covers all six
            # call sites without enclosing solver behaviour, and it makes the
            # channel correct by construction wherever a caller fires.
            _n_before = self._bulk_n_sum()
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
            # open-air with the constant ambient seed (anti-vacuum-pulse).
            on_edge_hull = was_hull and (
                fy < 1 or fy >= h - 1 or fx < 1 or fx >= w - 1)
            # BC (boundary_conditions_spec_2026-07-19 §1, joins-AMBIENT twin): on
            # a planetside map a breached edge hull opens to SKY, not to space —
            # the exposes rule reads and joins `is_ambient`, not `is_vacuum` (the
            # two masks are mutually exclusive; a space map's is_ambient is empty
            # and this is the exact prior behaviour). `breach_mask` is the map's
            # single boundary mask.
            breach_mask = (self.is_ambient if self._boundary == "ambient"
                           else self.is_vacuum)
            # "Exposing the boundary" == a 4-neighbour that is the EXPOSED
            # boundary (vacuum/ambient AND not solid — an intact hull tile is
            # vacuum AND solid and does NOT count; see the `breach` predicate).
            exposes = any(
                0 <= fy + dy < h and 0 <= fx + dx < w
                and breach_mask[fy + dy, fx + dx]
                and not self.solid[fy + dy, fx + dx]
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            if on_edge_hull or exposes:
                # True breach — the tile joins the boundary reservoir; the
                # solver's Dirichlet pin (P=0 vacuum / P=P_amb ambient) + donor-
                # cell venting drain/fill it natively (no hard zero).
                breach_mask[fy, fx] = True

            # --- the seed (P-M3 §3.1-§3.3) --------------------------------
            n_total_q, o2_amb_q, _n2_amb_q, pin_q = self.ambient_seed()
            if breach_mask[fy, fx]:
                # §3.2 — a tile that joined the boundary is Dirichlet-pinned;
                # seeding it would be pointless on a space map (measured: +10
                # this tick, -10 the next) and wrong on an ambient map (the
                # rail FILLS it to N_amb each substep and books the difference
                # to boundary_flux). Predicate is the LIVE mask, mirroring
                # `unseal_tiles`' `joins_boundary`, so the two cannot drift.
                #
                # But a SKIP alone is not enough. destroy_wall's gate is
                # `material != MAT_AIR`, and furniture is a NON-solid
                # destructible that already holds N. Chew a crate next to
                # vacuum and, with only a skip, its existing gas would never be
                # booked here — the next transport pass zeroes it via the
                # `solid || is_vacuum -> N = 0` clamp, which carries no
                # boundary_flux credit. Mass would vanish with a channel on
                # NEITHER side of the seam. So evacuate explicitly, at destroy
                # time, where the measured bracket above can see it.
                self.gas[O2][fy, fx] = 0
                self.gas[INERT_N2][fy, fx] = 0
            else:
                # §3.1.3 — CONSTANT total (`n_total_q`), composition INHERITED
                # from the open donors. A fixed 21/79 seed would inject fresh
                # oxidizer inside combustion's 2-hop draw radius on every tile
                # of a burning wall run; inheriting keeps the local mole
                # fraction. Donor set is PINNED to the same predicate the old
                # `_neighbor_mean` used — 4-neighbours that are `not solid and
                # not is_vacuum` (is_ambient cells DO count: on a planetside
                # map the reservoir composition is the right thing to inherit
                # toward).
                #
                # ONE exact int64 form, ONE rounding, no intermediate ratio.
                # Written naively as `sum_o2 / sum_n` this would be a float in
                # the sim path (iron rule); written via the house
                # mul_q16(reciprocal_q16(...)) idiom the reciprocal's rounding
                # can push the fraction to FP_ONE and make n2 NEGATIVE, which
                # bulk_transport then silently clamps to 0 — an unbooked mint
                # one substep after the books recorded a smaller delta.
                sum_o2 = 0
                sum_n = 0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = fy + dy, fx + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and not self.solid[ny, nx]
                            and not self.is_vacuum[ny, nx]):
                        d_o2 = int(self.gas[O2][ny, nx])
                        sum_o2 += d_o2
                        sum_n += d_o2 + int(self.gas[INERT_N2][ny, nx])
                if sum_n > 0:
                    # round-half-up; 0 <= o2_q <= n_total_q is GUARANTEED
                    # because sum_o2 <= sum_n (per-plane N >= 0 is held by
                    # bulk_transport's clamp), so n2 can never go negative.
                    o2_q = (n_total_q * sum_o2 + sum_n // 2) // sum_n
                else:
                    # §3.1.4 — NOT an edge case: this fires whenever no donor
                    # holds gas, most importantly when all four neighbours are
                    # solid, i.e. the interior tile of a >=2-thick slab, which
                    # is ordinary blast geometry. Erik's ruling: fall back to
                    # the map's AMBIENT composition, not pure N2 — the cave
                    # case in the docstring is the whole motivation for the
                    # ambient seed, and digging out a cave must not fill it
                    # with nitrogen and suffocate the player. ACCEPTED GAP:
                    # blasting a burning slab therefore briefly feeds the fire
                    # (bounded at ambient_N per tile, does NOT scale with
                    # pressure, and arguably the behaviour wanted anyway —
                    # breach a burning wall, air rushes in, the fire flares).
                    o2_q = o2_amb_q
                self.gas[O2][fy, fx] = o2_q
                self.gas[INERT_N2][fy, fx] = n_total_q - o2_q

            # §3.1.2 — `atmosphere` MUST be written, on every path including a
            # breach. A solid tile's atmosphere is a hard 0 (the MG solve zeroes
            # it), and the two callers that run AFTER the physics step — fire
            # burn-through and the burst valve — are refilled by nothing in that
            # tick. find_burst_walls then reads `atm[ny, nx]` for every
            # non-solid, non-vacuum, non-ambient neighbour: leave the hole at 0
            # and a wall between a 2.1 atm room and a 1.0 atm corridor sees
            # sides {2.1, 1.0, 0.0} -> spread 2.1 > burst_threshold 2.0 -> its
            # along-wall neighbours pop, up to burst_max_per_tick per event.
            # Dropping this write would remove the amplifier at the mass end and
            # install one at the burst end, same code path, same tick. Three
            # further same-tick readers of a stale 0 confirm it is not
            # display-only: the MG warm start (p_prev := atmosphere),
            # apply_wave_push (a ~1 atm phantom grad-P that can fire
            # KNOCKED_DOWN beside a burnt-through wall), and the PRESSURE sensor
            # channel (an RL observation).
            #
            # The value is the map's effective PIN, not n_total_q: p*(N_amb, 0)
            # is 65540 raw (1.000061 atm) at Earth defaults, not 65536. Do not
            # assert p* == N — the seed is stated purely in N.
            self.atmosphere[fy, fx] = pin_q
            # §3.1 — T := 0 explicitly, on EVERY destroyed tile including a
            # breach. The energy books sum `n_bulk * T_game` over a set that
            # skips solid/ts/vacuum/ambient with NO offset term, so a cell
            # joining at T = 0 contributes exactly 0 and the books need no
            # energy channel for the destruction (destroy_wall writes no
            # temperature today, so a burning wall currently joins the books
            # HOT — this closes a pre-existing energy-seam hole). On a breach
            # tile it also stops a burning hull tile joining is_ambient still
            # carrying the wall's heat: the c_local scan skips only
            # `solid || is_vacuum`, so that would inflate map-wide sound speed
            # and substep count for a tick.
            self.temperature[fy, fx] = 0
            # §3.3 — clear `fire`. The C++ burn-through path already zeroes it
            # on every tile it emits, but the OTHER callers (burst, explosion,
            # bullet chew, door) do not: `on_tile_changed` patches ten caches
            # and never `fire`. This is a stale display/sensor cleanup, and it
            # stops fire_simulation decrementing a now-air tile's wall_hp
            # forever. It is NOT a change to fire spread — cellular spread no
            # longer exists, apply_temperature_ignition is `flammable`-gated so
            # it cannot relight an air tile, and a destroyed burning tile is
            # already dropped as a radiation emitter. It DOES change the FIRE
            # sensor channel, which is an RL observation change.
            self.fire[fy, fx] = 0

            # §3.4 — book it. MEASURED (post - pre), not derived from a
            # formula, and SIGNED: on the solid path the prior N is 0 so the
            # delta is +n_total_q per tile, but chewing furniture in a 5 atm
            # room DELETES ~4 cell-equivalents, and evacuating a breached
            # furniture tile is negative too.
            self.n_destruction_seed_sum += self._bulk_n_sum() - _n_before

    # ------------------------------------------------------------------
    # EOS evacuation rule — seal / unseal (A5)
    #
    # The door-close half of the eos_refactor_design.md §2.2 occupancy-
    # transition rule (only the destroy direction existed before): a tile
    # leaving the open-air mask has its gas EVACUATED conservatively into
    # adjacent open cells before any solver pass sees the new mask — the
    # bulk-flux solver defensively zeroes N on solid every pass, so a seal
    # without evacuation silently deletes mass. The symmetric open half
    # (`unseal_tiles`) withdraws its seed from the donors, so opening a door
    # is exactly conservative. `destroy_wall` deliberately does NOT: it seeds a
    # CONSTANT cell of the map's ambient air and BOOKS it to
    # `n_destruction_seed_sum` (P-M3 §2 — an explosion redistributes matter
    # rather than eliminating it, and the sim carries no rubble, so charging
    # the player for the opened volume would make a blasted-open cave
    # suffocate him). The asymmetry survives, but it is now a NAMED bounded
    # source rather than the pressure-proportional mint it used to be.
    # Both primitives are pure-integer, order-pinned, and
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
        mean of the tile's PRE-call THERMAL-solid 4-neighbors' temperatures
        (``thermal_solid``, i.e. thermal_mass > 0 — the thermal-medium axis,
        addendum D5) — the door panel belongs to the wall assembly it slides
        from, so no instant "hot door" from post-grenade air — falling back
        to keeping the local air T only when the tile has no pre-existing
        thermal-solid neighbor (Erik's ruling 4, 2026-07-19; design §4a). ``is_vacuum``
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
        #
        # THERMAL-MASS AXIS (addendum 2026-07-30 D5): the donor test is
        # `thermal_solid`, not `solid` — the question this seed asks is "which
        # of my neighbours hold an OBJECT temperature I should inherit", which
        # is the thermal-medium axis, not the flow axis. A burning crate can now
        # seed the door panel that closes beside it. A no-op wherever
        # `thermal_solid == solid` (every furniture-free map), so it cannot move
        # gate (a).
        h, w = self._h, self._w
        close_t = {}
        for fy, fx in span:
            wall_ts = []
            for dy, dx in self._FACE_DIRS:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < h and 0 <= nx < w and self.thermal_solid[ny, nx]:
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
        ``destroy_wall``, which seeds one constant cell of the map's ambient
        air and books it to ``n_destruction_seed_sum``, opening never creates
        gas at all — a door is not a demolition. Atomic like ``seal_tiles``
        (``ValueError`` on caller bugs, no partial mutation).
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

            # Boundary join (destroy_wall's exposes predicate, minus its
            # unconditional mint): LIVE solid mask — chains down-span. BC
            # (joins-AMBIENT twin): on a planetside map the tile joins the
            # ambient SKY reservoir (is_ambient), not space (is_vacuum). A space
            # map's is_ambient is empty, so this is the exact prior behaviour.
            breach_mask = (self.is_ambient if self._boundary == "ambient"
                           else self.is_vacuum)
            joins_boundary = bool(breach_mask[fy, fx])
            if not joins_boundary:
                for dy, dx in self._FACE_DIRS:
                    ny, nx = fy + dy, fx + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and breach_mask[ny, nx]
                            and not self.solid[ny, nx]):
                        joins_boundary = True
                        break
            if joins_boundary:
                breach_mask[fy, fx] = True
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
            # displayed atmosphere (divisor k, deliberately NOT k+1). The
            # ORIGINAL justification — "the minted display value destroy_wall
            # also provides" — no longer holds as written: since P-M3
            # destroy_wall writes the map's effective ambient PIN there, not a
            # neighbour mean. The write survives on its own merits, which are
            # the load-bearing half anyway: `atmosphere` is not display-only.
            # It is read the SAME tick by the MG warm start (p_prev :=
            # atmosphere), by find_burst_walls' spread scan — where a stale 0
            # in a freshly-opened tile manufactures a phantom differential that
            # pops its neighbours — by apply_wave_push, and by the PRESSURE
            # sensor channel. Leaving it unwritten trades a mass artefact for a
            # burst artefact. It remains OUTSIDE the conservation ledger (the
            # solver rematerializes P from N next tick); the divisor stays k
            # because the seed here is withdrawn, so the opened tile should
            # display what its donors display, not their post-withdrawal value.
            # wave_p matches so the |P - P_prev| ripple splash sees no phantom
            # spike in the window before step 0.
            self.atmosphere[fy, fx] = (
                sum(int(self.atmosphere[d]) for d in donors) // k)
            self.wave_p[fy, fx] = int(self.atmosphere[fy, fx])

    # ------------------------------------------------------------------
    # Integer gas-N source/sink primitive — the pump feed (Arc B B4, §6/D10)
    # ------------------------------------------------------------------
    # A per-slice, zero-clamped, RNG-free, FLOAT-free add/remove of gas mass N
    # at ONE tile — the deliberate SOURCE/SINK the `pump` actuator drives (§6b).
    #
    # UNLIKE seal/unseal above (which REDISTRIBUTE — grid N is conserved to the
    # LSB), this primitive MINTS (inject) or DESTROYS (extract) mass ON PURPOSE:
    # an airlock pump is a gas source/sink, not a conservative transfer. UNLIKE
    # FieldEdit's gas path (a float bridge: dequantize -> float -> [0,1] clamp ->
    # requantize, plus an optional ``sim.rng`` noise draw — the 3-lens critique's
    # D10 rejection), EVERY step here is pure integer Q16.16 with a PINNED
    # remainder rule and a LOUD int32 overflow guard, so it is bit-identical
    # cross-machine and safe for synced sim state (determinism iron rules, §9).
    #
    # FORWARD-NOTE (§5b Rung-2 rider, NOT Arc B): this is a structural-ish field
    # write to ``gmap.gas`` (the CPU MIRROR). When GPU residency lands, the edited
    # port tile joins the §5b dirty-set H2D upload — pushing this edit to the
    # device is that LATER rider's job; Arc B edits only the host mirror. Do NOT
    # add a ``from_host()`` / device push here.

    @staticmethod
    def gas_proportional_split(holdings, want):
        """THE pinned proportional-floor-split-with-cascade (§6/D10, extended
        by the vent design's §3 plenum withdrawal — docs/vent_system_design_
        2026-08-23.md). Splits ``want`` (an int, ``0 <= want <= sum(holdings)``)
        across ``holdings`` (a list of non-negative ints, any length)
        proportionally: ``take_g = holdings[g]*want // sum(holdings)`` (a
        floor, no float divide), with the fractional shortfall (< len(holdings))
        cascaded in PINNED index order to slots with remaining headroom, so
        ``sum(take) == want`` EXACTLY. Pure — no bounds/overflow/zero-clamp of
        its own; callers own the array they're splitting (a tile's ``gas``
        column via :meth:`extract_gas_n_vec`, or a duct plenum's holdings list
        via the vent runtime) and apply the zero-clamp store themselves.

        Extracted from the original ``extract_gas_n`` body (D10) so the ONE
        pinned split algorithm serves both the tile primitive and the plenum
        ledger — never a second copy (CLAUDE.md canonical-systems rule).
        """
        n_total = sum(holdings)
        if want <= 0 or n_total <= 0:
            return [0] * len(holdings)
        take = [(holdings[g] * want) // n_total for g in range(len(holdings))]
        short = want - sum(take)
        for g in range(len(holdings)):
            if short <= 0:
                break
            room = holdings[g] - take[g]
            if room <= 0:
                continue
            grab = room if room < short else short
            take[g] += grab
            short -= grab
        return take

    def _inject_gas_n_core(self, fy, fx, add_vec):
        """The shared mutation core (§6/D10, extended §3): add ``add_vec``
        (a length-``N_GASES`` list of non-negative Q16.16 ints, ANY subset of
        slices non-zero) to tile ``(fy, fx)``, loud ``OverflowError``
        pre-mutation exactly like the original single-pair ``inject_gas_n``.
        No bounds check (callers check), no T-mix (the vent-facing
        :meth:`inject_gas_n_vec` wrapper owns that — §4).
        """
        limit = 1 << 31
        cur = [int(self.gas[g][fy, fx]) for g in range(N_GASES)]
        new = [cur[g] + int(add_vec[g]) for g in range(N_GASES)]
        for g in range(N_GASES):
            if new[g] >= limit:
                raise OverflowError(
                    f"inject_gas_n: tile ({fy}, {fx}) would overflow int32 "
                    f"on slice {g} ({cur[g]}+{add_vec[g]})")
        for g in range(N_GASES):
            if add_vec[g]:
                self.gas[g][fy, fx] = new[g]

    def inject_gas_n(self, fy, fx, delta_n):
        """Add ``delta_n`` (Q16.16 gas mass) at tile ``(fy, fx)``, split at the
        FIXED standard O2/N2 mix across the two bulk slices (§6, D10 inject half).

        Integer split (round-half-up O2, exact-remainder N2 — the ``__init__``
        air-seed idiom), so the injected mass is EXACTLY ``delta_n`` and the mix
        ratio is preserved within the one-LSB remainder rule. ``delta_n <= 0`` is
        a no-op. Bounds are STRICT (a primitive caller passing OOB is a bug —
        the seal_tiles precedent). Loud ``OverflowError`` pre-mutation: N is
        int32 on the mirror and a saturating store would SILENTLY inject less
        than ``delta_n`` (a source that quietly under-delivers).
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            raise ValueError(f"inject_gas_n: tile ({fy}, {fx}) out of bounds")
        dn = int(delta_n)
        if dn <= 0:
            return
        from simulation import gas_fixed as _gas_fx
        # 0.21 O2 by mole fraction (Earth-normal) — quantize_scalar(0.21) == 13763,
        # the P1 calibration constant the ambient air seed uses (13763 + 51773 ==
        # FP_ONE), so an inject reproduces calibrated air composition to the LSB.
        o2_frac_q = _gas_fx.quantize_scalar(0.21)
        o2_add = (dn * o2_frac_q + (1 << 15)) >> 16      # round-half-up (air seed)
        n2_add = dn - o2_add                              # exact remainder
        add_vec = [0] * N_GASES
        add_vec[O2] = o2_add
        add_vec[INERT_N2] = n2_add
        self._inject_gas_n_core(fy, fx, add_vec)

    def extract_gas_n(self, fy, fx, delta_n):
        """Remove up to ``delta_n`` (Q16.16 gas mass) at tile ``(fy, fx)``,
        PROPORTIONAL to the tile's current composition across ALL gas slices,
        zero-clamped so no slice can go negative (§6, D10 extract half). Returns
        the total mass actually withdrawn.

        The withdrawn total is ``min(delta_n, N_total)`` — the aggregate
        zero-clamp (a near-empty tile clamps at 0 and NEVER over-withdraws). The
        per-slice split is :meth:`gas_proportional_split` (a floor, no float
        divide, no dequantize) with the shortfall cascaded in PINNED slice-id
        order to slices with remaining holdings, so ``sum(remove_g) == want``
        EXACTLY (mirrors the seal-tiles remainder rule and unseal's balanced-
        then-greedy withdrawal). The explicit ``max(0, ...)`` per-slice
        zero-clamp before the store guarantees N can NEVER go negative into
        the Helmholtz solver. ``delta_n <= 0`` / an empty tile is a no-op.

        Thin wrapper over :meth:`extract_gas_n_vec` (§3 extension, vent design)
        — the SAME withdrawal, this call just collapses it to the scalar total
        pump_system.py has always consumed.
        """
        return int(sum(self.extract_gas_n_vec(fy, fx, delta_n)))

    def extract_gas_n_vec(self, fy, fx, delta_n):
        """§3 extension (docs/vent_system_design_2026-08-23.md): the SAME
        withdrawal :meth:`extract_gas_n` performs, returning the per-slice
        withdrawal as a length-``N_GASES`` list instead of collapsing it to a
        scalar — so a caller (the vent circulation sweep) can credit each bulk
        plane separately and apply the per-species filter, rather than
        re-deriving proportions the primitive already computed. ``delta_n <= 0``
        or an empty tile returns an all-zero vector (never negative slices).
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            raise ValueError(f"extract_gas_n_vec: tile ({fy}, {fx}) out of bounds")
        dn = int(delta_n)
        if dn <= 0:
            return [0] * N_GASES
        holdings = [int(self.gas[g][fy, fx]) for g in range(N_GASES)]
        n_total = sum(holdings)
        if n_total <= 0:
            return [0] * N_GASES            # empty tile — zero-clamp, nothing to take
        want = dn if dn < n_total else n_total          # aggregate zero-clamp
        remove = self.gas_proportional_split(holdings, want)
        for g in range(N_GASES):
            new = holdings[g] - remove[g]
            self.gas[g][fy, fx] = new if new > 0 else 0   # explicit zero-clamp
        return remove

    def inject_gas_n_vec(self, fy, fx, add_vec, t_dep, t_min_q, t_max_phys_q):
        """§3/§4 extension (docs/vent_system_design_2026-08-23.md): inject an
        ARBITRARY per-slice composition ``add_vec`` (length ``N_GASES``,
        non-negative Q16.16 ints — bulk AND trace slices alike, unlike
        :meth:`inject_gas_n`'s fixed 21/79 split), THEN mass-weighted-mix the
        tile's ``temperature`` toward ``t_dep`` (the deposit's characteristic
        temperature, Q16.16 ΔT-from-ambient — the vent plenum's own
        ``E_plenum/N_plenum``).

        T-mix (§4): weighted by BULK N only — ``N_old = gas[O2]+gas[INERT_N2]``
        BEFORE this call, ``ΔN = add_vec[O2]+add_vec[INERT_N2]`` — matching the
        EOS's own N_total convention (P-T0: n_total == n_bulk, trace planes
        carry no independent T). ``T_new = floor((N_old*T_old + ΔN*t_dep) /
        (N_old+ΔN))`` toward -inf (Python ``//`` on ints IS floor-toward-inf
        for a positive divisor — the same contract as C++ ``floordiv_q``,
        cpp/src/fixed_point.h:501, energy-books arc §2.1.5/§2.7: truncation-
        toward-zero would mint energy on a sub-ambient mix). Skipped entirely
        (T untouched) when ``N_old+ΔN <= 0`` (a bulk-free deposit — e.g. into a
        vacuum tile with only trace moving — has nothing to weight the mix by).
        Railed to ``[t_min_q, t_max_phys_q]`` like every other T writer;
        returns ``(t_new_raw, rail_hit)`` with ``rail_hit`` in
        ``{-1, 0, +1}`` (low / none / high) so the caller can count the hit
        (the plenum's runtime-row diagnostics) — the caller derives its OWN
        energy debit from ``(N_old, T_old)`` it already read and ``(N_old+ΔN,
        t_new_raw)``, so this primitive need not return N/T twice.

        The caller (the vent 9e(d) sweep) is responsible for the RUNTIME
        aperture guards (docs §4: never call this on a solid, thermal_solid,
        vacuum, ambient, or flooded tile) — this primitive trusts its caller,
        exactly like :meth:`inject_gas_n` trusts the pump port resolve.
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            raise ValueError(f"inject_gas_n_vec: tile ({fy}, {fx}) out of bounds")
        dn_total = sum(int(v) for v in add_vec)
        if dn_total <= 0:
            return int(self.temperature[fy, fx]), 0
        n_old = int(self.gas[O2][fy, fx]) + int(self.gas[INERT_N2][fy, fx])
        delta_n_bulk = int(add_vec[O2]) + int(add_vec[INERT_N2])
        t_old = int(self.temperature[fy, fx])
        self._inject_gas_n_core(fy, fx, add_vec)
        n_new = n_old + delta_n_bulk
        if n_new <= 0:
            return t_old, 0                  # nothing bulk to weight the mix by
        t_new = (n_old * t_old + delta_n_bulk * int(t_dep)) // n_new   # floor(-inf)
        rail_hit = 0
        if t_new < t_min_q:
            t_new = int(t_min_q)
            rail_hit = -1
        elif t_new > t_max_phys_q:
            t_new = int(t_max_phys_q)
            rail_hit = 1
        self.temperature[fy, fx] = t_new
        return t_new, rail_hit
