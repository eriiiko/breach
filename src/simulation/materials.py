"""Material ids + the material-property table (ch.02 — Material System).

Single source of truth for the ``MAT_*`` ids (previously duplicated in
``gamemap.py`` and ``level_loader.py``) and for the per-material property
table that derives every per-tile constant from the material id.

The table is the **data-driven** foundation described in
``docs/architecture/02_material_system.md``: adding a material is one
``config.toml`` row + one CSV mapping. Properties are stored as per-id numpy
arrays so the derived caches (see :mod:`simulation.gamemap`) can be built with
a single fancy-index lookup ``column[material]``.

CPU-only foundation: the optics (``light_atten``/``heat_atten``) and acoustics
(``wave_*``) columns are *stored* here but consumed by nobody yet — they wire
into the ray/wave passes in later chapters.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Material IDs — the single source of truth.
# Renderer keeps its own color table; these are the gameplay/physics ids.
# Order must match the CSV mapping in :func:`level_loader.materials_from_tilemap`.
# ---------------------------------------------------------------------------
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3
MAT_STEEL = 4
MAT_GLASS = 5

# Config-key <-> id mapping. The key is the ``[materials.<name>]`` table name.
# Listed in id order; ``MaterialTable`` validates contiguity.
MATERIAL_NAMES = {
    MAT_AIR: "air",
    MAT_HULL: "hull",
    MAT_WOOD: "wood",
    MAT_DOOR: "door",
    MAT_STEEL: "steel",
    MAT_GLASS: "glass",
}

# Scalar columns: name -> numpy dtype. ``light_atten`` is handled separately
# because it is a per-channel RGB triple, not a scalar.
_SCALAR_COLUMNS = {
    "hp": np.float32,
    "flammable": bool,
    "passable": bool,
    "conductivity": np.float32,
    "thermal_mass": np.float32,
    "ignition_temp": np.float32,
    "heat_atten": np.float32,
    "wave_reflect": np.float32,
    "wave_absorb": np.float32,
    "blast_resist": np.float32,
}


# Conduction log-bucket constants (engine/06 §2.4–§2.5, proposal §7.2
# [physics.thermal]). Defaults mirror config.toml; the live values are read from
# CFG and threaded in via :meth:`from_config` so the table tracks config edits.
# Kept here so a dict-built table (tests) and any config-less build still produce
# a valid face table.
_THERMAL_DEFAULTS = {
    "TEMP_SCALE": 65536,  # Q16.16, == HEAT_SCALE (shared temperature/heat domain)
    "SHIFT_AT_REF": 2,    # metal self-rate = 1/4 (fastest stable on 4-nbr)
    "SHIFT_MIN": 2,       # rate floor / stability bound (4 * 1/4 <= 1)
    "KAPPA_REF": 50.0,    # reference conductivity (hull) for the log bucket
    "NO_FACE": 63,        # sentinel: kappa==0 face / grid edge -> zero conduction
}


class MaterialTable:
    """Per-material property table, indexed by material id.

    Built from the ``[materials]`` section of ``config.toml`` (the named-key
    dict format from ch.02). Each scalar column is exposed as a 1-D numpy array
    indexed by material id (``table.hp[material_id]``); ``light_atten`` is an
    ``(N, 3)`` RGB array. Per-tile derived caches index these columns directly
    with the ``material`` grid: e.g. ``table.hp[gmap.material]``.

    Rebuild via :meth:`from_config` after a config hot-reload.
    """

    def __init__(self, materials_cfg, thermal_cfg=None):
        """Build from the ``CFG.materials`` namespace (or any equivalent).

        ``materials_cfg`` is the :class:`config.Namespace` for ``[materials]``;
        each attribute (``air``, ``hull``, ...) is itself a namespace of the
        named columns. A plain dict-of-dicts is also accepted (for tests).

        ``thermal_cfg`` is the optional ``[physics.thermal]`` namespace (or dict)
        carrying the conduction log-bucket constants (``SHIFT_AT_REF``,
        ``SHIFT_MIN``, ``KAPPA_REF``, ``NO_FACE``). When omitted the
        :data:`_THERMAL_DEFAULTS` are used so a dict-built table (tests) still
        produces a valid face-shift table.
        """
        ids = sorted(MATERIAL_NAMES)
        # Contiguity: ids must be 0..N-1 so an array indexed by id has no gaps.
        if ids != list(range(len(ids))):
            raise ValueError(f"MATERIAL_NAMES ids must be contiguous 0..N-1, got {ids}")
        self.n = len(ids)
        self.names = [MATERIAL_NAMES[i] for i in ids]

        rows = [self._get_row(materials_cfg, MATERIAL_NAMES[i]) for i in ids]

        for col, dtype in _SCALAR_COLUMNS.items():
            values = [self._get_field(row, name, col)
                      for row, name in zip(rows, self.names)]
            setattr(self, col, np.array(values, dtype=dtype))

        # heat_inv_shift: per-id log2(thermal_mass) (engine/06 §1.2). The
        # heat -> temperature conversion is `temperature += heat >> shift`, a
        # pure arithmetic right shift (no divide, bit-identical cross-machine),
        # so `thermal_mass` MUST be a power of two. Validate here and freeze the
        # integer shift; the per-tile cache (GameMap.heat_inv_shift) is this
        # column indexed by the material grid.
        shifts = []
        for tm, name in zip(self.thermal_mass.tolist(), self.names):
            tm_int = int(round(tm))
            if tm_int < 1 or (tm_int & (tm_int - 1)) != 0:
                raise ValueError(
                    f"materials.{name}.thermal_mass must be a power of two "
                    f">= 1 (it sits on the heat->temperature divide); got {tm!r}"
                )
            shifts.append(tm_int.bit_length() - 1)   # log2 of a power of two
        self.heat_inv_shift = np.array(shifts, dtype=np.int32)

        # --- Conduction face-shift tables (engine/06 §2.4–§2.5) ---------------
        # All log2 / harmonic-mean / division happens HERE, at LOAD, in float;
        # the runtime conduction pass is a pure signed-add + arithmetic shift.
        self._build_conduction_tables(thermal_cfg)

        # ignition_temp_q16: the per-material ignition threshold QUANTIZED ONCE
        # AT LOAD into the Q16.16 fixed-point domain shared by `heat` /
        # `temperature` (engine/06 §3, proposal §1.2/§7.1). Stored as
        # round(ignition_temp * TEMP_SCALE) with a pinned rounding mode, so the
        # ignition consumer's runtime test is a direct integer compare
        # `temperature[tile] >= ignition_temp_q16[material]` against the Q16.16
        # `temperature` field — no per-tick rescale, no float on the threshold
        # path. This is the single most determinism-critical conversion in the
        # system; it is fixed here, never recomputed per tick. int64 so the
        # multiply can't overflow before it lands (the field itself is int32, but
        # a threshold beyond INT32_MAX simply never fires, which is correct).
        temp_scale = self._thermal_get(thermal_cfg, "TEMP_SCALE")
        self.ignition_temp_q16 = np.array(
            [int(round(float(it) * temp_scale))
             for it in self.ignition_temp.tolist()],
            dtype=np.int64,
        )

        # light_atten: per-channel RGB, (N, 3) float32.
        atten = np.zeros((self.n, 3), dtype=np.float32)
        for idx, (row, name) in enumerate(zip(rows, self.names)):
            triple = self._get_field(row, name, "light_atten")
            arr = np.asarray(triple, dtype=np.float32)
            if arr.shape != (3,):
                raise ValueError(
                    f"materials.{name}.light_atten must be a [R,G,B] triple, "
                    f"got {triple!r}"
                )
            atten[idx] = arr
        self.light_atten = atten

        # permeability: gas + smoke flow coefficient (0 = sealed wall, 1 = open
        # air; partial = a leaky/porous material). OPTIONAL column — if a
        # material omits it, derive a behaviour-preserving default: sealed where
        # the material occludes light, open otherwise (== the legacy is_wall
        # set). Config may set it explicitly to decouple flow from optics — e.g.
        # a grill is opaque-ish to nothing but highly permeable, glass is opaque
        # to flow but clear to light. Consumed by the gas/smoke boundary
        # (ch.02/03/04); see docs/architecture/engine/03_material_system.md.
        occludes_per_id = self.light_atten.max(axis=1) > 0.0
        perm = []
        for idx, (row, name) in enumerate(zip(rows, self.names)):
            val = self._get_field_opt(row, "permeability")
            perm.append(float(val) if val is not None
                        else (0.0 if occludes_per_id[idx] else 1.0))
        self.permeability = np.array(perm, dtype=np.float32)

        # burst_threshold: max pressure DIFFERENTIAL a wall tile can hold across
        # its opposing sides before it fails (over-pressure relief valve, ch.04
        # §5). OPTIONAL column — a material that omits it defaults to 0.0, which
        # the burst scan treats as "n/a" (a 0-threshold material never bursts;
        # see GameMap.find_burst_walls). Interior pressure is ~1.0, so real
        # thresholds sit comfortably above 1 to never pop a normal ship.
        burst = []
        for row, name in zip(rows, self.names):
            val = self._get_field_opt(row, "burst_threshold")
            burst.append(float(val) if val is not None else 0.0)
        self.burst_threshold = np.array(burst, dtype=np.float32)

    # -- conduction face-shift tables (engine/06 §2.4–§2.5) --------------
    def _build_conduction_tables(self, thermal_cfg):
        """Build ``self_shift[N]`` and ``face_shift_table[N][N]`` from the
        per-material ``conductivity`` column (engine/06 §2.4–§2.5, proposal §2).

        These are the LOAD-TIME float computations (base-2 log buckets + the
        harmonic-mean face resolve). The runtime conduction pass only ever
        indexes ``face_shift_table[mat_a][mat_b]`` and shifts — no float, no
        division — so the whole spread is division-free and bit-identical
        cross-machine (proposal §2.7).

        ``self_shift[a]`` — the material's own log-bucket shift (§2.4):

            shift = clamp(SHIFT_MIN,
                          round(SHIFT_AT_REF - log2(kappa / KAPPA_REF)),
                          NO_FACE)            # NO_FACE if kappa == 0

        ``face_shift_table[a][b]`` — the shift for a face BETWEEN materials a, b,
        from the HARMONIC MEAN of their conductivities (§2.5), so two resistances
        in series add (a wood/metal face conducts at ~the wood, slow, rate; an
        arithmetic mean would leak heat into insulators too fast):

            hm = 2*ka*kb / (ka + kb)
            face = clamp(SHIFT_MIN, round(-log2(hm / KAPPA_REF)), NO_FACE)
                   NO_FACE if either kappa == 0

        Symmetric N×N. NO_FACE on every face a kappa==0 material touches makes
        the air no-op STRUCTURAL (not a runtime value-branch) — see §2.6.
        """
        shift_at_ref = float(self._thermal_get(thermal_cfg, "SHIFT_AT_REF"))
        shift_min = int(self._thermal_get(thermal_cfg, "SHIFT_MIN"))
        kappa_ref = float(self._thermal_get(thermal_cfg, "KAPPA_REF"))
        no_face = int(self._thermal_get(thermal_cfg, "NO_FACE"))
        self.no_face = no_face

        kappa = self.conductivity.astype(np.float64)
        n = self.n

        def _clamp_shift(s):
            s = int(round(s))
            if s < shift_min:
                s = shift_min
            if s > no_face:
                s = no_face
            return s

        # self_shift[a] — per-material log-bucket self-rate (§2.4).
        self_shift = np.empty(n, dtype=np.int32)
        for a in range(n):
            ka = kappa[a]
            if ka <= 0.0:
                self_shift[a] = no_face
            else:
                self_shift[a] = _clamp_shift(
                    shift_at_ref - math.log2(ka / kappa_ref)
                )
        self.self_shift = self_shift

        # face_shift_table[a][b] — harmonic-mean face resolve (§2.5), symmetric.
        face = np.full((n, n), no_face, dtype=np.int32)
        for a in range(n):
            ka = kappa[a]
            for b in range(n):
                kb = kappa[b]
                if ka <= 0.0 or kb <= 0.0:
                    face[a, b] = no_face        # kappa==0 either side -> no face
                    continue
                hm = 2.0 * ka * kb / (ka + kb)  # harmonic mean (one float div)
                face[a, b] = _clamp_shift(-math.log2(hm / kappa_ref))
        self.face_shift_table = face

    # -- accessors -------------------------------------------------------
    @staticmethod
    def _thermal_get(thermal_cfg, name):
        """Read one ``[physics.thermal]`` constant, falling back to
        :data:`_THERMAL_DEFAULTS` so a dict-built / config-less table still
        produces valid load-time tables. Accepts a namespace or a plain dict."""
        if thermal_cfg is None:
            return _THERMAL_DEFAULTS[name]
        if isinstance(thermal_cfg, dict):
            return thermal_cfg.get(name, _THERMAL_DEFAULTS[name])
        return getattr(thermal_cfg, name, _THERMAL_DEFAULTS[name])

    @staticmethod
    def _get_row(cfg, name):
        if isinstance(cfg, dict):
            if name not in cfg:
                raise KeyError(f"config [materials] missing required material '{name}'")
            return cfg[name]
        if not hasattr(cfg, name):
            raise KeyError(f"config [materials] missing required material '{name}'")
        return getattr(cfg, name)

    @staticmethod
    def _get_field(row, mat_name, col):
        if isinstance(row, dict):
            if col not in row:
                raise KeyError(f"materials.{mat_name} missing column '{col}'")
            return row[col]
        if not hasattr(row, col):
            raise KeyError(f"materials.{mat_name} missing column '{col}'")
        return getattr(row, col)

    @staticmethod
    def _get_field_opt(row, col):
        """Like ``_get_field`` but returns ``None`` for an absent column.

        Used for *optional* columns (e.g. ``permeability``) that carry a
        derived default when omitted, so existing configs need no edit.
        """
        if isinstance(row, dict):
            return row.get(col)
        return getattr(row, col, None)

    @classmethod
    def from_config(cls, cfg=None):
        """Build from the global :data:`config.CFG` (or a provided config).

        Threads the ``[physics.thermal]`` namespace (conduction log-bucket
        constants) into the table so the face-shift tables track config edits.
        Tolerates a config without that block (falls back to defaults).
        """
        if cfg is None:
            from config import CFG
            cfg = CFG
        thermal_cfg = getattr(getattr(cfg, "physics", None), "thermal", None)
        return cls(cfg.materials, thermal_cfg)

    def occludes(self, material_grid):
        """Static occlusion mask: a tile occludes if it attenuates any channel.

        Replaces the hardcoded ``np.isin(m, [HULL, WOOD, DOOR])`` — derived
        purely from the table so adding an opaque material needs no code edit.
        A door has ``light_atten = [1,1,1]`` so it occludes; air has
        ``[0,0,0]`` so it does not. (Glass attenuates but does not fully
        occlude — see note in ch.02; for the current behaviour-preserving
        set, only air is fully transparent.)
        """
        per_id = self.light_atten.max(axis=1) > 0.0
        return per_id[material_grid]
