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
    "ignition_temp": np.float32,
    "heat_atten": np.float32,
    "wave_reflect": np.float32,
    "wave_absorb": np.float32,
    "blast_resist": np.float32,
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

    def __init__(self, materials_cfg):
        """Build from the ``CFG.materials`` namespace (or any equivalent).

        ``materials_cfg`` is the :class:`config.Namespace` for ``[materials]``;
        each attribute (``air``, ``hull``, ...) is itself a namespace of the
        named columns. A plain dict-of-dicts is also accepted (for tests).
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

    # -- accessors -------------------------------------------------------
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
        """Build from the global :data:`config.CFG` (or a provided config)."""
        if cfg is None:
            from config import CFG
            cfg = CFG
        return cls(cfg.materials)

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
