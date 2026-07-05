"""Gas ids + the gas-property table (engine/05 §6.2 — Multi-gas system, M1).

Single source of truth for the ``GAS_*`` ids and the per-gas property table that
generalises the single ``smoke`` scalar field into N gas density fields. Mirrors
:mod:`simulation.materials` (the material table): a gas type is **data-driven**,
exactly like a material — one ``[gases.<name>]`` row in ``config.toml`` + one id
here. Properties are stored as per-id numpy arrays so a future per-tile lookup
(``table.absorption[gas_id]``) is a single fancy-index.

The five gases (engine/05 §6.2):

    white_smoke, black_smoke, poison, teargas, fuel_gas

``black_smoke`` is combustion soot — what fire and explosions emit. Its diffusion
(0.10) matches today's ``physics.d_smoke`` (0.1), so the existing generic smoke
field maps onto the ``black_smoke`` slice with **no behaviour change** (M1).

M1 scope: this table is **loaded** (data only). The per-channel ``absorption`` /
``scatter_albedo`` are summed density-weighted across coexisting gases in the
raycaster at M2; ``flammable`` / ``emits_when_hot`` / ``effect`` are read by fire
and mechanics at M2/M3. ``decay`` is loaded but **not yet applied** in transport
(the M1 C++ smoke solver has no decay term; applying it would break behaviour
preservation — see :mod:`simulation.physics_runner`).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gas IDs — the single source of truth (engine/05 §6.2). Order is the slice
# order of the dense ``gmap.gas`` (N, h, w) array; ids must be contiguous
# 0..N-1 so an array indexed by id has no gaps (validated in GasTable).
# ---------------------------------------------------------------------------
WHITE_SMOKE = 0
BLACK_SMOKE = 1
POISON = 2
TEARGAS = 3
FUEL_GAS = 4

# Config-key <-> id mapping. The key is the ``[gases.<name>]`` table name.
# Listed in id order; ``GasTable`` validates contiguity.
GAS_NAMES = {
    WHITE_SMOKE: "white_smoke",
    BLACK_SMOKE: "black_smoke",
    POISON: "poison",
    TEARGAS: "teargas",
    FUEL_GAS: "fuel_gas",
}

# Number of gas types (the N of the (N, h, w) gas array).
N_GASES = len(GAS_NAMES)

# Scalar columns: name -> numpy dtype. ``absorption`` / ``scatter_albedo`` are
# handled separately because they are per-channel RGB triples, not scalars.
_SCALAR_COLUMNS = {
    "diffusion": np.float32,
    "decay": np.float32,
    "glow": np.float32,
    "flammable": bool,
    "emits_when_hot": bool,
}


class GasTable:
    """Per-gas property table, indexed by gas id (engine/05 §6.2).

    Built from the ``[gases]`` section of ``config.toml`` (the named-key dict
    format, same as :class:`simulation.materials.MaterialTable`). Each scalar
    column is a 1-D numpy array indexed by gas id (``table.diffusion[gas_id]``);
    ``absorption`` and ``scatter_albedo`` are ``(N, 3)`` RGB arrays. ``effect``
    is a list of per-gas strings (a gameplay tag read unit-side in mechanics, not
    a numeric column).

    The id constants (:data:`WHITE_SMOKE`, :data:`BLACK_SMOKE`, ...) are the
    canonical slice indices into ``gmap.gas``; ``name_to_id`` gives the same map
    keyed by name.

    Rebuild via :meth:`from_config` after a config hot-reload.
    """

    def __init__(self, gases_cfg):
        """Build from the ``CFG.gases`` namespace (or any equivalent).

        ``gases_cfg`` is the :class:`config.Namespace` for ``[gases]``; each
        attribute (``white_smoke``, ``black_smoke``, ...) is itself a namespace
        of the named columns. A plain dict-of-dicts is also accepted (tests).
        """
        ids = sorted(GAS_NAMES)
        # Contiguity: ids must be 0..N-1 so an array indexed by id has no gaps.
        if ids != list(range(len(ids))):
            raise ValueError(f"GAS_NAMES ids must be contiguous 0..N-1, got {ids}")
        self.n = len(ids)
        self.names = [GAS_NAMES[i] for i in ids]
        self.name_to_id = {GAS_NAMES[i]: i for i in ids}

        rows = [self._get_row(gases_cfg, GAS_NAMES[i]) for i in ids]

        for col, dtype in _SCALAR_COLUMNS.items():
            values = [self._get_field(row, name, col)
                      for row, name in zip(rows, self.names)]
            setattr(self, col, np.array(values, dtype=dtype))

        # absorption: per-channel RGB, (N, 3) float32 (Beer-Lambert per-unit-
        # density absorption — summed density-weighted across gases at M2).
        self.absorption = self._load_rgb(rows, "absorption")
        # scatter_albedo: per-channel RGB, (N, 3) float32 (additive god-ray glow
        # gain, decoupled from absorption).
        self.scatter_albedo = self._load_rgb(rows, "scatter_albedo")

        # beam_absorb_q16: per-gas Q16.16 BEAM-absorption coefficient for the
        # HITSCAN laser (mechanics/03 §5, W2). DERIVATION OF RECORD, computed
        # ONCE at table build: the arithmetic MEAN of the gas's RGB absorption
        # triple — (r + g + b) / 3 in float64 (pure + and one correctly-rounded
        # divide on load-time constants: ingress door 3) — then quantized onto
        # the Q16.16 grid with the standard round-half-away-from-zero twin
        # (door 2). A laser has one energy channel, not three; the mean is the
        # panchromatic collapse of the same per-channel Beer-Lambert data the
        # renderer uses, so a gas that blocks light blocks beams to the same
        # degree. The beam consumes these in PURE INTEGER arithmetic (door 1):
        # per tile crossed, energy *= max(0, ONE - sum_g(absorb_q * density_q
        # >> 16)) >> 16 — no exp, no transcendentals (combat.fire_beam).
        # Plain Python ints (a tuple) so the march never touches numpy scalars.
        from simulation import unit_fixed as _ufx
        self.beam_absorb_q16 = tuple(
            _ufx.quantize_scalar(
                (float(self.absorption[i, 0]) + float(self.absorption[i, 1])
                 + float(self.absorption[i, 2])) / 3.0)
            for i in range(self.n)
        )

        # effect: per-gas gameplay tag string (read unit-side in mechanics; the
        # solver only transports the field). Stored as a plain list by id.
        self.effect = [
            str(self._get_field(row, name, "effect"))
            for row, name in zip(rows, self.names)
        ]

    # -- RGB column loader -----------------------------------------------
    def _load_rgb(self, rows, col):
        """Load an ``(N, 3)`` RGB column, validating each row is a [R,G,B] triple."""
        arr = np.zeros((self.n, 3), dtype=np.float32)
        for idx, (row, name) in enumerate(zip(rows, self.names)):
            triple = self._get_field(row, name, col)
            vec = np.asarray(triple, dtype=np.float32)
            if vec.shape != (3,):
                raise ValueError(
                    f"gases.{name}.{col} must be a [R,G,B] triple, got {triple!r}"
                )
            arr[idx] = vec
        return arr

    # -- accessors (mirror MaterialTable) --------------------------------
    @staticmethod
    def _get_row(cfg, name):
        if isinstance(cfg, dict):
            if name not in cfg:
                raise KeyError(f"config [gases] missing required gas '{name}'")
            return cfg[name]
        if not hasattr(cfg, name):
            raise KeyError(f"config [gases] missing required gas '{name}'")
        return getattr(cfg, name)

    @staticmethod
    def _get_field(row, gas_name, col):
        if isinstance(row, dict):
            if col not in row:
                raise KeyError(f"gases.{gas_name} missing column '{col}'")
            return row[col]
        if not hasattr(row, col):
            raise KeyError(f"gases.{gas_name} missing column '{col}'")
        return getattr(row, col)

    @classmethod
    def from_config(cls, cfg=None):
        """Build from the global :data:`config.CFG` (or a provided config)."""
        if cfg is None:
            from config import CFG
            cfg = CFG
        return cls(cfg.gases)


__all__ = [
    "WHITE_SMOKE", "BLACK_SMOKE", "POISON", "TEARGAS", "FUEL_GAS",
    "GAS_NAMES", "N_GASES", "GasTable",
]
