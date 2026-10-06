"""Gas ids + the gas-property table (engine/05 §6.2 — Multi-gas system, M1).

Single source of truth for the ``GAS_*`` ids and the per-gas property table that
generalises the single ``smoke`` scalar field into N gas density fields. Mirrors
:mod:`simulation.materials` (the material table): a gas type is **data-driven**,
exactly like a material — one ``[gases.<name>]`` row in ``config.toml`` + one id
here. Properties are stored as per-id numpy arrays so a future per-tile lookup
(``table.absorption[gas_id]``) is a single fancy-index.

The five TRACE gases (engine/05 §6.2):

    steam, smoke, poison, teargas, fuel_gas

``smoke`` is combustion soot — what fire and explosions emit. Its diffusion
(0.10) matched the retired ``physics.d_smoke`` (0.1), so the old generic smoke
field mapped onto the ``smoke`` slice with no behaviour change (M1).

SMOKE TRANSPORT v2 (#12, docs/smoke_transport_design_2026-10-04.md): the trace
planes RIDE THE AIR — each EOS substep moves them on the bulk planes' applied
face flux, priced at the donor's trace-per-air ratio, and once a tick they get
a conservative face diffusion (``diffusion``, stability-checked at this door),
stranded zeroing and ceil-rounded ``decay``. Nothing clamps them to [0, 1]: a
compressed pocket can hold more than 1.

EOS refactor P1 (docs/eos_refactor_design.md §1/§2, decisions log #11) appends
two **bulk** species, ``o2`` and ``inert_n2`` — ALWAYS appended, never
prepended/reordered (existing views like ``gmap.smoke = gas[SMOKE]`` are
index-bound). These are the ``conservative`` pair: they move by donor-cell
CONSERVATIVE flux (the trace planes ride it, see above)
(``cpp/src/bulk_transport.cpp``), and carry NO optics (invisible bulk air —
``absorption``/``scatter_albedo`` all-zero, ``glow`` 0), NO decay, are not
flammable, and set no gameplay ``effect`` tag.

M1 scope: this table is **loaded** (data only). The per-channel ``absorption`` /
``scatter_albedo`` are summed density-weighted across coexisting gases by the
radiation sweep's light channels (the ``light_absorb_q16`` / ``light_glow_q16``
columns below, ray-engine-v2 P6a; the render march that summed them first went
at P6c); ``flammable`` / ``emits_when_hot`` / ``effect`` are read by fire
and mechanics at M2/M3. ``decay`` is applied once a tick by the trace tail
(EOS P4; ceil-rounded and booked since smoke transport v2). ``conservative``
(P1) marks the bulk pair, read by
:class:`simulation.physics_runner.PhysicsRunner`.

Ray-engine-v2 P5a (docs/ray_engine_v2_design_v3_2026-09-15.md §6.3) adds
``heat_absorb``: the gas's HEAT extinction per unit of its density, quantized to
``heat_absorb_q16`` and read by the radiation sweep on every gas cell (the density
law: what thin smoke does not absorb continues down the stream). LIVE since P5c,
which opened the temperature fold's gas branch: smoke carries a DERIVED value
(config.toml [gases.smoke]); every other shipped row is 0.0 with its reason.

``light_absorb`` (optional RGB, #12 smoke x light handle 1, 2026-10-04): the
gas's VISIBLE extinction per unit of its density, per light channel. A row that
carries it is PHYSICS: ``light_absorb_q16`` is that triple as is, and the
``[smoke]`` dials do not touch it. A row without it keeps the older dial
product ``absorption x smoke_absorption x smoke_absorb_scale``. Smoke carries a
derived value (config.toml [gases.smoke]); ``absorption`` itself stays -- the
hitscan beam (``beam_absorb_q16``) and the render medium still read it.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# THE SOOT MASS ONE UNIT OF SMOKE DENSITY HOLDS (#12 smoke x light, handle 3,
# 2026-10-04) -- the "M" of the [gases.smoke] heat_absorb / light_absorb
# derivations (config.toml), now in code so a SOOT SOURCE other than the fire
# (a blast) can deposit a physical mass: combustion mints `soot_yield` units of
# smoke per unit of N_O2 burned, one unit of N_O2 burns KG_FUEL_PER_N_O2 kg of
# wood, and wood sheds SOOT_G_PER_G_WOOD g of soot per g (red oak, well
# ventilated: Tewarson, SFPE Handbook of Fire Protection Engineering,
# "Generation of heat and chemical compounds in fires"). M = 11.15 g at the
# shipped dials, at the reference tile (KG_FUEL_PER_N_O2 is tile-size
# dependent, materials.py).
# ---------------------------------------------------------------------------
SOOT_G_PER_G_WOOD = 0.015


def soot_g_per_smoke_unit(cfg=None) -> float:
    """Grams of soot in ONE unit of smoke density (one tile's worth at ONE)."""
    if cfg is None:
        from config import CFG as cfg
    from simulation.materials import KG_FUEL_PER_N_O2
    return (SOOT_G_PER_G_WOOD * KG_FUEL_PER_N_O2 * 1000.0
            / float(cfg.physics.combustion.soot_yield))


def smoke_units_of_soot(grams: float, cfg=None) -> float:
    """A soot MASS in grams as smoke-density units (summed over tiles)."""
    return float(grams) / soot_g_per_smoke_unit(cfg)


# ---------------------------------------------------------------------------
# Gas IDs — the single source of truth (engine/05 §6.2). Order is the slice
# order of the dense ``gmap.gas`` (N, h, w) array; ids must be contiguous
# 0..N-1 so an array indexed by id has no gaps (validated in GasTable).
# ---------------------------------------------------------------------------
STEAM = 0
SMOKE = 1
POISON = 2
TEARGAS = 3
FUEL_GAS = 4
# EOS refactor P1 (§1/§2): the bulk pair, APPENDED — never renumber the five
# ids above (they are index-bound: gmap.smoke = gas[SMOKE], the field
# digest's `gas` shape/order, etc.).
O2 = 5
INERT_N2 = 6

# TODO(B2): cpp/ still spells these `black_smoke`/`white_smoke` in identifiers
# and comments (e.g. cpp/src/combustion.h, cuda_combustion.h/.cu,
# bindings.cpp — `black_smoke_idx` and friends). Left as-is this beat: the
# C++/Python boundary passes gas planes positionally/by index, never by
# name string, so the mismatch is cosmetic and safe. Deferred cosmetic sweep,
# no rebuild needed (B2 design doc §1, §0 iron constraints).

# Config-key <-> id mapping. The key is the ``[gases.<name>]`` table name.
# Listed in id order; ``GasTable`` validates contiguity.
GAS_NAMES = {
    STEAM: "steam",
    SMOKE: "smoke",
    POISON: "poison",
    TEARGAS: "teargas",
    FUEL_GAS: "fuel_gas",
    O2: "o2",
    INERT_N2: "inert_n2",
}

# Number of gas types (the N of the (N, h, w) gas array). 5 trace + 2 bulk (P1).
N_GASES = len(GAS_NAMES)

# EOS refactor P1: the trace-gas id range is [0, N_TRACE_GASES) — the 5 M1
# gases (steam .. fuel_gas), always the low contiguous block since the
# bulk pair is APPENDED, never inserted. Callers that want "every gas EXCEPT
# the one under test, but NOT the always-present ambient O2/N2 bulk pair"
# (weapon/payload slice-isolation tests are the main consumer) iterate
# ``range(N_TRACE_GASES)`` instead of ``range(N_GASES)``.
N_TRACE_GASES = O2

# Scalar columns: name -> numpy dtype. ``absorption`` / ``scatter_albedo`` are
# handled separately because they are per-channel RGB triples, not scalars.
# ``conservative`` (P1): true only for the bulk pair (o2 / inert_n2) — read by
# PhysicsRunner to route that plane to the donor-cell flux transport. Since
# smoke transport v2 (#12) the trace gases RIDE that same flux (priced by
# air); ``conservative`` keeps its one meaning, bulk membership.
_SCALAR_COLUMNS = {
    "diffusion": np.float32,
    "decay": np.float32,
    "glow": np.float32,
    "flammable": bool,
    "emits_when_hot": bool,
    "conservative": bool,
    # ray-engine-v2 P5a (design v3 §6.3): the gas's HEAT extinction per unit of
    # its own density — validated and quantized to ``heat_absorb_q16`` below.
    "heat_absorb": np.float32,
}

# The ``heat_absorb`` door's upper bound (ray-engine-v2 P5a). NOT a physical
# guess: it is the largest power of two for which the radiation sweep's per-cell
# density sum ``Σ_g heat_absorb_q16[g] · N_g`` is EXACT in int64 for ANY int32
# densities on up to 16 gas planes (every term < 2^28 · 2^31 = 2^59, sixteen of
# them < 2^63) — no saturating arithmetic, no data-dependent branch; the sweep
# re-checks both numbers at its own door (radiation_sweep.h). Physically it
# constrains nothing: at 4096 a gas is opaque at 1/4096 of ambient density, a
# wisp. The integer reference mirrors it as ``sweep_ref_q.HEAT_ABSORB_MAX``.
HEAT_ABSORB_MAX = 4096.0


# SMOKE TRANSPORT v2 (#12, docs/smoke_transport_design_2026-10-04.md §2.2): the
# trace diffusion's STABILITY DOOR. The once-per-tick Jacobi face diffusion
# (bulk_transport.cpp::trace_tail) folds dd_q = quantize(d_g * dt) on the host
# and moves |F| = (mul_q16(dd_q, perm_face) * |dS|) >> 16 across each face; its
# exact positivity rests on sum_j c_ij <= ONE, i.e. 4 * dd_q <= ONE. Checked ON
# THE INTEGER, with the engine's own fold (float32 dial times float32 dt,
# multiplied in double, round-half-away), here at the gases door and again
# where PhysicsRunner binds dt. Shipped dials sit near 0.004 a tick.
def trace_diffusion_dd_q(diffusion: float, dt: float) -> int:
    """The engine's dd_q = quantize(d * dt) fold, bit for bit."""
    v = float(np.float32(diffusion)) * float(np.float32(dt)) * 65536.0
    return int(np.floor(v + 0.5)) if v >= 0.0 else int(np.ceil(v - 0.5))


def check_trace_diffusion_stable(names, diffusion, conservative, dt: float,
                                 where: str) -> None:
    """Refuse a trace diffusion dial whose integer step is not stable at ``dt``.

    Raises ValueError naming the gas when ``4 * dd_q > 65536`` or the dial is
    negative / not finite (a negative coefficient would anti-diffuse)."""
    for g, name in enumerate(names):
        if bool(conservative[g]):
            continue          # bulk planes do not diffuse through this law
        d = float(diffusion[g])
        if not np.isfinite(d) or d < 0.0:
            raise ValueError(f"{where}: gases.{name}.diffusion must be finite and "
                             f">= 0 (smoke transport v2, design 2.2); got {d!r}")
        dd_q = trace_diffusion_dd_q(d, dt)
        if 4 * dd_q > 65536:
            raise ValueError(
                f"{where}: gases.{name}.diffusion = {d!r} at dt = {dt!r} s gives "
                f"dd_q = {dd_q}, and 4 * dd_q > 65536: the trace diffusion's "
                f"Jacobi step would no longer be positivity-preserving (smoke "
                f"transport v2, design 2.2 -- needs diffusion * dt <= 0.25)")


class GasTable:
    """Per-gas property table, indexed by gas id (engine/05 §6.2).

    Built from the ``[gases]`` section of ``config.toml`` (the named-key dict
    format, same as :class:`simulation.materials.MaterialTable`). Each scalar
    column is a 1-D numpy array indexed by gas id (``table.diffusion[gas_id]``);
    ``absorption`` and ``scatter_albedo`` are ``(N, 3)`` RGB arrays. ``effect``
    is a list of per-gas strings (a gameplay tag read unit-side in mechanics, not
    a numeric column).

    The id constants (:data:`STEAM`, :data:`SMOKE`, ...) are the
    canonical slice indices into ``gmap.gas``; ``name_to_id`` gives the same map
    keyed by name.

    Rebuild via :meth:`from_config` after a config hot-reload.
    """

    def __init__(self, gases_cfg, smoke_cfg=None, tick_dt=None):
        """Build from the ``CFG.gases`` namespace (or any equivalent).

        ``gases_cfg`` is the :class:`config.Namespace` for ``[gases]``; each
        attribute (``steam``, ``smoke``, ...) is itself a namespace
        of the named columns. A plain dict-of-dicts is also accepted (tests).

        ``smoke_cfg`` (ray-engine-v2 P6a) is the ``[smoke]`` section whose three
        optics dials fold into the LIGHT columns below (one owner, the dials
        the render medium reads). ``None`` reads ``CFG.smoke`` -- so a table
        built from test rows still carries the shipped dials; a dial the
        section omits is neutral (1.0).
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
        # light_absorb: OPTIONAL per-channel RGB, (N, 3) float64 with NaN rows
        # where a gas does not carry it (#12 handle 1) -- see the light door.
        self.light_absorb = self._load_rgb_optional(rows, "light_absorb")

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

        # heat_absorb_q16 (ray-engine-v2 P5a, design v3 §6.3): the per-gas
        # Q16.16 HEAT-extinction coefficient the radiation sweep reads on a GAS
        # cell — the density law, a_gas = min(ONE, Σ_g heat_absorb_q16[g] · N_g
        # >> 16), 0 where the bulk N is below gas_energy.h's N_EPS_RAW. Validated
        # HERE, at the table door, then quantized ONCE in the beam_absorb_q16
        # idiom (door 2: the standard round-half-away-from-zero twin, no divide
        # this time — the column IS the coefficient). The door:
        #   * finite and >= 0 — a negative coefficient would make smoke a
        #     SOURCE (the sweep's positivity rests on a >= 0);
        #   * <= HEAT_ABSORB_MAX (4096) — the arithmetic's own limit, see the
        #     constant's comment. No [0, 1] cap: heat_absorb is a coefficient
        #     PER UNIT DENSITY, not a fraction, and min(ONE, ·) is what keeps
        #     a_gas a fraction — a soot that is opaque at a tenth of ambient
        #     density is heat_absorb 10, and physical.
        # Stored as a contiguous int32 ARRAY (not a tuple like the beam's): the
        # engine takes it by pointer every tick (PhysicsEngine.step_tail).
        # P5c opened the Pass-1 fold's gas branch, so a non-zero value now has a
        # book to land in (e_gas_deposit_sum): smoke ships its derived value,
        # every other gas 0.0 (tests/test_gas_heat_absorb.py pins both).
        heat_absorb_q16 = []
        for name, value in zip(self.names, self.heat_absorb.tolist()):
            v = float(value)
            if not np.isfinite(v) or v < 0.0 or v > HEAT_ABSORB_MAX:
                raise ValueError(
                    f"gases.{name}.heat_absorb must lie in [0, {HEAT_ABSORB_MAX:g}] "
                    f"(heat extinction per unit density: negative would make the "
                    f"gas a radiation SOURCE, above {HEAT_ABSORB_MAX:g} the "
                    f"radiation sweep's density sum is no longer provably exact in "
                    f"int64 — design v3 section 6.3, P5a); got {value!r}")
            heat_absorb_q16.append(_ufx.quantize_scalar(v))
        self.heat_absorb_q16 = np.ascontiguousarray(
            np.asarray(heat_absorb_q16, dtype=np.int32))

        # ---- ray-engine-v2 P6a (design v3 §4.1 / §4.4 / §6.3; the P6a brief,
        # decision 6): THE LIGHT CHANNELS' SMOKE TERM, the heat term's pattern
        # one channel at a time. Two (N, 3) int32 Q16 columns the radiation
        # sweep reads on every GAS cell, per unit of the gas's density:
        #   light_absorb_q16[g][c] = absorption[g][c] * smoke_absorption[c]
        #                            * smoke_absorb_scale
        #       -> a_gas_c = min(ONE, Σ_g light_absorb_q16[g][c]·max(0, N_g) >> 16),
        #          the density law and the N_EPS floor exactly as heat's; what
        #          thin smoke does not absorb continues down the stream, and hot
        #          smoke EMITS through L° (the arc's "black-body smoke")
        #   light_glow_q16[g][c]   = scatter_albedo[g][c] * smoke_scatter_albedo[c]
        #       -> the in-scatter glow coefficient (albedo x density, capped at
        #          ONE), render-only, never in the books
        # THE [smoke] DIALS ARE FOLDED IN HERE -- one owner, the same dials the
        # render medium reads (design v3 §4.4: renderer/gas_medium.py reads
        # [smoke] directly; the Raycaster that once held them is deleted,
        # P6c). Validated and quantized ONCE here,
        # in the heat_absorb_q16 idiom (door 3 -- the product of load-time
        # constants in float64 -- then door 2, the round-half-away twin): finite,
        # >= 0 (a negative coefficient is a light SOURCE), <= HEAT_ABSORB_MAX
        # (the density sum's own int64 bound, the heat door's).
        if smoke_cfg is None:
            from config import CFG as _CFG
            smoke_cfg = getattr(_CFG, "smoke", None)

        def _dial(name, default):
            if smoke_cfg is None:
                return default
            if isinstance(smoke_cfg, dict):
                return smoke_cfg.get(name, default)
            return getattr(smoke_cfg, name, default)

        s_abs = [float(v) for v in _dial("smoke_absorption", (1.0, 1.0, 1.0))]
        s_sca = [float(v) for v in _dial("smoke_scatter_albedo", (1.0, 1.0, 1.0))]
        s_scale = float(_dial("smoke_absorb_scale", 1.0))
        if len(s_abs) != 3 or len(s_sca) != 3:
            raise ValueError("[smoke] smoke_absorption / smoke_scatter_albedo must be "
                             "[R, G, B] triples")
        lab, lgl = [], []
        for i, name in enumerate(self.names):
            row_a, row_g = [], []
            physical = not np.isnan(self.light_absorb[i, 0])
            for c in range(3):
                # A row with its own light_absorb is PHYSICS (#12 handle 1):
                # used as is, never scaled by a [smoke] dial.
                va = (float(self.light_absorb[i, c]) if physical
                      else float(self.absorption[i, c]) * s_abs[c] * s_scale)
                vg = float(self.scatter_albedo[i, c]) * s_sca[c]
                for what, v in (("light_absorb" if physical
                                 else "absorption x [smoke] dials", va),
                                ("scatter_albedo x [smoke] dial", vg)):
                    if not np.isfinite(v) or v < 0.0 or v > HEAT_ABSORB_MAX:
                        raise ValueError(
                            f"gases.{name}: {what} channel {c} = {v!r} outside [0, "
                            f"{HEAT_ABSORB_MAX:g}] (a light extinction per unit "
                            f"density: negative would make the gas a light SOURCE, "
                            f"above {HEAT_ABSORB_MAX:g} the sweep's density sum is no "
                            f"longer provably exact in int64 -- P6a)")
                row_a.append(_ufx.quantize_scalar(va))
                row_g.append(_ufx.quantize_scalar(vg))
            lab.append(row_a)
            lgl.append(row_g)
        self.light_absorb_q16 = np.ascontiguousarray(np.asarray(lab, dtype=np.int32))
        self.light_glow_q16 = np.ascontiguousarray(np.asarray(lgl, dtype=np.int32))

        # SMOKE TRANSPORT v2 (#12, design §2.2): the trace diffusion's
        # stability door, on the integer, at the configured tick length
        # ([clock] ticks_per_second; PhysicsRunner re-checks at the dt it is
        # actually handed). ``tick_dt`` overrides it for a table built for
        # another clock.
        if tick_dt is None:
            from config import CFG as _CFG
            clock = getattr(_CFG, "clock", None)
            tps = getattr(clock, "ticks_per_second", None) if clock is not None else None
            tick_dt = (1.0 / float(tps)) if tps else None
        if tick_dt is not None:
            check_trace_diffusion_stable(self.names, self.diffusion,
                                         self.conservative, float(tick_dt),
                                         "gases door")

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

    def _load_rgb_optional(self, rows, col):
        """An OPTIONAL ``(N, 3)`` RGB column: a row without ``col`` reads NaN
        (float64, so a present value keeps its authored digits)."""
        arr = np.full((self.n, 3), np.nan, dtype=np.float64)
        for idx, (row, name) in enumerate(zip(rows, self.names)):
            present = (col in row) if isinstance(row, dict) else hasattr(row, col)
            if not present:
                continue
            triple = self._get_field(row, name, col)
            vec = np.asarray(triple, dtype=np.float64)
            if vec.shape != (3,) or not np.all(np.isfinite(vec)):
                raise ValueError(
                    f"gases.{name}.{col} must be a finite [R,G,B] triple, got {triple!r}")
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
        return cls(cfg.gases, getattr(cfg, "smoke", None))


__all__ = [
    "STEAM", "SMOKE", "POISON", "TEARGAS", "FUEL_GAS",
    "O2", "INERT_N2",
    "GAS_NAMES", "N_GASES", "N_TRACE_GASES", "GasTable", "HEAT_ABSORB_MAX",
]
