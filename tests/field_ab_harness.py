"""Field-level A/B determinism harness — PhysicsEngine unification, Patch 0.

The unification's safety argument is "behavior test-identical": every sim field,
every cell, every tick, the refactored path must match the old path. The existing
determinism check (``tests/test_simulation.py:_state_signature``) compares five
whole-grid MEANS — it cancels per-cell sign-flipped errors and re-runs the same
build, so it CANNOT detect the float-reorder desync a glue->C++ port risks
(panel finding, docs/physics_engine_unification_plan.md §1).

This module is the real gate: snapshot every sim field each tick, run two paths on
the same seed+inputs, and assert per-FIELD per-CELL equality (exact under the
/fp:precise 0-ULP plan, or within a stated tolerance for the fallback path).

Usage (Patch 1, same machine — float bit-identity is NOT cross-machine until the
fixed-point migration):

    from field_ab_harness import capture_trajectory, assert_trajectories_match
    base = capture_trajectory()            # pre-refactor code  (git checkout old)
    # ... land a refactor phase, rebuild ...
    new  = capture_trajectory()            # refactored code
    assert_trajectories_match(base, new)   # tol=0.0 == 0-ULP

Run:
    C:/Users/steen/anaconda3/python.exe tests/field_ab_harness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hashlib

import numpy as np

import breach_physics as bp
from level_loader import LevelData
from simulation import Simulation
from simulation import wave_fixed   # S2a: wave_source Q16.16 quantize helper
from simulation import gas_fixed     # S2b: smoke/gas Q16.16 quantize helper
from simulation.unit import Unit

SEED = 20260615

# Reserved key under which each per-tick snapshot carries its synced-unit digest
# (see _capture_unit_state). Chosen so it cannot collide with a gmap field name
# in SIM_FIELDS, and so diff_trajectories can special-case it for human-readable
# per-unit / per-event reporting instead of an opaque "hashes differ".
UNIT_DIGEST_KEY = "__unit_state__"

# The synced (lockstep-critical) unit fields the digest captures, per unit, in a
# STABLE id-sorted order. These are the fields a desync would corrupt: position
# (tile + float), HP, life status, faction, and the footprint offsets (a stamp
# slip would change occupancy). RENDER-only / AI-scratch fields (facing, orders,
# zombie_path, accumulators) are intentionally excluded — they are not part of
# the synced determinism contract and may legitimately differ.
SYNCED_UNIT_FIELDS = (
    "tile_x", "tile_y", "x", "y", "current_hp",
    "alive", "life_state", "faction", "offsets",
    "facing", "ap", "n_orders",
)

# Every field the physics writes — the sim state a structural refactor must
# preserve bit-for-bit. `gas` is (N,h,w) and covers `smoke` (a view into
# gas[BLACK_SMOKE]); `dyn_*` are the per-tick stamp_units outputs; material /
# is_vacuum / wall_hp catch any topology-path change. Render-only buffers
# (light_rgb, light_dir, smoke_glow) are intentionally EXCLUDED — they are not
# sim state and may legitimately differ between machines/builds.
SIM_FIELDS = (
    "atmosphere", "wave_p", "wave_v", "wave_source", "wind_x", "wind_y",
    "gas", "fire", "water_depth", "flow_vx", "flow_vy",
    "heat", "temperature", "ripple", "ripple_v",
    "dyn_permeability", "dyn_wave_absorb", "obstacles", "dyn_light_atten",
    "wall_hp", "material", "is_vacuum",
)


def _scenario_level() -> LevelData:
    """A 16x16 hull-walled room, border on the map edge (so a border breach
    exposes vacuum), interior air carved out — synthetic, no asset files."""
    h = w = 16
    tm = np.ones((h, w), dtype=np.int32)   # all hull
    tm[1:15, 1:15] = 4                       # carve interior air
    return LevelData(name="ab_harness", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def default_scenario_sim() -> Simulation:
    """The canonical A/B scenario — every solver activated, fully deterministic.

    Seeds smoke + fire + water + a wave pulse over the interior, spawns one static
    marine (so stamp_units stamps a footprint each tick), and opens a hull breach
    (so the sink-pull + venting engage). Returns an unpaused sim ready to step.
    """
    sim = Simulation(_scenario_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    # S2b: smoke is int32 Q16.16 — quantize the seed (a raw `= 0.6` would store 0
    # counts ~ no smoke). 0.6 density -> 0.6*65536 = 39322 counts. This exercises
    # the integer-SL advection + diffusion + venting on a real cloud.
    g.smoke[interior] = gas_fixed.quantize_scalar(0.6)  # smoke transport (gas[BLACK_SMOKE] view)
    g.fire[8, 8] = 0.8               # fire feedback -> heat deposit -> temperature
    g.fire[8, 9] = 0.5
    g.water_depth[10, 10] = 0.3      # water pipe model + W3 displacement + ripple
    g.water_depth[10, 11] = 0.3
    # S2a: wave_source is int32 Q16.16 — quantize the seed so the explicit wave
    # actually kicks + reflects off the hull (a raw `= 8.0` would store 8 counts
    # ~ 1.2e-4 real, i.e. no wave). Real magnitude 8 -> quantize -> 524288 counts.
    g.wave_source[4, 4] = wave_fixed.quantize_scalar(8.0)
    sim.add_unit(Unit("M1", x=7, y=7, team=0))   # stamp_units footprint
    g.destroy_wall(8, 0)             # hull breach on the map edge -> vacuum (venting)
    sim.set_paused(False)
    return sim


def _snapshot(gmap, fields):
    return {name: np.copy(getattr(gmap, name))
            for name in fields if hasattr(gmap, name)}


# ---------------------------------------------------------------------------
# Synced UNIT-STATE digest
# ---------------------------------------------------------------------------
# The field harness above hashes only gmap arrays — it is BLIND to unit state.
# But unit HP and life/death are lockstep-critical SYNCED state: two trajectories
# can agree on every gmap cell yet disagree on who lives, who dies, and how much
# HP each unit has (the exact leak combat HP/damage going integer must not open).
# These helpers capture that state per tick alongside the field snapshot.


def _unit_faction(u):
    """Stable faction identifier. ``faction_id`` is the spec field; fall back to
    ``team`` (its alias) so the digest works on bare Unit objects too."""
    return int(getattr(u, "faction_id", getattr(u, "team", -1)))


def _unit_record(u):
    """A canonical, JSON-ish dict of one unit's synced state.

    Floats are rounded into a fixed-point integer (1e-9 quantum) so the record
    serializes byte-stably and the digest is reproducible across runs on the
    same machine — without that, float repr() noise could perturb the hash.
    """
    def q(v):
        # Quantize a float to a stable integer (avoids repr/format drift).
        return int(round(float(v) * 1_000_000_000))

    life = getattr(u, "life_state", None)
    life_repr = getattr(life, "value", str(life))   # enum -> its .value string
    return {
        "id":         int(getattr(u, "id", -1)),
        "tile_x":     int(u.tile_x),
        "tile_y":     int(u.tile_y),
        "x":          q(u.x),
        "y":          q(u.y),
        "current_hp": q(getattr(u, "current_hp", 0.0)),
        "alive":      bool(u.alive),
        "life_state": str(life_repr),
        "faction":    _unit_faction(u),
        "offsets":    [[int(dx), int(dy)] for (dx, dy) in getattr(u, "offsets", ())],
        # Sim-derived synced state beyond the core: facing (drives LOS/shot
        # direction), action points (gate synced actions), and the count of
        # remaining orders (a divergence in order-consumption shows here). A
        # full per-order serialization is a future extension if needed.
        "facing":     q(getattr(u, "facing", 0.0)),
        "ap":         [int(a) for a in getattr(u, "ap", ())],
        "n_orders":   len(getattr(u, "orders", ())),
    }


def _event_record(ev):
    """Canonical tuple for one synced tick event. We key off the class name +
    its public scalar fields (unit_id, damage, source / killed_by). Only the
    SYNCED combat events (UnitHitEvent / UnitKilledEvent) carry HP-relevant
    state, but any event with a ``unit_id`` is recorded in emission order so a
    divergent kill/hit stream is caught."""
    kind = type(ev).__name__
    return {
        "kind":      kind,
        "unit_id":   getattr(ev, "unit_id", None),
        "damage":    getattr(ev, "damage", None),
        "source":    getattr(ev, "source", None),
        "killed_by": getattr(ev, "killed_by", None),
    }


# Only events that touch synced unit life/HP belong in the determinism digest.
# (ShotFiredEvent / ExplosionEvent / Door|WallDestroyed are render/topology
# signals; the topology ones already show up in the gmap fields.)
_SYNCED_EVENT_TYPES = ("UnitHitEvent", "UnitKilledEvent")


def _capture_unit_state(sim):
    """Snapshot the synced unit state for the tick that just stepped.

    Returns a dict with:
      - ``units``  : per-unit records (living AND dead), sorted by unit id so the
                     order never depends on list/dict iteration order.
      - ``events`` : the synced unit events emitted THIS tick, in emission order
                     (``sim.tick_events`` is cleared at the start of each step and
                     appended to in order, so its order IS the emission order).
      - ``hash``   : a stable hash over a canonical serialization of the above.
    """
    units = sorted(sim.units, key=lambda u: int(getattr(u, "id", -1)))
    unit_records = [_unit_record(u) for u in units]
    event_records = [
        _event_record(ev) for ev in sim.tick_events
        if type(ev).__name__ in _SYNCED_EVENT_TYPES
    ]
    payload = {"units": unit_records, "events": event_records}
    blob = repr(payload).encode("utf-8")   # repr of nested int/str/bool == stable
    digest = hashlib.sha256(blob).hexdigest()
    return {"units": unit_records, "events": event_records, "hash": digest}


def unit_digest_hash(snapshot_or_state):
    """Convenience: pull the stable hash out of a tick snapshot (which stores the
    unit state under UNIT_DIGEST_KEY) or out of a raw unit-state dict."""
    if UNIT_DIGEST_KEY in snapshot_or_state:
        return snapshot_or_state[UNIT_DIGEST_KEY]["hash"]
    return snapshot_or_state["hash"]


def capture_trajectory(make_sim=default_scenario_sim, n_steps=30, fields=SIM_FIELDS,
                       capture_units=True):
    """Run ``make_sim()`` for ``n_steps``, returning a per-tick list of snapshot
    dicts. Each dict holds the gmap field arrays AND (when ``capture_units``) the
    synced unit-state digest under UNIT_DIGEST_KEY. Forces unpause each step so a
    phase/round boundary cannot silently halt the trajectory (the round reset
    itself is deterministic)."""
    sim = make_sim()
    traj = []
    for _ in range(n_steps):
        sim.set_paused(False)
        sim.step()
        snap = _snapshot(sim.gmap, fields)
        if capture_units:
            snap[UNIT_DIGEST_KEY] = _capture_unit_state(sim)
        traj.append(snap)
    return traj


def _diff_unit_state(t, ua, ub):
    """Locate the first divergence in the synced unit state between two ticks.

    Reports WHICH unit / WHICH field, or WHICH event, so a desync points at the
    exact unit and quantity — not just "the unit hashes differ". Returns a list
    of human-readable mismatch lines (empty == match)."""
    if ua["hash"] == ub["hash"]:
        return []                      # fast path: identical
    out = []
    # --- per-unit field comparison (records are already id-sorted) ---
    ua_by_id = {r["id"]: r for r in ua["units"]}
    ub_by_id = {r["id"]: r for r in ub["units"]}
    for uid in sorted(set(ua_by_id) | set(ub_by_id)):
        ra, rb = ua_by_id.get(uid), ub_by_id.get(uid)
        if ra is None or rb is None:
            out.append(f"tick {t}: unit id {uid} present in only one run")
            continue
        for fld in sorted(set(ra) | set(rb)):
            if ra.get(fld) != rb.get(fld):
                out.append(
                    f"tick {t}: unit id {uid} field '{fld}' differs "
                    f"(a={ra.get(fld)!r} b={rb.get(fld)!r})")
    # --- event-stream comparison (emission order matters) ---
    ea, eb = ua["events"], ub["events"]
    if ea != eb:
        if len(ea) != len(eb):
            out.append(f"tick {t}: unit event count {len(ea)} != {len(eb)} "
                       f"(a={ea!r} b={eb!r})")
        else:
            for i, (xa, xb) in enumerate(zip(ea, eb)):
                if xa != xb:
                    out.append(f"tick {t}: unit event #{i} differs "
                               f"(a={xa!r} b={xb!r})")
    if not out:   # hashes differed but no structured field did — surface it loudly
        out.append(f"tick {t}: unit_digest hash differs ({ua['hash'][:12]} != "
                   f"{ub['hash'][:12]}) but no field located — serialization drift?")
    return out


def diff_trajectories(a, b, tol=0.0):
    """Per-field per-cell mismatches between two trajectories (empty list == match).

    ``tol == 0.0`` -> exact equality (the /fp:precise 0-ULP gate). ``tol > 0`` ->
    absolute tolerance (the fallback path). Each entry is a human-readable line
    locating the worst-offending cell — or, for the synced unit state, the exact
    unit id + field or event that first diverges (so a kill/HP desync that leaves
    every gmap cell identical still FAILS loudly)."""
    if len(a) != len(b):
        return [f"trajectory length {len(a)} != {len(b)}"]
    diffs = []
    for t, (sa, sb) in enumerate(zip(a, b)):
        for k in sorted(set(sa) | set(sb)):
            if k not in sa or k not in sb:
                diffs.append(f"tick {t}: field '{k}' present in only one run")
                continue
            if k == UNIT_DIGEST_KEY:
                diffs.extend(_diff_unit_state(t, sa[k], sb[k]))
                continue
            fa, fb = sa[k], sb[k]
            if fa.shape != fb.shape:
                diffs.append(f"tick {t}: '{k}' shape {fa.shape} != {fb.shape}")
                continue
            if tol == 0.0:
                if np.array_equal(fa, fb):
                    continue
            elif np.allclose(fa, fb, rtol=0.0, atol=tol, equal_nan=True):
                continue
            d = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
            idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(d)), d.shape))
            diffs.append(
                f"tick {t}: '{k}' differs — {int((d > tol).sum())} cell(s), "
                f"max|delta|={d.max():.3e} at {idx} (a={fa[idx]!r} b={fb[idx]!r})")
    return diffs


def assert_trajectories_match(a, b, tol=0.0, max_report=10):
    """Raise AssertionError with the first ``max_report`` mismatches if a != b."""
    diffs = diff_trajectories(a, b, tol=tol)
    if diffs:
        head = "\n  ".join(diffs[:max_report])
        more = "" if len(diffs) <= max_report else f"\n  ... +{len(diffs) - max_report} more"
        raise AssertionError(
            f"A/B trajectories differ ({len(diffs)} mismatch group(s)):\n  {head}{more}")


def save_trajectory(traj, path):
    """Persist a trajectory so the OLD path's golden survives a C++ rebuild.

    The unification gate is OLD-build vs NEW-build on the SAME machine; since a
    rebuild swaps the .pyd in-process, capture the golden, pickle it here, rebuild,
    then ``load_trajectory`` + ``assert_trajectories_match`` against a fresh capture.
    (Underscore-prefixed golden files are untracked dev artifacts, never committed.)
    """
    import pickle
    with open(path, "wb") as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_trajectory(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    a = capture_trajectory()
    b = capture_trajectory()
    assert_trajectories_match(a, b, tol=0.0)
    nfields = len(a[-1]) - (1 if UNIT_DIGEST_KEY in a[-1] else 0)
    # S2a — make the integer WAVE determinism explicit: wave_p/wave_v/wave_source
    # are now int32 Q16.16 (the synced wave state). Assert the dtype + bit-identity
    # run-to-run (np.array_equal is exact on int32). This is the S2a P1 gate.
    for f in ("wave_p", "wave_v", "wave_source"):
        assert a[-1][f].dtype == np.int32, f"{f} should be int32 Q16.16 (S2a), got {a[-1][f].dtype}"
        for t in range(len(a)):
            assert np.array_equal(a[t][f], b[t][f]), f"{f} not bit-identical at tick {t} (S2a P1)"
    nz = int((np.abs(a[-1]["wave_p"]) > 0).sum())
    print(f"OK: field-level A/B harness — {len(a)} ticks x {nfields} fields "
          f"+ synced unit digest, per-cell 0-ULP self-match "
          f"(final unit hash {unit_digest_hash(a[-1])[:12]})")
    print(f"OK: S2a wave fields int32 Q16.16, bit-identical run-to-run "
          f"(wave_p nonzero cells at final tick: {nz})")
    # S2b — the integer smoke + 5 gas planes are now int32 Q16.16 (the synced
    # smoke/gas state). Assert dtype + bit-identity run-to-run (np.array_equal is
    # exact on int32). This is the S2b P1 gate. Plus the deterministic-NON-
    # conservation check: total mass per plane identical run-to-run (NOT conserved
    # to the LSB — the integer-SL >>16 truncation is a deliberate gentle decay),
    # bounded and never blowing up. `gas` is (N,h,w); `smoke` is the int32 view.
    assert a[-1]["gas"].dtype == np.int32, \
        f"gas should be int32 Q16.16 (S2b), got {a[-1]['gas'].dtype}"
    for t in range(len(a)):
        assert np.array_equal(a[t]["gas"], b[t]["gas"]), \
            f"gas not bit-identical at tick {t} (S2b P1)"
    # Deterministic non-conservation: per-tick total gas mass identical A vs B,
    # and never amplifying past the seeded peak (a runaway would blow this up).
    mass_a = [int(a[t]["gas"].astype(np.int64).sum()) for t in range(len(a))]
    mass_b = [int(b[t]["gas"].astype(np.int64).sum()) for t in range(len(b))]
    assert mass_a == mass_b, "S2b gas mass trace not bit-identical run-to-run"
    peak = max(mass_a) if mass_a else 0
    final = mass_a[-1] if mass_a else 0
    assert peak < (1 << 62), "S2b gas mass overflow guard"
    nzg = int((a[-1]["gas"] != 0).sum())
    print(f"OK: S2b smoke+gas int32 Q16.16, bit-identical run-to-run "
          f"(gas nonzero cells final: {nzg}); deterministic non-conservation "
          f"mass peak={peak} final={final} kept={100.0*final/peak if peak else 0:.1f}%")
    # S2c — the integer ATMOSPHERE (pressure) + WIND are now int32 Q16.16 (the
    # synced atmosphere state) — the CLOSER of the S2 group. Assert dtype +
    # bit-identity run-to-run (np.array_equal is exact on int32). This is the S2c
    # P1 gate; the per-cell P2 conservation gate lives in
    # tests/test_atmosphere_saturation.py (the run-past-wave-death saturation).
    for f in ("atmosphere", "wind_x", "wind_y"):
        assert a[-1][f].dtype == np.int32, \
            f"{f} should be int32 Q16.16 (S2c), got {a[-1][f].dtype}"
        for t in range(len(a)):
            assert np.array_equal(a[t][f], b[t][f]), \
                f"{f} not bit-identical at tick {t} (S2c P1)"
    nza = int((a[-1]["atmosphere"] != 0).sum())
    nzw = int((np.abs(a[-1]["wind_x"]) + np.abs(a[-1]["wind_y"]) > 0).sum())
    print(f"OK: S2c atmosphere+wind int32 Q16.16, bit-identical run-to-run "
          f"(atmosphere nonzero cells final: {nza}; wind nonzero cells: {nzw}) "
          f"— the S2 group is now cross-GPU deterministic (only the FIRE bridge "
          f"remains, S3)")
