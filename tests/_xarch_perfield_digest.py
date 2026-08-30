"""X-ARCH per-field, per-tick digest dumper  (THROWAWAY diagnostic — underscore).

Purpose
-------
The aggregate 30-tick trajectory digest (``xarch_digest.py`` -> one hash) tells us
two machines DISAGREE, but not WHERE. This script breaks that one hash open: it
emits a hash for EVERY (field, tick) pair over the canonical A/B scenario, plus
the synced unit-state hash per tick — split (Q2-lift; +statuses P3) into
per-attribute sub-hashes (__unit_hp__ / __unit_facing__ / __unit_pos__ /
__unit_life_events__ / __unit_status__)
so a unit-state divergence NAMES the sub-field, not just "the unit hash". Run the IDENTICAL script on two machines and
``diff`` the two output files — the FIRST line that differs names the exact
(field, tick) where the trajectories first diverge, which names the responsible
solver (e.g. ``water_depth`` -> WaterSolver, ``atmosphere`` -> AtmosphereSolver).

This is the localization tool the X-ARCH Ada findings (docs/xarch_ada_beatB_findings)
asked for: per-field per-tick dumping to find the first diverging (field, tick).

How to run (NO ARGS — dead simple, same on every machine)
---------------------------------------------------------
    C:/Users/steen/anaconda3/python.exe tests/_xarch_perfield_digest.py

It loads the clean CPU build (``cpp/build/Release`` on sys.path), runs 30 ticks of
the default A/B scenario, and writes:

    tests/_xarch_perfield_<host>.txt     (host = platform.node())

Each line is::

    <tick>\t<field>\t<per-field-hash>

in a FIXED (tick, field) order, so two files line up for a plain ``diff``. The
script also prints, to stdout:
  - the host + the aggregate 30-tick trajectory digest (compare to the committed golden on
    Ampere — the P4-wave-push golden — as a sanity check that the build is clean), and
  - if a SECOND file from another host is present in tests/, the first diverging
    (field, tick) between this run and that file (a built-in cross-machine diff).

Cross-machine workflow
----------------------
  1. Run here (Ampere)  -> tests/_xarch_perfield_DESKTOP-0E98HUV.txt
  2. Copy that file to the Lenovo, run there -> tests/_xarch_perfield_<lenovo>.txt
  3. ``diff`` the two files (or let step 2's run auto-report the first divergence).

Leaves only diagnostics in the tree; commits nothing.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from field_ab_harness import capture_trajectory, UNIT_DIGEST_KEY
from field_digest import (
    DIGEST_FIELDS, DIGEST_SPEC_VERSION, _field_bytes, trajectory_digest,
)

N_STEPS = 30
# A stable, machine-independent label for the synced unit-state hash so it sorts
# and diffs alongside the gmap fields (it is not in DIGEST_FIELDS).
UNIT_FIELD_LABEL = "__unit_state__"
# Re-baselined 2026-07-05 (P3 statuses): the synced unit record grows the
# status list (__unit_status__ sub-hash); no field trajectory moved.
# (was ae1164ca163b4bf49a86694ba78ea5319f86cfff46301c6aa59190207e6c1a12)
# Re-baselined 2026-07-05 (P4 wave-push): shockwaves displace units + trigger
# KNOCKED_DOWN (exchange.apply_wave_push, step 9c2). The A/B wave pulse
# sub-tile-nudges the marine (~0.04 tiles before its heat death), so only
# __unit_pos__ moved; no tile crossing -> the occupancy stamp and ALL field
# trajectories are byte-identical (and the pulse's dv ~2.3 is below the
# knockdown threshold 6.0 -> __unit_status__ unmoved too).
# (was 6d690fda8259b392be9029082013623fbef0fc0322ed3089107d5db220e1b441)
# EOS refactor P3 (2026-07-10): whole-physics replacement (the compressible
# gamma*p* multigrid solver supersedes wave+diffusion) — goldens moved ONCE,
# by design, at the end of the patch (the P1+P2 merge discipline). Previous:
# 2bab9702 (P1+P2), 07c3f370 (pre-EOS), f7b8becd (P3 at the pre-re-pin
# N_SUB_MAX=16 — superseded same-day by the blessed 16->8 re-pin, decisions
# log #14, which legitimately moves trajectories).
# Re-baselined 2026-07-10 (EOS refactor P4 — combustion on real O2): the O2 gate
# re-point (FireSimulation + apply_temperature_ignition now read gas[O2], not
# atmosphere/P — item 3) and the newly-applied trace decay->inert_N2 credit
# (item 2, decisions.md #12 v2.1) both touch the default scenario (it seeds
# fire + smoke): fire[8,8]/[8,9]'s O2-gated intensity and smoke's decay
# both move the trajectory. Combustion itself (item 1) does NOT touch this
# scenario (no flammable/wood material in the level).
# (was 493645d34b01d7ad55e5f0e6ae7254e94989dc1b6dce5c1b7ee5e53acaff3e63)
# Re-baselined 2026-07-10 (eos-p3fix-thermal-ceiling, design v2.4):
# the plume shim's T_FLAME_MAX self-limiter fix, the saturating T/u
# writes, the T_MAX_PHYS/U_MAX rails, the absorption-proportional gas
# radiant deposit (Pass 1), and the O2-gate hot-zone-equilibrium
# rescale (P_min/P_full/o2_threshold) all touch the default scenario
# (it seeds fire + smoke): the fire tiles' heat->T->wind->O2 chain
# moves the trajectory. ONE re-baseline for the whole branch (the
# gate-h rule). DIGEST_SPEC_VERSION unchanged (values moved; no field
# added/removed/retyped).
# (was 7eeb41d431a79ba01cbafef37416188bbf1ecb2a194d92af5f4ede279c9f2758)
# P-R4 GOLDEN REBASE (2026-08-01, the arc's ONE deliberate rebase —
# ruling amendment 5 D2, Erik's approval). The canonical A/B scenario seeds
# fire at (8,8)/(8,9) on AIR tiles (material 0, heat_atten 0,
# flammable.sum() == 0) — a GHOST fire whose only observable was the retired
# painter's air deposit. Under Kirchhoff a body that cannot absorb cannot
# emit (a_s == 0), so that heat is now correctly ZERO and every trajectory
# carrying it moves. Folded into the SAME one-shot rebase: D1's demand
# accumulator (digest spec v2 -> v3, +dem_acc), D3's radiant-flux sensor and
# D4's per-tick fan rotation. ONE approved change-set, ONE rebase event.
# P-O2b GOLDEN REBASE (2026-08-02) - the fire-realism arc's OWN single
# deliberate rebase (design v5.2 section 5: "this arc carries its own
# single deliberate rebase"; the arc-local golden the design budgets).
# THE EXTENDED OXYGEN DRAW (Erik's Option 2b) widens `dem_acc` from the 4
# faces to the 2*R*(R+1) SOURCE OFFSETS within BFS hop-radius DRAW_R -
# (12, h, w) at the shipped DRAW_R = 2. The shape rides the hashed
# per-field header, so this is a DIGEST-SPEC VERSION BUMP (v3 -> v4) taken
# per tests/field_digest_spec.toml's own change procedure, with every
# committed golden regenerated in the same commit.
# The A/B scenario carries no flammable tiles, so the LAW itself moves
# nothing here: the entire delta is dem_acc's layout. That is deliberate
# and separately gated - at DRAW_R = 1 the offset table's ring 1 IS D4's
# order, so the plane is bit-for-bit the v3 plane and the full engine
# reproduces every pre-patch field, byte for byte, over 45 ticks.
# (was e73f130ea6f514fc285825d1efc828202bfc7e2e77dee3212bed2aa822e45f8a)
# PRESSURE-ARC GOLDEN REBASE (2026-08-18, Erik's ruling: re-scope the canary,
# then ONE re-baseline). This one clears a BACKLOG of THREE deliberate,
# separately-approved behavioural changes that were each left un-rebased, so
# the golden had been stale — and this file's lineage silent — since P-T0:
#   1. P-T0 (energy-books arc): trace_mass_scale RETIRED and the decay->N2
#      credit deleted. Moved the canonical scenario to 8203584350ae69a5...
#      (recorded in docs/archive/e1_p_e2a_asbuilt_2026-08-17.md, not here).
#   2. P-E5 (energy-books arc close): Erik shipped k_drag = 0.5 +
#      k_drag_heat_frac = 0.0014 at the HUMAN-TEST. Interior momentum drag is
#      a live sink now, so the scenario's wind trajectory moves. -> b4f7d86c...
#   3. THIS ARC: mg_cycles 2 -> 8 (docs/pressure_arc_root_cause_2026-08-17.md).
#      The MG V-cycle schedule was frozen at C=2 on 16^2/160^2 scenarios for
#      300 ticks; at map scale C=2 leaves ~0.28 atm of residual per tick, which
#      IS the storm (playground: P_max 103.2 atm, negative P_min, n_sub pinned
#      at the cap). C=8 converges (C=8 and C=16 agree to 0.3%), zeroes every
#      rail counter, and is 18% FASTER per tick because a converged solve
#      collapses n_sub 8 -> 1. Every pressure-coupled field therefore moves.
# The three are folded into ONE rebase event, per the gate-h rule. Reproducible
# across builds: the CPU build and the CUDA build's CPU path both produce the
# new value (tests/test_cuda_mg_solve.py PART 3 et al.). DIGEST_SPEC_VERSION
# unchanged — values moved; no field added/removed/retyped.
# NOTE the companion change: tests/test_w6_armory.py's single
# "golden_and_untouched_rng" test was SPLIT, because its "this is a bug, never
# a re-baseline" message was true of its RNG half and false of its digest half
# (the canonical scenario exercises the EOS, so physics arcs lawfully move it).
# The durable dormancy canary is now test_canonical_scenario_consumes_no_rng.
# (was 28678e9d6210533f63cc701bba8f93194e23df9ebbdfa5f75f5d26681e897040)
# VELOCITY-CLAMP GOLDEN REBASE (2026-08-20, Erik's ruling: canon-fold cadence
# deferred, but the accumulated re-baseline debt is settled ONCE at this arc's
# close, docs/golden_rebaseline_2026-08-20.md). Folds the ONE deliberate
# behavioural change this arc shipped (HUMAN-TEST PASS 2026-08-20): the
# global-scalar velocity cap replaced by a per-cell cap^2 plane, plus the
# Chebyshev-pretest diagonal-leak fix replaced by an exact squared-magnitude
# clamp (docs/archive/velocity_clamp_pv1_asbuilt_2026-08-19.md sites 1-18).
# The canonical A/B scenario engages the new clamp (u_clamp_hits = 4 over 30
# ticks, all in tick-0 startup transient), so every pressure-coupled field
# trajectory moves -- the predicted GOLDEN_AGGREGATE cascade (11 dependent
# test files), not a scattered regression. DIGEST_SPEC_VERSION unchanged --
# values moved; no field added/removed/retyped.
# (was a18e0dfb017b98cb2454857014a6531c5f69b0460ecda70e60b486f281274c1b)
# T_ABS COMPRESSION-WORK GOLDEN REBASE (2026-08-21, arc `tabs-compression-work`,
# RULING R1 (Erik, 2026-08-17) executed. HUMAN-TEST PASS 2026-08-21, Erik
# blessed; the standing ruling's ONE deliberate re-baseline at this arc's
# close, docs/tabs_compression_work_rebaseline_2026-08-21.md). Folds step 4c's
# reversible-work arithmetic onto ABSOLUTE T (t_abs = T_rel + t_amb_q) instead
# of ambient-relative T: the inverted sub-ambient compression branch (cold gas
# got COLDER under compression) is fixed, ambient air stops being an exact
# fixed point of 4c, and honest rarefaction cooling / compression warming both
# now register (docs/tabs_compression_work_design_2026-08-20.md §1/§2). The
# canonical A/B scenario exercises the EOS every tick, so this lawfully moves
# every pressure/thermal-coupled field trajectory -- the predicted
# GOLDEN_AGGREGATE cascade (12 dependent test files + test_b6_logic_golden's
# inline golden), not a scattered regression. DIGEST_SPEC_VERSION unchanged --
# values moved; no field added/removed/retyped.
# (was d575df33de5c2af37108d29b73853b465eda761b148c6b812f4a4c4da40e0bb0)
# GAS-ENERGY CONSERVATION ARC, P-G0 GOLDEN REBASE (2026-08-29, design
# §2.2/§5, digest spec bump event 1: v4 -> v5, +gas_energy int64). This is a
# SCHEMA move, not a physics move: `gas_energy` is a new field, so its bytes
# fold into every tick_digest whether or not anything else changed, and
# DIGEST_SPEC_VERSION is itself hashed into every per-field digest — so
# EVERY GOLDEN this suite carries moves on a spec bump, by construction, even
# when the underlying arithmetic is untouched. Verified NOT a physics move
# directly (not just inferred from the spec-bump argument): captured every
# SIM_FIELDS array (raw, not hashed -- immune to the version salt) for this
# exact 30-tick canonical scenario on a clean HEAD build and on this patch's
# build, and diffed with np.array_equal per (tick, field) -- 0 mismatches
# on all 24 pre-existing fields x 30 ticks (720/720 identical); the only new
# keys are the 30 `gas_energy` entries. mul128_shr's three-copy ->
# one-primitive promotion (design §2.5) and the C/T_AMB_K throw guard +
# unconsumed k_ke fold (design §2.1) are therefore confirmed
# behaviour-preserving on the CPU build. (was a2cbc77ac324db99e0fcf2dc76e9ca15b3187c220a6d5abc5f4a110022c65cea)
# GAS-ENERGY CONSERVATION ARC, P-G3 GOLDEN REBASE (2026-08-30, design §6
# P-G3, value-move event 2). Unlike P-G0's schema-only bump above, this one
# is arithmetic: the canonical scenario exercises the EOS every tick, and
# four patches on this branch moved that arithmetic since the P-G0 rebase:
#   - P-G1a: the kick-loop KE brackets (RAD_SAFE guard moved + tightened,
#     ∇p/absorb/sponge/clamp/drag each individually booked) and the
#     sub-cycled face-flux energy step (design §2.3/§2.4) replace step 4c
#     on the CPU EOS path -- every pressure- and thermal-coupled field the
#     scenario's fire+smoke exercise the EOS through moves.
#   - P-G1b: the writer seam went live for combustion, the thermal solver's
#     gas side, pump primitives, and seal/unseal/destroy_wall/on_tile_changed
#     (moved mass carries T_abs, minted mass is born at ambient) and the
#     EOS-entry re-sync was deleted, so `gas_energy` is D1-live across whole
#     ticks instead of being re-derived from the T mirror each tick.
#   - P-G1d: the solve's divergence stencil moved to face form (û = 0 at
#     solid faces, the exact discrete adjoint of the kick's pressure
#     gradient) -- interior cells are bit-identical, but every field
#     downstream of a wall-adjacent pressure cell moves (the "feel-adjacent"
#     fix: BLAST peak |u| 8.7 -> 18.9 m/s in the P-G1d bench).
#   - P-G2: CUDA twins only (K1/K3, bulk transport, combustion, temperature
#     kernels) -- zero CPU source touched (`git diff d3c6689 -- '*.cpp'`
#     empty), so it contributes nothing to the CPU trajectory captured here;
#     named for completeness since it sits between P-G1d and P-G3.
# DIGEST_SPEC_VERSION unchanged (v5, set at P-G0) -- values moved; no field
# added/removed/retyped.
# (was df1f5153c9ce60a4de8e9c2198ff8eab3eb8d8267cf8be43d3ede03650b236bd)
GOLDEN_AGGREGATE = "f6daf44f4c2f563fc88bdb4465fb681a776141a9079d0e7c0f62f5c2b7fbb306"

# Q2-lift: the single unit-state hash is additionally SPLIT into per-attribute
# hashes so a cross-machine diff NAMES the diverging sub-field (hp vs facing vs
# pos vs life/events vs statuses) instead of the opaque "__unit_state__ differs"
# that the Ada Beat-B run left us localizing by hand. Order is FIXED (these
# five, then the aggregate) so files diff line-by-line. Payloads reuse the
# id-sorted unit records the harness already captures
# (field_ab_harness._unit_record: floats pre-quantized at 1e-9 -> byte-stable
# repr). __unit_status__ appended by P3 (2026-07-05): the status/condition
# list (mechanics/06 §4) is synced state — [[kind, magnitude_q16,
# remaining_ticks, source_id], ...] per unit, in list order, all ints.
UNIT_SUBFIELD_LABELS = (
    "__unit_hp__", "__unit_facing__", "__unit_pos__", "__unit_life_events__",
    "__unit_status__",
)


def _perfield_hash(name: str, arr: np.ndarray) -> str:
    """blake2b-256 over ONE field's spec bytes (name|dtype|shape + raw bytes).
    Same byte recipe as field_digest so a (field,tick) hash here is meaningful."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"PERFIELD_V{DIGEST_SPEC_VERSION}\n".encode("ascii"))
    h.update(_field_bytes(name, arr))
    return h.hexdigest()


def _unit_subfield_payloads(ustate: dict) -> dict[str, object]:
    """Split one tick's unit-state dict (harness format: id-sorted records +
    emission-ordered synced events) into the four named sub-payloads."""
    units = ustate.get("units", [])
    return {
        "__unit_hp__":     [(r["id"], r["current_hp"]) for r in units],
        "__unit_facing__": [(r["id"], r["facing"]) for r in units],
        "__unit_pos__":    [(r["id"], r["tile_x"], r["tile_y"], r["x"], r["y"])
                            for r in units],
        "__unit_life_events__": {
            "life":   [(r["id"], r["alive"], r["life_state"]) for r in units],
            "events": ustate.get("events", []),
        },
        "__unit_status__": [(r["id"], r["statuses"]) for r in units],
    }


def _unit_subfield_hash(label: str, payload) -> str:
    """blake2b-256 over one unit sub-attribute payload (label-pinned, repr of
    nested int/str/bool/tuples == byte-stable, same idiom as the harness)."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"PERUNIT_V{DIGEST_SPEC_VERSION}\n".encode("ascii"))
    h.update(label.encode("ascii"))
    h.update(b"|")
    h.update(repr(payload).encode("utf-8"))
    return h.hexdigest()


def build_perfield_lines(traj) -> list[str]:
    """One '<tick>\\t<field>\\t<hash>' line per (tick, field), in FIXED order
    (tick ascending; fields in DIGEST_FIELDS order, then the four unit
    sub-attribute hashes, then the aggregate unit-state hash)."""
    lines: list[str] = []
    for t, snap in enumerate(traj):
        for name, _dtype in DIGEST_FIELDS:
            if name not in snap:
                lines.append(f"{t}\t{name}\tMISSING")
                continue
            lines.append(f"{t}\t{name}\t{_perfield_hash(name, snap[name])}")
        # Synced unit state (HP/life/event stream) rides alongside the fields so a
        # combat/kill desync that leaves every gmap cell identical still localizes
        # — split per attribute so the diff NAMES the culprit sub-field.
        ustate = snap.get(UNIT_DIGEST_KEY, {})
        payloads = _unit_subfield_payloads(ustate)
        for label in UNIT_SUBFIELD_LABELS:
            if ustate:
                lines.append(f"{t}\t{label}\t{_unit_subfield_hash(label, payloads[label])}")
            else:
                lines.append(f"{t}\t{label}\tNO_UNIT_STATE")
        uhash = ustate.get("hash", "NO_UNIT_STATE")
        lines.append(f"{t}\t{UNIT_FIELD_LABEL}\t{uhash}")
    return lines


def first_divergence(lines_a: list[str], lines_b: list[str]):
    """Return the first (line_index, a_line, b_line) where two perfield dumps
    differ, or None if identical. Files share a fixed order so index lines up."""
    if len(lines_a) != len(lines_b):
        return (-1, f"<{len(lines_a)} lines>", f"<{len(lines_b)} lines>")
    for i, (la, lb) in enumerate(zip(lines_a, lines_b)):
        if la != lb:
            return (i, la, lb)
    return None


def main() -> int:
    import breach_physics as bp
    pyd = getattr(bp, "__file__", "?")
    host = platform.node()

    traj = capture_trajectory(n_steps=N_STEPS)
    lines = build_perfield_lines(traj)
    agg = trajectory_digest(traj)

    out = ROOT / "tests" / f"_xarch_perfield_{host}.txt"
    header = [
        f"# X-ARCH per-field per-tick digest",
        f"# host={host}",
        f"# pyd={pyd}",
        f"# spec_v={DIGEST_SPEC_VERSION}  n_steps={N_STEPS}",
        f"# aggregate_trajectory_digest={agg}",
        f"# columns: tick<TAB>field<TAB>blake2b256",
    ]
    out.write_text("\n".join(header + lines) + "\n", encoding="utf-8")

    print(f"host                = {host}")
    print(f"pyd                 = {pyd}")
    print(f"wrote               = {out}")
    print(f"per-field lines     = {len(lines)}  ({N_STEPS} ticks x "
          f"{len(DIGEST_FIELDS) + len(UNIT_SUBFIELD_LABELS) + 1} fields)")
    print(f"aggregate digest    = {agg}")
    print(f"matches golden      = {agg == GOLDEN_AGGREGATE}  "
          f"(Ampere clean-build sanity)")

    # Built-in cross-machine diff: if any OTHER host's perfield file is present,
    # report the first diverging (field, tick) against it.
    others = sorted(
        p for p in (ROOT / "tests").glob("_xarch_perfield_*.txt")
        if p.name != out.name
    )
    if others:
        print("\n--- cross-machine first-divergence vs other host files ---")
    for other in others:
        other_lines = [
            ln for ln in other.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")
        ]
        this_lines = [ln for ln in lines]  # already header-free
        d = first_divergence(this_lines, other_lines)
        if d is None:
            print(f"  {other.name}: IDENTICAL (no divergence over {N_STEPS} ticks)")
        elif d[0] == -1:
            print(f"  {other.name}: length mismatch {d[1]} vs {d[2]}")
        else:
            i, la, lb = d
            print(f"  {other.name}: FIRST DIVERGENCE at line {i}")
            print(f"      this ({host}) : {la}")
            print(f"      other         : {lb}")
            tick = la.split('\t', 1)[0]
            field = la.split('\t')[1] if '\t' in la else '?'
            print(f"      => first diverging (field, tick) = ({field}, tick {tick})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
