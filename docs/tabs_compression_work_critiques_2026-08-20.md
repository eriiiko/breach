# T_abs compression work — adversarial critique round 1 (2026-08-20)

Three lenses + one test-blast-radius sweep, run against design v1 (`3444941`).
**All findings are FOLDED into design v2** (`docs/tabs_compression_work_design_2026-08-20.md`)
— this capture exists so worktree subagents can read the full evidence and
derivations without re-deriving them. Finding numbers (A1..., B-F1..., C1...)
are referenced from the design.

---

## Lens A — Q16 arithmetic / determinism / ABI (verbatim keepers)

**A1 (MUST-FIX, folded §2):** `t_abs` must be int64. `T_MAX_PHYS` is
def_readwrite (`bindings.cpp:2222`) with no 4c-entry rail; at T > 32477 real,
`T + t_amb_q` overflows int32 — UB on host vs wrap on device = determinism
break. Idiom precedent: `eos_solver.cpp:424`.

**A2 (MUST-FIX, folded §2):** the real fold is
`std::max<q16>(1, quantize((double)T_AMB_K))` (`eos_solver.cpp:372`,
`cuda_eos_step.cu:153`) — the A7 divide-by-zero floor; T_AMB_K=0 is reachable.
Transcribe verbatim everywhere; live CPU path reuses the in-scope local.

**A3 (MUST-FIX, folded §5):** six ABI sites missing from v1 — see design §5
for the complete list. Key trap: `cuda_kick_check.py:481` threads solver dials
into the reference and MUST pass `t_amb_k=float(eos.T_AMB_K)`;
`test_thermal_mass_axis.py:645+` calls positionally.

**A4+A5 (MUST-FIX, folded D-3):** `assert` is dead under Release/NDEBUG
(`cpp/CMakeLists.txt:14`), and `s_eos_q` is not in scope at two of three
twins. Companion invariant `T_MIN > −T_AMB_K` must be guarded too: at
`T_MIN ≤ −T_AMB_K` the compression branch silently re-inverts (t_abs < 0).

**A6 (NOTE — the reversibility proof, folded §2):** with `a = t_abs > 0`:
compression `C(a) = a + ceil(w·a) = ceil(a(1+w))` (mul_q16 floors the negative
dT ⇒ heating rounds UP); expansion `E(a) = floor(a/(1+w))`. Then
`E(C(a)) = a` EXACTLY (from `a(1+w) ≤ ceil(a(1+w)) < a(1+w)+1`) and
`C(E(a)) ∈ {a, a−1}`. The ±t_amb shift cancels across a cycle. Worked at
T_rel=0, w=0.5 (w_q=32768, t_amb_q=19005440): compress→expand residual 0;
expand→compress `19005440/1.5 → 12670293` (floor), back
`ceil(12670293·1.5) = 19005440` → residual 0 (the −1 appears at odd t_abs).
Consequence: `test_p_e4_reversible_work.py:159-177`'s at-clamp assertions
should STAY GREEN — a red there is a real transcription bug (STOP).
Per-branch bias is NOT one-way (compression rounds up ≤1 count/cell/tick);
only the composition is one-way.

**A7 (NOTE, folded P-W1b):** the below-clamp bound
`test_p_e4_reversible_work.py:214` (`max(8, |T0_raw|//200)`) must re-key on
`|T0_raw + t_amb_q|`.

**A8 (MUST-FIX, folded P-W1b):** the ~97 oracle cannot be
`quantize(-96.67)`. Exact raw value:
`floordiv_q(19005440<<16, 98304) = 12670293`, minus 19005440 →
**−6,335,147 raw = −96.66665**. Dial-derived form per B-F10.

**A9 (checks that PASSED):** int64 headroom (`<<16` ≈ 7.0e13); q16 narrow safe
at T_MAX_PHYS+290; sat_add correct-but-dormant at shipped dials; k==0 exact
identity for ALL int32 T incl. negatives; three transcriptions confirmed, no
fourth; no float in the per-cell path; T_MIN floor catches the one-tick
undershoot (max (w/(1+w))·t_abs); T cannot legitimately sit below −290 at 4c
entry under shipped dials (pre-4c writers: vacuum wipe writes 0, recovery
floors at t_min_q, SL bounded, drag deposit non-negative).

**A10 (NOTE, folded §9):** `T_AMB_K` becomes a *physics* dial for 4c, not
just the pstar offset — retuning it changes work strength everywhere.

---

## Lens B — physics / energy books (verbatim keepers)

**B-F1 (MUST-FIX, folded STOP set):** `test_e1_hot_rail.py:220-222` is a
strict `t_max_phys_hits == 0` property gate (flipped strict at P-E4) on the
compounding scenario — not a digest; its determinism tests are self-A/B and
cannot red. Why green is the honest expectation: the trust gate zeroes w as
the evacuated block's N collapses, unchanged by this patch.

**B-F2 (MUST-FIX, folded §6):** `test_e1_cold_rail.py` is one self-A/B test —
in NO manifest set. The window scenario's new behavior is P-W2's capture.

**B-F3 (MUST-FIX, folded §3):** RISK-1 was argued on the wrong quantity.
`test_ambient_gate2` runs at T ≡ 0 on HEAD (`_ambient_gmap` never seeds T) —
its rail assert was VACUOUS until this patch. Real new hazard: T=0 stops
being a fixed point; sustained rail-rate compression from ambient gives
`T_k = 290·(1.5^k − 1)` → T_MAX_PHYS in ~10 rail ticks. Quasi-static rush-in
estimate: `N_f ≈ 0.52·N_amb`, T_abs ≈ 562 K ⇒ interior ≈ +272 game-deg —
three orders under the ceiling; risk is the converging-front transient.
Pre-agreed rule: red on the rail ⇒ clamp ratchet ⇒ levers T_WORK_CLAMP /
n_work_ref, never a re-pin.

**B-F4 (MUST-FIX, folded §3 RISK-2):** two residuals, not one. The ≤1-LSB
ratchet is symmetric-cycle only. The O(k²) shape-asymmetry term
≈ ½(Σ_c k² − Σ_e k²)·(T+290) (named in archived §2.7) multiplied T_rel≈0
before, 290 now. Steepened acoustics ⇒ Σk²_c > Σk²_e. Estimate: one
half-cycle k=0.02 vs two at k=0.01 ⇒ ~0.03 game-deg/cycle ⇒ ~7 game-deg per
1000 ticks under a sustained standing mode — 10⁴× the ratchet, NOT invisible.
Needs a bound gate that can fail (P-W2 quiet room).

**B-F5 (BLOCKER, folded D-3):** as lens A4/A5. Chosen fix: keep `T + t_amb_q`
in the twins (bit-identical to the s_eos idiom at the frozen value) + one
always-compiled per-tick host check at the two live fold sites, throwing a
named error; reference twin exempt by contract.

**B-F6 (MUST-FIX, folded §3/§8):** conduction relaxation timescale computed
from the shipped table (air κ=0.024, hull κ=50 ⇒ harmonic 0.0480 ⇒
face_shift 10): τ ≈ 1024 ticks ≈ **43 s** per hull face; air↔air shift 11 ⇒
**21 s** lateral. The FAST cold-fade channel is advective mixing (refill
flow), not conduction. Corroborated: the energy-books hull leg was visible
only over 4800-tick runs.

**B-F7 (MUST-FIX, folded §3 + D-6):** the T_MIN pressure collapse.
`p* = C·N·(T + T_AMB_K)` (`eos_solver.h:60`); floor cell (t_abs = 1 K) loses
×290 of p* beyond N loss, reachable in ln(290)/ln(1.5) ≈ 14 rail ticks ≈
0.6 s of sustained expansion. Near-zero-p* cell beside normal cells ⇒ giant
∇P kick ⇒ velocity spike ⇒ pile-up — a PRESSURE route into the flash class,
not covered by D-1's Mach argument. Cheap lever if it bites: raise T_MIN
(t_abs = 30 K caps collapse at ×10) — Erik's D-6.

**B-F8 (MUST-FIX, folded §3):** `e_vac_wipe_sum`/`e_ring_pin_sum` are signed
channels that have only ever destroyed (0 sub-ambient open cells in 4.86M
cell-snapshots). Sub-ambient gas at wiped/pinned cells now CREATES —
bounded ≤ 290·N_vented/tick; named + priced at P-W2.

**B-F9 (MUST-FIX, folded §7):** the KE↔eth deferral argument rebuilt: the
work term is now the dominant new channel, same-signed with the kick's KE
mint during blasts; deferral stays legitimate because it cancels over closed
cycles (A6 proof) and accumulates only under sustained one-sided divergence,
where `eth_compression_delta` + the P-W2 bound gate instrument it.

**B-F10 (MUST-FIX, folded P-W1b):** dial-derived clamp oracle:
`expected = floordiv(t_amb_q<<16, FP_ONE + quantize(T_WORK_CLAMP)) − t_amb_q`;
shipped-default sanity comment −96.6666. Tuning T_WORK_CLAMP must not red it.

**B-F11 (MUST-FIX, folded §3):** cap CEILING goes live: T=700 ⇒ cap ≈ 554 m/s;
`u_max_hits` reachable at T ≳ 2930. `n_sub` insensitive below T ≈ 2930
(derives from max(c_LOCAL, u_max), u_max = 1000).

**B-F12 (MUST-FIX, folded §1/§8):** the trust gate does NOT starve the §2.9
window pocket (measured n_bulk 1.7–9.3 ≫ n_work_ref 0.25 — fade saturated)
but DOES fade the vent core (below 0.125·N_amb ⇒ zero work): **cold reads as
a ring, not a core.** Correct bookkeeping; `n_work_ref` is the reach knob.

**B-F13 (MUST-FIX, folded D-7):** `HeatFieldOverlay` is additive-emissive
("cold tiles stay invisible" by construction) — no mapping change can darken.
Cold display = separate pass (diverging ramp, pressure-overlay machinery,
alpha-blended under the additive heat pass).

**B-F14–17 (NOTEs, folded):** reuse the existing t_amb_q local on CPU;
factory + reference twin take `float t_amb_k`; ~97 is the scheme's number
(adiabatic ~114, 15% first-order gap deliberate); the census instrument is a
new `--mach-census` mode on `analyze_blowup_dump.py` + a deliberate recorder
capture (small tool patch, not free).

**B on D-1 (verdict — keep the floor, sharpened):** cold and fast ARE
colocated (a rarefaction is a nozzle), but numerically the floor re-opens
nothing: even the ambient cap runs ~4.7 tiles/substep at shipped scale (the
audit's "~14× over resolvable Courant during blasts" is owned by N_SUB_MAX,
ruled 8). c(1 K) = 17.6 m/s would throttle venting (k_drag=10 precedent).
The genuinely re-opened class is B-F7's pressure collapse — measure THAT.

---

## Lens C — scope / gates / process (verbatim keepers)

**C1 (BLOCKER, folded):** = B-F1 (independent convergence).

**C2 (BLOCKER, folded P-W0):** no baseline red set exists post-velocity-clamp
close (last recorded list is pre-P-V1: `velocity_clamp_pv1_baseline_2026-08-19.md`,
31 names at d9ae647). P-W0 produces the artifact; every set-diff cites it.

**C3–C5 (MUST-FIX, folded §2/§5):** = A2/A3 + the positional-caller audit;
param position pinned BEFORE trailing defaulted pointers;
`kick_compression_launch_resident` needs NO change (takes the folds struct).

**C6 (MUST-FIX, folded D-3):** = A4/B-F5.

**C7 (MUST-FIX, folded STOP set):** `test_ambient_gate1_flat_interior_holds`
(`test_air_boundary.py:767-787`, ≤1 raw LSB drift over 60 quiescent ticks) is
the direct consumer of the k==0 identity — STOP semantics, red means the
identity is broken.

**C8 (MUST-FIX, folded STOP set):** `test_thermal_mass_axis.py:577`
(`eth_transport_delta ≤ 0` strict) + `test_e1_hot_rail.py:177-208` books
gates — STOP, not triage.

**C9 (MUST-FIX, folded P-W1b gates):** add `cuda_bulk_flux_check.py` (PART 3
seeds a −120 sub-ambient pocket + closure identity `:457-473`),
`cuda_ambient_check.py` (`:186-189` asserts ambient ring pin absolutely),
`cuda_thermal_mass_check.py` to the parity sweep.

**C10 (MUST-FIX, folded §6):** single-sourcing VERIFIED fixed — 12 importers
of `GOLDEN_AGGREGATE`, sole literal `_xarch_perfield_digest.py:168`. The
Arc-B trio `test_b1_signal_bus:99`/`test_b2_nodes:524` duplicate
`DOORTEST_NOPHYS_TRAJ_DIGEST` inline but run physics=None — MUST NOT flip
(STOP if they do). `test_b6_logic_golden:83` is inline AND physics-live —
EXPECTED red; its latency pins `:92-94` re-derive at close with rationale.

**C11 (MUST-FIX, folded P-W0):** entry points corrected: cold-rail instrument
is `tools/bench_two_room.py::run_bench` (WINDOW dials PF1B + k_wind_strip=0.5,
damp 0.005) — the test pins nothing; hot-rail docstring numbers are STALE
(pre-energy-books) — measure fresh; quiet-room drift instrument DOES NOT
EXIST — P-W0 builds it (spec in design §6); storm-ledger command lineage:
`--ticks 4800 --damp 0.005 --pf1b` (see `e1_p_e0_asbuilt_2026-08-17.md:119`).

**C12 (MUST-FIX, folded D-7):** the render Kelvin frame (K = 293 + 3·T_game)
vs the EOS 290-frame diverge below ambient (−96.67 game-deg reads "3 K" via
the render map). The taste-free instrument Erik already asked for
(TODO:814-819) is the unwired `renderer/hover_readout.py`. Sim-side radiation
is safe: `e_bucket_of` maps negative T to bucket 0 (`raycaster.h:208-215`).

**C13 (MUST-FIX, folded P-W3):** close ritual gains the tag (precedent
`velocity-clamp-close`) and egregore-collect-transcripts.

**C14 (NOTE→adopted):** split P-W1 into ABI-plumb (digest-neutral, gate =
set-diff EMPTY) + law flip. The plumb run doubles as the strongest possible
ABI gate.

**C15 (NOTEs, folded P-W1b):** t_abs type pinned int64; at-clamp exactness
degradation = STOP (per A6); the −100 probe rationale is void under the new
law ((−200+290)·1.5−290 = −155 no longer crosses T_MIN); oracle lives in
test_p_e4_reversible_work.py; tier+effort pinned per patch; P-W0 records
CUDA rebuild + skip count (vacuous-green detector).

**C16 (process):** no house-rule violations found. (a) golden re-baseline
(one digest event) ≠ scenario re-pins — the latter get individual measured
rationale in the close doc; (b) the render change lands pre-HUMAN-TEST —
procedurally fine, named as provisional.

---

## Test-blast-radius sweep (the P-W1b manifest's evidence base)

**Deterministic will-fail:** `cuda_kick_check.py:297+364-367+421-425` — the
T_MIN rail forcer seeds −289 expecting compression to cross the floor
(−433.5 old law); new law warms it to −288.5 ⇒ floor never fires ⇒ coverage
hole. Re-derive: the floor is crossed by EXPANSION at the boundary (seed
−289.0, positive div: t_abs 65536 → floordiv → 43690 → T_rel −289.33 <
t_min_q). | `test_p_e4_reversible_work.py` — full rewrite (§8 of the sweep:
probe temps, analytic 2.8%/10.4% figures halve on the (T+290) base, below-
clamp bound re-key). | The 12 GOLDEN_AGGREGATE importers + `test_w6_armory:583`.
| `test_b6_logic_golden:83` + latency pins `:92-94`.

**High-risk (GRAY, measure then classify):** `test_air_boundary.py:785,787`
(gate 1 — but see STOP semantics), `:850,854` (gate 3 reflection ratio picks
up ambient-T evolution via `_ambient_reflection.py`'s different-size maps),
`test_level_water_physics.py:128` (byte-identical atmosphere after 1 tick —
should hold: 4c feeds NEXT tick's p*), `test_wall_failure.py:166,251,284`
(pressure-differential pops), `test_b5_airlock.py:413-421` (pressure band
pins), `test_velocity_clamp_property.py:281` (gate 3's 1.5× within-tick
slack), `test_fire_heat_source.py:422` (strict xfail → XPASS risk),
`test_p_e3_drag.py:122-158` (near-T_MAX clip cases shift the drag identity),
`test_thermal_mass_axis.py:651` (non-vacuous control),
`test_fire_feedback.py:347`, `test_water_boil.py:145,205,286`,
`test_eos_p1_species_transport.py:134-135,211`.

**NOT affected (verified):** `test_atmosphere_conservation.py` +
`test_atmosphere_saturation.py` (legacy AtmosphereSolver, not EOS);
`test_recorder_dump.py` (fake gmap); self-A/B digest tests (§3e of the sweep:
both halves move together); `test_continuous_o2_law.py`, `test_pr3_capacity_law.py`,
`test_cool_shift_axis.py` (T is an input plane, no EOS).

**Digest artifacts to regenerate at close (not pytest-asserted):**
`tests/_xarch_perfield_DESKTOP-0E98HUV.txt`, `tests/_xarch_perfield_erik_lenovo.txt`,
`tests/digest_erik_lenovo_cpu_cpu.txt`. `field_digest_spec.toml` is NOT
bumped (values move; no field added/removed/retyped).

**Meta-gate:** `test_cuda_check_scripts_are_wired.py` fires only on
add/delete/rename of `cuda_*_check.py` — keep filenames stable.
