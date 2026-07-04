# Q2-LIFT — the last determinism patch (→ `cuda-breached`)

**Status:** EXECUTED on branch `q2-lift` (3 staged commits, 2026-07-04). Erik green-lit
2026-07-04; NO feel-check (all deltas are quantization-scale, ~1/65536 — imperceptible;
Erik pre-approved the raycaster trig swap explicitly). The ONE re-baseline moved the
golden `60bd331f…` → `453829a67a38d79e0befd01d591cb19bdeb19f49d9234fb4d27a5083d126501a`.
Remaining: the Lenovo cross-machine confirm (roadmap 0.2, Erik) → then `cuda-breached`.
Note one sequencing adjustment vs the plan below: the HP-delta quantization landed with
Patch 3 (not Patch 2), because the A/B scenario's env damage is ACTIVE — quantizing HP
moves the golden, and the tree stays green at every commit only if that lands together
with the single re-baseline. Facing + bullet-trig + the tool split (digest-neutral,
verified) landed as Patch 2 as planned.

**Goal:** remove the last cross-machine nondeterminism from the synced trajectory —
the Q2-fenced Python-float unit state. The prime culprit is `facing = math.atan2(...)`
(`src/simulation/unit.py` ~278), the ONLY transcendental in the synced unit state;
libm transcendentals differ at the last ULP across Python/CRT versions (py3.11 desktop
vs py3.12 Lenovo), and the digest's 1e-9 quantization amplifies that into a hash flip.
Plain float `+−×÷` and `sqrt` are IEEE-correctly-rounded → already cross-version
stable → HP arithmetic is quantized only as belt-and-suspenders.

**Background:** the physics fields are already proven cross-machine deterministic
(cross-compiler MSVC 14.50≡14.44 + cross-arch Ampere≡Ada, digest `60bd331f…`). Only
the unit-state hash diverges (Lenovo `fe8eddda…`). See
`docs/xarch_ada_beatB_findings_2026-06-29.md` + `docs/roadmap_2026-07.md` Phase 0.

---

## Patch 1 — the deterministic trig kit (`cpp/src/fixed_point.h`)

Add three pure-integer, FP_HD (host+device) helpers, following the house style
(documented invariants, `tan_poly` as the precedent at ~576):

- `q16 atan2_q16(q16 y, q16 x)` → angle in Q16.16 **radians**, range (−π_q, π_q].
  Range-reduce: t = min(|y|,|x|) / max(|y|,|x|) via ONE rounded integer divide
  (`((int64)min<<FP_SHIFT + max/2) / max` — define the rounding exactly, document it),
  so the poly argument is in [0,1]; evaluate an odd minimax/Taylor-refined polynomial
  for atan on [0,1] (degree ~9–11, integer Horner via `mul_q16`/`mul_wide`+`narrow` —
  pick ONE narrowing discipline and document it); then the standard fixups:
  `atan(1/t) = π/2 − atan(t)` when |y|>|x|, quadrant offsets with **checked-in
  quantized constants** `PI_Q = quantize(π)`, `PI_2_Q = quantize(π/2)` (computed once
  in double, round-to-nearest — the locked "load-time constant" idiom). Edge cases
  pinned: x=0, y=0, both 0 (define, e.g. 0), axes, y=±x.
- `q16 sin_q16(q16 a)`, `q16 cos_q16(q16 a)` — input assumed within ~one wrap
  (document the valid range; a single conditional ±2π_q reduction is fine — our
  callers pass facing ∈ (−π,π] and ray angles ∈ [0, 2π+ε)); quadrant-reduce to
  [0, π/2_q]; odd/even minimax poly (degree ~7–9), integer Horner. cos via
  sin(a + π/2_q) or its own even poly — pick one, document.
- Expose all three via pybind in `bindings.cpp` (plain `m.def`, available in BOTH CPU
  and CUDA builds — no `#ifdef`).

**Accuracy gate (part of the pytest, not hand-waving):** sweep ≥1M angles/pairs
(dense + edge cases + random) against double libm; assert a pinned max-error bound —
target ≤ 2e-5 rad for atan2 and ≤ 1e-5 for sin/cos (tune the degree until met, then
PIN the achieved bound in the test with a small margin). Also: exact symmetry checks
(sin odd, cos even, atan2 quadrant signs), and determinism (pure integer body — no
float/double anywhere between the q16 inputs and the q16 output).

## Patch 2 — wire the unit state

- `src/simulation/unit.py` (~278): `self.facing = dequantize(bp.atan2_q16(quantize(-dy),
  quantize(dx)))` — quantize the float deltas at the boundary, integer atan2,
  dequantize back to float radians (exact n/65536 doubles → cross-machine identical;
  ALL downstream facing consumers — sprite compass, renderer — unchanged). Find every
  other `math.atan2/cos/sin` writing SYNCED unit state (grep; the combat shooting
  cos/sin at ~492-503 write bullet trajectories — check whether those reach synced
  state (HP via hits: yes, damage application) — if a transcendental feeds a synced
  outcome, route it through the kit the same way; if render-only, leave it).
- `src/simulation/combat.py`: quantize each damage delta before it applies:
  `dmg = dequantize(quantize(dmg))` at the `current_hp -=` sites (env damage ~234-238,
  blast ~130-132, and any other) — HP stays a float field, every delta becomes a
  multiple of 1/65536 (belt-and-suspenders; behaviour change ≤1.5e-5 HP per hit).
- `tests/_xarch_perfield_digest.py`: split the single unit-state hash into per-
  attribute hashes (hp / facing / pos / life+events) so a cross-machine diff NAMES the
  sub-field. Regenerate the Ampere baseline files after the re-baseline (below).

## Patch 3 — the raycaster trig swap (pre-approved)

- `cpp/src/raycaster.cpp`: replace `std::cos/std::sin` with
  `dequantize_f(cos_q16(quantize(angle)))` (or an equivalent documented boundary) at
  ALL SIX sites — the ray dirs (~17-18, ~140-141, ~555-556) and the cone
  `angular_atten` cos (~99, ~392, ~538) — so ray geometry + heat become cross-machine
  deterministic (the arc's deferred "integer cos/sin" item). `build_ray_list` and the
  live cast share the same code → the CPU-ref and GPU consume the SAME dirs → the S2/S2b
  heat bit-identity gates stay valid (re-run them). NOTE `raycaster.cpp` is /fp:strict;
  the angle math feeding quantize stays as-is.

## The re-baseline + gates (ONE re-baseline at the end)

The trajectory legitimately moves (facing/HP quantization + ray-dir poly deltas):
1. Rebuild BOTH builds (`fixed_point.h` touches every TU): CPU via the golden recipe
   (VS18/14.50 + anaconda cp311 → `cpp/build/Release`) and CUDA (`cpp/build_cuda.bat`).
2. Regenerate the 30-tick golden (`tests/xarch_digest.py`); update the GOLDEN constant
   in EVERY file that pins it (grep `60bd331f` — the S1..S7 checks, S2b, S8-adjacent,
   harness docs). Record old→new in the commit message.
3. Re-run ALL gates green: the full suite
   (`C:/Users/steen/anaconda3/python.exe -m pytest tests/ --ignore=tests/test_main_smoke.py
   --ignore=tests/test_renderer_smoke.py`) + every CUDA gate (S0–S7 + S2b live) —
   GPU==CPU must hold everywhere with the new trig.
4. Determinism self-check: capture the trajectory twice in separate processes → same
   digest. Regenerate `tests/_xarch_perfield_ampere.txt` (+ the host-named file).
5. The CROSS-MACHINE proof is Erik's Lenovo run (`_xarch_perfield_digest.py` after
   pulling this) — expected ALL-GREEN vs the new baseline. The `cuda-breached` tag
   goes on AFTER that confirms (not part of this branch).

**Discipline:** work on `q2-lift` only; staged commits (kit / wiring / swap+re-baseline);
each ends with the suite green; no physics change beyond the documented quantization
deltas; don't touch `wave_solver.cpp` (dead) or anything unrelated. If the poly can't
meet the accuracy bound at reasonable degree, STOP and report (don't ship a sloppy kit).
