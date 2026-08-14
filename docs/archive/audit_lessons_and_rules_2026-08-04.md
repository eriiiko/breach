# What we learned, and the rules worth writing (2026-08-04)

**Status: proposal. Erik decides which rules to adopt; nothing has been written
into CLAUDE.md or any skill.** Evidence: `docs/codebase_audit_2026-08-03.md` and
`docs/fire_atmosphere_oscillation_analysis_2026-08-03.md`. Two independent
passes over that evidence (one by the session, one by a fresh agent that had not
written the audit) converged on the same patterns; this document is the merge.

**Method note.** Every rule below is scored by an **acid test**: which specific
findings would it have caught, and — stated just as plainly — which it would
*not*. A rule that catches nothing concrete is a slogan and has been cut. Three
were cut on those grounds (§5).

---

## 1. The two findings that matter most

**(1) Gate coverage, not language, predicts quality.** The audit's grades:
Python core B+, renderer A−, C++ engine A−, CUDA A− — and the two weakest areas
are `bindings.cpp` (C+) and `tools/` (B−). Erik's hypothesis ("Python is
probably worse") is not supported, and the *reason* is the useful part:
everything under a digest/golden gate is excellent in any language; everything
outside one has drifted. Note that a folder-or-language split of the audit would
have *confirmed* the wrong hypothesis. The split that produced the insight was
by exposure to an automated oracle.

**(2) ★ In an agent-built repo, a technique that is not in a file agents must
read does not exist.**

This is the sharpest thing to come out of the whole exercise, and it explains
the recurrence pattern mechanically. Compare:

- **Dormancy discipline** — every new subsystem ships behind a gate making a
  level without it byte-identical — propagated to *every* arc across ~15 arcs.
  It lives in the **canon chapters** (`16_entity_system.md:98,210,243,273`;
  `04_atmosphere_and_pressure.md:97`; `03_combat_and_weapons.md:354,600,635`),
  which project CLAUDE.md instructs every session to read.
- **`RC_HD`** (one definition compiled host+device), **`cudaMemcpyToSymbol`
  canonical tables**, **`_ep`** (the C++ struct member *is* the Python
  fallback), **non-vacuousness controls** — each solves a whole defect class,
  each is genuinely excellent, and each stayed local to the patch that invented
  it. They live in header comments and individual test files. Nobody is
  instructed to read those.

The mechanism: `autonomous-patch-workflow` runs each patch in a fresh subagent
with its own context; the orchestrator keeps only the plan and short summaries.
A human accumulates habits across patches. **A fresh subagent cannot.** So in
this workflow, propagation is not cultural — it is purely a function of what the
next agent is *required* to read.

**Consequence for every rule below: the home matters as much as the rule.** A
rule in CLAUDE.md governs sessions. A technique belongs additionally in the
canon chapter of its system, because that is the artifact with a proven
propagation record here.

---

## 2. The recurring failure patterns

Ranked by damage actually incurred. Each produced many findings at once.

| # | Pattern | Why it stays invisible | Worst instances |
|---|---|---|---|
| **P1** | **Green is the same colour as blind** — gates armed on a world where the thing cannot happen | A passing gate emits one bit; "correct" and "the scenario lacks the phenomenon" both map to it. The *stronger* the suite, the more confidence a vacuous green buys | The canonical golden has **no flammable tiles** — every combustion change was golden-preserving *by construction*, during the month the fire arc changed combustion. Every fire bench is single-room; the storming is a two-room mode. `cuda_fire_check.py:170` compares **sets**, so it cannot see the confirmed ordering divergence. Two complete CUDA gates never collected |
| **P2** | **The unnamed quantity** — a field exists in state; its *meaning* was never written down, so each consumer invented one | A unit conversion is a function evaluated in two languages, never on the same input. Each half is locally correct; only the **pair** is wrong, and no file owns pairs | The two Kelvin maps. `tile_size_m = 1.0` shipped in a level while `rad_scale`/`burn_rate` freeze 0.333 m — **canon already forbids this in writing** |
| **P3** | **Identified, recorded, then lost** | Arc close folds the ***as-built*** result and archives the brainstorm. Nothing harvests what was decided-and-**not**-built | `k_drag`: correctly diagnosed, remedy named and sized, **zero hits in TODO.md or the ledger**. Plus 6 more instances; 217 `DEFERRED/PROVISIONAL/flagged pending` occurrences across 95 doc files |
| **P4** | **The good pattern has no home** — see §1(2) | Canon records *what the system does*, not *what technique the patch invented* | `_ep` solves dial drift structurally; the three solvers still on hand-written literals are exactly where every live drift is |
| **P5** | **Calibrated once, at a point the game no longer occupies** | The dependency link is one-way and in prose. The anchor value carries no back-pointer to what was solved against it | Fire calibration derived at `cool_shift = 9`, shipped at 5 → **16× off its own anchor**. Package-A sizing measured at zero air damping |
| **P6** | **The perimeter is nobody's system** | Digest gates compare arrays from *legal* calls. Nothing constructs an illegal call, so the boundary is never measured | `bindings.cpp`: zero shape/dtype/contiguity checks in 3141 lines. `config.py`: 126 lines, zero validation |
| **P7** | **Instruments are second-class** | Benches have no gate and no user but their author. A bench reading a wrong dial emits a *number*, which enters a design doc as measured fact | Erik's tuning loop cannot execute. `apply_overrides` non-atomic. No bench stamps provenance |
| **P8** | **Load-bearing prose nothing can falsify** | An invariant stated in prose is the one assertion that cannot go red — and agents reason from it as ground truth | `eos_solver.h:453` "replays **EXACTLY**" (false since B3c — and this is *how* P1's ambient gap survived) |

---

## 3. The rules — with honest scope

### Adopt now (three lines, ~15 minutes of Erik's time)

**R1 — Every gate ships with a demonstration that it can go red.**
> No gate is accepted until it has been *seen failing*. Parity/feature gates
> re-run with the feature omitted and assert the comparison **fails**
> (template: `cuda_combustion_check.py:397`). Scenario gates — goldens, benches,
> digests — assert mechanically that the scenario *contains* what is being gated
> (template: `cool_shift_axis_gate_a_capture.py:210,230`). Every
> `cuda_*_check.py` is referenced by some `test_*.py`, enforced by a meta-test.

- **Catches:** the empty golden; the two orphan gates; the set-vs-list
  blindness; the drifted P6.4 reference.
- **Misses, stated plainly:** **the storming.** Every single-room bench is
  non-vacuous — it measures a real thing on a real geometry. A failability rule
  cannot tell you your *scenario set* omits a geometry class. That is R2's job.
  Also misses all of P2 and P6.
- **Cost:** 20–40% more effort per new gate.
- **Home:** project CLAUDE.md iron rules **+ `14_determinism_and_number_ingress.md`**
  (per §1(2) — CLAUDE.md alone governs sessions, canon governs patches).
- **Why this one first:** both templates already exist in-repo. This is
  codification, not invention.

**R2 — A config default is never a hand-written literal; retiring a name
includes sweeping its consumers.**
> Every config read takes its default from the other side's definition — the
> C++ struct member (`_ep` idiom) or a named constant imported from its owner.
> If neither exists, the key is required and its absence raises. Retiring a key
> or symbol is not done until a repo-wide grep across
> `src/ cpp/ tools/ tests/ docs/ config.toml levels/` returns only the tombstone.

- **Catches:** `k_p` 0.0-vs-0.5 (water head silently off); `max_source_per_step`
  20×; the supply bench publishing measurements of a *retired* law; the ten
  `getattr(CFG.physics, "<section>", None)` sites; **and Erik's bricked tuning
  loop** — `k_fire_heat` was tombstoned correctly at the definition and left
  live in `tools/`.
- **Misses:** the Kelvin maps, the CPU/CUDA re-declarations, the golden.
- **Cost:** near zero (the `_ep` conversion is behaviour-preserving today);
  ~30 lines for a config schema. Real cost: a missing key now crashes instead of
  degrading — scope `_REQUIRED` to physics sections.
- **Home:** project CLAUDE.md, one line. **Best benefit-to-cost in the list.**

**R3 — A decision you do not build gets one grep-optimised line, today.**
> `docs/deferred_register.md`, one line per deferral, written **in the same
> commit that records the finding**:
> `[ID] date · SYMPTOM in the words a future person would type · diagnosis ·
> remedy named · where the detail lives · trigger to revisit`
> Arc close may not archive a brainstorm until every
> `DEFERRED/ACCEPTED GAP/PROVISIONAL/flagged pending/revisit` inside it has a
> register line. Any symptom investigation greps the register first.

- **Catches, as *retrieval*:** `k_drag` — on 2026-08-03,
  `grep -i "oscillat\|damp"` returns the July line and the evening is 20 minutes
  instead of a night. Also the `cool_shift = 9` re-tune note, and the test
  pinned to a config state that never arrived.
- **Misses — and this bound is important:** the blast-tuple wart **was** in
  TODO.md *and* the priority ledger since 2026-07-19, fully designed, and is
  still unbuilt. **A register buys retrieval, not execution.** Do not expect it
  to make work happen.
- **Design constraint that is the whole point:** symptom-first phrasing, in the
  words you would type when confused. Erik typed *"it oscillates"* — that word
  had to be in the index. One line, hard; its purpose is grep, not reading.
  Otherwise it becomes TODO.md (674 append-only lines, live items from March).
- **Cost:** one line per deferral, ~20 min per arc close.
- **Home:** project CLAUDE.md — amend the arc-close ritual from *"fold the
  as-built result"* to *"fold the as-built result **and harvest every not-built
  decision into the register**."* The one-line habit is universal enough for
  master CLAUDE.md too.

### Adopt with the fire work (they cost real effort)

**R4 — Calibrate on a shipped geometry, or write down what your fixture cannot
show.** A calibration bench must include one fixture structurally matching a
shipped level (multi-room, doored, shipped `tile_size_m`). Every bench artifact
carries a `BLIND:` header naming what its fixture cannot exhibit. A tuning
ruling may not cite a bench whose BLIND list includes the axis being ruled on.
- **Catches:** the storming, a month earlier — this is the *only* rule that
  does. Also package-A's silent dependence on zero air damping.
- **Misses:** not one finding in the code audit. This is pure physics
  methodology.
- **Cost:** the most expensive rule here — roughly doubles bench runtime and
  curves to read. Still second-most valuable, because it prevents the one
  failure that reached Erik's eyes.

**R5 — One owner per physical quantity; every constant names its anchor and its
dependents.** A conversion between game and physical units is defined once; a
second spelling is a bug even when the two agree (deliberate duplicates carry
`DELIBERATE DUPLICATE — <contract requiring it>`). A constant ships with an
anchor: unit · source · owner · the dial it was solved against. The anchor link
is **bidirectional** — the anchor carries a back-pointer to its dependents.
- **Catches:** both Kelvin maps; the `cool_shift` 9-vs-5 gap (the back-pointer
  puts the dependent list where the editor sees it); `tile_size_m = 1.0`; `dx`
  spelled two ways; `0.21` ×6; `T_MAX_PHYS` ×6.
- **Misses:** the `FP_ONE` reimplementations and the 13762 fixture — those are
  *rounding-convention* duplicates, not unit conversions.
- **⚠ The caveat that makes or breaks it:** canon ch.01:55-58 *already says*
  "a solver that assumes 1/3 m is silently wrong at any other resolution" — and
  a level shipped `1.0` anyway. **A units rule without a mechanical check is
  empirically a slogan in this repo.** So the rule is only adopted together with
  its gate: a ~10-line test recomputing the Kelvin map from config and comparing
  against the baked emissive table.
- **Cost:** half a day for the canon page, plus an anchor line per new constant.

**R6 — Every measured number carries its provenance.** Bench artifacts write
git commit + dirty flag, UTC timestamp, fixture id, and the full override dict.
A number quoted in a design doc cites the artifact that produced it.
- **Catches:** the supply bench measuring a retired law; `burn_rate = 1.0` vs
  shipped 0.02; `OPERATING_POINT_I = 0.192`'s untracked-CSV origin.
- **Misses:** any correctness defect — this is a *falsifiability* rule, not a
  bug rule. Which, in a PhD context, is the point.
- **Cost:** one shared helper, ~3 lines per bench. Trivial.
- **Home:** a new project skill `physics-bench`, together with R4's BLIND header.

---

## 4. What we should copy rather than invent

Four techniques in this repo already solve a whole pattern each, and each stayed
local. **Promoting them is cheaper than any new practice** — and per §1(2), the
promotion must be into a file agents are required to read.

| Technique | Solves | Where it must go |
|---|---|---|
| Non-vacuousness controls (≥6 CUDA gates) — *and the scenario form also already exists* in `cool_shift_axis_gate_a_capture.py:210,230` | P1 entirely. **Both halves are written; neither reached the golden** | R1, verbatim, with both templates named |
| `_ep`/`_cp` — the C++ struct member IS the Python fallback | P4/dial drift, structurally | R2 + do the three-solver conversion |
| `RC_HD` host/device shared helpers; `cudaMemcpyToSymbol` canonical tables (*"transcription drift is therefore structurally impossible"*) | the CPU/CUDA transcription seam | R5 clause 1, **with the DELIBERATE-DUPLICATE label** — without it the rule over-fires and someone "fixes" the tol-0 contract |
| Tombstones at the deletion site (12 sites in `src/`, `cpp/`, `tests/` — **zero in `tools/`**) | dead-vs-scaffolding | R2's sweep clause. The grade gap matches exactly: `tools/` B− against A−/B+ elsewhere |

---

## 5. Rules deliberately rejected

- **"Validate inputs at boundaries."** Catches all of P6 in principle, nothing
  in practice — the fix is one `require_2d()` helper. That is a *task*, not a
  rule.
- **"Keep comments in sync with code."** Unenforceable. Only the narrow form
  bites — *an invariant a comment asserts is a `static_assert`, a runtime check,
  or explicitly labelled `PROSE-ONLY:`* — and that is folded into R1.
- **"Add CI."** ⚠ Flagged as cost-exceeding-benefit *as currently framed*: there
  is no CUDA runner, so CI would go green on the CPU half while all 22 CUDA
  gates skip — **manufacturing a fresh, institutionalised instance of P1.**
  Either get a GPU runner or do not call it a gate.

---

## 6. The cross-repo idea

Erik has twelve projects. Worth knowing before spending time: **none of the
eight patterns is Breach-specific.** P1, P3, P4, P6, P7, P8 are universal to any
codebase. P5 (calibrated at an operating point the shipped system no longer
occupies) and half of P2 are specific to *tuned simulation* work — which means
**they transfer directly to `pkpd-tools` and `hes`**, where a parameter fitted at
one operating point and shipped at another is the same bug with a publication
attached. That is probably where the second audit belongs, not in another game.

For results to be **comparable** across repos, a `codebase-audit` skill must fix
five things:

1. **The area split, defined by gate coverage, not language or folder** — (a)
   code under an automated oracle, (b) under a human-eye gate only, (c) under no
   gate, (d) the measurement/tooling layer, (e) the boundaries between (a) and
   everything else. A folder split would have confirmed Erik's wrong hypothesis.
2. **A grading rubric with written anchors**, every grade justified by ≥3
   `file:line` citations — otherwise "B+" compares two auditors' moods.
3. **A machine-comparable finding schema**, with `pattern_id` drawn from the
   fixed P1–P8 taxonomy. ★ **The taxonomy is the actual cross-repo product** —
   counts-per-pattern-per-repo is the only number that means anything across
   projects. Prose summaries are not comparable at all.
4. **A required input: "what have the last three arcs been changing?"** The
   biggest finding here — an empty golden during a month of fire work — is only
   visible to an auditor who knows what the project has been changing.
5. **Two mandatory sections: "what is genuinely good — do not clean this up"**
   (an audit without it is a hazard; §3 of the audit explains why) and **"what
   this audit could not see"** — R1 applied to the audit itself, and the only
   thing that makes two audits' *silences* comparable.

Plus a required follow-on artifact: the **handover brief**
(`audit_handover_patch_a_2026-08-04.md`) — the thing that converts an audit into
work, with an explicit "these need the human" list.

**One caution:** this audit's value came substantially from six parallel passes
by a reader who understood the physics. Before comparing repos, re-run the skill
on **Breach** — you now know the answers, so it is the calibration case, and any
spec that fails to re-surface the empty golden and the two Kelvin maps is
under-specified.
