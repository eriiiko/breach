# SEED — doc-freshness policy + staleness sweep (2026-07-30)

**For: a fresh session.** Written at Erik's request ("i fear lots of docs are stale now …
i think we need some kind of project policy of how to keep our docs up to date as well").
Self-contained.

**This is a process/tooling task, not a physics task.** Keep it in its own session — it
has nothing to do with the fire work and will only dilute both.

---

## 1. The trigger: a concrete, measured failure

On 2026-07-30 a blessed design doc (`docs/thermal_mass_axis_design_2026-07-25.md`) was
handed to a build session. Its **§1 was titled "Current state (file:line, verified
2026-07-25 by the read-only agent)"** — and it was wrong. It described the codebase as it
existed *before* commit `97b3de8` (2026-07-08), three weeks earlier:

- It said "add a per-material `thermal_mass`" — the column **already existed**.
- It said "the value IS the convert divisor (today's global 8)" — the convert was
  **already per-tile** via `heat_inv_shift`.
- It prescribed "every currently-solid material = 8" — actual values were hull/steel
  **32**, glass **16**. Following it literally would have **broken the design's own
  byte-identity gate**.
- Its blessed predicate `thermal_mass > 0` was **unsatisfiable** as written, because air
  was 1 and the loader rejected 0.

None of this was caught by review; it was caught by an implementer reading the code. The
cost was contained (a build addendum, one patch cycle) but it is exactly the failure mode
that gets expensive when nobody checks.

**The deeper instance in the same arc:** the doc's own instruction was *"grep first; the
agent found none, but **verify**"* — and a grep was treated as the verification. It
structurally could not see the file it needed to (mask definition and field writes
hundreds of lines apart). Two more `temperature[]` writers were found later, one by the
design pass and one by the build. See `docs/thermal_mass_eos_escalation_2026-07-30.md` §7
and `docs/thermal_mass_eos_ruling_2026-07-30.md` §5.

## 2. The finding to start from: policy exists, enforcement does not

`CLAUDE.md` **already states the doc culture**:

> `docs/architecture/` chapters are **canon, live-edited**; everything else in `docs/` is
> **append-only capture** (dated notes, patch docs, specs) — add new dated docs, don't
> rewrite old ones. At the close of every arc, fold the as-built result into the canon
> chapters and archive the brainstorms (`docs/archive/`).

So the gap is **not a missing policy**. It is that (a) nothing verifies a doc's "current
state" claims against the code, (b) the arc-close fold is easy to skip when an arc ends in
a play-test rather than a merge, and (c) a dated capture doc has no way to say "this was
true on date X and is now superseded by Y" other than a reader happening to know.

Design for that, rather than writing the policy again.

## 3. Known-stale items (a starting inventory, not exhaustive)

**Canon holes from the thermal-mass arc** — 15 items enumerated in
`docs/thermal_mass_axis_bench_report_2026-07-30.md` §8, across `engine/06`, `/03`, `/02`.
Headlines:
- `engine/06` still documents **`solid`** as the thermal medium key. It is now
  `thermal_solid` (`thermal_mass > 0`).
- **`thermal_mass` and `heat_inv_shift` are documented NOWHERE** — not in canon at all,
  despite existing since 2026-07-08 and now being load-bearing.
- The **ownership rule** ("on `thermal_solid` tiles `temperature[]` is owned by the
  TemperatureSolver; every other system is a reader") is a ruling, not yet canon.
- The **combustion object-path deposit** and the **T-only EOS occluder** are as-built and
  undocumented.
- A per-material **`cool_shift`** axis is landing now and will need the same treatment.

**Deliberately deferred, not forgotten:** that fold waits until Erik play-tests and merges
`thermal-mass-axis` — folding unmerged behaviour into live canon inverts the order. **A
policy should say what to do in this gap**, because "done but unmerged" is where canon
rot actually starts.

**Other suspected staleness** (unverified — the sweep should check, not assume): the
untracked dated docs in `docs/` accumulated over several arcs; `docs/TODO.md` vs git
history; `docs/priority_ledger.md` currency; whether retired mechanisms (e.g. `is_wall`,
`o2_threshold`, `P_min` — config marks several dials "RETIRED") are still described as
live anywhere.

## 4. What the session should produce

1. **A staleness audit** — sweep `docs/`, classify each doc: canon / live capture /
   superseded / archive-now. Verify canon's factual claims against code where they are
   checkable (esp. field names, masks, dial names, file:line references).
2. **A doc-freshness policy** — added to `CLAUDE.md` (concise) and/or a canon meta-chapter.
   Questions it must answer: how does a dated capture doc get marked superseded? What is
   the arc-close checklist, and what happens when an arc ends unmerged? Who/what verifies a
   "current state" section — and what counts as verification (the §6 lesson: a grep is
   not)? Should design docs cite the commit SHA they were verified against, so drift is
   detectable rather than invisible? (That one change would have caught this arc's failure
   outright and is cheap.)
3. **Something checkable, if cheap.** Ideas, to be weighed not assumed: a doc header with
   `verified-against: <sha>`; a test or script that flags canon referencing identifiers
   that no longer exist; a pre-arc-close checklist in the autonomous-patch-workflow skill.
   **Bias to the simplest honest mechanism** — for a single-author project, a documented
   convention plus one cheap check beats machinery.
4. **Execute the archival part** of the existing policy where it is unambiguous (move
   closed-arc brainstorms to `docs/archive/`).

## 5. Constraints

- `docs/architecture/` is **canon, live-edited** — it may be rewritten. Everything else in
  `docs/` is **append-only capture**: add new dated docs, do not rewrite old ones. A
  staleness sweep marks and supersedes; it does not silently rewrite history.
- **Do not fold the thermal-mass arc into canon in this session** unless Erik has merged
  `thermal-mass-axis` by then — check `git log main` first. The fold list is ready in the
  bench report §8.
- Many `docs/*.md` files are **untracked on purpose** (Erik's working notes). Do not
  bulk-commit them; ask before tracking anything that is currently untracked.
- **Never `git add -A`** — the tree carries untracked art, notes and prototypes
  deliberately.
- Never quote sensitive personal files into shared docs (see `CLAUDE.md`).

## 6. One lesson worth writing into the policy verbatim

**For a routing or ownership question, verify by enumerating writers of the field — not by
grepping the mask name near topic keywords.** A line-oriented grep cannot see a file where
the mask definition and the field writes are far apart. This is now standing practice
(`docs/thermal_mass_eos_ruling_2026-07-30.md` §5) and generalises past code: *a search that
confirms your expectation is not a verification of it.*
