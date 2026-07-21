# Breach — project instructions

Breach is an ML/RL project wearing a game's clothes: a deterministic GPU physics
engine as a state space, then agents trained in it for emergent strategy. Every
rule below serves that.

## Orient first

1. `docs/priority_ledger.md` — what we're working on and in what order.
2. `docs/architecture/README.md` — how the engine works (canon chapters).
3. `docs/TODO.md` — open items; git history has what's done.

Doc culture: `docs/architecture/` chapters are **canon, live-edited**; everything
else in `docs/` is **append-only capture** (dated notes, patch docs, specs) — add
new dated docs, don't rewrite old ones. At the close of every arc, fold the
as-built result into the canon chapters and archive the brainstorms
(`docs/archive/`).

## Environment (machine-agnostic — specifics live in docs/dev_setup.md + docs/lenovo_dev_setup.md)

- **Python: always the conda env `data`** (same env name on all dev machines;
  verified desktop + Lenovo). Bare `python` is a different install and fails on
  breach imports with a misleading ModuleNotFoundError. Run via the env's
  python or `conda run -n data python ...`.
- **pytest: always `pytest tests -q`** — never bare `pytest` from repo root.
- C++/CUDA: `cpp/` via CMake; per-machine build scripts are `cpp/build_cuda*.bat`.
- Do NOT add machine specs (paths, GPU models, toolchains) to this file — they
  belong in the per-machine dev-setup docs.

## Iron rules

- **Determinism is a hard requirement** (multiplayer + distributed training +
  portfolio). Synced sim state is Q16.16 integer only: no floats, no libm
  transcendentals in the sim path — use `cpp/src/fixed_point.h` (incl.
  `atan2_q16`/`sin_q16`/`cos_q16`). Digest/golden gates guard this; goldens are
  re-baselined only deliberately, once per approved behavioral change, with
  written rationale. Render-layer code is exempt.
- **Never `git add -A`** — the tree carries untracked art, notes, and prototypes
  on purpose. Stage explicit paths.
- **Feel-adjacent changes never auto-merge.** Anything touching game feel
  (weapons tuning, physics behavior, visuals) gets a HUMAN-TEST gate: built,
  gated, pushed, then Erik plays it before merge. Mechanical digest-gated
  changes may auto-merge on green only when that's been pre-authorized.
- **Credit the source**: any file implementing a published technique carries an
  author + paper citation in its header comment; archive the paper under
  `docs/papers/`.

## Working style

- Big changes run as arcs: design doc → adversarial critique → patches with
  gates → CUDA lockstep → canon fold at close.
- Repo hygiene: delete merged branches and finished worktrees (local + remote);
  never touch Erik's parked branches (check the ledger/TODO for which).
- Commit design docs to the branch BEFORE spawning worktree agents that depend
  on them — agents can't see your uncommitted working tree.
