# Deep-research workflow — cost post-mortem (2026-07-08)

> Recorded at Erik's request. The EOS lit-search (`docs/eos_research_report.md`) **exhausted the
> 5-hour token window TWICE** and never produced its own verified+synthesised report — the tail
> was finished in-thread by Opus instead. Lit-searches are extremely valuable to Erik AND
> extremely expensive; he is already on the top tier (180 EUR/mo), so the fix is **process, not
> plan size**. This doc is the evidence + concrete refinement proposals, to return to later.

## What actually ran (measured)

| Run | Agents done / errored | Subagent tokens | Tool uses | Wall clock | Died on | Produced |
|---|---|---|---|---|---|---|
| 1 `wzvkg55se` | 17 / 87 | **772,897** | 162 | ~11.0 min | session limit (10:10) | scope+search+fetch+claims; **0/25 verified** |
| 2 `wg6mb8a0z` (resume) | 36 / 69 | **1,689,020** | 411 | ~9.4 min | session limit (16:30) → **out of credits** | cached prefix; **2/25 verified**; synth FAILED |
| **Total** | — | **~2.46M** | ~573 | ~20 min | — | claims + 2 confirmed; report done in-thread |

The searches (5), scope (1), and fetches (23 sources → 112 claims) completed and **cache-replayed
for free** on the resume. What repeatedly died is the **tail: verification + synthesis** — which
is also **the whole point of the exercise** (the adversarial fact-check is *why* you run this
rather than just asking a model).

## Root cause — the cost is back-loaded, and the window can't hold it

Per-phase token estimate (from this run's per-agent data — one fetch/extract agent measured at
**57,780 tokens / 15 tool calls / 291 s**):

| Phase | Rough cost | Notes |
|---|---|---|
| Scope | ~5 k | one agent, cheap |
| Search (×5) | ~50 k | cheap, cache-replays |
| **Fetch/extract (×23)** | **~1.0–1.3 M** | ~40–58 k *each*; the first big block |
| **Verify (25 claims × 3 votes = 75 agents)** | **~1.5–2.2 M** | the biggest block, each re-reads a source, **and it runs LAST** |
| Synthesize | ~50 k | also last |

So an uninterrupted run is **~3 M tokens** — more than a single 5-h window holds at this tier.
**Fetch alone (~1.2 M) plus a partial verify is enough to exhaust the window before verify
finishes.** Resume-from-cache replays the finished prefix for free, but the *tail never gets a
fresh full window to itself*, so it starves every time. The three worst properties stack on the
same phase: verification is **most valuable + most expensive + dead last**.

## Refinement proposals (for a future session)

1. **Partition into separately-invoked stages with a disk checkpoint between them.**
   Stage 1 (scope+search+**fetch**) writes a `*_claims.json` + fetched **source text** to disk.
   Stage 2 (verify) reads that artifact. Stage 3 (synthesise). Each is a bounded, restartable unit
   with a known envelope; a verify failure never re-pays the 1.2 M fetch, and the tail can be run
   in its *own* fresh window.
2. **Budget by expectation value up front.** Print the estimate table above before launching, and
   if the total exceeds one window, force partitioning (or shrink) rather than launching a run that
   *cannot* finish.
3. **Shrink the verify block — it's the cost driver.** Any of: top **8–10** claims not top 25;
   **2 votes** (or 1 skeptical vote) not 3; a **cheaper verifier model (Haiku)** — verification is a
   bounded judgement task. Combined, this could cut ~1.9 M → ~0.3–0.5 M.
4. **Cache fetched source TEXT, not just extracted claims.** Verifiers are ~25 k each largely
   because they re-fetch; reading local cached text would collapse that.
5. **Make in-thread synthesis the default tail (this run's accidental success).** Have the workflow
   STOP after the claims+sources artifact and let the **main-thread model do verify+synthesise with
   domain judgment**. For well-known-literature questions (this one: Kwatra / Feldman-O'Brien /
   Nguyen-Fedkiw), the main model can verify against what it knows *better* than generic rate-limited
   verifier agents — and far cheaper. The report we shipped is proof it works.

## Silver lining

The failure was **infrastructure (rate/credit limits), not research quality** — 0 claims were
refuted, the 2 load-bearing claims *were* adversarially confirmed, and the fetched primary sources
were all high quality. So the fix is tractable: checkpoint the expensive middle, shrink or reroute
the tail. That's a much better failure mode than "the search found garbage."
