# Sky-exchange P3 — τ sweep + gate-d notes (2026-07-24, Opus build)

Handoff for the JOINT fire re-tune (`sky_exchange_design_2026-07-24.md` §5 /
`fire_tuning_plan_2026-07-22.md` §8 item 4). Produced by `tools/sky_exchange_bench.py`
on the `sky-exchange` branch (CPU build). Append-only capture, not canon.

## The bench

40×84 sponge-safe planetside room, a central 4×4 O2 sink (a *composition-only*
combustion proxy: remove O2, bank inert, N_total conserved), 180 s burn + 180 s
recovery. Probe = mean O2 mole fraction over the 14×14 halo around the sink (the
locally depleted zone). Full engine tick each step (`PhysicsRunner.step`, so the
sky pass runs where it does in the game).

## Result (the τ menu)

| config  | far0  | min_far (burn) | end_burn | end_recover | τ_recover |
|---------|-------|----------------|----------|-------------|-----------|
| sky OFF | 0.210 | 0.193          | 0.193    | **0.193**   | — (never) |
| τ = 30  | 0.210 | 0.193          | 0.193    | 0.210       | **30 s**  |
| τ = 60  | 0.210 | 0.193          | 0.193    | 0.209       | **59 s**  |
| τ = 120 | 0.210 | 0.193          | 0.193    | 0.206       | **120 s** |

**Reading:**
1. **Sky OFF the depleted halo does not recover** (stuck at 0.193) — the edge
   reservoir cannot refill a volumetric deficit. This is the mechanism the
   feature exists to fix.
2. **Sky ON recovers toward ambient with τ_recover ≈ τ** (30 / 59 / 120 s) — the
   design's "returns … with time-constant ≈ τ", confirmed end-to-end in the full
   engine.
3. **The recovery CEILING falls as τ grows** (0.210 → 0.209 → 0.206). This is the
   P1 round-to-nearest **deadband** (≈ 0.5·N_total/λ counts) made visible: a
   bigger τ = smaller λ = bigger deadband = rests further below 0.21. Unit-test
   estimate of the resting far-field fraction: τ=30 ≈ 0.203, τ=60 ≈ 0.199,
   τ=120 ≈ 0.188 (**under the 0.19 floor** — τ=120 is likely too slow).

## τ recommendation for the re-tune

**τ = 60 s** is the sweet spot on this evidence: recovery in ~1 min (fast enough
to matter across a burn), deadband small enough that the resting field stays
≥ 0.19. τ = 30 recovers faster and rests higher (0.203) if the re-tune wants more
headroom; τ = 120 is deprecated by the deadband (rests ~0.188). Erik blesses the
final value at the joint re-tune.

## What this bench does NOT show (deliberately)

The whole-field far-field **suffocation** of §7 Q2 (0.21 → 0 over ~5 min) is
driven by the fire's **heat → pressure → outward-wind** coupling — combustion
raises P = C·N·T, bulk wind points outward the whole burn, and O2 can only
diffuse in against it (§7 Q2 point 2). A composition-only sink has no pressure
footprint, so it can't reproduce that symptom; the near-field diffuses back fine.
The **full gate-d acceptance** (locked-combo crate burn, far field ≥ 0.19 for the
whole burn) therefore runs at the **joint re-tune** with the fire-tuning harness
+ the locked fire dials — exactly where the design (§5) places it. This bench's
job is narrower and done: prove the pass is wired end-to-end and that its
replenishment dynamics match τ.

## Smoke sanity (design P3)

Smoke is untouched by this build (the conservative O2/inert pair only; smoke's
sky-λ is deferred). Confirmed indirectly: `sky_tau_s` defaults to 0 (dormant), so
every existing level — including `planetside_demo` — is byte-identical (full
suite 1679 passed, no golden/digest moves). No smoke plane is read or written by
`sky_exchange_step`.
