# Fire realism design — ROUND 3 (final gate) verdicts (2026-08-02)

Two reviewers on v4 (post Erik's Q-0/2b blessing). Outcome: **direction faithful,
mechanics not yet buildable — v5 closure required (written same day, see the v5
block in the design doc). Both endorse committing the document set as the dated
record.**

## Physics+determinism lens — verdict: BUILDABLE: NO (4 blockers)

T1's four limits RE-VERIFIED (ambient/equal-T/partial/hot-receiver all hold for
non-emitting receivers; self-cell exclusion confirmed mandatory against the
actual march). Blockers:
- **F1 — mutual-emitter 2×:** T1's pair is a net term containing the receiver's
  back-emission — correct only when r does not cast. When BOTH ends are
  emitters, each pays its own sink AND eats the other's net pair ⇒ 2× exchange,
  with a rate DOUBLING at T_emit_gate (R2-3's cliff reborn). Present in the
  SHIPPED law too; load-bearing now that flashover = warm-crate radiation.
  Fix: emitter-membership mask — r non-emitter: T1 unchanged; r emitter:
  one-way pair a_s·a_r·τ·w·(E_emit(s) − E°[0]), NO credit at s (s's inflow
  arrives on r's own cast). Gate (iv) restated with ΔT (equal-T is blind).
- **F5 — 2b deposit site vs ruling (4):** Pass A deposits heat+soot at the
  donor air cell; radius-3 draw would heat/smoke every cell within 3 tiles —
  the vicinity ruling forbids exactly this. 2b must re-site the deposit to the
  fire's own tile + faces via a second gather (single-writer preserved).
- **F11 — suppressed conduction interface has no refund:** with crate κ > 0
  (ruling 3) the suppression fires on every crate contact; keep-credit ⇒
  energy created, drop-both ⇒ destroyed (the Ω-sink still charges that
  direction). Fix: refund the emitter its own direction share
  a_s·τ·w·(E_emit − E°[0]) and terminate the ray (distinct from F1(c)'s
  cavity refund).
- **F13 — no 2b patch exists in §5.**
Majors: F2 clamp/ledger bookkeeping (fix: book post-clamp integers at grid AND
ledger sites — conservation structural); F3 cavity closure misses range/cull
termination (sealed rooms wider than max_range shred energy — sealed
equal-T-room gate case + range floor or direction-share refund on range
termination); F4 φ/flame_lift alive-but-unruled under the 2b package (source-
cell exclusion must cover the PAIR too; gate (iii) fire-free); F6 alloc_face
4-plane doesn't generalize (offset-keyed plane ~24·n or symmetric re-walk;
MAX_CLAIMANTS cap + hard assert; CUDA register spill priced); F7 the o2f
SENSOR stays radius-1 (knee/smother semantics preserved — must be stated as
the chosen lump; the supply curve measures delivery, not I); F8 "connected
open" must be permeability-multiplicative (else the draw passes through crate
stacks unattenuated, defeating gate (viii)); F9 distance-weight form IS the
supply multiplier (uniform ≈6×, 1/d ≈3×, 1/d² ≈1.8× at r=3) — baked integer
table by BFS hop, radius+table = new authored dials for §9; F10 the
supply-vs-radius sweep must run INSIDE the 2b patch (P-F4a → P-O2b → P-F4b);
F12 stale R1/Q-0 language sweep; F14 dem_acc stale-debt ALREADY FIXED in tree
(drop from P11); F15 the hysteresis phrase = an unpriced persistent latch
(delete; single threshold + boundary gate).

## Intent-fidelity lens — verdict: FAITHFUL: NO (direction yes, coverage no)

All major rulings present and correct. Three blockers: **B1** 2b blessed but
unspecified (same as F13/F5-F9); **B2** per-family burn durations have no
mechanism — fuel IS combat hp today; a 30-min crate = 4.5× combat hp, an
unblessed balance change → v5 adopts a per-material FUEL column decoupled from
combat hp (recommended to Erik, veto open); **B3** the campfire-scale
reference object (1–3 kg class) exists nowhere — material row + tile id +
bench scenario land in P-F4a; oracles re-pointed at it. Five annotations had
no disposition (now folded): the constant heat-split blessing (named lump),
the T_emit_gate raise lean (tuning-session item with its accuracy cost), the
Problem-5 arena replan (STILL-AIR REFERENCE + FORCED-WIND become named F4
modes with tuned-parameter lists and a literature slot), the smoulder
threshold question (answered in the ember charter), decision-4's reframing
(room-feel targets = damage-onset curves). M8 the fizzle gate contradicts
crate conduction (fizzle case re-sited to SPACED crates; touching 2×2 now
expected to spread 30–60 s/hop and die by knee at the cluster edge); M9 the
blessed-shape oracle re-pointed (reference object; hard death-by-knee on the
reference only; per-family monotone elsewhere; 61.7% = historical label;
timed AFTER 2b); M10 kill "cellulosic band ≡ lone-crate band" (band per
material); M11 state fuel-energy ≠ nominal mass (30 kg crate ≈ effective
50 MJ ≈ 3 kg burn) + the PERIMETER-scaling delivery prediction as the bench
falsifier; M12 rebuild the touchpoint table (+ the radius/power sizing call,
2b design-gate, ember review; restore ε feel + ~5 play tests); M13 pore gas =
bookkeeping over open faces (ruled, no new state); m14–m24 small captures
(15–30 vs 30–60 note — the §5 answer governs; "hot spreads faster" =
expectation not gate; ships-O₂-is-map-design scope ruling recorded — the
reason 2b is the ONLY supply term; furniture→crate rename slotted in P-M1;
decision-5's unit-temperature half + destination named; flicker = a scalable
feel dial; the smoke slow-tick idea filed with the R3-future folder; req-1/9
rephrasings added to plain §10). Corrected patch order (Q4):
**P-F4a → P-O2b → P-F4b (supply sweep + smother check) → Erik: radius+power
sizing → P-F1a/b → P-F3+F-BO (+forced-wind level) → P-EMBER → P-F5′ → P-M1
(classes+doors+rename) → P-R5 → P-PAYLOAD (molotov+grenades) → fold + the
single arc rebase.**
