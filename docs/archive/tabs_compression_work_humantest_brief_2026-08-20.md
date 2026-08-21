# T_abs compression work — HUMAN-TEST brief (P-W3, 2026-08-20)

**Arc:** `tabs-compression-work`. For Erik, before playing at P-W3. Sources:
design §8 seeds (`docs/tabs_compression_work_design_2026-08-20.md`), the
P-W1b manifest's two open findings
(`docs/tabs_compression_work_manifest_pw1b_2026-08-20.md`), and this patch's
own measurements (`docs/tabs_compression_work_pw2_measurements_2026-08-20.md`
— cited by section below). **Feel-adjacent: nothing merges before you play.**

## What changed, in one line

Step 4c's compression/expansion arithmetic now runs on absolute temperature
(`T + 290`) instead of ambient-relative `T`. Ambient air (T_rel = 0) used to
be a fixed point of compression work — no heat signal near ambient, ever —
and cold gas below ambient INVERTED under compression (colder gas cooled
further, the cold-rail spiral). Both are fixed. Nothing else moved: same
rails, same counters, same clamp values.

## Play list

**1. Breach venting — cold as a RING, not a core.** Vent a sealed room to
space. Expect the coldest gas at the WORK CLAMP figure, −96.67 game-deg
(290 K → 193.3 K at w = 0.5 — the scheme's number; exact adiabatic would be
~−114, the ~15% gap is a deliberate first-order form, not a bug). You will
NOT see this at the vent core — the P-E4 trust gate fades compression work
to zero below 12.5% of ambient density, and a hard-vented core is thinner
than that. The cold reads as a RING where the wake still holds ≥12.5%
ambient density. This is correct bookkeeping (no temperature unbacked by
energy); `n_work_ref` is the knob if you want the cold to reach deeper.

**2. How long cold lingers.** Two clocks: advective mixing (fast — the
refill flow itself clears it) and conduction (slow floor — harmonic-mean κ
gives ≈43 s against a hull face, ≈21 s laterally through air). Expect fast
fade where air is actively flowing back in, and cold lingering tens of
seconds in stagnant corners the refill doesn't reach.

**3. Breach-mouth flashes — the B-F7 pressure-collapse route.** The genuinely
new hazard this patch opens: a cell driven near the T_MIN floor (t_abs → 1 K)
loses up to ×290 of pressure beyond its mass loss (`p* = C·N·T_abs`), and a
near-zero-p* cell beside normal ones is a giant pressure gradient → velocity
spike. **Measured on a 1500-tick sustained venting bench (§6 of the
measurements doc): NOT OBSERVED.** Cells got close to the floor (min t_abs
6.81 K, ~7x the 1 K floor) but min pressure stayed within a few×10⁻⁴ atm of
zero — nowhere near the deep-negative signature of the old flash class. Max
Mach vs a cell's OWN local sound speed reached 2.99, but only ~0.1% of
open cell-snapshots exceeded Mach 1, and the counted velocity clamp
(`u_clamp_hits` = 327) handled them. **This is a measured "no" for this one
geometry and 1500 ticks, not a general clearance** — watch for it anyway;
D-6 (raise T_MIN) is the cheap lever if you find it where this bench didn't.

**4. Grenade compression warmth.** A blast compresses the air around it;
expect the interior to read honestly warmer post-blast where the old law
would have shown nothing (ambient's old fixed point).

**5. Warm, pressurized rooms after violent transients.** Design §0b R-4
(water-displacement bench): a flood transient's compression work honestly
warms a sealed room — measured +23.5 game-deg mean, settling **+0.070 atm**
above ambient (spatial spread ≤0.0001 atm — reaches a single flat level, not
a lumpy mess). This generalizes past water: any violent transient in a
sealed room should now leave it slightly warm and slightly over-pressure.
Gameplay-visible, honest physics — not a bug if you notice it.

**6. Fire sanity.** The hot rail now has an entry point from ambient itself
(sustained compression from a seed of T_rel = 0 reaches T_MAX_PHYS in ~10
rail ticks, vs needing an already-hot seed before). Gates held green through
P-W1b (hot-rail HOT scenario: 4 ceiling hits over 2000 ticks, one 13-tick
transient climb, equilibrium ~5342 game-deg — well under the 16000 ceiling).
Does fire still feel right, or does warmed ambient air read as tipping
things too easily?

**7. The quiet-room drift number — READ THIS ONE.** A sealed room, quiet, no
fire, just a one-time pressure bump: the net thermal mean should sit near
zero forever. It does not, quite. **Measured at 2000 ticks: +4.646 game-deg
net drift** (design's own founding measurement cited +0.004 — that number
does not reproduce on this recipe; see the manifest's §6 for the full
mechanism note. Sign, order of magnitude, and mechanism match the design's
own named RISK-2 shape-asymmetry term). **P-W2 extended this to 10000 ticks
(~7 real minutes at 24 tps) and the drift does NOT saturate** — it grows
close to linearly, decelerating only ~23% in slope over that horizon, and
reaches **+10.39 game-deg by tick 10000** — past the mint-guard gate's own
`≤10.0` budget (which was sized from the 2000-tick number with "2x
headroom"). No gate in the suite currently reds on this (the committed gate
only runs 2000 ticks), but a sealed room left alone for several minutes will
keep drifting warm past what the current gate's margin implies. This is the
single most load-bearing number in this brief for any drift-tolerance
decision.

**8. The two render instruments — look, provisional.** (a) **Hover readout**:
hold the cursor over any tile (F6, the existing debug-coords toggle) — shows
T in game-deg + pseudo-Kelvin + fire/O2/trace-gas densities under the
tile/material line. (b) **Cold tier**: T_rel < 0 now shows as a diverging
blue tint under the heat glow (T key, same toggle as the heat overlay).
Placeholder colors/thresholds — judge the LOOK, not just whether it's there.
**Kelvin-frame trap:** trust the readout's game-deg number sub-ambient, not
Kelvin — the canonical render map (K = 293 + 3·T_game) reads −96.67 game-deg
as "3 K" (misleadingly near absolute zero) when the EOS's own frame says
193 K.

## Two decisions for you, with P-W2's data

### D-1 — keep the ambient cap²-plane floor (c ≥ c(T_abs = 1 K) ≈ 17.6 m/s)?

**Question:** the velocity-clamp arc's cap law has an ambient floor built
in — a cold cell's sound-speed cap never drops below what T_abs = 1 K (the
T_MIN floor) gives. Should this arc touch it?

**Measured data:** venting-bench mach census (measurements doc §6) — 2136
peak sub-ambient open cells, min T −283 game-deg, max Mach 2.99 vs a cell's
OWN sound speed, min P only −0.00026 atm (no collapse observed). The design
argued (§4 D-1) that a colder cap would throttle breach venting to a crawl
(the k_drag=10 probe already showed killing venting is a bad feel outcome),
and that at shipped scale the ambient cap already runs far over resolvable
Courant during blasts (owned by N_SUB_MAX, separately ruled to stay 8) — so
a cold pocket at the ambient cap is not distinguishable from any other cell
numerically.

**Options:** (a) keep the floor as-is (design's own recommendation, not
re-argued here); (b) lower the floor's temperature reference if venting
still feels too crawly; (c) revisit only if a future scenario DOES show the
B-F7 flash this patch's one venting bench did not.

### D-6 — is T_MIN = −289 (t_abs floor 1 K) still the right value?

**Question:** now that the floor is reachable (it was structurally dead
under the old relative law), is its depth right?

**Measured data:** the venting bench got cells to t_abs = 6.81 K — about 7x
above the floor — without a flash. §3's estimate: a cell needs
`ln(290)/ln(1.5) ≈ 14` sustained rail ticks (~0.6 s) to go from ambient to
the floor under continuous expansion. Raising T_MIN (e.g. to 30 K) would cap
the worst-case pressure collapse at ×10 instead of ×290 — a cheaper lever
than touching the cap plane (D-1) if flashes DO show up in play that this
bench's one geometry didn't surface.

**Options:** (a) keep T_MIN = −289 (no measured need to change it yet — this
patch's one bench found no flash); (b) raise it now as a preemptive
cushion if you'd rather not wait for a flash to show up in play; (c) leave
it exactly where the design left it — on the table, revisited if P-W3 play
finds a flash the bench didn't.

## Also worth knowing (not a decision, just named)

- **The vac/ring creation channel (§5 of the measurements doc) was never
  exercised.** `e_vac_wipe_sum`/`e_ring_pin_sum` measured exactly 0 in every
  P-W2 scenario, including 1500 ticks of sustained venting — but that is
  because none of the measured scenarios include a mid-run wall break (a
  live `destroy_wall` that exposes previously-warm/cold gas to vacuum mid-
  flight). The ≤290·N_vented/tick bound (design §3) is a design-time
  argument, not yet an observed one. If you break walls mid-fight during
  play and something reads oddly hot/cold right at the breach moment, this
  channel is the first suspect.
- **The XPASS finding** (P-W1b manifest §5):
  `test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood`
  now passes when it's marked `xfail(strict=True)` — the air-separated plank
  reaches ignition temperature. This was already a pre-existing baseline red
  (a `MaterialTable.ignition_to_ext_delta` drift bug is the suspected root
  cause), and whether warmer ambient air under this arc's law is what tipped
  it was not disentangled — worth a look at fire/materials triage, not
  blocking this arc's close.
