# All-systems scenario harness — spec + first findings (2026-08-27)

Erik's spec (evening session): one deterministic playground run where every
major system fires — water tanks, fire, blasts, hull breach, decompression —
with per-property PASS/FAIL so "does the engine behave?" becomes automatable.
Harness: `tests/_scenario_all_systems_bench.py` (headless, seed 1, 45 s sim,
vents stripped in-memory like the #54 hot-plate bench). This doc is the
capture of the design + the first run's findings.

**Superseded by arc #54 (gas-energy conservation, 2026-08-30)**: the
`k_drag_heat_frac` dial (line 114), `T_WORK_CLAMP` (line 148), and
"compression work" (step 4c, line 129) this capture measured are all
RETIRED — replaced by the derived `k_ke` constant and the conservative
face-flux energy step, `docs/gas_energy_conservation_design_2026-08-29.md`.
The scenario's own PASS/FAIL findings for the water/blast/breach properties
are re-measured fresh in that design doc's P-G1a/AS results, not here.

## The scenario

Two water tanks (0.3 m fill), both meant to keep every drop:

- **Aquarium** — a NEW 9x7 glass box built at t=0 on open arena floor via
  `seal_tiles` (the canonical topology primitive), rows 50–58 × cols 24–32.
  Erik plans glass aquariums as a level element; this is their physics debut.
- **Pressure lab** — the sealed hull room (#54 bench region R5), which
  makes it double as a false-heating probe.

Script (24 tps): t=2 s `ignite_ring` on the mid-arena crate stack (26,41) —
the only declared heat source · t=8 s + 9 s two `frag_standard` blasts at
(35,30)/(35,34) via the payload executor · t=20 s `breach_focus` on the
north hull (2,30) — the main hall vents to space · end 45 s.

Payload note (Erik, mid-session): grenades are untuned; the shipped
`frag_standard` row is used as-is, and the properties are written
payload-independent — no shipped blast may leak a tank. When #31 (generic
explosion archetype — the June "pressure-footprint" design, egregore
`concept:breach-generic-explosion-profile-design`) lands, the rows swap.

## First-run verdicts (2026-08-27, HEAD `3e6b89d`)

| # | Property | Verdict | Number |
|---|---|---|---|
| P1a | Water conserved through fire + 2 blasts (#10) | **PASS** | exact to the integer LSB, 0 stray |
| P1b | Water intact end-to-end | FAIL* | *all loss is post-breach design-burst + flash-boil, see P5 |
| P2 | Steel enclosure wave-silent (transmit=0 in config) | **FAIL — real bug signal** | peak 1.17 atm INSIDE the steel bunker; glass pen reads 1.16 — wall material barely matters |
| P3 | No undeclared heat | KNOWN-FAIL **#54** | bunker +82 game-deg in 18 s from a crate fire ~40 tiles away; fresh aquarium +124 deg / +52% pressure with NO source inside |
| P4 | Breach vents the hall | **PASS** | arena 0.009 atm in ~4 s through a 3-tile hole |
| P4b | Vacuum kills the fire | KNOWN-FAIL **#7** | crate burns 25 s at 0.011 atm, still lit at t=45 s |
| P5 | Tanks hold air post-breach | BURST(design) ×2 | both tanks' glass gave way at the 1.0 atm decompression differential; water then flash-boiled (<0.3 atm) |
| P6 | No undeclared structural damage | **PASS** | 53 changed tiles = declared 3-tile breach hole + 50 glass design-bursts; 0 undeclared, 0 chipped tank walls |

## What this settles / opens

1. **Wall wave transmission is per-material BY DESIGN** —
   config.toml [materials]: `transmit = 1 − wave_reflect − wave_absorb`:
   hull/steel 0%, wood/doors 10%, glass 60%. Erik's "shockwaves don't
   respect walls" is design for wood/glass — but P2 shows steel is NOT
   silent (1.17 atm interior, open-tiles-only measurement), nearly equal to
   glass. Either the reflect/absorb machinery under-attenuates or energy
   crosses walls by another path (the #54 MG-through-thin-walls suspect).
   → Bisection candidate for the #54 diagnosis session.
2. **The runtime-sealed aquarium is the cleanest #54 probe yet**: a box
   born sealed at t=0 with no doors and no history heats +124 game-deg and
   self-pressurizes 1.0→1.52 atm in 18 s (ratio consistent with constant-N
   heating — energy, not mass, appears inside) while the arena around it
   cools. Smaller and cleaner than the hot plate; no held-T forcing at all.
3. **#10 reframe hypothesis**: water conservation through blasts is EXACT;
   the only water loss anywhere came from design-legal glass bursts followed
   by vacuum flash-boil. The original "water escapes sealed aquarium" may
   have been a burst/boil chain misread as a leak — #10's verify-seal step
   should check its aquarium's pane differentials first.
4. **Decompression shatters every pane on the ship** (~50 tiles: gallery,
   pen, both tanks) because glass bursts at 1.0 atm differential — so NO
   glass enclosure can face vacuum. Game-design call for Erik: is shipwide
   glass cascade on hull breach wanted? If aquariums should survive, they
   need a tougher pane row (one `[materials.*]` row, per the table rule).
5. **#7 confirmed headlessly**: fire burns indefinitely at 0.011 atm.
   The harness is its standing repro (P4b flips when #12/#7's O2-law lands).
6. Crate-fuel sanity (Erik's dial check): the crate stack burned 43+ s and
   was still lit — matches the ">45 s" expectation.

Promotion path: each KNOWN-FAIL becomes a real pytest gate as its issue
closes; P1a/P4/P6 are green today and could gate now (as a `test_*` wrapper)
once the #54 fix stops moving the thermal numbers.

## Systems (rules lifecycle)

**(a) Existing canonical systems used**: payload executor
(`execute_payload` / `ignite_ring` — the one event entry), GameMap
`seal_tiles` (the only topology write), `water_fixed` quantize (the Q16
boundary), level loader, tests-conventions (`_*` harness, bench-style region
slices shared verbatim with `_hotplate_heating_bench.py`), materials table
(burst thresholds + wave columns read, never hardcoded).

**(b) New system**: the all-systems scenario harness itself. Draft rule
(enters CLAUDE.md when promoted to a gate): *scenario-level "does the ship
behave" checks extend `_scenario_all_systems_bench.py`'s script + property
list — never a second parallel scenario bench.*

---

## Addendum (same evening): tough-glass run + the #54 bisection

**Tough-glass scenario** (`--tough-glass`: glass burst_threshold 1.0→3.0,
the candidate aquarium-pane row): with panes that survive the decompression
differential, **both tanks hold 100.00% of their water at every checkpoint
through 25 s of ~0.03 atm vacuum — exact to the integer LSB, zero stray,
zero wall changes beyond the declared breach hole.** Erik's original
spilled-container incident (shockwave pushing water through intact glass)
does not reproduce on the S1 integer water substrate. Shockwaves still
cross walls (P2 unchanged); water never does. Note: the aquarium's P5
reads 1.65 atm at end — NOT a leak; that is #54's false heating inflating
the sealed box (dT +314 by 42 s, fire still burning in vacuum feeding it).

**Sealed-box bisection** (`tests/_sealedbox_bisect_bench.py` — minimal
probe: crate fire + dry sealed glass box, 18 s, one EOSSolver field per
run):

| variant | box dT | box P | bunker dT | pen dT | u_max |
|---|---|---|---|---|---|
| baseline | +115.0 | 1.00→1.50 | +72.7 | +66.4 | 19.1 |
| k_drag_heat_frac=0 | +124.4 | →1.51 | +74.8 | +66.2 | 19.2 |
| k_drag=0 | +165.6 | →1.43 | +133.7 | +139.9 | 16.5 |
| **adiabatic_index=1.0** | **+0.0** | →1.28 | **+0.0** | **+0.0** | **1.8** |
| use_multigrid=False | −286.4 | →0.02 | −274.0 | −283.0 | 277.4 |
| U_MAX=1e9 | +115.0 | →1.50 | +72.7 | +66.4 | 19.1 |

**Verdict: the compression-work term (−P∇·u, γ=1.4) is THE #54
mechanism** — switching it off (γ=1.0) zeroes every false-heating channel
at once. Drag-heat and the velocity rail are exonerated (no effect);
momentum drag is a partial damper of the mechanism (removing it worsens
heating ~+45%); the flat point-GS path blows up at current settings and
cannot serve as the MG control. Composed story consistent with P2: wave
energy crosses walls it shouldn't (1.17 atm inside 0%-transmit steel),
and inside closed cavities the compression-work term rectifies that
oscillation into net heat while the open source region cools. Residual
worth the mass-books look: with compression work OFF, the sealed box still
rose to 1.28 atm at dT 0 — pressure without temperature suggests some mass
does cross.

Next per the #54 session plan: design ruling with Erik — energy-conserving
fix of the compression-work term in place (and/or the wall-crossing wave
path it feeds on) vs an additive second method. Erik's standing ruling
holds: the Kwatra solver is not discarded either way.

---

## Erratum + narrowing (2026-08-29)

- **"Transmission by design" above is WRONG.** `wave_reflect` had no
  consumer in C++ (column deleted 2026-08-29, Erik's ruling); `wave_absorb`
  is a velocity-damping weight, not a transmission model. On the EOS
  substrate a wall is a perm=0 face: designed transmission is ZERO for
  every material. Any wave inside a sealed enclosure is a solver leak.
- **The γ=1 bisection toggle was confounded** (`adiabatic_index` also sets
  kick stiffness K = c_max²/γ). Clean switch `T_WORK_CLAMP = 0` gives the
  identical zeroing — verdict stands, unconfounded.
- **No mass crosses sealed walls** (box N ×1.000 in every variant). The
  pressure FIELD alone drifts inside sealed cavities (1.00→1.28 atm at
  ΔT 0, N const), growing for thicker walls / smaller pockets
  (1.50/1.69/1.82 for 1/2/3-tile walls, term on) while the work-term
  heating shrinks (+115/+92/+61). The pressure solve is contaminated across
  solid faces; the compression-work term is the rectifier, not the source.
- **Erik's ruling 2026-08-29**: the compression-work term is physics we
  keep (explosives will become physical, not injected). Fix the
  pressure-solve wall leak first, then re-measure with the term on.
