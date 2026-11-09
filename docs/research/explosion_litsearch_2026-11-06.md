> RESEARCH REPORT — NOT CANON. Issue #31, 2026-11-06. Input to Erik's explosion design session. It decides nothing.

# The generic explosion: literature search (#31)

> Sources and what was read of each: [`explosion_litsearch_2026-11-06_ledger.md`](explosion_litsearch_2026-11-06_ledger.md). PDFs: [`docs/papers/explosions/`](../papers/explosions/).
> `lit_research_2026-08-17_multigrid_and_blast.md` §B (blast at a distance) is not repeated. This report asks how energy enters a coarse grid.

**Three kinds of statement below.** "Source:" is a published result. "Derived:" is my own arithmetic from published physics. "Toy:" is a scratch one-tick numpy replica of Breach's pressure step on a sealed 58 x 38 room, which is not the engine. Anything marked Derived or Toy still needs a measurement in the real solver.

## TL;DR

1. **Breach's solver is Kwatra 2009, and one pressure solve per tick puts a ceiling on what a heat deposit can do.** The solve enforces `div(u)·dt = (p* − p)/(γ·p*)`, which saturates at **1/γ = 0.71** of a cell's volume per tick however hot the cell is. Toy: 3.5 MJ, 34.5 MJ and 3450 MJ in one disc give almost the same first-tick wind (3.9, 4.6, 4.7 m/s). True adiabatic expansion from 55 atm is a factor of 17. Above about 0.3 MJ per tile per tick, heat buys temperature and no wind. This is the likely reason `frag_standard` (34.5 MJ, 8 m/s) kills by heat and knocks nobody down.
2. **The graphics literature's standard answer is a divergence source in the pressure solve** (Feldman 2003, Nielsen 2019, Houdini Pyro, and FDS in fire science). It has no such ceiling (toy: wind linear in energy). It conforms to walls and doors by construction. Our Helmholtz form also handles the sealed room, which a pure incompressible solver cannot. It must be paired with heat, or the core springs back.
3. **Real blast waves do not deliver air.** Yngve 2000 says so outright. An honest grenade's total expansion is about 2.5 m³, or 1.3 m/s at 3 m in a 2.5 m room. Reaching the 245 Pa knockdown at 3 m takes the volume of about 14 MJ, roughly 15x the detonation energy. Erik's dramatic shock is a deliberate departure, and the literature gives no reason to avoid it.
4. **Sealed-room end state:** energy deposit leaves `ΔP = (γ−1)E/V`, which matches experiment and decays as heat is lost. Hot mass leaves permanent extra N. A divergence source alone leaves nothing. A velocity kick is mostly removed by the next solve.
5. **Quasi-static pressure, venting, channelling and flow around corners come free** from a flow solver. Shock arrival, reflection spikes, the shadow behind a corner and the Mach stem do not exist at 24 tps and need a rule if wanted.
6. **TNT 4.56 vs 15.0 MJ/kg is confirmed** (Dobratz & Crawford, NIST). RDX 6.32 vs 9.5, Comp B 5.02 vs about 11.7. Afterburn takes about 10–20 ms, inside one 41.7 ms tick, and needs oxygen: an M67 burns the O2 of about **one tile of air**. In nitrogen a confined TNT charge's pressure drops by a factor of about 3.3.
7. **Heat + low O2 is a defensible smoke proxy, with two caveats.** It is the shape of the standard empirical rate (Khan–Greeves: `p_fuel · φ^r · exp(−E/RT)`), and Yngve proposed it as a heuristic. But the fuel term is what the published form depends on, and temperature is a window (inception needs about 1250–1650 K, soot falls off again above about 2000 K), not a threshold.
8. **Glowing material in vacuum makes no smoke proper:** no combustion and no gas to suspend particles. Organics still pyrolyse to vapour, and a detonation in vacuum does leave carbon (Ornellas). Smoke needs a carbon source, and suspension needs gas.
9. **Knockdown is an impulse, not a pressure.** The steady 245 Pa threshold matches the industrial "1–2 psi knocks you down", but only for long waves. Breachers stand at 4 psi (q about 2.6 kPa) for milliseconds and stay up. For a one-tick pulse the physical range is about 0.7–4 kPa (Derived).
10. **A per-explosive contact-damage number is physically sound.** Detonation pressure (21 GPa for TNT) falls about five orders of magnitude within 2 m. Contact damage is local, and a charge a breacher stands near is limited by 4 psi at his position. Sachs: overpressure scales with ambient pressure, and in vacuum there is no air blast at all.

---

## 1. Delivering a blast into a coarse implicit solver

### 1.1 The scale of the problem

An explosion has two natural scales, both Derived from the Sachs quantities in §7:

- **Energy radius** `R0 = (E/p0)^(1/3)`. Overpressure is about one atmosphere here. 1 MJ gives 2.15 m (6.5 tiles), 35 MJ gives 7 m. The grid resolves this. Everything inside R0 is a strong blast.
- **Time** `R0/c`, about 6 ms. The tick does not resolve this. A real blast has crossed the room and started reverberating before tick 1 ends.

The acoustic CFL number is `c·dt/dx` = 37.5, with `c·dt` = 12.5 m per tick. So anything that needs a resolved wave is out, as #31's measurement showed.

A third scale matters for "delivering air": the **expansion volume** at constant pressure, `V_exp = (γ−1)E/(γ·p0)` = 2.82 m³/MJ. In a 2.5 m room that is 1.13 m² per MJ, or about 10 tiles. This is the air the explosion actually has to push aside.

### 1.2 What the solver does to a pressure spike

`eos_solver.h` names Kwatra et al. 2009, and the code matches: `rhs = p* − (γ·p*)·dt·div(û*)` with a mass term `1/(γ·p*·K·dt²/dx²)` (`eos_solver.cpp` around line 1812). Written out, each tick solves

```
(p − p*)/(γ·p*·dt²) − div(grad p / ρ) = −div(u*)/dt
u = u* − dt·grad p / ρ
```

which is Kwatra's equations 4–5 with `ρc² = γ·p*` frozen per cell. Three consequences follow. They are Derived, and the second is my main finding.

**The solve is a screened Poisson problem with screening length `c·dt` = 12.5 m.** For a room smaller than that it is nearly the incompressible projection, which is why `P` is flat at tick ends. This is the limit Kwatra 2010 uses deliberately: as c → ∞ the equation "reduces to the standard incompressible Poisson equation". A hot cell therefore does not launch a wave. It acts as a local **divergence source** in a nearly incompressible flow. The engine is already doing Feldman's method, with the source strength set implicitly by `(p* − p)/(γ·p*)`.

**That implicit source saturates.** Eliminating the velocity gives, per cell,

```
div(u)·dt = (p* − p)/(γ·p*) = (1 − p/p*)/γ  →  1/γ = 0.71 as p*/p → ∞
```

The operator is linearised about `p*`. It is first order in the strain per step, and Kwatra's own time step keeps that strain small: "as with all implicit schemes, taking too large a time step can lead to inaccurate results", so they clamp `dt` to a fixed multiple of the explicit step in their acoustic test. Breach solves once per 41.7 ms, and the strain per tick is not small. The true adiabat from `p*/p` = 55 (the `T_MAX_PHYS` rail at ambient N) is a volume ratio of `55^(1/1.4)` = 17.5. The solve allows 1.7.

Toy, one tick from rest, 0.333 m tiles, sealed 58 x 38 room, heat deposited in `p*`:

| Disc | Energy | `p*` peak | Core strain | Peak wind | q at peak | Wind at 3 m |
|---|---|---|---|---|---|---|
| radius 1.5 (9 cells) | 0.9 MJ | 2.4 atm | 0.42 | 2.7 m/s | 4.5 Pa | 0.5 m/s |
| | 3.5 MJ | 6.4 atm | 0.60 | 3.9 m/s | 9 Pa | 0.8 m/s |
| | 34.5 MJ | 55 atm | 0.70 | 4.6 m/s | 13 Pa | 0.9 m/s |
| | 3450 MJ | 5400 atm | 0.71 | 4.7 m/s | 13 Pa | 0.9 m/s |
| radius 5 (81 cells) | 34.5 MJ | 7 atm | 0.61 | 11.9 m/s | 86 Pa | 7.1 m/s |
| | 1470 MJ | 259 atm | 0.71 | 14.0 m/s | 117 Pa | 8.3 m/s |

So the first-tick wind depends on the **area of the deposit**, not its energy: `u(r) ≲ A/(2π·γ·r·dt)`. The strain is at 80 % of its ceiling by `p*/p0` = 5, which for a tile at ambient is about **0.3 MJ per tile per tick**. `frag_standard` deposits about 3.8 MJ per tile.

The toy's peak of 4.6 m/s at tick 1 sits beside #31's measured 8 m/s for the whole run. Pressure in the toy stays flat to about 1 %, as measured. `frag_hot_mass`'s measured 56 m/s is higher than the toy's first-tick 14 m/s, which needs later ticks and the added mass to explain. A single tick of a toy cannot settle that.

Hedged reading of the rest of the event: the core expands by about 1.7 per tick, so reaching 1 atm from 55 atm takes about 5 ticks instead of a few milliseconds. For those ticks it sits near the rail and radiates, while real gas would already have cooled by doing work. That may be why `frag_standard` loses most of its 34.5 MJ by tick 24. This is a hypothesis.

**Wind in a sealed room is slaved to the expansion.** In the near-incompressible limit a sealed, simply connected room has no divergence-free irrotational flow, so when the source stops the potential part of the wind has nothing to sustain it. Only rotational flow survives. Acoustic ringing is damped too: the implicit step's amplification is about `1/sqrt(1+(ω·dt)²)`, about 0.45 per tick for a 20 m room's lowest mode. Yngve states the general point: implicit methods "achieve stability by damping high frequencies", which is exactly what a blast wave is made of.

### 1.3 The options

**Energy deposition.** The compressible codes do this: Yngve 2000 initialises "a region of the volume with higher temperature or pressure" (2900 K, 1000 atm for a chemical charge) and integrates explicitly. It is correct where the time step resolves it. Conserves N and momentum, adds E through the seam. In the sealed room it leaves `ΔP = (γ−1)E/V`, which matches experiment (§2), and the pressure falls again as heat leaves. In Breach it is subject to the §1.2 ceiling, so wind grows with area and the rest of the energy becomes temperature.

**Hot mass injection.** Adds N and E. Real products are about 0.2 m³ at STP for a grenade, about two thirds of the air in one tile (Derived: the #31 note's "under a tenth of a tile" looks like a slip), so this is a cheat at the scale of a blast disc. Same ceiling as energy, since the operator only sees `p*`, though the higher N raises `ρ` and lowers the kick. In the sealed room it leaves extra moles that never leave, so repeated charges ratchet the pressure upward. This is the "static over-pressure" of the 2026-07-30 note and the mass-buildup artifact of the EOS report. The one place mass is the honest term is vacuum (§7).

**Divergence source in the pressure solve.** Feldman, O'Brien & Arikan 2003 is the reference, and its scale is close to ours: 0.13–0.5 m cells at 30 fps, on Matlab. They replace `div u = 0` with `div u = φ` and solve `∇²p = (ρ/dt)(div u − φ)`. For a detonation, φ "rises rapidly until it reaches a peak value determined by the strength of the detonation", then "decays back to zero, goes slightly negative, and finally stabilizes again at zero". "The flow will correctly conform to obstacles", shown in their Figure 2 for open and constraining arrangements of obstacles, and in a coal-mine portal comparison. Their parameters "do not have any particular physical significance".

Nielsen et al. 2019 (Autodesk Bifrost) keeps the method and derives φ from first principles: `div v = −(1/ρ)Dρ/Dt`, with density from the ideal gas law evaluated at a constant pressure, which decouples density from pressure. Houdini's Pyro works the same way: a `divergence` field, considered in the projection, and the projection is "responsible" for "causing the sudden and rapid expansion for an explosion". Its Configure Fireball page warns that too little does nothing and too much "rapidly explodes" (I read that line only in a search excerpt).

FDS is the fire-science version of the same idea, from Rehm & Baum: drop acoustic pressure from the equation of state and let heat release set the divergence. In Breach terms the matching source for an isobaric deposit is `S·dt = (γ−1)E/(γ·p0·V)`. It references ambient pressure instead of `p*`, and that is the whole difference from §1.2.

Toy, same disc, radius 1.5 tiles: 0.9 MJ gives 6.6 m/s, 3.5 MJ gives 25 m/s, 34.5 MJ gives 252 m/s, linear in E. Mean room pressure rises by the right amount (0.2 atm for 34.5 MJ). The toy has no velocity clamp and no drag.

What goes wrong, from the sources and the arithmetic:

- **A source creates volume, not mass or energy.** Feldman's fluid has constant density, so nothing checks this. Breach's does: a source with no heat moves N out of the core, `p(N,T)` falls, and the core springs back next tick. Source and heat must be one event, and the energy must go through the seam so the closure identity holds.
- **The sealed room.** In a pure incompressible solver, `∫φ dV ≠ ∮u·dS` has no solution. FDS calls its eq. 3.31 "essentially a consistency condition" and gives each sealed zone a background pressure that rises for exactly this reason. Breach does not need that machinery: the identity term in the Helmholtz solve absorbs the mismatch as a rise in mean pressure. This is a real advantage of our form over Feldman's.
- **It does not capture shocks.** Kwatra 2010 on these methods: "they discard the compression waves".
- **Symmetry.** Feldman warns that a zero initial velocity gives "undesirably symmetrical results", and perturbs it with small random seeds. Breach can do this deterministically through the sim seed.
- **Feldman's combustion ignores oxygen** ("combustion occurs irrespective of oxygen availability"), justified for dust. Nielsen handles fuel-lean to fuel-rich. Erik's O2-dependent heat is on Nielsen's side.

**Initial radial velocity kick.** Feldman found direct outward force "largely unnecessary", and says a jet can be made by forcing velocity. Here, Derived: the Helmholtz right-hand side contains `−γ·p*·dt·div(u*)`, so the divergent part of a kick becomes pressure in the same tick and is projected away, leaving only what a source would have produced. A kick alone therefore delivers little in a room smaller than 12.5 m. It would also put kinetic energy into the books outside the seam, and that energy later returns as heat through `k_ke`. I found no paper using it as the primary mechanism.

**Multi-tick release.** Feldman's φ(t) is multi-frame. The blast-load engineers measure the same thing: Oswald's 2025 DDESB method has detonation and afterburn heat each enter the room gas over a separate rise time scaled on room size and shock speed, because treating it as instantaneous overpredicts vented gas pressure. Hager et al. 2006 is the root of this form (cited in Oswald, not read). Honest rise times are one or two of our ticks (§3). In Breach, spreading a fixed energy over K ticks keeps `p*` near ambient and avoids the ceiling, at the cost of duration: a steady source sets the wind and does not accumulate it (§1.2). So it trades peak for length, unless a divergence term carries the peak.

**More pressure solves in a blast tick.** Derived: the ceiling is per solve, so K solves in a tick give K/γ and shrink the screening length. Kwatra 2009 Figure 3 is relevant: with the implicit solve outside the Runge–Kutta stages, strong shock tubes show "spurious overshoots". Kwatra's strong shock tube also shows oscillations near the rarefaction. This costs a per-tick variable schedule, and the RL-batch habits may object.

**A local change of sound speed.** Kwatra 2010 drives c to infinity non-physically over time and the operator accepts it. Evaluating `ρc²` at a reference pressure in hot cells instead of at `p*` is the same formal change as the divergence source. I found no paper doing this for blast delivery, so treat it as an idea only.

### 1.4 What shipped games do

Almost nothing I found delivers a grenade through a gas solver at game scale. Practice divides in two.

**Rule on tiles, gas as trigger or bystander.**

- **SS13 (tgstation):** three radial bands. A turf is hit if `dist + block·K < range`, with `block` the summed `explosive_resistance` along the straight step toward the epicentre (`subsystem/explosions.dm`). A gas tank rupture only computes `power = volume·(pressure − TANK_FRAGMENT_PRESSURE)/TANK_FRAGMENT_SCALE` and calls that table (`tanks.dm`). The atmos never carries the blast.
- **SS14:** a flood fill with a budget. `totalIntensity` is "the final sum of the tile intensities", plus `slope` and a per-tile `maxIntensity`. Airtight entities block a direction, and the tiles behind them join later, once the explosion is strong enough there. The budget gives confinement for free: fewer open tiles means more per tile (my reading of the code).
- **Cataclysm-DDA:** Dijkstra over walkable tiles, `force = power·distance_factor^distance`. Each tile is bashed with `force + 2·force/empty_neighbors`, commented "Up to 200 % bonus for shaped charge". This is the cheapest confinement term I found.

**Air solver with sources: The Powder Toy** is the one clear game in the other camp. 4-pixel cells, pressure and velocity on a grid, updated explicitly: `dp` from the velocity differences at `AIR_TSTEPP = 0.3`, velocity from pressure differences at `AIR_TSTEPV = 0.4`, self-advection at `AIR_VADV = 0.3`, slow loss toward ambient. An exploding BANG particle does `pv += 0.5`, BOMB `+= 0.1` per destroyed pixel, and FIRE adds increments of its own. Waves are resolved because the wave step is explicit and small. Pressure has no equation of state and is not conserved: it is an acoustic toy, not a gas. It looks dramatic and it respects walls, and it is close in spirit to Breach's old `wave_p` field, whose fate is in #31's inbox. It is the precedent if Erik wants a visible shock: a separate, explicit, non-conserving wave layer.

Noita, Teardown and Oxygen Not Included: I found no technical account of explosion delivery. Noita's developers describe turning hit pixels into collapsing sand, a destruction rule. For SS13 atmos and ONI artifacts see the EOS report §1.

### 1.5 What this means for Breach

| Route | Wind | Sealed room | Cost and risk |
|---|---|---|---|
| Heat only, wide disc | Set by area; 8-tile disc for 20 m/s at 3 m | Honest `ΔP` | No solver change. Appears at once over a wide disc. Fits "hot gas alone" most closely, but only to a ceiling |
| Heat plus divergence source | Linear in strength, no ceiling | Honest, mean P rises | New RHS term on CPU and CUDA, new books. Tunable into blow-up. Most literature behind it |
| Adiabatic strain per cell | As above, derived, no extra dial | Honest | As above plus a baked table for the `1/γ` power. No precedent found |
| Multi-tick release | Trades peak for duration | Honest | Cheap. Combines with any row here |
| Hot mass | As heat | Permanent N, ratchets | Exists. Keep for vacuum and perhaps big bombs |
| Velocity kick | Weak, removed by the solve | Nothing | Needs a KE booking. A garnish at best |
| Tile rule (flood or Dijkstra) | None, it is a damage rule | Budget gives confinement | Erik: only very short range. Independent of the air |
| Explicit wave layer | Visible front | Non-conserving | A second field, close to the old `wave_p` |

Measurement owed before any of this is trusted: run one payload at E, 10E, 100E in the real engine and read tick-1 and peak wind. If the ceiling is real, wind barely moves. This takes minutes and it is the toy's only real test.

## 2. Confined and geometry effects

**Quasi-static pressure (QSP) is natural, and it is the part that matters most.** After reverberation the room holds a nearly uniform pressure set by energy, volume and heat capacity. Farrimond 2024 gives the ideal-gas form, `ΔP = (γ−1)E/V`. The UFC 3-340-02 curve descends from Weibull's 1968 correlation of peak gas pressure on charge mass over volume. Kuhl's test checks the arithmetic (Derived): 0.875 kg TNT in 16.6 m³ gives `0.4·4.0 MJ/16.6 m³` = 0.96 bar against about 0.85 measured in nitrogen, and `0.4·13 MJ/16.6 m³` = 3.2 bar against 2.8 measured in air. Breach's EOS gives this by construction, provided the energy stays in the gas. In a 100 m³ room an honest M67 gives about 3.7 kPa on detonation alone, and about 8.6 kPa with full afterburn.

**Venting is natural and resolved.** Derived: blowdown time is of order `V/(A·c)`. 100 m³ through a 2.5 m² door is about 0.1 s for one e-fold, or 3 ticks. This is the one explosion timescale the tick resolves comfortably, so a room emptying through a door should work as is, drag permitting. UFC and FRANG are empirical here. Oswald reports that they are conservative for large, lightweight vents, because they assume the peak is reached at once.

**Channelling is natural.** In the near-incompressible limit the flux must go somewhere, so a corridor carries the whole expansion at `Q/(w·h)`, with no 1/r spreading until drag takes it (e-fold about 4.4 m for a fast jet). A dead-end corridor gets pressure and no wind. This is the behaviour Erik likes. Field and small-scale studies do show slow decay along tunnels, and a cube-root scaling with a geometry factor gave reasonable agreement in one thesis (abstract only). I found no open numbers for doors or side branches.

**Flow around corners is natural, shock diffraction is not.** Feldman's Figure 2 shows the source flow conforming to obstacles. A potential flow turns corners, and the pressure reaches behind them at once. So there is no shadow in pressure. There is a shadow in wind, because flux divides at junctions.

**Reflection is not available.** Real reflected overpressure is `2·Ps·(7p0+4Ps)/(7p0+Ps)` (Rankine–Hugoniot, γ = 1.4): twice the incident value when weak, up to 8x when strong, for milliseconds. Yngve's solver shows the Mach stem, with pressures "twice that of the initial wave". Nothing in a 41.7 ms tick can show it. If wall damage should exceed what QSP implies, that needs a rule on the wall's distance from the charge.

**The negative phase** could appear in Breach from inertia or cooling after the hot gas expands, as the EOS report guessed. Feldman writes it in by hand, with φ going slightly negative. Given §1.2's damping I would not assume it survives.

**For Breach:** QSP, venting, channelling and turning come from the solver, and they are the effects that make doors and corridors matter. Shock arrival delay, reflection doubling and the corner shadow would each be a rule. Most of the difference between real and simulated blast is in the first 10 ms, which a player cannot see.

## 3. Afterburn

**Numbers.** Heat of detonation is from Dobratz & Crawford 1985, Table 5-2, experimental, liquid water. Combustion is from the NIST WebBook. Comp B combustion and all oxygen balances are mine, by mass mix.

| Explosive | Detonation, MJ/kg | Combustion, MJ/kg | Afterburn potential, MJ/kg | Oxygen balance |
|---|---|---|---|---|
| TNT | 4.56 (4.27 H2O gas) | 15.0 | 10.4 | −74 % |
| RDX | 6.32 (5.94) | 9.42–9.54 | 3.2 | −21.6 % |
| Comp B | 5.02 (4.69) | about 11.7 | 6.7 | −42.6 % |
| C-4 | 6.65 maximum, calculated (5.86) | not found | not found | not found |

**TNT about 4.5 vs about 15 MJ/kg: confirmed.** Kuhl quotes the same pair from Ornellas, 1093 vs 3575 cal/g, which is 4.57 and 14.96 MJ/kg. His measured ratio of mean pressures, air over nitrogen, is 3.3, matching the ratio of heats, 3.27. Afterburn is about 2.3x the detonation energy for TNT, about 1.3x for Comp B and about 0.5x for RDX.

**Timescale.** It is mixing-limited, not chemistry-limited, and it depends on confinement:

- Kuhl 1998: 0.875 kg TNT in 16.6 m³ of air, exothermic lifetime **18.9 ms**, fuel consumption an exponential decay after a sharp start.
- Farrimond 2024: 30 g PE10 in a 276 L chamber, inert atmospheres reach peak QSP in well under 1 ms, air takes "the subsequent 10 ms or so". The CFD agrees over 20 ms, by which time "the majority of the afterburn process has occurred".
- Kuhl 2000 (numerical, 1 g TNT): the burn rate depends on chamber size. Combustion raises peak pressure about 3.5x regardless of volume. In the smallest chamber it stopped at about half the fuel because the air ran out.

So afterburn fits inside one or two ticks. It does not need its own timescale in Breach.

**Oxygen.** Edri's figure, via Farrimond: 3.17 kg air per kg TNT, which my oxygen balance reproduces (3.19). For an M67 (Derived): 0.184 kg Comp B needs 0.078 kg O2, the oxygen of about **0.28 m³ of air, one tile**. Air-starved events stop early: Kuhl 2000's smallest chamber, and a 26 m³ chamber series that reached 86 % of equilibrium, attributed to imperfect mixing (a search excerpt around Donahue 2013, whose full text I did not read). In tubes, a larger L/D means "confinement suppresses mixing and combustion" (Kuhl's gasdynamic model paper). Unconfined, small charges burn less: the config comment on `[explosives.*]` already says the products "freeze".

**Thermobarics** push further the same way: fuel-rich charges, often with reactive metal, designed to release most energy by afterburn. In a 2024 *Case Studies in Thermal Engineering* study, impulse and positive duration markedly exceed TNT's. Türker 2016 is the review (abstract level only for both).

**For Breach:** one tile of O2 is small. At honest strength a grenade barely dents the O2 field, so a rule tied to it would be visible only at scaled-up strengths or in a very thin atmosphere. At 40x it is forty tiles. An O2-dependent heat term is physically sound, and the most useful effect is the extreme case: a TNT charge in pure nitrogen gives about 1/3 of the pressure, an RDX charge about 2/3. Erik's strength-1 grenade could carry this as one extra number per explosive, its afterburn fraction, or derive it from the heat table. RDX's small afterburn and zero soot agree, so one column could drive both.

## 4. Soot and smoke proxies

**Is temperature + low O2 defensible? Yes, as a shape.**

- **Khan & Greeves' one-step model** (ANSYS Fluent's default) is `R_form = C_s · p_fuel · φ^r · exp(−E/RT)`: fuel, equivalence ratio, temperature. Hiroyasu's two-step and Moss–Brookes are refinements. Fluent's own documentation calls the result "qualitative indicators". So Erik's proxy is this form with `p_fuel` removed.
- **Yngve 2000** proposed it for graphics: "Incomplete combustion at lower temperatures results in smoke, and that observation could be used as a heuristic to determine where smoke should be created in the fireball".
- **Fire data agree on the O2 half.** Tewarson 2008: combustion efficiency is about constant for equivalence ratio up to 0.7, and below that "combustion becomes less efficient" and smoke and CO emission increase. Köylü & Faeth: in well-ventilated flames, CO and soot generation are uniform and independent of residence time, at about 0.37 kg CO per kg soot (abstract). So a fire's soot yield is a fuel constant until it starves, which is how FDS treats it (a fixed yield per fuel, zero by default).
- **Nielsen 2019** on why the linear rule fails: at stoichiometric conditions all fuel and oxidiser react, "with no excess soot". Soot is not proportional to burn rate.

**Where the proxy needs care.**

- **The fuel term.** Sunderland 2002: soot inception needs both temperature above a critical value and C/O above a critical value, "necessary, but not sufficient", because residence time matters too. Without a carbon term, hot N2 after a blast smokes as readily as a starving fire. In Breach the carbon source exists: a burning tile's fuel, a charge's explosive mass. A multiplier on those keeps the rule local.
- **Temperature is a window.** Soot does not form at low temperature in diffusion flames, "with the threshold ranging from about 1250–1650 K" (Sunderland). I could not verify a single sharp value, and a review's point is that it depends on method. The high side falls off too: about 1800–2000 K in diesel conditions (thesis abstract), a 2203 K peak in shock-tube methane pyrolysis (arXiv abstract). The peak moves with pressure and fuel. Oxygen and high temperature burn soot off. Erik's "heat + low O2" fits the low side and needs a ceiling only if the hot core should stay clean.
- **Low O2 is a late signal for a blast.** The real incompleteness comes from the explosive itself: TNT is −74 % oxygen balance, and its products are rich in carbon dust and CO, about 20 % each (Kuhl). Well-mixed in air that carbon burns. This suggests a single balance for both fire and explosion: **carbon available, split by local O2 into heat and smoke**. It gives Erik's rule, and it ties the #12 soot numbers (`soot_yield` 0.193 TNT, 0 RDX) to afterburn.
- **Under-ventilated smoke differs.** A NIST-indexed study finds it mainly organic and agglutinated rather than graphitic (abstract). This is relevant if Erik ever wants a look difference, for instance pale grey starved fires against black well-fed ones.

**One rule or two?** Literature supports one form with per-source constants. It does not support one constant: TNT's 0.193 against RDX's zero is the clearest case, and the source's composition is the main variable. Minting from nothing is accepted by Erik, and the size of the error is small compared to these spreads.

**Does a glowing hot material make smoke in vacuum?** Not smoke in the usual sense:

- There is no O2, so no combustion and no soot from flames.
- There is no gas to suspend particles, so anything released flies ballistically.
- A glowing metal or ceramic releases almost nothing at ordinary temperatures.
- An organic solid does decompose: vacuum pyrolysis yields volatile fragments and char (Madorsky and Straus at NBS, search-level reading). Those are vapours, and they leave as a thin expanding gas. They would condense on cold surfaces.
- Very dense vapour can condense in free expansion: laser ablation plumes make nanoparticles in vacuum, with size set by the expansion (abstract). This is far from a hot wall.
- **An explosive carries its own oxygen**, so detonation in vacuum does leave carbon: Ornellas's 25 g TNT vacuum calorimetry gives 3.65 C(s) per mole (via Kuhl and the config comment).

**For Breach:** gate any smoke rule on gas present, so a heat-only blast in a breached room stops minting smoke into near-vacuum. This is the "smoke 1.0 into near-vacuum" finding from the 2026-10-02 session. An explosive's own soot is the honest exception and can stay on the charge's mass.

## 5. Knockdown and blast wind on a person

**The physics is impulse.** Bowen et al. 1961 treat a body as a drag object: `dv/dt = α·q`, with `α = C_d·A/m`. For a 168 lb man standing facing the wind, `α` = 0.052 ft²/lb, which is **0.0107 m²/kg** and `C_d·A` = 0.81 m², in line with the 0.75 m² of the wind note. A tumbling man is 0.030 ft²/lb. Overpressure loading is small next to drag for body-sized objects, because the wave engulfs the object quickly. A dummy at 5.3 psi with a 0.96 s positive phase reached 21.4 ft/s (6.5 m/s) in 0.5 s, about 8 ft of travel. So `Δv = α·∫q dt`.

**Published anchors.** Derived for q: `q = 2.5·Ps²/(7p0+Ps)`.

- An industrial table gives 1–2 psi (7–14 kPa) as "knocks personnel down" (secondary, vapour cloud explosions, so long durations; I could not verify it against Lees or Clancey). That is q = 170–680 Pa, and **the steady-wind fall threshold of 245 Pa sits inside it**.
- FM 8-9: eardrum rupture threshold about 22 kPa, lung threshold about 69 kPa. These are primary injury levels. The manual gives no displacement threshold.
- According to PubMed, military breachers work to a 4 psi (27.6 kPa) exposure limit and a 5-year study found no measurable effect when kept within it ([Kamimori 2018](https://doi.org/10.1093/milmed/usx186)). That is q about 2.6 kPa, for milliseconds. They are standing.
- The TNO Green Book displacement probit I could not retrieve. Nor could I retrieve Baker's or the "Improved Displacement Velocity" report (DTIC blocked).

**A threshold range, Derived.** According to PubMed, Hof et al. 2005 give the stability condition: position plus velocity times `sqrt(l/g)` must stay inside the base of support ([DOI](https://doi.org/10.1016/j.jbiomech.2004.03.025)). With l = 1 m, base margin 0.1 m gives 0.3 m/s without stepping, and a 0.6 m step margin gives about 1.9 m/s. Through α, that is `∫q dt` of **about 30–180 Pa·s**.

| Pulse | q to fall |
|---|---|
| Steady, long wave | 245 Pa (#140) |
| One tick, 41.7 ms | about 0.7–4.3 kPa |
| Breacher at the 4 psi limit, 2.6 kPa for an assumed 3 ms | 8 Pa·s, stays up |

**For Breach:** the shipped law fires at 245 Pa on any tick. #31's measured peaks are 40 Pa, 1900 Pa, 2500 Pa. Erik's plan to tune unit thresholds in Pa is fine, and the honest range to tune in is 245 Pa to a few kPa. A one-tick 245 Pa knockdown is three to twenty times easier than life, which may be what the game wants: it makes a dramatic shock read. An option, if a long weak vent and a short sharp blast must differ, is a threshold on a decaying `∫q dt`. Reading q from the air keeps it a function of geometry, which is the point.

## 6. Breaching charges

Why a contact charge destroys a wall and a person metres away survives:

- **Pressure.** TNT's detonation pressure is about 210 kbar, 21 GPa (Kuhl). At the energy radius, 2 m for 1 MJ, overpressure is about one atmosphere (§1.1). That is five orders of magnitude over two orders of distance, the steepest fall of anything here.
- **Contact couples the detonation into the solid.** Standoff couples only an air shock with a tiny impedance. Damage from contact is predominantly local, with negligible global response (Carleton thesis, abstract), and empirical blast charts underpredict below about Z = 0.4 m/kg^(1/3).
- **Empirical breach rule:** `P = R³·K·C`, charge mass by the cube of breaching radius, K a material factor, C a tamping factor with values 1.0–3.6 by placement (secondary, from manual excerpts and study aids, I could not read FM 3-34.214 itself). Tamped charges work with much less explosive.
- **People.** Minimum safe distance is computed for the 4 psi limit. Field measurements ran from under 1 psi to over 12 psi, and the authors call the calculation "often inaccurate" (abstract). Water tamping cut overpressure by 18 % on average in a 2022 study (search excerpt).
- **Spall.** I did not get a primary source for the stress-wave mechanism at a level I would cite. It is an open point.

**For Breach:** two numbers on a row are physically justified, since blast and brisance are distinct properties. Heat of detonation governs the air. Detonation pressure, which grows with density and velocity of detonation, governs the contact. A breach charge is a small energy with a large contact number, and it sits exactly at Erik's "tiny blast, very high damage on its own tile".

## 7. Thin air

**Sachs scaling** (Wang 2018 states it): `R̄ = R·(p0/E)^(1/3)`, `p̄ = Δp/p0`, `t̄ = t·a0·(p0/E)^(1/3)`, `Ī = I·a0/(E^(1/3)·p0^(2/3))`. At half an atmosphere, overpressure is half at a given scaled distance, and that distance is 26 % further out. A 1960s DTIC test series went to about 0.0007 atm (search excerpt only, the report is blocked).

**At zero pressure** there is no air blast. Products expand freely, their speed limited by their energy. A hull breach or vacuum room has nothing to shock.

**For Breach, Derived:** the strain ceiling of §1.2 is dimensionless, so wind speed from a deposit does not fall with N. Density does, and `q` follows through the #140 law, so q falls in proportion to N as Sachs wants. The room rise `(γ−1)E/V` does not fall, however, because energy per mole climbs: this is the 7350 K air and the vacuum fireball of the 2026-10-02 session. In a thin room the charge should arguably deposit less into the gas, or scale with N, since the real energy goes to products that fly off. In vacuum the honest term is the products' own mass, about 0.2 m³ at STP for a grenade.

---

## What the literature does not settle

- **Whether §1.2's ceiling is what limits Breach.** The form is in the code and the toy shows it, but the engine has rails, radiation, advection substeps and a velocity clamp. It needs one measurement.
- **Coarse-grid blast at a fixed tick.** Nobody publishes this. Feldman's cell size is ours, his physics is not. Kwatra stops at the time steps he can trust.
- **How much a source should be strengthened.** All graphics values are by eye. Strength 1 = 245 Pa at 3 m is a game decision.
- **Doors and side branches.** I found no open quantitative data.
- **A one-tick knockdown threshold.** My range is a derivation. No published test gives knockdown against duration in the 10–100 ms band.
- **Soot at blast temperatures.** Flame soot models are not validated at 16000 K, and the rule's high-temperature side is a game call.
- **Whether a divergence term survives CUDA lockstep and the energy closure unchanged.** This is engineering, not literature.

## Sources

Full notes in the ledger. Key items:

- Feldman, O'Brien, Arikan 2003, *Animating suspended particle explosions*, ACM TOG 22(3). https://doi.org/10.1145/882262.882336
- Yngve, O'Brien, Hodgins 2000, *Animating explosions*, SIGGRAPH. https://doi.org/10.1145/344779.344801
- Kwatra, Su, Grétarsson, Fedkiw 2009, JCP 228, 4146–4161. `docs/papers/ADA492343.pdf`
- Kwatra, Grétarsson, Fedkiw 2010, *Practical animation of compressible flow*, SCA. https://diglib.eg.org/handle/10.2312/SCA.SCA10.207-215
- Nielsen, Bojsen-Hansen, Stamatelos, Bridson 2019, *Physics-based combustion simulation in Bifrost*, SIGGRAPH Talks. https://doi.org/10.1145/3306307.3328149
- McGrattan et al. 2024, *FDS Technical Reference Guide* vol. 1, NIST SP 1018-1. https://doi.org/10.6028/NIST.SP.1018
- SideFX, Pyro background. https://sidefx.com/docs/houdini/pyro/background.html
- Source code read 2026-11-06: The Powder Toy (`Air.cpp`, `BANG.cpp`, `FIRE.cpp`, `BOMB.cpp`), Cataclysm-DDA (`explosion.cpp`), Space Station 14 (`ExplosionSystem*.cs`), tgstation (`explosions.dm`, `tanks.dm`)
- Kuhl et al. 1998, UCRL-K-131748. https://www.osti.gov/biblio/3648 · Kuhl et al. 2000, UCRL-JC-137418. https://www.osti.gov/biblio/793989
- Farrimond et al. 2024, Proc. R. Soc. A 480. https://doi.org/10.1098/rspa.2024.0061 · Barr et al. 2025, PSEP 202. https://doi.org/10.1016/j.psep.2025.107678
- Oswald 2025, IESSE. https://www.denix.osd.mil/ddes/denix-files/sites/32/2026/02/045p_IESSE-2025_Oswald_Improved-Gas-Pressure-Methodology_REV-1.pdf
- Dobratz & Crawford 1985, *LLNL Explosives Handbook*, UCRL-52997 chg. 2, Table 5-2. https://www.osti.gov/biblio/6530310 · NIST WebBook. https://webbook.nist.gov/
- Sunderland et al. 2002, NASA. https://ntrs.nasa.gov/citations/20020027073 · Tewarson 2008, Fire Safety Science 9. https://publications.iafss.org/publications/fss/9/1153/view/fss_9-1153.pdf
- Bowen, Albright, Fletcher, White 1961, *A model designed to predict the motion of objects translated by classical blast waves*, US AEC Civil Effects Test Operations
- According to PubMed: Kamimori et al. 2018, https://doi.org/10.1093/milmed/usx186 · Hof et al. 2005, https://doi.org/10.1016/j.jbiomech.2004.03.025
- Wang, Liu, Gong 2018, *A scaled air blast model for high altitudes*, WIT Trans. https://www.witpress.com/elibrary/wit-transactions-on-the-built-environment/180/36801
