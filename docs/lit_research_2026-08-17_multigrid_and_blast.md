# Literature research — 2026-08-17

Two topics, both from the ledger, both needing published work rather than
reasoning from our own code.

**A. Multigrid transfer operators** — the immediate follow-up to tonight's root
cause (`docs/pressure_arc_root_cause_2026-08-17.md`). Question: *why* is our
convergence ~×0.55/cycle when textbook multigrid is ~×0.1, and is there a
principled fix that buys the cycles back?

**B. Generic explosion archetype** — ledger item, "ONE parameterized explosion
archetype, weapons are scaled instances". Question: is there an established
scaling law that gives us exactly that, and what are its limits?

Nothing here is implemented. No code changed.

---

# A. Why our multigrid converges at ×0.55/cycle

## A1. The literature says our prolongation is the known-deficient one

Our solve uses **piecewise-constant (PC) injection** prolongation, chosen as
the exact transpose of the summing restriction to keep the transfers
variational (`eos_solver.h:425-436`).

The cell-centred multigrid literature identifies exactly this as a weak point:
piecewise-constant interpolation *"may lead to multigrid solver breakdown
because it lacks sufficient accuracy to transfer the correction from the coarse
grid to the fine grid"*, mitigated only by applying enough smoothing after
coarse-grid correction — and the documented consequence is that the best
cell-centred solvers need **2–4 smoothing iterations** where node-centred
schemes get away with one ([Cell-centred multigrid
revisited](https://link.springer.com/article/10.1007/s00791-004-0137-0);
[Cell-Centered Multigrid with Higher-Order Transfer
Operators](https://www10.cs.fau.de/publications/theses/2013/Martin_BT_2013.pdf)).

We run V(2,2) — inside that band — which is presumably why the scheme works at
all rather than diverging. But "works" and "converges fast" are different
things, and ×0.55/cycle is the price.

## A2. The mechanism, stated precisely

This is the part worth internalising, because it reconciles our own header
comment with the measurement.

Galerkin (variational) coarsening is **optimal in the sense that it minimizes
the error in the range of interpolation** ([Dendy & Moulton,
2010](https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.705)). Read that
qualifier carefully. The coarse-grid correction is the best possible
correction *drawn from the range of the prolongation operator* — and no
better.

So our header's claim, *"the two-grid correction is an energy-norm projection
and cannot amplify, at ANY pyramid depth"*, is **true and also weak**. It
guarantees we never diverge. It guarantees nothing about how much error we
remove, because with PC injection the range of interpolation is the space of
piecewise-constant functions — which cannot represent smooth error well.

**That is the ×0.55.** Not a bug; a ceiling built into the choice of transfer.
And it is why raising `mg_cycles` works so cleanly: each cycle removes its
fixed modest fraction, so more cycles is the only lever the current transfer
operator leaves us.

## A3. The named, published fix: BoxMG (matrix-dependent interpolation)

Our situation is not generic. The operator coefficient `m_i = 1/aK_i` scales
with `N`, and `N` on a Breach map runs from near-vacuum to dense gas across a
single breach face. **That is a jumping-coefficient problem**, and the
literature is unambiguous that this is the regime where standard multigrid
degrades worst and where PC-type interpolation fails hardest: convergence
*"often deteriorates when the coefficients in the differential equation are
discontinuous"*, and the established remedy is **operator-dependent
(matrix-dependent) prolongation with Galerkin coarse operators**
([De Zeeuw / matrix-dependent
prolongations](https://www.sciencedirect.com/science/article/pii/037704279090252U);
[robust multigrid for nonsmooth
coefficients](https://www.sciencedirect.com/science/article/pii/S0377042700004118)).

The specific method built for our exact configuration is **Black Box Multigrid
(BoxMG)**, Dendy 1982:

- **cell-centred** on logically rectangular grids — ours is cell-centred
- **operator-induced interpolation** built from the matrix coefficients, which
  *approximately preserves continuity of the normal flux* across a coefficient
  jump — this is precisely the property PC injection lacks
- **restriction = transpose of interpolation** — we already do this
- **Galerkin coarsening** — we already do this
- reported *"robustness with respect to discontinuous diffusion coefficients,
  boundary conditions, and grid dimension"*

**The gap between our solver and BoxMG is one component: the prolongation.**
Everything else in our pyramid already matches. That makes this a much smaller
change than it sounds — we would keep the SPD row form, keep Galerkin
coarsening, keep transpose restriction, and replace PC injection with
operator-induced weights derived from the `gE`/`gS` face conductances we
already build every tick.

Note this also retires the old objection. The gate rejected bilinear
prolongation when the operator was still **nonsymmetric** and the result was
divergent (`eos_p3_gate_measurements.md` §B1). Against the symmetric operator
now in use, operator-induced interpolation *is* variational — the failure mode
that killed the earlier attempt does not apply.

## A4. What I would and would not do with this

**Recommend now:** ship `mg_cycles = 8` (tonight's measurement — correct *and*
18% faster). It is a one-line schedule change against a well-understood
ceiling.

**Recommend as its own arc, later:** BoxMG-style operator-induced prolongation.
Payoff if it lands: convergence at a rate that makes C=2 genuinely sufficient,
recovering the cycles and probably improving the breach/vent cases (jumping
coefficients are where it helps most, and breaches are where our gate was
always marginal).

**Do NOT treat A3 as measured.** The ×0.1-class convergence figure is the
literature's general claim for good transfers, not a measurement of our
operator in Q16.16 integer arithmetic. Fixed-point truncation at every level
could easily eat part of the gain. This needs its own measurement gate — the
honest first step is a float prototype of the prolongation to see what
convergence factor we actually get before touching the shipped path.

**Citations to archive** (`docs/papers/`, per the credit-the-source rule) if we
pursue this: Dendy, J.E. (1982) *Black box multigrid*, J. Comput. Phys.;
Dendy & Moulton (2010) *Black Box Multigrid with coarsening by a factor of
three*, Numer. Linear Algebra Appl.; De Zeeuw (1990) *Matrix-dependent
prolongations and restrictions in a blackbox multigrid solver*, J. Comput.
Appl. Math.

---

# B. Generic explosion archetype — the scaling law already exists

Erik's ask: one parameterized explosion archetype (yield / radius / heat /
pressure profile), with grenades, bazooka shots and future ordnance as *scaled
instances*, so we get variation without hand-balancing each weapon.

**Blast physics has exactly this, and it is a similarity law, not a fit.**

## B1. Hopkinson–Cranz: the one-parameter family

The **Hopkinson–Cranz cube-root scaling law** (Hopkinson 1915, Cranz 1926)
states that explosions at equal *scaled distance* produce geometrically similar
blast waves. Scaled distance:

```
Z = d / W^(1/3)          [m / kg^(1/3)]
```

where `d` is distance to the charge and `W` the TNT-equivalent yield.

This is the archetype Erik described, derived rather than invented: **every
blast parameter is a function of `Z` alone**, so one curve set serves all
charge sizes, and a weapon is fully specified by its yield `W`. Different
explosives fold in through the **relative effectiveness (RE) factor** — e.g.
C-4 ≈ 1.34 × TNT — so a weapon row needs one number, not a tuning profile.

## B2. Kingery–Bulmash: the actual curves

The standard curve set is **Kingery & Bulmash (1984)** — piecewise polynomials
in log-space giving peak incident overpressure `Pso`, reflected overpressure
`Pro`, positive impulse `is`/`ir`, arrival time, shock front velocity, and
positive phase duration `to`, all as functions of `Z`. These underpin
UFC 3-340-02 and essentially every engineering blast tool.

Useful caveat found in the review literature: impulse is *not* a pure function
of scaled distance the way pressure is, so impulse needs its own curve rather
than being derived.

## B3. Friedlander: the pressure–time shape

The waveform at a point is the **modified Friedlander equation**:

```
P(t) = Pso · (1 − t/to) · exp(−b·t/to)
```

`Pso` peak overpressure, `to` positive phase duration, `b` a dimensionless
decay coefficient. Note the `(1 − t/to)` factor drives P negative after `to` —
the **negative (suction) phase**, which is real and is the physical cousin of
the negative `P_min` we spent today chasing as a bug. Worth keeping straight:
a real blast *should* produce a rarefaction behind the front.

## B4. The missing piece, now published: `b(Z)`

`b` was historically given as tables/diagrams, with sources disagreeing
substantially. **Karlos, Solomos & Larcher (2016)**, *Analysis of the blast
wave decay coefficient using the Kingery–Bulmash data*, Int. J. Protective
Structures 7(3), 409–429, closes this: they derive `b` as a closed-form
function of `Z` for four cases (incident/reflected × spherical free-air /
hemispherical surface burst), fitted to within 3% of the underlying data.

Form (their equation 7, deliberately mirroring Kingery–Bulmash's own):

```
Y = C0 + C1·U + C2·U² + … + Cn·Uⁿ ,   U = K0 + K1·T
Y = log10(b) ,  T = log10(Z)
```

**Spherical free-air burst constants (their Table 1):**

| | 0.4 ≤ Z < 2.0 incident | 0.4 ≤ Z < 2.0 reflected | 2.0 ≤ Z < 40.0 incident | 2.0 ≤ Z < 40.0 reflected |
|---|---|---|---|---|
| K0 | −1.21918 | −1.19297 | 0.91029 | −0.81111 |
| K1 | −1.02211 | −2.15780 | −1.30156 | −0.79605 |
| C0 | −48.18977 | 2.06997 | −0.09812 | 36.02749 |
| C1 | −149.9999 | 16.80778 | 1.35537 | 131.07634 |
| C2 | −98.92512 | 72.93669 | 0.30350 | 149.9999 |
| C3 | 62.44699 | 149.9999 | −1.25125 | 24.56940 |
| C4 | 30.85296 | 144.25099 | −2.43775 | −45.12465 |
| C5 | −37.17606 | 32.36555 | −2.88558 | −1.25611 |
| C6 | 15.40019 | −33.83582 | 6.77186 | 15.84096 |
| C7 | 10.37512 | −4.03564 | 5.93303 | −0.97900 |
| C8 | −9.26648 | 14.50199 | −11.10183 | −0.14570 |
| C9 | 9.01084 | 1.03770 | 7.47192 | 4.87916 |
| C10 | 1.86709 | −0.74065 | 15.76656 | 2.68134 |
| C11 | 0.15419 | 4.07724 | −17.02748 | 0.41271 |
| C12 | 5.08837 | −2.76428 | −16.90636 | — |
| C13 | 1.07710 | 0.49798 | — | — |
| C14 | 2.46682 | — | — | — |
| C15 | 3.61837 | — | — | — |
| C16 | 1.10167 | — | — | — |

Continuity at the Z = 2.0 join is good: incident b(2.0−) = 2.49 vs
b(2.0+) = 2.42; reflected 3.70 vs 3.69. (Table 2 in the paper gives the
hemispherical surface-burst set, split at Z = 2.5 — less relevant to us,
since Breach explosions are mostly free-air inside rooms.)

## B5. Limits — read these before designing

- **Valid range `0.4 ≤ Z < 40.0 m/kg^(1/3)`**, and the authors explicitly warn
  that values for **`Z < 1.0` should be used with caution**.
- **Below the range the single-peak Friedlander shape is simply wrong** —
  expanding detonation products give *multiple* peaks (blast front, then
  products). So very close to a charge, the archetype does not apply.
- The underlying Kingery–Bulmash data itself carries scatter: independent
  studies report peak overpressure and impulse differing by **>40%**, and
  positive duration by **>60%**, from the K–B values.
- Karlos et al.'s own validation against experimental recordings shows
  *"substantial differences"* for incident waves, attributed partly to
  measurement asymmetry.

**Does our use case sit in the valid band?** Roughly yes, which is the good
news. A grenade at ~0.2 kg TNT-equivalent gives `W^(1/3) ≈ 0.585`, so a target
2 m away is at `Z ≈ 3.4`, 1 m away `Z ≈ 1.7`, 0.5 m away `Z ≈ 0.85`. Typical
room engagements land in the well-behaved part of the curve; point-blank does
not, and would need a separate near-field rule.

## B6. How this should meet the engine

Two notes specific to Breach's constraints, both of which I'd want in the
design doc before anyone writes code:

1. **Do not evaluate a 16th-order polynomial in the sim path.** It is
   ill-conditioned, and the iron rule forbids libm in Q16.16 anyway. The repo
   already has the right pattern: the raycaster **bakes** the blackbody
   `E°(T)` table at load and indexes it at runtime (`config.toml:398-403`).
   Bake `Pso(Z)`, `to(Z)`, `is(Z)` and `b(Z)` to LUTs the same way, in float,
   at load time — then the sim path is a table lookup plus interpolation, fully
   deterministic, and the polynomial conditioning problem disappears entirely.
2. **This is a pressure/impulse source, and today the EOS receives blast as a
   static over-pressure deposit.** The TODO already carries Erik's 2026-07-30
   note that grenades *"dump too much static pressure into the room"* and that
   the deposit should be re-split toward HEAT plus an initial radial **wind**
   (velocity initial condition). Friedlander gives exactly the missing shape:
   a peak, a decay constant, a duration, and a physically-correct negative
   phase. That is a much better-founded input than a hand-tuned pressure spike.

**Citations to archive** (`docs/papers/`): Kingery & Bulmash (1984)
*Airblast parameters from TNT spherical air burst and hemispherical surface
burst*, ARBRL-TR-02555; Karlos, Solomos & Larcher (2016), IJPS 7(3) 409–429
(open access); Hopkinson (1915) / Cranz (1926) for the scaling law itself.

---

## Sources

- [Cell-centred multigrid revisited](https://link.springer.com/article/10.1007/s00791-004-0137-0)
- [Cell-Centered Multigrid with Higher-Order Transfer Operators (thesis)](https://www10.cs.fau.de/publications/theses/2013/Martin_BT_2013.pdf)
- [Black Box Multigrid with coarsening by a factor of three — Dendy & Moulton, 2010](https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.705)
- [Matrix-dependent prolongations and restrictions in a blackbox multigrid solver — De Zeeuw](https://www.sciencedirect.com/science/article/pii/037704279090252U)
- [Robust multigrid methods for nonsmooth coefficient elliptic linear systems](https://www.sciencedirect.com/science/article/pii/S0377042700004118)
- [On local Fourier analysis of multigrid methods for PDEs with jumping and random coefficients](https://arxiv.org/pdf/1803.08864)
- [Analysis of the blast wave decay coefficient using the Kingery–Bulmash data — Karlos, Solomos & Larcher, 2016](https://journals.sagepub.com/doi/10.1177/2041419616659572)
- [Scaled distance — overview](https://www.sciencedirect.com/topics/engineering/scaled-distance)
- [IATG 01.80 — Formulae for ammunition management (blast scaling)](https://data.unsaferguard.org/iatg/en/IATG-01.80-Formulae-ammunition-management-IATG-V.3.pdf)
