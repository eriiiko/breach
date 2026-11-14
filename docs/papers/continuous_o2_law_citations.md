# Citations — continuous O₂→combustion law (2026-07-24)

The continuous-O₂ law files carry these in-header citations (iron rule: "any
file implementing a published technique carries an author + paper citation").
The PDFs themselves should be dropped into `docs/papers/` alongside this note —
**ACTION FOR ERIK: archive the two PDFs below** (not fetched here; Huggett is
paywalled, Peatross & Beyler is an IAFSS symposium paper often free via
iafss.org / publications.iafss.org).

## Papers

1. **Peatross, M.J. & Beyler, C.L. (1997).** "Ventilation effects on compartment
   fire behavior." *Fire Safety Science* 5:403–414 (IAFSS 5th International
   Symposium). — The empirical result that compartment burning rate declines
   **~linearly** with O₂ volume (mole) fraction below ambient: the shape the
   continuous law adopts (linear, not a step). Extinction-limit context
   (~13–16 vol-% O₂) from Beyler, *SFPE Handbook of Fire Protection Engineering*,
   flammability-limits chapter.
   - Cited in: `cpp/src/fire_simulation.cpp`, `cpp/src/fire_simulation.h`,
     `cpp/src/combustion.cpp`, `cpp/src/combustion.h`.

2. **Huggett, C. (1980).** "Estimation of rate of heat release by means of
   oxygen consumption measurements." *Fire and Materials* 4(2):61–65. — Oxygen-
   consumption calorimetry (~13.1 MJ per kg O₂ consumed): the physical anchor
   for the `burn_rate` / `H_fuel` (heat-per-unit-O₂) scale, motivating the drop
   to the ceiling_h-anchored `burn_rate = 0.02`.
   - Cited in: `cpp/src/combustion.cpp`, `cpp/src/combustion.h`.

## Issue #7 — the pressure factor on the law (2026-09-23)

Erik's ruling multiplies the fraction law by `g = clamp01((p − p_ext) /
(p_full − p_ext))` with `p_ext = 0.1 atm`, `p_full = 0.5 atm`
(`[physics.fire] p_ext_atm / p_full_atm`; law in
`cpp/src/o2_pressure_factor.h`). What the edges rest on, and how firmly:

3. **Harper, S.A., Juarez, A., Perez, H., Hirsch, D.B. & Beeson, H.D. (2016).**
   "Oxygen Partial Pressure and Oxygen Concentration Flammability: Can They Be
   Correlated?" *Journal of ASTM International* (NASA NTRS 20160001047). —
   NASA-STD-6001 Test 1 self-extinguishment thresholds of 22 aerospace
   materials over 2.8–119.3 kPa: "little relation to total pressures above
   41 kPa (6 psia). Below 41 kPa (6 psia), MOCs and required oxygen partial
   pressures show increased dependence on total pressure." The knee `p_full`
   stands for. VERIFIED from the full text. ARCHIVED:
   `harper2016_o2_partial_pressure_vs_concentration_flammability_nasa.pdf`.
   Caveat: polymers, not cellulose.

4. **Hirsch, D., Williams, J. & Beeson, H. (2006).** "Pressure Effects on
   Oxygen Concentration Flammability Thresholds of Materials for Aerospace
   Applications." *Journal of Testing and Evaluation* (NASA NTRS 20070005041).
   — Maximum O₂ concentrations for self-extinguishment move only slightly
   between 48.2 and 101.3 kPa (e.g. polyethylene 17 → 18 %). VERIFIED from the
   full text. ARCHIVED:
   `hirsch2006_pressure_effects_o2_flammability_thresholds_nasa.pdf`.
   Caveat: polymers, not cellulose.

5. **He, X., Wang, J. & Fang, J. (2021).** "Flammability limits and near-limit
   chemistry controlled flame spread over thermally thin paper under
   sub-atmospheric pressure." *Fire Safety Journal* 120:103042,
   doi:10.1016/j.firesaf.2020.103042. — Thin CELLULOSE (paper): limiting O₂
   concentrations quantified across 4–45 kPa; flame spread there is near-limit
   and chemistry-controlled. VERIFIED: the abstract only (Semantic Scholar);
   the LOC-vs-pressure numbers — and so the exact pressure at which the LOC
   reaches 21 % (the air extinction pressure) — are behind the paywall and
   were NOT read. **ACTION FOR ERIK: archive the PDF.**

Not cited in the config (seen only through search-engine summaries, not
verified against the papers): Fang, J., He, X., Li, K., Wang, J. & Zhang, Y.
(2018), *Combustion and Flame* 188:90–93, doi:10.1016/j.combustflame.2017.09.010
— reported to place a ~25 kPa transition between the power-law and the
extinction-limit regime of flame spread over thin paper; and Nakamura, Y. &
Aoki, A. (2008), *Advances in Space Research* 41(5):777–782 — irradiated
ignition of thin cellulosic paper from 101 to 20 kPa.

Reading against the ruled edges: the measured knee (~41 kPa for the NASA
materials; ~25 kPa reportedly for paper) sits at or below `p_full = 0.5 atm`,
so the ruled ramp starts weakening a fire somewhat EARLIER than the
literature's knee — conservative, not contradictory. `p_ext = 0.1 atm`
(10 kPa) lies inside the 4–45 kPa band that study mapped for paper, but where
paper's AIR crossover sits within it could not be read (paywall), so 0.1 atm is
neither confirmed nor contradicted. No material disagreement found; the values
stand as ruled.
