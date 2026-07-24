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
