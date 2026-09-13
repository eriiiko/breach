# Papers behind the ray engine v2 arc (archived 2026-09-13)

Erik: *"the papers we base these things off, do we have access to them, can we
download them and put them into docs/papers? I want to make sure we've read
them in full."* All eight are here, verified complete, and text-extractable.

**Reading PDFs on this machine:** `pdftoppm` is not installed, so the Read tool
cannot render pages. `pdftotext` **is** available at `/mingw64/bin/pdftotext`,
so the route in is:

```
pdftotext -q docs/papers/<name>.pdf <out>.txt
```

Every paper below extracts cleanly that way.

---

## Reading order, and why each one matters

### 1. Osborne & Sannikov 2024 — read this one first
`osborne_sannikov2024_radiance_cascades_nonLTE_bilinear_fix.pdf` · arXiv
2408.14425 · 21 pp · RAS Techniques and Instruments

**The most important paper for this arc, and it partly rewrites the design's
framing.** Sannikov (who invented radiance cascades for Path of Exile 2) with an
astrophysicist, presenting cascades not as a rendering trick but as a **formal
solver for the radiative transfer equation**, validated against a solar
prominence model.

Why it matters to us, in their own words:

- They name our exact situation: *"This global and recursive problem does not
  occur uniquely in astrophysical radiative transfer but is also a core problem
  of energetic neutron transport in reactor design, and the problem of global
  illumination in computer graphics."* All three are the same problem. We had
  been treating the graphics literature and the transport literature as separate
  worlds; they are not.
- **The ray effect is worst exactly where we live.** *"This method is known to
  fail, producing so-called ray effects clearly visible in media without
  scattering … but these effects are dramatically reduced in diffusive media."*
  Erik ruled clear air transparent (R-L), so our medium is non-scattering — the
  worst case for discrete ordinates, and the reason our 8-ray fan struck.
- They put short characteristics + discrete ordinates head to head with cascades
  on a scene of *"opaque emissive circles, two small opaque absorbing circles,
  and a central square absorbing region … the remaining space is pure vacuum"*.
  That is a ship deck with fires in it. Short characteristics *"exhibits very
  strong ray effects that are not present in our radiance cascades solutions."*
- **Cascades are offered as the cure for ray effects**: *"a credible route for
  routinely performing multidimensional radiative transfer calculations free
  from so-called ray effects."*

**And the finding that decides our architecture** (§ around Fig. 7): the ringing
artifact *"is due to parallax between the different probes of cascade i+1 and
cascade i, leading to **non-conservation of energy** at locations where cascades
overlap a light source."*

So: **cascades do not conserve energy exactly, by the authors' own statement.**
That is the justification for the design's two-solver split. Heat must close the
arc-#54 ledger exactly and therefore cannot use cascades; light has no such
constraint and should. The split was proposed on intuition and is now on record.

Also carries the **bilinear fix** — the published remedy for the leak visible in
our own prototype render (`docs/ray_engine_v2_scheme_study_2026-09-13/cascade_render.png`,
where light appears in rooms that should be dark).

### 2. Sannikov 2023 — the original
`sannikov2023_radiance_cascades.pdf` · 59 MB slide deck from ExileCon ·
[source](https://radiance.wiki/papers/sannikov-original)

The primary source for radiance cascades: the penumbra hypothesis, the cascade
hierarchy, the merge. A deck rather than a paper, so it is light on formalism
and heavy on intuition and pictures. Text extracts fine.

### 3. Yates et al. 2025 — Holographic Radiance Cascades
`yates2025_holographic_radiance_cascades.pdf` · arXiv 2505.02041 · 9 pp

The current refinement, and the source of the performance numbers the design
quotes: **1.85 ms at 512×512, 7.67 ms at 1024×1024 on an RTX 3080 Laptop, with
runtime constant for a given grid size regardless of scene complexity.** Fixes
distant-light penumbras by aligning the high-resolution probe axis with the
incoming light direction. Read for the cost model and the limitations list.

### 4. Camminady et al. 2019 — ray effect via quadrature rotation
`camminady2019_ray_effect_quadrature_rotation.pdf` · arXiv 1808.05846 · 24 pp

The paper behind the design's per-tick rotation, and the claim that rotation
buys equivalent quality at **under a third** of the quadrature points. Worth
reading carefully against our own measurement, which found rotation *helps*
apertures and *hurts* point-source isotropy in a grid scheme — the paper may
explain the asymmetry we measured.

### 5. Random Source Iteration, 2025 — the other ray-effect remedy
`random_source_iteration_ray_effect_2025.pdf` · arXiv 2502.15454 · 25 pp

A newer alternative to rotation. Likely incompatible with us as written, since
it is stochastic and our ingress rules put randomness behind door 4, but worth
knowing what it buys before ruling it out.

### 6. Davis et al. 2012 — short characteristics in Athena
`davis2012_short_characteristics_athena.pdf` · arXiv 1201.2222 · 20 pp

A production short-characteristics solver, described concretely enough to
implement. The sharp-transport candidate in our bake-off, and the place to look
for how a real code handles the interpolation and stability choices Auer &
Paletou flagged.

### 7. PLUTO 2020 — radiation transport module
`pluto2020_radiation_transport_module.pdf` · arXiv 2010.00457 · 19 pp

Another production sweep implementation. Its discussion of shadow quality is
directly relevant: it names *"the protrusion of the ionization front into the
shadow"* as numerical diffusion, which is the artifact our room render shows as
step differencing losing half the energy down a corridor.

### 8. Osborne 2025 — a simple ray acceleration structure
`osborne2025_ray_acceleration_structure_nonLTE.pdf` · arXiv 2511.08498 · 12 pp

Follow-up to (1). Relevant if cascade tracing cost becomes the bottleneck.

---

## Already here from earlier arcs

- `ADA492343.pdf` — see `README_radiation_2026-08.md`.
- `continuous_o2_law_citations.md` — the combustion-law citations.

## The rule this satisfies

CLAUDE.md: *"Credit the source: any file implementing a published technique
carries an author + paper citation in its header; archive the paper under
`docs/papers/`."* When the sweep and the cascade passes get written, their
headers cite (1), (2), (4) and (6) by name.

## Still unread in full

Downloaded and verified, but only (1) has been read closely so far, and only its
introduction, method framing and the conservation statement. **The full read is
the next session's first job**, per Erik's instruction. Highest value per page,
in order: (1) the merge and the bilinear fix sections; (4) the whole paper,
against our rotation measurement; (6) the interpolation and stability choices.
