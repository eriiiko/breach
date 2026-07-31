# TO ARCHIVE — the P-R4 radiation references (2026-08-01)

Project iron rule: *"any file implementing a published technique carries an
author + paper citation in its header comment; archive the paper under
`docs/papers/`."*

The citations are in the headers (`cpp/src/raycaster.h`, and the exchange /
limiter sites in `cpp/src/raycaster.cpp` + `cpp/src/cuda_raycaster.cu`). **The
PDFs are NOT here**: neither could be fetched from the machine this patch was
built on. This file is the honest placeholder — no fabricated archive files were
created. Drop the PDFs beside it and delete this note.

## 1. Net exchange, view factors, `E°(T) = σT⁴`

> J. R. Howell, M. P. Mengüç, R. Siegel, **"Thermal Radiation Heat Transfer"**,
> 6th ed., CRC Press, 2016. ISBN 978-1-4665-9326-8.
> (Ch. 5 — radiative exchange between grey diffuse surfaces; Ch. 4 — the
> configuration/view factor `F`.)

**What P-R4 uses.** The net radiative exchange between two grey diffuse
surfaces,

```
Q_net(1→2) = a_1 · a_2 · F_12 · A · ( E°(T_1) − E°(T_2) ),   E°(T) = σT⁴
```

is the deposit law in `march_ray_directional`'s radiation block. Two properties
of that form are load-bearing here and are why the arc adopted it:

* it is **antisymmetric** — the same quantity leaves one end and arrives at the
  other, so two equal-temperature emitters exchange exactly zero and the
  divergence hazard is impossible by construction, not by tuning;
* with **Kirchhoff's law** (ε = a, so the pair coefficient `a_1·a_2` is
  symmetric in 1↔2) the two directions of a pair exchange at the same rate,
  which is what lets the engine apply ONE truncated integer `+` to one end and
  `−` to the other and conserve exactly.

The 8-ray fan is our **discrete view-factor sampler**: "how many rays connect
the pair" is `F`. It is a sampler with visible aliasing (a tile is either on a
ray line or not) — a documented approximation, not a hidden one.

Free equivalents if the book is unavailable: R. Siegel & J. Howell,
*Thermal Radiation Heat Transfer* (NASA SP-164, 1968–71) is in the NASA
Technical Reports Server and covers the same net-exchange derivation.

## 2. The flux limiter

> C. D. Levermore, G. C. Pomraning, **"A Flux-Limited Diffusion Theory"**,
> *The Astrophysical Journal* **248**, 321–334 (1981).
> DOI 10.1086/159157 — open on ADS (bibcode 1981ApJ...248..321L).

**What P-R4 uses.** Radiative transfer linearised about a temperature has an
effective coefficient that steepens as `T³` (`d(T⁴)/dT = 4T³`), so an explicit
update stable at one temperature is not stable at another. The flux-limited
treatment caps the transfer at a fraction of the physically available flux.
Ours is the discrete analogue: per pair, per ray, per tick, the transfer may not
exceed `1/2^RAD_LIM_SHIFT` of the temperature gap through either end's own
thermal mass (`RAD_LIM_SHIFT = 4`, i.e. 1/16 of the gap per ray; 8 rays ⇒ half
the gap per tick worst case — 2× inside conduction's own monotone line and 4×
from divergence). A power-of-two shift for the same reason `cool_shift` and
`face_shift` are: one arithmetic shift, no multiply, deterministic.

It is a **stability constant, not a feel dial**, and it is inert in normal
operation — the T⁴ net sits far below the budget at game temperatures. It exists
as a rail against the T³ steepening at `T_MAX_PHYS`-scale gaps.

## 3. Already archived, still relevant

* `docs/papers/continuous_o2_law_citations.md` — Peatross & Beyler 1997 (linear
  burning rate vs O₂ volume fraction) and Huggett 1980 (oxygen-consumption
  calorimetry). P-R4's `H_bed` is Huggett-**shaped** (∝ O₂ consumed) but
  deliberately not Huggett-**valued**; see `cpp/src/combustion.h`.
