# TO ARCHIVE — the swarm larva law's references (arc #63 P3, 2026-11-15)

Project iron rule: *"any file implementing a published technique carries an
author + paper citation in its header comment; archive the paper under
`docs/papers/`."*

The citations are in the headers (`cpp/src/swarm.h`, `cpp/src/swarm_larva.h`,
`cpp/src/swarm_larva.cpp`, `cpp/src/swarm_larva.cu` and the module docstring of
`src/simulation/swarm_larva.py`). **The PDFs are NOT here**: the three open-access
papers are in PubMed Central, but the automated fetch from the build machine was
refused (Europe PMC's render endpoint answered with an HTML page, not the PDF).
This file is the honest placeholder, the `README_radiation_2026-08.md` precedent:
no fabricated archive file was created. Drop the PDFs beside it under the names
below and delete their sections here.

## 1. The head sweep (`luo2010_drosophila_thermotaxis_jneurosci.pdf`)

> Luo L, Gershow M, Rosenzweig M, Kang K, Fang-Yen C, Garrity PA, Samuel ADT.
> **"Navigational decision making in *Drosophila* thermotaxis."** *J Neurosci*
> 2010;30(12):4261-72. doi:10.1523/JNEUROSCI.4090-09.2010 (PMC2871401).

What P3 uses: larvae regulate run length (the tumble rate, raised to the alarm
rate when both sweep points are worse) and turn direction (toward the better
side with probability `turn_bias`). Honest framing: Luo describes the input as
TEMPORAL variation during head sweeps; the law's two-point spatial comparison is
the discrete stand-in for one sweep, not a claim that larvae compute a spatial
gradient. Turn SIZE regulation is an ACCEPTED GAP.

## 2. Sensory dynamics (`klein2015_thermotaxis_sensory_dynamics_pnas.pdf`)

> Klein M, Afonso B, Vonner AJ, ... Garrity PA, Samuel ADT (the full author
> list is the paper's). **"Sensory determinants of behavioral dynamics in
> *Drosophila* thermotaxis."** *PNAS* 2015;112(2):E220-9.
> doi:10.1073/pnas.1416212112 (PMC4299240).

What P3 uses: nothing directly; it is the graded response the sign-only law omits
(ACCEPTED GAP, ruling 2).

## 3. Chill coma (`macmillan2011_chill_coma_jip.pdf`, if a legal copy is obtainable)

> MacMillan HA, Sinclair BJ. **"Mechanisms underlying insect chill-coma."**
> *J Insect Physiol* 2011;57(1):12-20. doi:10.1016/j.jinsphys.2010.10.004.
> (Elsevier, not in PMC.)

What P3 uses: CTmin and reversible chill coma, the `T_ctmin` / `T_coma_exit`
rows (ruling 5: enter below 10 C, wake above 12 C).

## 4. Heat limits (`monzon2022_blowfly_high_temperature_limits_insects.pdf`)

> Monzon MA, Weidner LM, Rusch TW, Nehrozoglu S, Hamilton G. **"High Temperature
> Limits of Survival and Oviposition of *Phormia regina* (Meigen) and *Lucilia
> sericata* (Meigen)."** *Insects* 2022;13(11):991. doi:10.3390/insects13110991
> (PMC9693050).

What P3 uses: adult blowflies over 24 h exposures (about 100 % survival at 37 C,
about 50 % at 41 C, 0 % at 44 C). The instant kill above 50 C (ruling 4) is
Erik's game compression of an exposure-time law, above the 24 h total-mortality
point.

## 5. Philox — already archived

`salmon2011_random123_sc11.pdf` (Salmon, Moraes, Dror, Shaw, SC'11): the vendored
`cpp/src/philox32.h` every larva draw goes through.
