# Explosion litsearch 2026-11-06 — source ledger (#31)

Companion to [`explosion_litsearch_2026-11-06.md`](explosion_litsearch_2026-11-06.md). One row per source.

**Status:** FULL = full text read · PART = the sections that matter read from the full text, the rest scanned · ABS = abstract, search excerpt or a secondary account only · PAY = paywalled, cited only · NF = not found or blocked.

**Archived** = PDF in [`docs/papers/explosions/`](../papers/explosions/). "no" means free online but not copied.

## Q1. Delivering energy into a coarse solver

| Id | Source | Status | Where | Archived | Supports |
|---|---|---|---|---|---|
| A1 | Feldman, O'Brien, Arikan 2003, Animating suspended particle explosions, ACM TOG 22(3) 708–715. DOI 10.1145/882262.882336 | FULL | https://escholarship.org/uc/item/4jk7m50m | `feldman_obrien_arikan2003_suspended_particle_explosions.pdf` | Divergence source, its schedule, cell sizes, conforming flow, heuristic values, symmetry warning, oxygen ignored |
| A2 | Yngve, O'Brien, Hodgins 2000, Animating explosions, SIGGRAPH. DOI 10.1145/344779.344801 | PART (method, initial conditions, discussion) | https://arxiv.org/abs/2303.10541 | `yngve_obrien_hodgins2000_animating_explosions.pdf` | Compressible explicit route, hot or high-pressure initial region, wave does not transport fluid far, implicit methods damp it, Mach stem, smoke heuristic |
| A3 | Kwatra, Su, Grétarsson, Fedkiw 2009, A method for avoiding the acoustic time step restriction in compressible flow, JCP 228 4146–4161 | PART (derivation, time step, tests) | repo | already `docs/papers/ADA492343.pdf` | The scheme Breach implements, large dt warning, implicit solve inside RK stages, strong-shock oscillations |
| A4 | Kwatra, Grétarsson, Fedkiw 2010, Practical animation of compressible flow for shock waves, SCA. DOI 10.2312/SCA/SCA10/207-215 | PART (sections 1–3) | https://diglib.eg.org/handle/10.2312/SCA.SCA10.207-215 | no (copyright Eurographics) | Helmholtz form, c → ∞ limit, divergence-term methods "discard the compression waves" |
| A5 | Nielsen, Stamatelos, Bojsen-Hansen, Bridson 2019, Physics-based combustion simulation in Bifrost, SIGGRAPH Talks. DOI 10.1145/3306307.3328149 | FULL (2 pp.) | https://history.siggraph.org/wp-content/uploads/2022/09/2019-Talks-Nielsen_Physics-Based-Combustion-Simulation-in-Bifrost.pdf | no | Divergence derived from the gas law at constant pressure, soot not linear in burn rate, fuel-lean to fuel-rich |
| A6 | Nielsen, Bojsen-Hansen, Stamatelos, Bridson 2022, Physics-based combustion simulation, ACM TOG 41(5) 176 | PAY | ACM | — | Named only; the talk stands in for it |
| A7 | McGrattan et al. 2024, FDS Technical Reference Guide vol. 1, NIST SP 1018-1, 6th ed., FDS-6.9.1. DOI 10.6028/NIST.SP.1018 | PART (ch. 3 pressure, zones, eq. 3.30–3.31, soot yields) | https://github.com/firemodels/fds/releases | `mcgrattan2024_fds_technical_reference_vol1.pdf` | Background pressure, sealed-compartment consistency condition, Rehm & Baum decomposition, fixed soot yield |
| A8 | SideFX, Pyro background (Houdini docs) | FULL via summary tool | https://sidefx.com/docs/houdini/pyro/background.html | web | Divergence field in the projection, explosions |
| A9 | SideFX, Pyro Configure Fireball | ABS (search excerpt) | https://www.sidefx.com/docs/houdini/pyro/pyro_configure_fireball.html | web | "too much and the simulation rapidly explodes" |
| A10 | Rasmussen, Nguyen, Geiger, Fedkiw 2003, Smoke simulation for large scale phenomena, ACM TOG 22(3) | ABS (scanned; no explicit divergence source found in the extracted text, Kwatra 2010 groups it with those methods) | https://graphics.stanford.edu/papers/smoke-sig03/ | no | Context only |
| A11 | Sewall, Galoppo, Tsankov, Lin 2008, Visual simulation of shockwaves, SCA | ABS (scanned) | https://gamma.cs.unc.edu/SHOCK/sewall-sca08.pdf | no | Explicit Riemann solver route, not applicable at our dt |
| A12 | Rehm & Baum 1978, The equations of motion for thermally driven, buoyant flows | ABS (via A7) | J. Res. NBS | — | Low Mach decomposition |
| A13 | Hager, Needham, Doolittle 2006, quasistatic rise-time algorithm, 32nd DoD Explosives Safety Seminar | ABS (via B3) | — | — | Rise time form |

## Game source code (read directly, 2026-11-06)

| Id | Source | Status | Where | Supports |
|---|---|---|---|---|
| G1 | The Powder Toy: `src/simulation/Air.cpp`, `elements/BANG.cpp`, `FIRE.cpp`, `BOMB.cpp`, `SimulationConfig.h` | PART (update loop, constants, pressure writes) | https://github.com/The-Powder-Toy/The-Powder-Toy | Explicit pressure and velocity at 4 px cells; explosives add to `pv` |
| G2 | Cataclysm-DDA: `src/explosion.cpp`, `do_blast` | PART | https://github.com/CleverRaven/Cataclysm-DDA | Dijkstra flood, power law, empty-neighbour bonus |
| G3 | Space Station 14: `ExplosionSystem.cs`, `.TileFill.cs`, `ExplosionGridTileFlood.cs` + API docs | PART (signatures, comments) | https://github.com/space-wizards/space-station-14 | Intensity budget, slope, cap, airtight blocking. The confinement reading is mine |
| G4 | tgstation: `code/controllers/subsystem/explosions.dm`, `code/game/objects/items/tanks/tanks.dm` | PART | https://github.com/tgstation/tgstation | Three bands, summed resistance, tank pressure feeds a table |
| G5 | Noita: 80.lv article, GDC talk page | ABS | https://80.lv/articles/noita-a-game-based-on-falling-sand-simulation/ · https://gdcvault.com/play/1025695/Exploring-the-Tech-and-Design | Pixels become collapsing sand. GDC notes gated |
| G6 | Teardown, Oxygen Not Included | NF | — | No technical account of explosion delivery found |
| G7 | Red Faction Geo-Mod, Rainbow Six Siege RealBlast | NF | — | Search returned nothing usable; not pursued |

## Q2. Confined and geometry effects

| Id | Source | Status | Where | Archived | Supports |
|---|---|---|---|---|---|
| B1 | Farrimond et al. 2024, Experimental studies of confined detonations of plasticized high explosives in inert and reactive atmospheres, Proc. R. Soc. A 480. DOI 10.1098/rspa.2024.0061. CC BY | PART | https://eprints.whiterose.ac.uk/id/eprint/215029/ | `farrimond2024_confined_detonations_inert_reactive.pdf` | QSP form, Weibull and UFC lineage, 3.17 kg air/kg TNT via Edri, afterburn duration |
| B2 | Barr et al. 2025, Fast-running thermochemical predictions of quasi-static pressure, PSEP 202. DOI 10.1016/j.psep.2025.107678. CC BY (AAM) | PART | https://eprints.whiterose.ac.uk/id/eprint/231418/ | `barr2025_fast_running_thermochemical_qsp.pdf` | UFC curve and TNT conversion, UFC overprediction up to 13 % |
| B3 | Oswald 2025, Improved method to calculate gas pressure histories from confined detonations with venting, IESSE | PART | https://www.denix.osd.mil/ddes/denix-files/sites/32/2026/02/045p_IESSE-2025_Oswald_Improved-Gas-Pressure-Methodology_REV-1.pdf | `oswald2025_improved_gas_pressure_method.pdf` | Separate rise times, FRANG conservatism |
| B4 | UFC 3-340-02 (2008, chg. 2 2014), Structures to resist the effects of accidental explosions | NF (WBDG's PDF is a front-matter stub; everyspec gated) | https://www.wbdg.org/ | — | Only through B1, B2, B3 |
| B5 | Baker et al. 1983, Explosion hazards and evaluation, Elsevier | PAY (book) | — | — | Not read |
| B6 | Weibull 1968; Anderson et al. 1983, Int. J. Mech. Sci. 25; Edri et al. 2011–12 | ABS (via B1) | — | — | QSP lineage |
| B7 | Tunnel and corridor studies: Colorado School of Mines exploding-wire thesis; long road tunnel field tests; barrier studies | ABS (search excerpts) | https://repository.mines.edu/entities/publication/d71d01e9-a617-4f84-9a0b-1b48f02ac89e | — | Cube-root scaling with a geometry factor. No door data |

## Q3. Afterburn

| Id | Source | Status | Where | Archived | Supports |
|---|---|---|---|---|---|
| C1 | Dobratz & Crawford 1985, LLNL Explosives Handbook, UCRL-52997 chg. 2, Table 5-2 (scanned; read by eye) | PART (table) | https://www.osti.gov/biblio/6530310 | `dobratz_crawford1985_llnl_handbook_table5-2_heats_of_detonation.pdf` (4 pp. excerpt; the whole is 44 MB) | Heats of detonation for TNT, RDX, Comp B, C-4 |
| C2 | NIST Chemistry WebBook, TNT and RDX, condensed phase thermochemistry | FULL via summary tool | https://webbook.nist.gov/ | web | ΔcH° TNT −3410 ± 20 kJ/mol; RDX −2092 to −2120 kJ/mol |
| C3 | Kuhl, Forbes, Chandler, Oppenheim, Spektor, Ferguson 1998, Confined combustion of TNT explosion products in air, UCRL-K-131748 | PART | https://www.osti.gov/biblio/3648 | `kuhl1998_confined_combustion_tnt_products.pdf` | 18.9 ms lifetime, 2.8 vs 0.85 bar, 1093 vs 3575 cal/g, 210 kbar, carbon and CO richness |
| C4 | Kuhl, Oppenheim, Ferguson, Reichenbach, Neuwald 2000, Effects of confinement on combustion of TNT explosion products in air, UCRL-JC-137418 | FULL | https://www.osti.gov/biblio/793989 | `kuhl2000_effects_of_confinement_tnt_combustion.pdf` | 3.5x regardless of volume, half burned in the smallest chamber |
| C5 | Kuhl, Ferguson, Oppenheim, Seizew 2000, Turbulent mixing & combustion in TNT explosions, UCRL-JC-141574 | FULL (short) | https://www.osti.gov/servlets/purl/15006079 | `kuhl2001_turbulent_mixing_combustion_tnt.pdf` | Mixing-limited picture |
| C6 | Kuhl et al., Gasdynamic model of turbulent combustion in TNT explosions | PART | https://www.osti.gov/servlets/purl/988949 | no | L/D suppresses mixing and combustion |
| C7 | Ornellas 1982, Calorimetric determinations of the heat and products of detonation, UCRL-52821 | ABS (via C1, C3 and the config comment) | — | — | Not found directly |
| C8 | Donahue, Zhang, Ripley 2013, Numerical models for afterburning of TNT detonation products in air, Shock Waves 23 | PAY; ABS (search excerpt; a proxy copy was refused and is not used) | DOI 10.1007/s00193-013-0467-2 | — | 86 % figure, attribution uncertain |
| C9 | Türker 2016, Thermobaric and enhanced blast explosives, Defence Technology 12(6) 423–445. DOI 10.1016/j.dt.2016.09.002 | ABS (open access, full text not retrieved) | https://doi.org/10.1016/j.dt.2016.09.002 | — | TBX vs EBX |
| C10 | Case Studies in Thermal Engineering, Nov 2024, thermobaric after-burning study | ABS | https://doaj.org/article/9d71eedf6c014acb8e569582b2c9015e | — | Impulse and duration vs TNT |
| C11 | FAS, Chemical explosives (Navy ES310 notes) | ABS | https://man.fas.org/dod-101/navy/docs/es310/chemstry/chemstry.htm | web | Comp B-3 OB −40.5 %, as a check on my −42.6 % for 60/40 |

## Q4. Soot and smoke

| Id | Source | Status | Where | Archived | Supports |
|---|---|---|---|---|---|
| D1 | Sunderland, Urban, Stocker, Chao, Axelbaum 2002, Sooting limits of microgravity spherical diffusion flames | PART (introduction) | https://ntrs.nasa.gov/citations/20020027073 | `sunderland2002_sooting_limits_spherical_diffusion_flames.pdf` | 1250–1650 K, C/O criterion, residence time |
| D2 | Tewarson 2008, Smoke emissions in fires, Fire Safety Science 9 | PART | https://publications.iafss.org/publications/fss/9/1153/view/fss_9-1153.pdf | `tewarson2008_smoke_emissions_in_fires.pdf` | Efficiency against equivalence ratio, CO and soot correlation |
| D3 | Köylü & Faeth 1991, Carbon monoxide and soot emissions from liquid-fueled buoyant turbulent diffusion flames, Combust. Flame | ABS | https://scholarsmine.mst.edu/mec_aereng_facwork/2197 | — | Uniform generation, 0.37 kg CO/kg soot |
| D4 | ANSYS Fluent theory guide, soot model section (ENEA mirror of Fluent docs) | PART (equation via alt text) | https://www.afs.enea.it/project/neptunius/docs/fluent/html/th/node229.htm | web | Khan–Greeves form. Constants not shown |
| D5 | Khan & Greeves; Hiroyasu; Moss–Brookes; Kennedy 1997 review | ABS (via D4 and search) | — | — | Model family. Kennedy not read |
| D6 | Soot bell curve: arXiv 2603.14314 (methane pyrolysis, 2203 K peak); MIT thesis dspace 1721.1/30330 (1800–2000 K) | ABS | https://arxiv.org/abs/2603.14314 | — | High-temperature fall-off |
| D7 | Under-ventilated laminar flames, smoke mainly organic (NIST-indexed) | ABS | https://firedoc.nist.gov/article/LHcyXYQBWEcjUZEYlqIU | — | Character of starved smoke |
| D8 | Gottuk & Lattimer, Tewarson, SFPE Handbook chapters | PAY | — | — | The φ correlation for yields was not verified, so it is not quoted |
| D9 | Madorsky, Straus and co-workers, vacuum pyrolysis of polymers, J. Res. NBS | ABS | https://nvlpubs.nist.gov/nistpubs/jres/066/5/V66.N05.A04.pdf | — | Volatile fragments and residue in vacuum |
| D10 | Formation of nanoclusters in expanding laser plume (ANU) | ABS | https://researchportalplus.anu.edu.au/en/publications/formation-of-nanoclusters-in-expanding-laser-plume/ | — | Size set by expansion in vacuum |

## Q5. Knockdown

| Id | Source | Status | Where | Archived | Supports |
|---|---|---|---|---|---|
| E1 | Bowen, Albright, Fletcher, White 1961, A model designed to predict the motion of objects translated by classical blast waves, US AEC | PART (ch. 1, 5, Table 5.1) | https://archive.org/details/DnaEm1CapabilitiesOfNuclearWeapons | `bowen1961_motion_of_objects_translated_by_blast.pdf` | Acceleration coefficients, drag dominates, dummy test |
| E2 | Hof, Gazendam, Sinke 2005, The condition for dynamic stability, J Biomech 38. DOI 10.1016/j.jbiomech.2004.03.025 | ABS (PubMed) | https://doi.org/10.1016/j.jbiomech.2004.03.025 | — | Extrapolated centre of mass |
| E3 | Kamimori et al. 2018, Mil Med 183 suppl 1. DOI 10.1093/milmed/usx186 | ABS (PubMed) | https://doi.org/10.1093/milmed/usx186 | — | 4 psi limit |
| E4 | Carr et al. 2016, Mil Med 181. DOI 10.7205/MILMED-D-15-00137 | ABS (PubMed) | https://doi.org/10.7205/MILMED-D-15-00137 | — | Breacher context |
| E5 | FM 8-9, part I ch. 4 | PART | https://nuke.fas.org/guide/usa/doctrine/dod/fm8-9/1ch4.htm | web | Eardrum and lung thresholds, no displacement figure |
| E6 | safteng.net, VCE impact table | ABS, secondary, unverified against Lees or Clancey | https://safteng.net/?p=2300 | web | 1–2 psi knockdown |
| E7 | `docs/research/wind_on_a_person_2026-10-31.md` | FULL | repo | — | 245 Pa, C_dA, q as the right input |
| E8 | TNO Green Book probits; "Improved Displacement Velocity" (DTIC ADA508082); thesis core.ac.uk/download/286448125 | NF (403 from DTIC and CORE) | — | — | Not used |

## Q6. Breaching

| Id | Source | Status | Where | Supports |
|---|---|---|---|---|
| F1 | FM 3-34.214 | NF (excerpts only) | — | — |
| F2 | Secondary excerpts of the breaching formula: 2011 demolitions manual excerpt, study cards | ABS, unverified | https://studfile.net/preview/1715212/page:10/ | `P = R³KC`, C values 1.0–3.6 |
| F3 | Occupational overpressure exposure of breachers and military personnel | ABS | https://opact.sdu.edu.tr/vufind/SummonRecord/FETCH-LOGICAL-c364t-c15c12020ee33a47d7701af7fad5a6150a39739895677a6395cf868dbbea43833 | <1 to >12 psi measured, MSD often inaccurate |
| F4 | Water tamping study, JSOM Winter 2022 | ABS | https://www.jsomonline.com/Flipbook/Journals/2204-Winter/files/basic-html/page59.html | 18.4 % reduction |
| F5 | Reinforced concrete element resistance to contact explosion effects, Carleton thesis | ABS | https://curve.carleton.ca/231956e1-b8aa-4cf2-8b5f-aba48cc547ec | Damage local, charts below Z = 0.4 |
| F6 | Spall mechanism (McVay and similar) | NF | — | Open |

## Q7. Thin air

| Id | Source | Status | Where | Archived | Supports |
|---|---|---|---|---|---|
| H1 | Wang, Liu, Gong 2018, A scaled air blast model for high altitudes, WIT Trans. Built Env. 180 | FULL | https://www.witpress.com/elibrary/wit-transactions-on-the-built-environment/180/36801 | `wang2018_scaled_air_blast_high_altitude.pdf` | Sachs scaled variables |
| H2 | DTIC AD0469014, reduced-pressure blast tests | ABS (search excerpt; 403 on the PDF) | https://apps.dtic.mil/sti/pdfs/AD0469014.pdf | — | Test range to about 0.0007 atm |
| H3 | Formby 1995, IChemE Symposium 139 | ABS (scanned) | https://www.icheme.org/media/25320/major-hazards-onshore-and-offshore-ii-icheme-symposium-series-139-1995-12-formby.pdf | — | Sachs correction in practice |
| H4 | Sachs 1944 | NF | — | — | Original not read |

## Repo inputs read

`docs/lit_research_2026-08-17_multigrid_and_blast.md` §B · `docs/research/eos_gas_model_literature_2026-07.md` · `docs/research/wind_on_a_person_2026-10-31.md` · issue #31 with comments · `config.toml` `[explosives.*]`, `[payloads.*]`, `[physics.eos]` · `cpp/src/eos_solver.h` header, `eos_solver.cpp` Helmholtz assembly · `src/simulation/physics.py::apply_explosion`.

**Toy:** a throwaway float, linear, one-tick replica of the pressure step (uniform initial density, sealed 58 x 38 room, direct sparse solve), run in the session scratchpad. It was not committed. Equations and results are in §1.2 of the report.

**Not available this session:** egregore dropped out partway, so report misses were not filed. `recall` found nothing on divergence sources for #31 beyond the EOS rung nodes.
