# Levelgen litsearch 2026-10-29 — source ledger (#132)

Companion to [`levelgen_litsearch_2026-10-29.md`](levelgen_litsearch_2026-10-29.md).
One row per source. **Status:** FULL = full text read · ABS = abstract,
summary or a secondary account only · PAY = paywalled (listed below for
Erik to fetch through Lund) · NF = not found. **Archived** = PDF in
[`docs/papers/levelgen/`](../papers/levelgen/).

## (a) Grammars and cyclic generation

| Id | Source | Status | Where | Archived |
|---|---|---|---|---|
| A1 | Dormans 2010, Adventures in level design: generating missions and spaces for action adventure games, PCGames. DOI 10.1145/1814256.1814257 | FULL | https://www.pcgworkshop.com/archive/dormans2010adventures.pdf | `dormans2010_adventures_missions_spaces.pdf` |
| A2 | Dormans 2011, Level design as model transformation: a strategy for automated content generation, PCG workshop | FULL | https://www.pcgworkshop.com/archive/dormans2011design.pdf | `dormans2011_level_design_as_model_transformation.pdf` |
| A3 | Togelius, Shaker, Dormans, Grammars and L-systems with applications to vegetation and levels, PCG book ch. 5 (Springer 2016). DOI 10.1007/978-3-319-42716-4_5 | FULL | http://pcgbook.com/chapter05.pdf | no (8 MB; free online) |
| A4 | Boris the Brave 2021, Dungeon generation in Unexplored (blog) | FULL | https://www.boristhebrave.com/2021/04/10/dungeon-generation-in-unexplored/ | web |
| A5 | Thompson 2021, Unexplored's secret: cyclic dungeon generation (Game Developer) | FULL | https://www.gamedeveloper.com/design/unexplored-s-secret-cyclic-dungeon-generation- | web |
| A6 | van Rozen & Heijn 2018, Measuring quality of grammars for procedural level generation, FDG. DOI 10.1145/3235765.3235821 | FULL | https://www.pcgworkshop.com/archive/rozen2018measuring.pdf | `vanrozen_heijn2018_measuring_quality_of_grammars.pdf` |
| A7 | Vijayakumar & Mathew 2024, On graph grammars and games, arXiv 2403.07607 | FULL | https://arxiv.org/abs/2403.07607 | `graph_grammars_and_games_2024_arxiv2403.07607.pdf` |
| A8 | Coxon, Metazelda (open-source key-level dungeon generator), README | FULL | https://github.com/tcoxon/metazelda | web |
| A9 | Boris the Brave 2019, Dungeon generation in Enter the Gungeon (blog) | FULL | https://www.boristhebrave.com/2019/07/28/dungeon-generation-in-enter-the-gungeon/ | web |
| A10 | Benard 2017, Building the level design of a procedurally generated Metroidvania: a hybrid approach (Game Developer) | FULL | https://www.gamedeveloper.com/design/building-the-level-design-of-a-procedurally-generated-metroidvania-a-hybrid-approach- | web |
| A11 | Shaker, Liapis, Togelius, Lopes, Bidarra, Constructive generation methods for dungeons and levels, PCG book ch. 3. DOI 10.1007/978-3-319-42716-4_3 | FULL | http://pcgbook.com/chapter03.pdf | no (free online) |
| A12 | Dormans & Bakkes 2011, Generating missions and spaces for adaptable play experiences, IEEE TCIAIG 3(3):216–228. DOI 10.1109/TCIAIG.2011.2149523 | PAY | IEEE Xplore | — |
| A13 | van der Linden, Lopes, Bidarra 2014, Procedural generation of dungeons, IEEE TCIAIG 6(1):78–89. DOI 10.1109/TCIAIG.2013.2290371 | ABS (author copy https://graphics.tudelft.nl/~rafa/myPapers/bidarra.TCIAIG.2014.pdf unreachable 2026-10-29) | — | — |
| A14 | Karavolos, Bouwer, Bidarra 2015, Mixed-initiative design of game levels: integrating mission and space into level generation, FDG | ABS (TU Delft server refused) | https://graphics.tudelft.nl/Publications-new/2015/KBB15 | — |
| A15 | D. Adams 2002, Automatic generation of dungeons for computer games, BSc thesis, U. Sheffield | NF (URL dead, archive.org not reachable from this machine) | — | — |
| A16 | Lavender 2015, The Zelda dungeon generator (cited in Gutierrez & Schrum 2020, arXiv 2001.05065) | NF (citation only) | — | — |
| A17 | Smith, Whitehead, Mateas 2011, Tanagra: reactive planning and constraint solving for mixed-initiative level design, IEEE TCIAIG 3(3):201–215. DOI 10.1109/TCIAIG.2011.2159716 | PAY (G. Smith's dissertation chapter is open: https://escholarship.org/uc/item/0fn558gq) | — | — |

## (b) Embedding a planar room graph into a tile ship

| Id | Source | Status | Where | Archived |
|---|---|---|---|---|
| B1 | Ma, Vining, Lefebvre, Sheffer 2014, Game level layout from design specification, Computer Graphics Forum 33(2):95–104. DOI 10.1111/cgf.12314 | FULL (author preprint) | http://chongyangma.com/publications/gl/2014_gl_preprint.pdf · code https://github.com/chongyangma/LevelSyn | `ma2014_game_level_layout_from_design_specification.pdf` |
| B2 | Nepožitek & Gemrot, Fast configurable tile-based dungeon level generator (Charles Univ.) + Edgar docs (intro, performance tips) | FULL | https://artemis.ms.mff.cuni.cz/main/papers/Fast_Configurable_Tile_Based_Dungeon_Level_Generator.pdf · https://github.com/ondrejnepozitek/edgar-dotnet · https://ondrejnepozitek.github.io/Edgar-DotNet/docs/other/performance-tips | `nepozitek_gemrot_fast_configurable_tile_dungeon_generator.pdf` |
| B3 | Enter the Gungeon layout (= A9) | FULL | see A9 | web |
| B4 | Unexplored grid embedding (= A4) | FULL | see A4 | web |
| B5 | Kazemi, Spelunky generator lessons, parts 1–2 (interactive) | FULL | https://tinysubversions.com/spelunkyGen/ · https://tinysubversions.com/spelunkyGen2/ | web |
| B6a | Mirahmadi & Shami 2012, A novel algorithm for real-time procedural generation of building floor plans, arXiv 1211.5842 | FULL | https://arxiv.org/abs/1211.5842 | `mirahmadi_shami2012_floor_plans_treemap_corridors_arxiv1211.5842.pdf` |
| B6b | Marson & Musse 2010, Automatic real-time generation of floor plans based on squarified treemaps algorithm, Int. J. Computer Games Technology. DOI 10.1155/2010/624817 | ABS (open access; PDF did not download from Hindawi here) | https://doi.org/10.1155/2010/624817 | — |
| B6c | Merrell, Schkufza, Koltun 2010, Computer-generated residential building layouts, ACM TOG 29(6). DOI 10.1145/1882261.1866203 (from memory — verify) | FULL | https://www.cs.princeton.edu/courses/archive/spring11/cos598A/pdfs/Merrell10a.pdf | no (9 MB) |
| B6d | Lopes, Tutenel, Smelik, de Kraker, Bidarra 2010, A constrained growth method for procedural floor plan generation, GAME-ON 2010 | ABS (https://graphics.tudelft.nl/~rafa/myPapers/bidarra.GAMEON10.pdf timed out 2026-10-29 — retry) | — | — |
| B7 | Buchsbaum, Gansner, Procopiuc, Venkatasubramanian, Rectangular layouts and contact graphs (ACM Trans. Algorithms 2008), arXiv cs/0611107 | FULL (intro, definitions, results; proofs skimmed) | https://arxiv.org/abs/cs/0611107 | `buchsbaum2008_rectangular_layouts_contact_graphs_arxiv_cs0611107.pdf` |
| B8a | Nelson & Smith, ASP with applications to mazes and levels, PCG book ch. 8. DOI 10.1007/978-3-319-42716-4_8 | FULL | http://pcgbook.com/chapter08.pdf | no (free online) |
| B8b | Smith & Mateas 2011, Answer set programming for procedural content generation: a design space approach, IEEE TCIAIG 3(3):187–200. DOI 10.1109/TCIAIG.2011.2158545 | ABS (author-site copy is a 4-page excerpt) → PAY for full | https://adamsmith.as/papers/tciaig-asp4pcg.pdf | — |
| B8c | Smith, Padget, Vidler 2018, Graph-based generation of action-adventure dungeon levels using answer set programming, FDG 2018. DOI 10.1145/3235765.3235817 | ABS (Bath repository returned 403) | https://researchportal.bath.ac.uk/en/publications/graph-based-generation-of-action-adventure-dungeon-levels-using-a/ | — |
| B8d | Karth & Smith 2017, WaveFunctionCollapse is constraint solving in the wild, FDG 2017. DOI 10.1145/3102071.3110566 | FULL | https://escholarship.org/uc/item/1f29235t | `karth_smith2017_wfc_constraint_solving_in_the_wild.pdf` |
| B9 | Dead Cells (= A10); Tanagra (= A17) | see A10 / A17 | — | — |
