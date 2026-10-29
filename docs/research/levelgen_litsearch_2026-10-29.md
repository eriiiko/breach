# Breach levelgen — literature and practice digest (#132)

*Dated capture, 2026-10-29. Append-only research synthesis, not canon. One
agent, sequential, strand by strand (Erik's research rules). Framed AGAINST
the nine locked decisions of [`docs/breach_levelgen_design_v0.1.md`](../breach_levelgen_design_v0.1.md)
(L1–L9): this digest answers HOW, never WHETHER. The per-source ledger
(full text / abstract only / paywalled, with DOIs for Erik to fetch through
Lund) is [`levelgen_litsearch_2026-10-29_ledger.md`](levelgen_litsearch_2026-10-29_ledger.md).
Open PDFs relied on are archived under [`docs/papers/levelgen/`](../papers/levelgen/).*

**Feeds:** #131 (levelgen umbrella) and its children — #27 vocabulary page,
#133 prefab pieces, #134 vent rules, #135 interactivity rules, #136
ship-design adjacency, #137 pressure cells. **Relates:** #120
(`tools/gen_ship.py`, the v1 generator whose stage 2 `embed()` and gates
`validate()` the grammar reuses), #124 (vents in v1), #48 (vent system),
#76 (RL-ready).

**How to read an entry.** Each source gets: what it is · what it says that
matters to Breach · the ledger status (FULL = I read the full text; ABS =
abstract, summary or secondary account only; PAY = paywalled, listed for
Lund). Every recommendation names the sources it rests on.

---

## Contents

- (a) Grammars and cyclic generation
- (b) Embedding a planar room graph into a tile ship
- (c) PCG for RL generalisation
- (d) Real ship, submarine and station interiors; how games encode "a ship"
- What this means for the design session
- Ten questions for the design session, with recommended answers
- Reading list

---

## (a) Grammars and cyclic generation

### Sources

**A1. Dormans 2010, "Adventures in level design: generating missions and
spaces for action adventure games"** (PCGames workshop, DOI
10.1145/1814256.1814257). FULL, archived.
The founding paper of the mission/space split. A level is two structures:
a *mission* (directed graph of tasks, "which tasks are made available by the
completion of a preceding task") and a *space* (rooms and connections). The
same mission maps to many spaces and vice versa. Missions are made by a graph
grammar (node-numbered LHS → RHS, the 7-step replacement: select, number,
remove edges between matched nodes, replace numbered nodes, add new nodes,
add RHS edges, drop numbers). The mission grammar has two edge kinds —
*tight* coupling (the subordinate must be placed *behind* its superordinate
in space) and *loose* coupling — and that distinction is "very important for
the space generation algorithm". Space is grown by a *shape grammar* whose
rules are each associated with a mission terminal ("each rule in the shape
grammar was associated with a terminal symbol from the mission grammar");
placement locations are chosen by "relative fitness"; dynamic parameters
shift rule weights (progressive difficulty, "registers"); a reconnect step
closes loops when two open connections are "in close proximity and in the
correct alignment" — with a guard that closes off open connections so the
mission cannot be short-circuited. Size control: a minimum-length rule
("Dungeon → Obstacle ×4 + treasure").
*Matters to Breach:* (i) Breach's L9 adjacency-strength classes are the
same idea as Dormans' tight/loose edges — an edge carries an instruction to
the embedder. (ii) The reconnect step with a short-circuit guard is exactly
the risk L4 forbids ("add edge between arbitrary nodes"); Dormans had to
patch it with commands; Breach avoids it by construction. (iii) The paper's
own space grammar is the weak half; Dormans says the mission→space step is
the hard one (see A3).

**A2. Dormans 2011, "Level design as model transformation: a strategy for
automated content generation"** (PCG workshop 2011, ACM; DOI not
verified — the archived copy carries none). FULL, archived.
Reframes generation as a chain of small model transformations, each a
rewrite system that codifies one design principle. Lock-and-key is two
rules that turn a linear task list into a branching structure without
changing the task count ("the size of a level is dictated by the length of
the initial mission"). Rules with wildcards (`?` nodes). A third rule
"moves a lock towards the goal" is shown and then flagged as breaking design
wisdom — a rule can be legal and still wrong. Recommends *small*
transformations ("it is advantageous to make the individual transformations
small") and a mixed-initiative tool where the designer selects the sequence
of rewrite systems (recipe) or applies rules by hand.
*Matters to Breach:* this is the precedent for L6's phases with per-phase
rule subsets, and for L7's recipe library: Dormans' "recipe" is a sequence
of (rule, count) applications.

**A3. Togelius, Shaker & Dormans, "Grammars and L-systems with applications
to vegetation and levels"**, ch. 5 of Shaker, Togelius & Nelson,
*Procedural Content Generation in Games* (Springer 2016, DOI
10.1007/978-3-319-42716-4_5; free at pcgbook.com). FULL (linked, not
archived: 8 MB).
The textbook version. Three practical lessons: (1) "graph grammars can be
hard to control" — uncontrolled application gives one to unbounded tasks;
the fix is to "specify a sequence of rules so that each rule in the sequence
is applied once" (start ×1, add task ×6, add boss ×1, define task ×6, move
lock ×5). (2) Break generation into steps: one step makes a deliberately
random graph, a second restructures it into sense ("designing one simple
step to create a highly randomised graph and using a second step to
restructure that graph ... is an effective strategy"). (3) Three strategies
for mission vs space: transform mission→space; mission→building
instructions; or **build space first and distil the mission from it**, which
"works well for strategic games, levels that take place in locations that
require some consistent architecture (such as castles, dwarf fortresses,
police stations, or space ships)". Its downside: the geometry must carry
enough "mission potential" (doors to lock, bottlenecks for traps).
*Matters to Breach:* this is the textbook's own endorsement of v0.1 L2
(space grammar only) for ships — and its named risk is exactly what L2's
*tactical patterns* phase (L6 phase 2) and L8's validators exist to cover.

**A4. Boris the Brave, "Dungeon generation in Unexplored"** (blog,
2021-04-10). FULL (web).
The best technical account of Dormans' shipped generator. ~50 PhantomGrammar
modules, ~5,000 find-replace rules per floorplan generator. Rules match a
fixed number of nodes, so variable-size structures are written as
*initialise → grow (repeated) → clean up*. **Topology lives on a fixed 5×5
grid graph whose nodes are "never deleted or moved, just annotated"**, which
keeps the graph 2D-embeddable by construction. The generator draws one big
loop with entrance and goal splitting it into two arcs; one of **24 cycle
types** defines how the arcs are used (alternative routes with different
obstacles; ThreeLeverHub; Hubs for multi-exit levels; Lake). Then
*embellishments*: minor cycles, cycle lengthening, dead ends, "until the
level reaches the target size". Lock/key are a non-terminal pair joined by a
special edge so "the pair stays consistent as nodes are shuffled"; hard vs
soft locks; a key can be knowledge or a potion. Abstraction resolves in
stages (path → cave/tunnel/room → typed room → items); *biomes* are early
annotations that coordinate later choices. Embedding: double the grid
(corridor cells between nodes), scale 5×, shape each 5×5 block (rooms grown
to rectangles with a boundary wall; caves by small CA), then set pieces by
pattern matching ("against a wall, not blocking a door"). Quote from
Dormans: "it is so much more interesting to generate small levels than it is
to generate large ones." Boris: the tooling makes content cheap to add "but
it still represents a lot of design work that procedural generation does not
shortcut".
*Matters to Breach:* three direct borrowings — (i) *grid-anchored topology*
is the cheapest planarity-and-embeddability guarantee there is, and Breach's
v1 already has a grid (the BSP leaves); (ii) relationship edges (lock→key,
hint→target) are how annotations survive rewriting — Breach's equivalents
are airlock door pairs, vent supply↔return, spawn↔objective; (iii) biomes =
Breach's lore-flavoured recipes ("Grays-refitted freighter").

**A5. Thompson 2021, "Unexplored's secret: cyclic dungeon generation"**
(Game Developer / AI and Games). FULL (web).
Secondary but adds lock taxonomy: conditional vs non-conditional locks;
temporary, relockable, unidirectional (collapsing bridge) locks; "unsafe"
locks used off the main path. New structure is added *into* existing cycles,
not beside them; average maps contain only a couple of embedded cycles.
*Matters to Breach:* the lock taxonomy maps cleanly onto Breach's edge
types: weldable door = relockable lock; airlock = conditional
(pressure-state) lock; breach charge on a wall = a soft lock that the
*player* can always open at a cost (gas, decompression).

**A6. van Rozen & Heijn 2018, "Measuring quality of grammars for procedural
level generation"** (FDG 2018, DOI 10.1145/3235765.3235821). FULL,
archived.
The authoring-practice paper on Ludoscope (Unexplored's tool). Designers
author "a level transformation pipeline ... by authoring modules, grammars
and rewrite rules"; **recipes "parameterize modules with instructions, that
determine the ordering of rules and limit how often rules work"**. The
observed failure modes: "levers in walls, blocked pathways, missing
encounters and lava adjacent to water"; "Grammars are brittle ... Fixing one
level with a rule that prevents an occurrence may introduce new problems in
others"; bugs "remain unknown until they are observed during playtesting".
Proposes MAD (does a rule add or remove detail — rules should gradually add
detail) and SAnR (a property language evaluated over the *generation
history*, so a broken property is traced to the rule that broke it).
*Matters to Breach:* L7 says LLM-authored rules sandboxed per phase. This
paper says the next thing you need: **provenance** (which rule produced which
node/tile) and **property checks evaluated after every rule application, not
only at the end** — so a bad LLM-authored rule is caught and named. Cheap to
add to a deterministic executor.

**A7. Vijayakumar & Mathew 2024, "On graph grammars and games"** (arXiv
2403.07607). FULL, archived.
Formal-language view: node-replacement (eNCE-family) graph grammars with
*regular control* (a regular language over rule names restricts the order
rules apply) and confluence results, applied to lock-and-key puzzle graphs
(solvable iff a directed begin→end path exists with keys before locks).
*Matters to Breach:* "regular control" is the formal name for L6's phases +
recipes; confluence (order-independence) is worth wanting *within* a phase
so that rule order under a seed does not matter for correctness, only for
variety.

**A8. Metazelda (T. Coxon, open-source generator, GitHub). FULL (README).**
*Key-levels:* "The set of rooms accessible with n keys but inaccessible with
n−1 keys is known as key-level n." Grows a tree on a grid, increments the
key-level at intervals, locks the edge into each new level, places each key
in the highest-*intensity* room of the previous level, then "graphifies" by
adding edges between grid neighbours whose preconditions match (or differ by
one symbol → a conditional edge). Constraints interface (shape, counts,
post-checks) with `RetryException`, up to 20 retries.
*Matters to Breach:* the clean, implementable form of lock-and-key. For
Breach the "key" is rarely an item; it is an **access state** (pressure in
cell k, a door welded, a vent fan on). The key-level idea becomes
*pressure-cell levels*: rooms reachable without cycling an airlock, with one
cycle, etc. — a measurable profile metric.

**A9. Enter the Gungeon (Dodge Roll), via Boris the Brave, "Dungeon
generation in Enter the Gungeon"** (blog 2019). FULL (web).
The most transferable *industrial* pattern for rule authoring: a handful of
**hand-authored flows per stage** (4 in the first, 8 in the second), each a
graph built around one feature (a big loop, a multi-way fork, a route
through the shop), plus **injection rules** (what to insert, where, spawn
probability, prerequisites — e.g. secret rooms prefer dead ends, 90 %
spawn). Concrete rooms chosen from tables without repeats. Generation order:
"they generate the parts of the map that are hardest / most important
first."
*Matters to Breach:* a Breach *archetype* (L2) is very close to a Gungeon
flow + injection table. The flow is the axiom (L3), injections are L4 leaf
attachments with probabilities and prerequisites. "Hardest first" is the
embedding order recommended in strand (b).

**A10. Dead Cells (Benard 2017, "Building the level design of a
procedurally generated Metroidvania")** (Game Developer). FULL (web).
Procedural generation "in the most restrained way possible": the world frame
is hand-made, each biome has a *concept graph* that fixes length, special
rooms, labyrinthness and entrance-to-exit distance, and nodes are filled
with hand-made rooms from a biome pool by try-until-match on entrance count
and type.
*Matters to Breach:* the archetype profile (L2) should carry exactly these
numbers — room-count band, special rooms, loop count, spawn-to-target graph
distance — as *profile parameters*, not as emergent outcomes.

**A11. Shaker, Liapis, Togelius, Lopes & Bidarra, "Constructive generation
methods for dungeons and levels"**, ch. 3 of the PCG book (DOI
10.1007/978-3-319-42716-4_3). FULL (linked).
Background for BSP, agent-based digging, CA and grammar methods; useful
only for vocabulary here.

**A12. Dormans & Bakkes 2011, "Generating missions and spaces for adaptable
play experiences"** (IEEE TCIAIG 3(3):216–228, DOI
10.1109/TCIAIG.2011.2149523). PAY — the journal version of A1/A2.
**A13. van der Linden, Lopes & Bidarra 2014, "Procedural generation of
dungeons"** (IEEE TCIAIG 6(1):78–89, DOI 10.1109/TCIAIG.2013.2290371).
ABS (author copy at TU Delft unreachable today). The survey's thesis: what
dungeon generators lack is "more powerful, accurate and richer control",
not speed. **A14. Karavolos, Bouwer & Bidarra 2015, "Mixed-initiative
design of game levels: integrating mission and space into level generation"**
(FDG 2015). ABS (TU Delft server refused). Reported Ludoscope use on a
platformer and a dungeon crawler. **A15. D. Adams 2002, *Automatic
generation of dungeons for computer games*** (BSc thesis, Sheffield) —
the first graph-grammar dungeon generator (Dormans cites it); NOT FOUND
(the Sheffield URL is dead and the archive is not reachable from here).
**A16. "Lavender & Thompson"** — I found only a citation to Lavender 2015,
"The Zelda dungeon generator", in later work (Gutierrez & Schrum 2020,
arXiv 2001.05065, a GAN-rooms + graph-grammar Zelda generator; Durrheim
2022 Wits MSc on node-replacement lock-and-key grammars); the original NOT
FOUND.

### What the strand says, in one place

1. **Space-only is the textbook's own recommendation for ships** (A3,
   strategy 3). The two-tier mission grammar buys pacing and causality, which
   an action-adventure needs and a tactical/RL arena does not; its cost is the
   hard mission→space step (A1, A3). Breach's *tactical patterns* are the
   honest replacement — but A3's warning applies: the space must be generated
   *with mission potential*, so patterns must be inserted by rules, not hoped
   for.
2. **Cycles are the unit of design, not rooms** (A4, A5). Unexplored starts
   from one cycle with two arcs and a cycle type, then adds embellishment
   cycles. Breach's L4 move 2 (edge→cycle) is the same operation; the *cycle
   type* is what Breach should name and author (retreat loop, flank pair,
   airlock bypass, maintenance ring).
3. **Control comes from recipes, not from the grammar** (A2, A3, A6, A7):
   a sequence of (rule, count) applications per phase, with grow/clean-up
   idioms for variable-size structures (A4). Uncontrolled random rule
   application is the documented failure.
4. **Relationship edges keep pairs consistent through rewriting** (A4, A8):
   lock↔key, hint↔target. Breach needs them for airlock door pairs (L5),
   vent supply↔return (Erik's rule, strand d) and spawn↔target.
5. **Grammars are brittle; provenance is the cure** (A6). Record which rule
   made what; check properties after every application.
6. **Grid-anchored topology makes planarity and embeddability free** (A4).

### Recommendation for Breach (strand a)

Keep L2 (space-only), and make the *cycle type* the main authored object:
an archetype is an axiom plus a small library of named cycle types and
injection rules (A4, A9), applied by a **recipe** — an ordered list of
(phase, rule, count-range) entries — exactly the Ludoscope/PCG-book control
idiom (A2, A3, A6). Write variable-size structures as initialise → grow →
clean-up rule triples (A4). Give the grammar **relationship edges** that are
not adjacencies — `pair` (the two doors of an airlock), `serves` (a vent
supply and its return), `guards` (chokepoint → target) — so pairs survive
rewriting (A4, A8). Make the executor record **provenance per node and per
edge** (rule id + application index) and evaluate the archetype's
invariants after every rule application, so a failing LLM-authored rule is
named, not just rejected (A6). Express Breach's version of key-levels as
*pressure-cell levels* (number of airlock cycles needed to reach a room) and
expose it as a validator metric (A8). Finally, keep the v1 generator's grid
as the topology substrate where it helps: Unexplored's fixed annotated grid
graph is the cheapest way to make every rewrite planar and embeddable (A4) —
see strand (b) for how that combines with the rigid hull.

---

## (b) Embedding a planar room graph into a tile ship

The problem, stated in Breach's terms: given the grammar's graph (nodes with
L9 footprint hints and shape classes; edges with adjacency strength
`must-share-wall` | `may-become-corridor` | `routed-freely`), produce
axis-aligned room rectangles inside a **rigid hull** such that every door
edge is a shared wall at least `DOOR_W` (+ clearance) long, the result is
left-right mirrored, and the v1 gates (`validate`: airtight, A*, reach,
symmetry) pass. The literature splits into three families: **layout by
search over room shapes** (Ma, Nepožitek/Edgar, Gungeon), **subdivision of a
given envelope** (floor-plan literature, BSP/treemaps), and **constraint
solving** (ASP, WFC). Breach's rigid hull puts it in the second family; the
first is right for station fragments.

### Sources

**B1. Ma, Vining, Lefebvre & Sheffer 2014, "Game level layout from design
specification"** (Computer Graphics Forum 33(2):95–104, Eurographics 2014,
DOI 10.1111/cgf.12314). FULL, archived (author preprint).
Input: a planar connectivity graph + a set of polygonal building blocks.
Output: many layouts where blocks don't intersect and every graph edge is a
shared boundary segment "long enough to place a doorway through". Two tools:
(1) **configuration spaces** — for a fixed block and a moving one, the set
of positions where they touch along a parallel edge and don't overlap is a
union of line segments, precomputed per block pair; the configuration space
w.r.t. several placed neighbours is the intersection. (2) **chain
decomposition** — chains (nodes with ≤ 2 neighbours) are easy; decompose
the graph via a planar embedding into faces, lay out *the smallest face
first*, then neighbouring faces, then trees breadth-first; each chain is
added by simulated annealing that emits *several* extended layouts and
**backtracks** to a stored partial layout when an extension fails. The
paper's related-work section is the useful map for Breach: graph drawing
"scale the drawing until blocks no longer touch" is easy but **the contact
requirement is what makes it hard**; architectural floor-plan methods work
"for predefined envelopes, rooms are composed of axis-aligned walls, and no
unoccupied voids are allowed between rooms" — Ma deliberately leaves that
setting, and the general rectangles-in-a-rectangle version is PSPACE-complete
(Hopcroft, Schwartz & Sharir 1984).
*Matters to Breach:* this is the canonical answer for *loose* archetypes
(station fragments, wrecks) where voids are vacuum and there is no hull to
fill. It does not fill a hull.

**B2. Nepožitek & Gemrot, "Fast configurable tile-based dungeon level
generator"** (Charles University; open-source as Edgar). FULL, archived.
Ma on an integer grid with rectilinear polygons: ">100×" faster, layouts in
< 1 s. Adds what Breach needs from Ma: **explicit door positions per room
template** (the configuration-space generator works from declared door
positions — a *port* list), **hard and soft custom constraints** ("make sure
that the whole layout does not exceed some defined boundaries" = a hull
bound), and **short corridors as a second configuration space** (rooms a
fixed distance apart, corridors greedily inserted) instead of corridor
nodes, because corridor nodes "almost twice as many nodes ... slows the
generation down". Edgar's docs: "not suitable for large levels — we
recommend less than 30 rooms"; "Cycles are very hard to lay out so make sure
that there are not too many of them"; give dead-end-only templates only to
degree-1 nodes.
*Matters to Breach:* the port declaration and the hull-bound constraint are
the two features to copy whatever embedder is chosen; the < 30 rooms and
few-cycles limits are a warning against using search-based layout for ships.

**B3. Enter the Gungeon** (via A9). FULL (web).
Pre-authored rooms with **exits** (ports); split the flow graph into **loop
composites** (repeatedly cut out the smallest loop) and **tree
composites**; trees placed depth-first by aligning an exit pair, preferring
exits far from used ones; loops grown alternately from both ends, preferring
opposite walls early and pairs that close the gap later; a loop is closed by
a small rectangular room or, failing that, a **pathfound corridor** (4–30
units); tree failures backtrack up to 3 times. "Generate the parts of the map
that are hardest / most important first."
*Matters to Breach:* the industrial version of B1/B2 — and the source of
the three rules any Breach embedder should adopt: cycles before trees,
hardest/mandatory first, corridor as the loop-closing escape valve (= L9
`may-become-corridor`).

**B4. Unexplored's grid embedding** (A4). FULL (web).
Topology on a fixed grid graph; embed by doubling the grid (connector cells
between nodes), scaling 5× and shaping each block (rectangle with boundary
wall, CA for caves, door shrunk to one tile). Embedding cannot fail because
the topology was never allowed off the grid.
*Matters to Breach:* the strongest guarantee available, at the cost of a
uniform cell size. A ship can recover non-uniform room sizes by letting one
graph node own *several* grid cells (a 2×1 cargo bay), which is exactly what
a slicing tree produces (B6).

**B5. Spelunky** (Kazemi's interactive write-up of Derek Yu's generator,
parts 1–2). FULL (web).
A 4×4 grid of rooms; a random-walk **solution path** that assigns each room
a *type by its guaranteed exits* (0: none; 1: L/R; 2: L/R/bottom; 3: L/R/top);
each type has 8–16 **10×8 tile templates** with probabilistic tiles and 5×3
obstacle chunks; rooms randomly mirrored. Solvability comes from the path,
not from search.
*Matters to Breach:* rooms typed by their port set, templates per type,
re-rollable probabilistic tiles — the minimal prefab system (#133). The
"type by guaranteed exits" idea is how a prefab declares which graph degree
it can serve (B2's dead-end advice in another form).

**B6. Floor-plan subdivision of a fixed envelope.**
- **Mirahmadi & Shami 2012, "A novel algorithm for real-time procedural
  generation of building floor plans"** (arXiv 1211.5842). FULL, archived.
  Rectangular outline; rooms arranged in a **hierarchy tree** by function
  (outside → living → kitchen …; social / service / private areas); the
  outline is divided by a **squarified treemap** with room areas drawn from
  per-function distributions ("divide a region of space into several smaller
  regions with predefined area without any unused space"); then the
  required connections that are not contacts get a **corridor routed along
  the room walls**: build a graph of the walls of the rooms needing
  connection (outer walls excluded), prune degree-1 edges, take the shortest
  path through it, carve it from the rooms. Corridor area is "wasted area"
  and is minimised.
- **Marson & Musse 2010, "Automatic real-time generation of floor plans
  based on squarified treemaps algorithm"** (Int. J. Computer Games
  Technology, DOI 10.1155/2010/624817). ABS (open access, but the PDF did
  not download here). The treemap method B6a extends.
- **Merrell, Schkufza & Koltun 2010, "Computer-generated residential
  building layouts"** (ACM TOG 29(6), SIGGRAPH Asia; DOI
  10.1145/1882261.1866203, from memory — verify). FULL (Princeton course copy, 9 MB, linked).
  A Bayesian network (trained on 120 real programs) expands requirements
  into an *architectural program* (rooms, areas, aspect ratios, which pairs
  connect); a Metropolis optimiser realises it from "a canonical high-cost
  configuration, in which equally-sized rectangular rooms are packed on a
  grid", with two **proposal moves: slide a wall segment** (split + move,
  snap to nearby walls) **and swap two rooms' identities**; cost terms:
  *accessibility* (each required connection must share a sufficient wall),
  area/aspect against the program's distributions, near-convex shapes,
  floor-to-floor compatibility. ~35 s per house.
- **Lopes, Tutenel, Smelik, de Kraker & Bidarra 2010, "A constrained growth
  method for procedural floor plan generation"** (GAME-ON 2010). ABS (TU
  Delft server unreachable today; listed for retry). Rooms *grown* from
  seeds on a grid inside a given outline, with functional constraints, to
  guarantee reachability and connectivity — the hull-as-input growth
  variant the v0.1 doc lists as "incremental growth".
*Matters to Breach:* this family is the one that fills a rigid hull with
no voids — v0.1 §5's "hull as input". Breach v1's `plan_graph` is already a
member (BSP of the wing, then a spanning tree over shared walls). The grammar
version keeps the subdivision and **drives the cut tree from the grammar's
nonterminal hierarchy** (SECTOR → ZONE → rooms), as Mirahmadi drives the
treemap from the room hierarchy; Merrell's wall-slide/swap moves are the
repair step; Mirahmadi's wall-graph corridor is the `may-become-corridor`
realiser.

**B7. Rectangular layouts and duals — the theory under B6.**
**Buchsbaum, Gansner, Procopiuc & Venkatasubramanian 2008, "Rectangular
layouts and contact graphs"** (ACM Trans. Algorithms; arXiv cs/0611107).
FULL, archived.
A *rectangular dual* tiles a rectangle with no gaps so that contacts are
exactly the graph's edges; a *rectangular layout* allows gaps. Results that
bound what any rectangle embedder can promise: a planar triangulation has a
rectangular dual only if it has no separating triangles (Koźmiński &
Kinnen); "the proscription of gaps ... severely limits the class of graphs
that admit rectangular duals"; a planar graph admits a rectangular layout
(gaps allowed) iff it has an embedding with **no filled triangles**
(a 3-cycle with a vertex inside), checkable and constructible in O(n);
minimum-area layout is NP-hard.
*Matters to Breach:* three consequences. (i) Breach does **not** need a
dual: a door edge must be a contact, but a contact need not be a door (two
rooms may share a blank wall), and gaps are allowed (steel, service space,
duct chases). So the requirement is the weaker one. (ii) The L4 moves keep
the graph **triangle-free** provided edge→cycle *replaces* A—B rather than
keeping it alongside A—C—B (subdivision and cycle insertion create cycles of
length ≥ 4; leaf attachment creates none) — and a triangle-free planar graph
has no filled triangle, so a rectangular layout always exists. That is a
one-line rule for the vocabulary (#27) worth writing down: *no rule may
create a 3-cycle of door edges*, or, if one is wanted (three rooms meeting
at a corner), it must be realised as a prefab. (iii) Area is the hard part,
not existence: a hull of fixed size can always fail to fit; the reroll
policy (L8) is the honest answer.

**B8. Constraint solving: ASP and WFC.**
- **Nelson & Smith, "ASP with applications to mazes and levels"**, PCG
  book ch. 8 (DOI 10.1007/978-3-319-42716-4_8). FULL (linked). The
  method: *choice rules* define a design space, *integrity constraints* cut
  it, recursive rules express reachability (the perfect-maze example
  enforces "the root is reachable from every tile"). Solvers return valid
  maps or prove none exist.
- **Smith & Mateas 2011, "Answer set programming for procedural content
  generation: a design space approach"** (IEEE TCIAIG 3(3):187–200, DOI
  10.1109/TCIAIG.2011.2158545). ABS (the author copy online is a 4-page
  excerpt). PAY for the full text.
- **Smith, Padget & Vidler 2018, "Graph-based generation of action-adventure
  dungeon levels using answer set programming"** (FDG 2018, DOI
  10.1145/3235765.3235817). ABS (Bath repository 403 today). Acyclic dungeon
  graphs with lock-and-key constraints solved in ASP, with expressive-range
  analysis.
- **Karth & Smith 2017, "WaveFunctionCollapse is constraint solving in the
  wild"** (FDG 2017, DOI 10.1145/3102071.3110566). FULL, archived. WFC is a
  greedy, non-backtracking CSP search with a minimum-remaining-values
  ("lowest entropy") heuristic and global restart on contradiction; adding a
  *global* constraint makes restart-only WFC fail where a backtracking ASP
  solver succeeds; reachability is the global constraint level designers
  will want, and practice (Caves of Qud) "uses a multi-pass approach that
  adds doorways and connections after WFC runs to ensure connectivity".
*Matters to Breach:* ASP is the right tool for a *small* combinatorial
sub-problem with hard constraints — e.g. placing the furniture and fuel
clusters of one room so they don't block a door and a 3×3 unit can still
cross it — not for the whole ship (solver time is unpredictable; and L7's
"(ruleset hash, seed) → map" would also have to pin the solver version).
WFC belongs **inside** a room (decoration, wrecked sections, the CA/WFC
shape classes of L9), never across room boundaries, because it cannot hold
the graph's global constraints.

**B9. Dead Cells** (A10) — concept graph + hand-made rooms by
try-until-match on entrances; and **Tanagra** (A17, PAY) — constraint
solver for geometry under a reactive planner, mixed-initiative. Both
confirm the split "graph by rules, geometry by a solver or a template
pool"; neither handles a hull.

### How the families handle Breach's three hard requirements

| | mandatory room set | rigid hull | mirror symmetry |
|---|---|---|---|
| Ma / Edgar (B1, B2) | yes (graph is input) | only as a bound constraint; leaves voids | not supported |
| Gungeon (B3) | yes (flow is input) | no | no |
| Unexplored grid (B4) | yes | grid = hull | trivial on a symmetric grid |
| Treemap / BSP (B6) | yes (leaves are assigned) | **yes, fills it** | **yes: subdivide one half, reflect** (gen_ship v1) |
| Merrell (B6) | yes (program) | yes (footprint) | not addressed |
| ASP (B8) | yes (constraint) | yes (constraint) | yes (constraint) — at solver cost |

### Recommendation for Breach (strand b): the stage-2 pipeline

Ships: **hull as input, subdivision as the layout algorithm, search only as
repair.** Concretely, for the embedding session:

1. **Hull.** A rectilinear hull polygon from a small silhouette library
   (rectangle first — v1 already has one — then rectangle-with-notches,
   tapered bow as steps). Decompose it into a few rectangles (the bow, the
   midbody, the stern) which become the top level of the cut tree. Hull
   silhouettes are art-directed data; the gates already check airtightness.
   (B6; v0.1 §5 "hull as input"; Merrell's footprint.)
2. **Embed half, reflect.** Lay out the port wing and the axis band only,
   reflect (gen_ship v1 does this; no other method in the literature offers
   symmetry as cheaply). The grammar should therefore run on the **quotient
   graph** — wing rooms + axis rooms — and every rule applied to a wing
   node is applied to its twin by construction.
3. **Cut tree from the grammar.** The nonterminal hierarchy (L4 / §4.1:
   SECTOR → ZONE → rooms) *is* the slicing tree: each SECTOR gets a hull
   rectangle by area (treemap with the L9 footprint hints as weights), each
   ZONE a sub-rectangle, each room a leaf. Children are laid out along the
   cut in the order of the graph's chains (B1: chains are easy) so
   consecutive siblings share a wall. Adjacency constraints from strand (d)
   (bridge forward, engineering aft, airlock on the hull) pin which side of
   a cut a child goes. (B6 Mirahmadi/Marson; B1 chains.)
4. **Rooms are axis-aligned rectangles; an L-shape is two leaves of one
   node.** One-tile steel walls between leaves, as in v1. Mandatory and
   prefab rooms are placed **first**, at their fixed footprint, and the
   treemap fills around them. (B2, B3 "hardest first".)
5. **Doors on shared walls.** A `must-share-wall` edge is realised on the
   shared wall segment, at an offset inside both rooms' *port ranges* (a
   prefab declares its ports — side + offset interval, B2/B5; a parametric
   room's port range is its whole wall minus corner clearance). An airlock
   pair (L5) is a prefab with exactly two ports.
6. **Repair before reroll.** If a `must-share-wall` edge is not a contact
   (or too short for `DOOR_W`), run a few **Merrell moves** (slide a cut,
   swap two leaf identities — mirrored) with cost = missing contacts +
   footprint deviation (B6 Merrell). If it still fails and the edge is
   `may-become-corridor`, carve a corridor from the cheaper neighbour along
   the **wall graph** (B6 Mirahmadi) and record it as a corridor node in the
   as-built graph (v0.1 §3 step 6). Otherwise reroll the seed (L8; v1's
   `MAX_ATTEMPTS`; Metazelda's `RetryException`, A8).
7. **Corridors are rooms (nodes), not edges.** They carry gas volume,
   chokepoints, sensors and spawn tags; the grammar's edge subdivision makes
   them (L4 move 1). Nepožitek's objection (corridor nodes slow the search,
   B2) applies to search embedders, not to a subdivision embedder. The
   as-built graph then contains only rooms (incl. corridors) and openings.
8. **Interiors by shape class.** Inside each leaf: prefab (template with
   probabilistic tiles, B5), parametric, CA for wrecks, WFC for decoration —
   all confined to the leaf so they cannot break the graph (B8).
9. **Vents are not routed today.** In the current engine a duct is an
   abstract plenum entity and a vent is a wall-mounted entity (`gen_ship`
   `_build_vents`); there is no duct tile to route. L4's vent leash and the
   L9 `routed-freely` class matter only if ducts become tiles; until then the
   embedder's only vent duty is to find a wall site (as v1 does). This is a
   finding for #134, not a literature result.
10. **Station fragments use the other family.** No hull to fill, voids are
    vacuum, `torn` edges dangle: Ma/Edgar chain layout with configuration
    spaces and backtracking (B1, B2) — with Edgar's < 30 rooms and
    few-cycles limits respected.


---

## (c) PCG for RL generalisation

*(in progress)*

---

## (d) Real ship, submarine and station interiors; how games encode "a ship"

*(in progress)*
