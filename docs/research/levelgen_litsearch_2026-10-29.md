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

The question for Breach: what must the generator *expose* so the RL track
(#76 onward) can train on one set of ships and be measured on another, and
what evaluation protocol should go with it. The prior RL digest
([`rl_litsearch_2026-07-20.md`](rl_litsearch_2026-07-20.md) §1 item 5)
already established *that* layouts must be randomised; this strand is about
*how* the level distribution should be shaped and handed over.

### Sources

**C1. Cobbe, Klimov, Hesse, Kim & Schulman 2019, "Quantifying
generalization in reinforcement learning"** (ICML 2019, arXiv 1812.02341).
FULL (7 MB, linked).
CoinRun: each level "generated deterministically from a given seed"; train
on sets of 100 … 16,000 levels, test on unseen levels from the same
distribution. "Substantial overfitting occurs when there are less than 4,000
training levels"; IMPALA-CNN, L2, dropout, data augmentation and
stochasticity each help. Levels "vary widely in difficulty, so the
distribution of levels naturally forms a curriculum".

**C2. Cobbe, Hesse, Hilton & Schulman 2020, "Leveraging procedural generation
to benchmark reinforcement learning"** (Procgen, ICML 2020, arXiv
1912.01588). FULL, archived.
Sixteen games, each with **two difficulty settings, `easy` and `hard`, that
are distribution-level, not per-level** ("This difficulty refers to the level
distribution and not to individual levels; in both settings, the difficulty
of individual levels has high variance"). Protocol: "train on a finite set of
levels and we test on the full distribution"; the recommended benchmark is
**500 training levels (200 on easy), test on held-out**. "Agents strongly
overfit to small training sets in almost all cases. To close the
generalization gap, agents need access to as many as 10,000 levels"; past a
threshold *training* performance also rises with set size (an implicit
curriculum). The deterministic-sequence ablation: agents trained on a fixed
level sequence look competent but "have in fact learned almost nothing
about the underlying level distribution". Hand-picked hard seeds are
provided for exploration studies.

**C3. Justesen, Torrado, Bontrager, Khalifa, Togelius & Risi 2018,
"Illuminating generalization in deep reinforcement learning through
procedural level generation"** (NeurIPS Deep RL workshop, arXiv 1806.10729).
FULL, archived.
GVG-AI games with a generator taking a **difficulty parameter in [0, 1]**;
fixed-level training overfits badly; introduces **Progressive PCG**, raising
the generator's difficulty "to match the agent's performance"; agents
trained only on hard levels fail where a difficulty ramp succeeds, and
policies trained on generated levels transfer poorly to human-designed ones
(the distribution matters, not only its size).

**C4. Juliani, Khalifa, Berges et al. 2019, "Obstacle Tower: a
generalization challenge in vision, control, and planning"** (IJCAI 2019,
arXiv 1902.01378). FULL, archived.
The closest precedent to Breach's plan: floors generated by a
**Dormans-style mission graph grammar** with *access levels* (number of locked
doors to reach a room), then a shape grammar to a room grid, then
**Spelunky-style room templates**; later floors use different grammar
**recipes** "to generate more complex floor layouts". Evaluation is defined
in three tiers: *no generalisation* (one fixed tower, five dynamics seeds);
**weak generalisation — "trained on a fixed set of 100 seeds ... tested on a
held-out set of five randomly selected tower configuration seeds", each run
five times with different dynamics seeds**; *strong generalisation* — also a
**held-out visual theme**. Note the separation of the *level seed* from the
*agent/dynamics seed*.

**C5. Khalifa, Bontrager, Earle & Togelius 2020, "PCGRL: procedural content
generation via reinforcement learning"** (AIIDE 2020, arXiv 2001.09212).
FULL, archived.
The inverse direction: an RL agent *generates* levels as an MDP of small
edits (narrow / turtle / wide representations; a change-percentage budget),
rewarded by the problem module's quality measures. *Matters to Breach:*
not a generator Breach needs, but its "problem module" (size, tile set,
quality metrics as reward) is the same object as an archetype's validation
profile — useful later if Erik wants a learned generator to propose
grammar parameters.

**C6. Wang, Lehman, Clune & Stanley 2019, "Paired open-ended trailblazer
(POET)"** (arXiv 1901.01753). FULL, archived.
Co-evolves environments (parameterised obstacle courses) with their
agents; a **minimal criterion** keeps environments neither too easy nor too
hard; agents **transfer** between environments as stepping stones.
*Matters to Breach:* POET needs an environment *encoding* that can be
mutated — Breach's (recipe, seed, parameters) triple is one.

**C7. Dennis, Jaques, Vinitsky, Bayen, Russell, Critch & Levine 2020,
"Emergent complexity and zero-shot transfer via unsupervised environment
design"** (PAIRED, NeurIPS 2020, arXiv 2012.02096). FULL, archived.
Names the three options for an *underspecified* environment: **domain
randomisation** (uniform, ignores the agent; fails on specific complex
configurations), **minimax adversary** (makes unsolvable levels), **minimax
regret** (PAIRED: an antagonist allied with the generator keeps levels
solvable; regret is maximised by "the simplest possible levels that the
student cannot currently solve"). *Matters to Breach:* validators (L8)
guarantee solvability by construction, which removes the reason minimax
adversaries fail; Breach could use the cheaper replay methods (C8, C9)
instead of training a generator.

**C8. Jiang, Grefenstette & Rocktäschel 2021, "Prioritized level replay"**
(PLR, ICML 2021, arXiv 2010.03934). FULL (11 MB, linked).
A drop-in replacement for uniform level sampling: each visited level gets a
score from the agent's own TD errors (L1 value loss / positive value loss,
"learning potential"), mixed with a **staleness** term
(`P_replay = (1−ρ)·P_S + ρ·P_C`); the next level is a new seed or a replayed
one. Requires that a level can be regenerated *exactly* from its index or
seed — "a PCG environment as any computational process that, given a level
identifier (e.g. an index or a random seed), generates a level".

**C9. Parker-Holder, Jiang, Dennis, Samvelyan, Foerster, Grefenstette &
Rocktäschel 2022, "Evolving curricula with regret-based environment design"**
(ACCEL, ICML 2022, arXiv 2203.01302). FULL (6.7 MB, linked).
PLR's curator plus **small edits (mutations) to previously curated
high-regret levels**, so complexity compounds from simple base levels
(MiniGrid mazes, MiniHack lava, BipedalWalker terrain).
*Matters to Breach:* a graph-grammar rule application **is** a small,
meaning-preserving edit. A level can be represented as (recipe, seed, list of
extra rule applications), and ACCEL's mutation operator becomes "apply one
more L4 move and re-embed" — edits that keep every L8 invariant by
construction, which is exactly what tile-level mutations cannot promise.

**C10. Kirk, Zhang, Grefenstette & Rocktäschel 2023, "A survey of zero-shot
generalisation in deep reinforcement learning"** (JAIR 76:201–264, arXiv
2111.09794). FULL (selected sections; 6.6 MB, linked).
The formal frame: a *contextual MDP*, the context being what the generator
takes. Distinguishes **black-box PCG** (one seed decides everything) from
**controllable** environments whose context is a product of named factors —
*cardinal* (a choice among options: archetype, theme) and *ordinal* (counts,
sizes, continuous dials) — and argues the purely-PCG approach is weak for
research because the factors of variation cannot be varied independently
(no interpolation/extrapolation studies, no per-factor held-out tests). The
generalisation gap alone is a poor metric ("a random policy is likely to get
a gap of 0"); report test performance in specific settings.

**C11. Ellis, Hughes, Gopalakrishnan, et al. 2023, "SMACv2: an improved
benchmark for cooperative multi-agent reinforcement learning"** (NeurIPS
2023 D&B, arXiv 2212.07489). FULL (12 MB, linked).
"If the initial state and transition dynamics of a Dec-POMDP are
deterministic, then open-loop policies suffice, i.e., there exists an
optimal joint policy that ignores all information but the timestep and agent
ID"; SMAC's fixed start positions and unit types let an open-loop policy do
well; SMACv2 randomises team composition and start positions per episode.
*Matters to Breach, sharply:* Breach is **bit-deterministic by iron rule**.
Every bit of closed-loop pressure must therefore come from the *episode
context* — level, spawn placement, roster, opponent seed — because the
dynamics add none.

### Recommendation for Breach (strand c): the knobs and the protocol

**Expose a white-box context, not only a seed** (C10). A Breach level is
`LevelSpec = (archetype, recipe_hash, size_band, params, level_seed)`, where
`params` are *named* factors the generator honours and records in the level
folder: cardinal — archetype (ship / station fragment / bomb map), hull
silhouette, lore recipe/theme; ordinal — room-count band, loop count (number
of L4 edge→cycle applications), mean corridor length, chokepoint count,
pressure-cell count and airlock count, vent connectivity (how many rooms
share a duct), spawn-to-spawn graph distance, spawn-to-target distance,
fuel/fire-load density, zombie count. The grammar's recipe already has
these as (rule, count) entries (strand a), so this costs little.

**Keep the level seed separate from the episode seed** (C4). The RL-ready
scope's "one seed drives levelgen, spawn, bots and zombies" is right for a
*match*, but a training run needs the level fixed while spawns, rosters and
opponent seeds vary: derive both from the match seed with
`SeedSequence.spawn` (the v1 generator already uses `SeedSequence([seed,
attempt])`), and record both.

**A difficulty ladder, distribution-level** (C2, C3, C4): two or three named
profiles (`easy`, `standard`, `hard`) that shift the ordinal factors (fewer
loops and cells on easy; more cells, longer routes, more fuel and zombies on
hard), with per-level difficulty left to vary inside each — the Procgen
definition. Progressive PCG (C3) and Obstacle Tower's per-floor recipes (C4)
are then just a schedule over profiles.

**Seed banks with held-out splits, published as data** (C2, C4):
- `train` — a large bank (Procgen's numbers say 500 is the benchmark size and
  ~10k is where gaps close; start with 1,000 ship seeds per profile, cheap
  since `gen_ship` needs no solver);
- `test-iid` — held-out seeds from the same archetype and profile (Obstacle
  Tower's *weak* generalisation);
- `test-ood` — a held-out **factor value**: an unseen hull silhouette, an
  unseen lore recipe, a loop count above the training range (Obstacle
  Tower's *strong* generalisation, held-out theme; C10's extrapolation);
- `probe` — a few hand-picked seeds that isolate one skill (an airlock
  fight, a vent-fire scenario), as Procgen's exploration seeds (C2).
  Each bank is a list of `LevelSpec`s plus the generator version; a
  generator change that alters any bank's output is a re-baseline, like a
  golden.

**Never a fixed level sequence** (C2's ablation) and **never a single map**
(C11): resample per episode.

**Sampling hooks for curricula** (C8, C9): make level generation a pure
function of `LevelSpec` (it is, by L7) and cheap enough to call per episode,
so PLR can replay by spec; and expose a `mutate(spec, rule)` entry point that
applies one extra grammar rule and re-embeds, so ACCEL-style editing is
available without a learned generator (C7 is not needed while validators
guarantee solvability).

**Report per factor, not just the gap** (C10): test return on `test-iid` and
on each `test-ood` axis separately, plus the as-built graph metrics of the
levels actually played (rooms, loops, cell levels — strand a's key-level
analogue) so a result can be tied to the factors that produced it.

---

## (d) Real ship, submarine and station interiors; how games encode "a ship"

The design session needs this strand for #136 (adjacency), #137 (pressure
cells), #134 (vent rules) and the L5 airlock recipe. What it can give is the
*logic* behind real subdivision — why bulkheads, zones and ducts are where
they are — so Breach's rules read as "a ship" for reasons, not by imitation.

### Sources

**D1. SOLAS II-1 subdivision, via "Compartment (ship)"** (Wikipedia,
read 2026-10-29). FULL (web; a secondary summary — the IMO text itself is
PAY, see ledger).
Watertight subdivision "limits loss of buoyancy and freeboard in the event
of damage"; ships are rated by how many adjacent compartments may flood (a
*two-compartment* ship survives any two). Three door classes: **watertight**
(rated like the bulkhead, "must, of course, be kept closed to effectively
contain flooding"), **weathertight**, and **joiner** doors (privacy only).
*For Breach:* the door classes map one to one onto the vocabulary's edge
types — bulkhead/airlock (pressure-rated), door (gas-tight when closed),
and a non-sealing opening (curtain, grating) — and the *n-compartment
standard* is a ready-made validator metric: "how many adjacent cells can
be breached before the spawn rooms are in vacuum".

**D2. SOLAS II-2 regulation 9, "Containment of fire"** (consolidated text
as published by the UK Maritime & Coastguard Agency). FULL, archived.
The fire analogue of watertight subdivision. Passenger ships are cut into
**main vertical zones** by A-60 divisions, extendable "to a maximum of
48 m" if the zone stays within 1,600 m² per deck (the normal 40 m figure is
in SOLAS's definitions, not read here), their boundaries "as far as practicable ... in line
with watertight subdivision bulkheads" — **fire zones and flood zones
coincide**. Spaces are classed by fire risk (control stations, stairways,
corridors, accommodation, service spaces, machinery spaces of category A,
galleys …) and the class pair fixes the division between two adjacent
spaces. Ventilation rules: "the ventilation fans shall be so disposed that
the **ducts reaching the various spaces remain within the main vertical
zone**"; machinery, galley, vehicle and cargo spaces have ventilation
systems "separated from each other and from the ventilation systems serving
other spaces"; "**stairway enclosures shall be ventilated and served by an
independent fan and duct system**"; a duct through an A-class division
over 0.075 m² free area gets a fire damper "capable of being closed manually
from both sides"; duct steel gauges are specified for widths "up to and
including 300 mm" and "760 mm and over"; galley exhausts have shut-offs
"close to the entrance to the galley".
*For Breach:* (i) **one duct system per cell** is the real rule — the L5
"vent leak path across a cell boundary" is exactly what SOLAS forbids, so a
cross-cell duct is a deliberate archetype choice (a sabotage map), never a
default; (ii) **escape routes get their own ventilation** (the stair rule)
— a corridor spine with its own duct is both realistic and tactically
interesting; (iii) **dampers** are the realistic version of a vent that can
be shut: a damper entity on a duct at a cell boundary is the natural L5
"leak path" control; (iv) the class-pair table is a template for the
vocabulary's adjacency matrix (which room types may share a plain door,
which need a bulkhead).

**D3. US Navy material conditions of readiness** (NAVEDTRA 14057,
*Damage Controlman*, via tpub; and secondary summaries). FULL (the tpub
page) / ABS (the rest).
Every closure (door, hatch, valve, vent) carries a class mark: **X-RAY**
(closed at all times except in use), **YOKE** (also closed at sea in
peacetime), **ZEBRA** (also closed at general quarters — "set immediately,
without further orders, when manning general quarters stations"; "used to
isolate and control fires and flooding"); circle-X / circle-Y fittings may
be opened briefly under authority. Compartment check-off lists enumerate
each space's fittings.
*For Breach:* a ready-made **door-state policy**: every door edge gets a
readiness class in the annotation schema, and a ship-wide condition
(peace / alert / battle) closes the classes in order. This gives the rule
AI and RL agents a structured, physically meaningful lever (set ZEBRA →
cells seal → fire and decompression stay local), and the grammar a reason to
emit it (the L4 "interactive/logic layer").

**D4. Submarine escape trunks and watertight hatches** (Wikipedia "Escape
trunk"; secondary sources). ABS.
An escape trunk "operates on a principle similar to an airlock": sealed
from the boat, flooded and pressurised to sea pressure before the outer
hatch opens, drained and equalised before the inner. The forward trunk is
also the boat's normal entry point for people and equipment and the docking
point for a rescue vehicle. Watertight hatches are dogged from one side;
internal doors normally stand open and are closed when conditions require.
*For Breach:* the airlock recipe (L5: two doors + a (de)pressurisable room)
is exactly the escape-trunk sequence; and the "normally open, closed on
alarm" behaviour is D3's readiness logic in a submarine.

**D5. ISS hatches — NASA-STD-3001 Technical Brief OCHMO-TB-028 "Vehicle
hatches"** (NASA OCHMO, Rev B, 2023). FULL, archived.
A hatch = hatchway + cover that isolates environments; a *pressure* hatch
has different atmospheres on its two sides. Requirements: operable by one
crewmember "in no more than 60 seconds, from both sides of the hatch";
**"Each side of each hatch shall have manual pressure equalization
capability with its opposite side"** (the MPEV, a valve that "allows air
from either side of the hatch to mix"); pressure indication and visibility
across the hatch; "The vestibule must be pressurized prior to crew members
opening the hatches." US modules connect through Common Berthing Mechanism
hatches (50 × 50 in, ≈ 1.27 m square, per the NASA human-factors material
and Wikipedia's CBM page — ABS for that number).
*For Breach:* an airlock prefab should carry, as data, an **equalisation
valve on each door** (a slow controlled flow — the realistic "bleed" lever),
a **pressure readout on both sides**, and a window; and a door between two
cells at different pressure should be *hard to open* (the force rule) —
which the gas solver already provides if the door's opening is gated on
|Δp|.

**D6. ISS Node 1 temperature and humidity control and intermodule
ventilation (IMV)** (Boeing/NASA, NTRS 20070018272). FULL, archived.
"The purpose of the Node 1 IMV is to provide air exchange between Node 1 and
the adjacent elements. It allows for **centralized atmosphere revitalization**
and oxygen partial pressure control." Node 1 carries IMV fans, **supply and
return valves for each adjacent element**, and **IMV isolation valves** "used
to isolate the Node 1 IMV from the adjacent pressurized elements during an
emergency event". Return paths are explicit and sometimes the hatch itself:
"Return atmosphere from the Cupola will be through the open hatch"; "Return
atmosphere from the MPLM will be through the open hatch." Flow rates are of
order 140 cfm (≈ 4 m³/min) per IMV fan.

**D7. Laurini et al. 1999, "Environmental control and life support for the
Multi-Purpose Logistics Module"** (ESA Bulletin 98). FULL, archived.
The cleanest statement of Erik's supply-and-return rule: air enters through
the IMV supply port, is distributed through the cabin and "returns it to the
ISS for revitalisation"; "Normally, air flows back to the Station through the
open hatch. **If the hatch is closed, the air travels through the separate
dedicated IMV return duct.** To isolate the module from the Station, a
motorised valve in each of the IMV supply and return ducts closes off the
airflow." A restrictor keeps the return at lower pressure "so air continues
to be sucked from the Station". Fire response escalates: portable
extinguisher → closed hatch → **the Cabin Depressurisation Assembly "can vent
the cabin air to space in less than 10 minutes as an integral element of
the ... fire suppression scheme"**; a Duct Smoke Detector sits in the cabin
return duct.
*For Breach (D6 + D7):* (i) **every module has a supply and a return, and the
return may be the open doorway** — so a vent rule (#134) can be "each room
has a supply; its return is either its own return vent or an always-open
doorway into a room that has one"; (ii) **isolation valves on both** = the
damper entity of D2; (iii) **venting to space is a sanctioned fire
response** — Breach's decompression-clears-smoke mechanic is not a game
liberty, it is the MPLM's last line, and a generator can place a
depressurisation valve on rooms whose archetype wants that play; (iv) the
smoke detector belongs in the return duct, which says where a sensor entity
goes.

**D8. Klote 1991, *Design manual for smoke control systems*** (NISTIR 4551;
superseded by Klote & Milke, *Design of smoke management systems for
buildings*, ASHRAE — PAY). FULL (selected chapters; 26 MB, linked).
Smoke management works by compartmentation, dilution, airflow,
pressurisation and buoyancy; barriers are only as good as their leakage
paths. **Zoned smoke control**: the building is divided into smoke zones; "in
the event of a fire, pressure differences produced by mechanical fans are
used to limit the smoke spread to the zone in which the fire initiated";
the fire zone is vented/exhausted and the neighbours pressurised — the
**"pressure sandwich"**; venting the smoke zone matters because it "prevents
significant over-pressures due to thermal expansion". **Pressurised
stairwells** keep escape routes clear; tall stairs are themselves
compartmented with normally-closed doors and an air injection point per
section.
*For Breach:* this is the gameplay of a vent network with fans, stated as
engineering: exhaust the burning cell, pressurise its neighbours. A
generator that gives each cell its own duct system (D2) and a pressurised
spine (the stair rule) makes this playable by construction.

**D9. Ducts versus crawlability.** SOLAS reg 9 (D2) for real duct sizes;
MythBusters episode 54 (mythresults.com) and HVAC-trade commentary — ABS.
Ship ventilation ducts are specified in bands "up to and including 300 mm"
and "760 mm and over"; the damper threshold is 0.075 m² (≈ 27 cm square).
The ISS's IMV flows (D6) are a few m³/min. MythBusters' "crawling through air
ducts" was **busted on noise** (magnets and suction cups held the weight but
were too loud; breaking the grate was too loud); trade sources add that bends
carry turning vanes and dampers, and that light-gauge or flexible duct does
not carry a person. The human-sized exceptions are large trunks.
*For Breach:* at 0.333 m tiles a **one-tile duct is ≈ 0.11 m²** — already
above SOLAS's damper threshold, a large real duct, and far below a marine's
3×3-tile (≈ 1 m) footprint or a 1.27 m CBM hatch. So "vents too small for
marines but open to smoke and small creatures" is *physically right* at one
tile; a crawlable two-to-three-tile **maintenance trunk** is a different
object (a narrow corridor room in the graph, SS13's "maintenance"), not a
vent. Recommend the vocabulary keep the two words apart: `vent`/`duct`
(gas + small creatures, never marines) and `crawlway` (a corridor-class
room, marines prone, if ever wanted).

**D10. Nuclear propulsion layout** (NASA NTP mission & system analysis
2006, NTRS 20240013888; Maddock et al. IAC 2021). ABS.
Crewed nuclear-thermal vehicles put the reactor at the far end of a truss
and the habitat inside the reactor's **shadow shield** cone; distance buys
shield mass.
*For Breach:* the one physical reason a *space*ship has "engineering aft":
thrust line plus radiation shielding, not a propeller shaft. It argues for
a *gradient* (crew spaces as far from the reactor as the hull allows, with
stores/tanks between as shielding), not just "engine at one end".

**D11. Ship general arrangement practice** (TU Delft OCW MTP101 lecture 4;
SNU ISOD notes; Marine Insight) — ABS.
Arrangement is compartments + equipment + structure; deckhouse ends and
sides line up with hull bulkheads; machinery placement follows propulsion;
the bridge is placed for the SOLAS view requirements; there is "no specific
procedure with rules stated for general arrangement, since almost every
ship is unique". Not found: an open source on sick-bay or galley adjacency —
those conventions are owner standards (Watson, *Practical Ship Design*, PAY).

**How games encode "a ship"**

**D12. Space Station 13 mapping requirements** (Paradise Station devdocs,
"Mapping requirements"; Paradise and Unitystation mapping guides). FULL
(web).
Per-room requirements as data: "Each room needs **at least one air vent and
one scrubber**" (supply and return, colour-coded blue supply loop / red
scrubber loop), at least one fire alarm, intercom, camera, light switch;
"Each station area should contain only **one APC and one air alarm**";
**firelocks "at area boundaries over doors"**, may break up long hallways,
never doubled, never on maintenance doors; maintenance hallways need no
vent pump; external/depressurised areas use airless turf. Department areas
and access levels structure the map; maintenance tunnels are a separate
network behind the rooms.
*For Breach:* SS13's rules are the closest existing "ship vocabulary" with
per-room mandatory fittings — and they are the same as Erik's rule
(supply + return per room) and D2 (a fire boundary at every area boundary
door). Copy the *form*: a room type's `requires` list in the vocabulary
(#27) — e.g. `{vent_supply: 1, vent_return_or_open_door: 1, lamp: 1,
sensor: 1}` — checked by the validator.

**D13. Barotrauma submarine editor** (official manual excerpts; community
guides). ABS (wiki 403).
Rooms are **hull** objects (volume for water and oxygen); **gaps** connect
neighbouring hulls; doors and hatches create gaps automatically; anything
not inside a hull is ocean; AI **waypoints are generated from the hulls,
gaps, doors and ladders**; bots prefer hulls named "airlock" to enter and
leave. Oxygen generators feed rooms through vents.
*For Breach:* the same split v0.1 §3 step 6 calls for — the *as-built*
graph (hulls + gaps) derived from geometry, and navigation derived from it
— and a precedent for naming conventions as data that the AI reads.

**D14. FTL: Faster Than Light** (FTL wiki "Door system"; player forums).
ABS.
Rooms hold oxygen; open doors equalise; **airlock doors vent rooms to space,
and venting is the standard way to put out fires**; plain doors do not stop
fire spread, upgraded blast doors do; an oxygen system replenishes. Ships
are a few dozen rooms grouped by system, each system in one or two rooms.
*For Breach:* FTL taught a generation of players to read "airlock = fire
extinguisher, door = fire barrier you can upgrade" — Breach's physics gives
the same reading for free, which is a reason to keep vent-to-space rooms
(airlocks off hull rooms) common in ship archetypes.

**D15. Objects in Space; Starsector.** NF / not applicable: no usable source
on Objects in Space's interior rules was found; Starsector ships have no
walkable interiors (hull mods are stat modifiers), so it says nothing about
adjacency.

### Recommendation for Breach (strand d)

Make the **pressure cell the unit that every other subdivision lines up
with**, as SOLAS makes fire zones line up with watertight bulkheads (D1, D2):
one cell = one duct system (no duct crosses a cell boundary by default, D2)
= one alarm zone (one air alarm per area, D12) = one readiness group (D3).
Cell boundaries are bulkheads or airlock pairs (L5); every door on a cell
boundary carries a damper-like closure class (ZEBRA, D3). Inside a cell,
**every room has a supply vent and a return** — its own return vent or an
always-open doorway into a room that has one (D6, D7, D12; Erik's rule).
Give the **corridor spine its own duct** (the pressurised-stair rule, D2, D8)
so it can be held clear while a side cell burns — the "pressure sandwich"
as a map feature. Airlocks are prefabs carrying both-side equalisation
valves, pressure readouts and a window (D5), and the hull rooms of
archetypes that want it get a depressurisation valve (D7). For adjacency
(#136), use *reasons* as rules: reactor/engineering at one end with
stores/tanks between it and quarters (D10), the bridge where the hull gives
it a view (D11), galley and machinery on separate ventilation (D2), airlocks
and cargo on the hull (D4, D6: nodes and berthing ports are where things
dock). And keep `vent` and `crawlway` as two words (D9): at one tile a vent
is already a large real duct and correctly too small for a marine.
