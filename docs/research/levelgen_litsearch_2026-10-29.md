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

*(in progress)*

---

## (c) PCG for RL generalisation

*(in progress)*

---

## (d) Real ship, submarine and station interiors; how games encode "a ship"

*(in progress)*
