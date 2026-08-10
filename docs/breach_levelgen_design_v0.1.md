# Breach — Level Generation Architecture, Design Doc v0.1

**Status:** Design phase. No implementation until the full system is specified.
**Date:** 2026-08-10
**Origin:** Claude.ai design sessions (level generation architecture). This doc is the handoff into Claude Code for the vocabulary consolidation pass.

---

## 1. Vision

Levels in Breach are tile maps (the same tiles the gas/water/fire fields live on), generated **graph-first**: a graph grammar grows an annotated room graph, which is then embedded into tile geometry. The room graph is a first-class engine citizen with (at least) five consumers:

1. Commander AI location scoring
2. Rat economy
3. Level-generation validation (the annotations double as the fitness function)
4. Graphics generation (via room type labels)
5. Per-room field aggregates (GPU-mirrored)

Guiding principle learned the hard way: **when adding a system entangled with other systems, design the whole system before implementing it.** This doc is that design, iterated until satisfied.

---

## 2. Locked architectural decisions

These are settled. Reopening one requires an explicit decision, not drift.

### L1 — Graph/mission grammar is the spine
All other techniques (BSP, cellular automata, WFC, prefabs) are demoted to **embedding or decoration strategies** invoked per-node/per-region. Nothing is discarded; everything gets a slot. CA remains alive for cave/underground levels and wrecked sections.

### L2 — Space grammar only, with archetype profiles (no separate mission tier)
No Dormans-style mission→space two-tier system. Instead, tactical *patterns* (chokepoint, retreat loop, food circuit, two-routes-to-target) play the role missions play for Dormans.

**Map archetype = axiom + rule subset + validation profile.** Examples:
- *Bomb map* (counter-strike-like): one high-value target room, ≥2 distinct routes to it, defensible chokepoints on each route, spawn zones at graph-distance ≥ X from target. Annotations (`target_candidate`, `spawn_zone: attacker/defender`) tell the objective system where to attach.
- *Ship*: mandatory room set present, rigid hull.
- *Station fragment*: looser layout, `torn` edges at severance points.
- Future: extraction map, infestation map, …

### L3 — Axiom per map archetype
- **Ships:** axiom = the mandatory set as a pre-connected skeleton (cockpit—corridor—quarters—…) so mandatory rooms cannot be grammared away.
- **Station fragments:** looser axiom, e.g. N sectors + M dangling edges marked `torn` (severed connections → free lore, free vacuum boundary conditions).

### L4 — Restricted, planarity-preserving rule set
Allowed rewrite moves (all planar-safe by construction):
1. **Edge subdivision** — A—B → A—C—B (insert room mid-connection; corridor → corridor—chokepoint—corridor)
2. **Edge→cycle** — A—B → A—C—B plus A—D—B (loops: retreat routes, flanking routes, two-routes-to-bomb-site)
3. **Leaf attachment** — add A—N, N new (dead-end rooms: closets, airlocks off the hull)
4. **Face insertion** — add a cycle of rooms inside an enclosed empty face (district-like annexes)

**Excluded:** the unrestricted "add edge between arbitrary existing nodes" (the only move that can create crossings).

**Vent exception:** vents may connect nearby nodes under a leash ("nearby" defined so the embedder can always route them through walls). Vents are the designated future z-layer crossings if we ever want vents-over-corridors; until then they too are constrained planar-safe. Z-layer vents are explicitly **parked, not priority** (would require two coupled tile lattices in the gas solver + layer-aware ray marching).

**Consequence:** stage-1 planarity checking vanishes — planarity holds by construction.

### L5 — Pressure cells by construction
Pressure cells (sets of rooms sharing atmosphere) are a **grammar-level concept**, not computed after the fact. During growth, subgraphs are tagged `cell: k`. Invariant enforced by the grammar:

> Every edge crossing a cell boundary must be an airlock-pair or sealed bulkhead — never a plain door.

Guarantees on every map: compartmentalization gameplay (breach cargo → cargo vents, quarters stay pressurized), airlock fights are always meaningful, and the gas solver receives its Dirichlet boundaries as data.

**Vent leak paths:** a vent crossing a cell boundary is a deliberate leak path — forbidden or allowed per-archetype (sabotage-the-fan gameplay).

**Airlock recipe:** an airlock must contain two doors and a room that can be (de)pressurized. Realized as prefab-or-parametric with that invariant.

### L6 — Phased application semantics
Generation runs in phases, each with its own rule subset and budget:
1. **Grow topology** (nonterminal expansion)
2. **Insert tactical patterns** (chokepoints, loops, cells)
3. **Terminalize** (assign concrete room types)
4. **Annotate** (emit the full annotation schema)

Benefits: trivial termination (per-phase budgets), and LLM-authored rules are sandboxed — a rule declares its phase and can only wreck its own phase.

### L7 — LLM recipe boundary: offline authorship, deterministic execution
- LLM (Claude) authors rules **offline** into a declarative rule format (JSON or similar).
- Runtime executor is deterministic and seeded.
- Every map reproducible from **(ruleset hash, seed)**.
- No LLM calls at generation time.
- Recipes are versioned, diffable, and accumulate into a **recipe library** over time (e.g. "Grays-refitted freighter"). Recipes may be lore-flavored (SPACE COM ledger, Grays, faction variants).

### L8 — Grammar/validator division of labor
- **By construction (grammar guarantees):** planarity, mandatory rooms, pressure-cell airlock invariant.
- **Post-validation (metric checks):** rats-reach-food distances, retreat-room quality for the location scorer, chokepoint counts within bounds, connectivity sanity.
- **On validation failure:** reroll seed. Generation is cheap; the architecture stays simple.

### L9 — Grammar→embedder output contract (shape locked, details deferred)
Per node: **footprint size hint** (min/preferred area) + **shape class** (`rectilinear | organic | prefab-id`) — shape class dispatches the realizer (prefab / parametric / CA / WFC).
Per edge: **adjacency strength** — `must-share-wall` (door) | `may-become-corridor` (embedder's patch-miss escape valve) | `routed-freely` (vent).
Field details finalized in the embedding session.

---

## 3. The pipeline (reference)

1. ~~Planarization check~~ — eliminated by L4 (planar by construction)
2. **Spatial layout** — assign nodes positions/footprints (algorithm choice open; candidates: incremental growth with backtracking, BSP-assignment, force-directed, constraint solver; likely per-archetype)
3. **Room realization** — node footprint → tiles, dispatched by shape class (prefab | parametric | CA | WFC)
4. **Edge realization** — doors on shared walls; missed adjacencies → A* corridors; vents routed through wall interiors; weldable annotation stamped onto tiles
5. **Simulation-field initialization** — pressure cells → pressurized/vacuum regions (Dirichlet boundaries from L5 data); fans/ventilators as momentum-injecting tiles turning vent networks into circulation systems; food density seeded in cantinas/warehouses
6. **Validation + annotation writeback** — recompute the **as-built graph** from tiles (rooms may merge, corridors add edges); verify contract still holds; the as-built graph, not the intended one, ships to all consumers

---

## 4. The vocabulary page (next work package — do in Claude Code)

One document, five sections, read by grammar, validator, embedder, and all graph consumers. **Hard rule: one word, one meaning.** If the existing level generator uses a word (e.g. "zone") with a different meaning, either the meanings are reconciled or one side is renamed.

### 4.1 Sections
1. **Nonterminals** — abstract growth symbols, all gone by end of generation: SECTOR, ZONE(habitat|industrial|medical), CHOKEPOINT?, PRESSURE_CELL boundaries. Open: hierarchy depth (lean: two-level but shallow — SECTOR → ZONE → rooms).
2. **Terminals** — concrete room types with mandated attributes: quarters, cockpit, med bay, cantina, warehouse, corridor, engineering, cargo, airlock. Each carries requirements as data, e.g. cantina = `{food_density: high, sealable: false, min_area: 6×6}`. Rat-economy needs baked in here.
3. **Edge types** — door, weldable_door, bulkhead, vent(min_size), airlock_pair, torn. Each with a traversability matrix (marine / rat / drone / …) and simulation behavior (gas passes when closed? weldable? destructible?).
4. **Annotation schema** — per-node fields in three ownership classes:
   - static grammar-authored (type, cell id, food sources, target_candidate, spawn_zone, …)
   - static embedder-authored (actual area, tile bounds)
   - dynamic runtime (current gas mass, occupancy — GPU-mirrored aggregates)
5. **Interactive/logic layer** *(new — discovered from existing level generator)* — buttons, sensors (e.g. motion sensor toggling lights on room entry), actuators, signal/logic links between them. Lives on tiles and rooms. Future payoff: the grammar can *emit* logic ("motion sensor at every chokepoint", "detonator + defusal interactions on bomb maps").

### 4.2 The consolidation pass (process)
1. Draft the vocabulary as we want it (from this doc).
2. Inventory the existing level generator: extract every term, entity, and mechanism from its code and design docs (known so far: manual level creation, buttons, sensors, logic communication).
3. Reconcile: reuse where meanings match; **rename where they collide**; adopt existing terms where they're better than ours.
4. Mine the existing design docs for features worth adding to the target design.
5. Deliverable: one page of definitions + a JSON schema.

---

## 5. Embedding session (after vocabulary)

Decisions parked for a dedicated session:
- **Layout algorithm** (stage 2): incremental growth w/ backtracking (prior for ships) vs BSP-assignment vs force-directed vs constraint solver; possibly per-archetype (ships rigid, fragments looser, caves force-directed/CA-regional)
- **Hull handling:** hull as input (library of silhouettes, fill them — art control, iconic shapes) vs output (grow rooms, wrap hull — never fails). Open.
- **Grid resolution & realizers:** footprint hints → tiles; realizer interfaces (prefab / parametric / CA / WFC)
- **Corridor & vent routing:** A* for patch-miss corridors; vents through wall interiors ⇒ minimum wall thickness decision, ripples into grid resolution
- **Failure protocol:** embedder can't place despite backtracking → strong lean: reroll seed (not re-enter grammar)
- Finalize L9 field details

---

## 6. Design-loop process

Iterate until satisfied, then implement:

1. **Describe** all the features we want
2. **Plan** how this integrates (with existing systems: level generator, gas solver, ray marcher, room-graph consumers, GPU contract)
3. **Review** what we have — add features, change things
4. **Return to 1** and repeat

No implementation before the whole system is designed.

---

## 7. Work items / roadmap

- [ ] **Vocabulary page + consolidation pass** (Claude Code; §4) — next
- [ ] JSON schema for the vocabulary / annotation contract
- [ ] Declarative rule format spec (JSON) + phase declarations (L6, L7)
- [ ] Archetype profile format (axiom + rule subset + validation profile) with first three archetypes: ship, station fragment, bomb map
- [ ] Embedding session (§5) → lock layout algorithm, hull policy, realizer interfaces, routing, failure protocol
- [ ] Deterministic seeded executor design (reproducibility: ruleset hash + seed)
- [ ] Validator: metric checks list + threshold tuning plan (L8)
- [ ] Recipe library conventions: versioning, diffing, naming, lore tagging
- [ ] First LLM-authored recipe as pilot (suggested: "Grays-refitted freighter")
- [ ] Integration map against existing systems: gas/water/fire tile fields, ray marcher, commander scorer, rat economy, GPU aggregate mirroring
- [ ] Event queue + room graph schema alignment (ties into open questions from the AI session)

## 8. Open questions (carried)

- Nonterminal hierarchy depth (flat vs two-level)
- Vent leak paths across cell boundaries: per-archetype policy defaults
- "Nearby" definition for the vent leash (embedder-routability criterion)
- Fan/ventilator placement: grammar-emitted or embedder/decoration-stage?
- How much logic (sensors/buttons) should phase 4 emit in v1 vs later
- Target counts: rooms per map per archetype (affects budgets in L6)
