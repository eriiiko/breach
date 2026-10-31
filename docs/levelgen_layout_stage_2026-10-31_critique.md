# Critique: levelgen layout stage (stage 1.5), revision 1

> Adversarial critique of `docs/levelgen_layout_stage_2026-10-31.md` (branch
> `136-layout-stage`, commit `b0af174`), 2026-10-31, Opus, read-only. Checked
> against recipe doc rev 2 and its critique, design v0.1 (L1-L9, §5, §9),
> litsearch strand (b) (B6, B7, Recommendation), `tools/gen_ship.py` (line
> numbers from the worktree, identical to main for this file) and
> `src/simulation/match.py`. Every number below was recomputed from `SIZES`
> and the constants; nothing was run.

Reference arithmetic used throughout (from `SIZES`, `BAND = 3`, insets 0-2
each, `_spawn_grid`, `FORMATION_PITCH = 5`, `FORMATION_GAP = 2`):

| | medium (100 x 70) | large (160 x 110) |
|---|---|---|
| interior h (`70 - 6 - 2 - insets`) | 58-62 | 98-102 |
| band `wm` | 12-18 | 14-22 |
| wing air width `(interior_w - wm)/2 - 1` | 36-39 | 64-68 |
| slab for `max_per_team = 6` (c = 2, r = 3) | 8-10 wide, 13-16 tall | same |
| row length (wing - slab - 1 wall) | 25-30 | 53-59 |
| row depth with a 6-across passageway | <= 27 | <= 47 |
| row depth with the 13-across bypass passageway | <= 23 | <= 43 |

## Findings

### 1. BLOCKER -- slack has nowhere to go: rigid rectangles against recipe size RANGES

> "the entry stub: a `strip` of `across` (drawn) cut from the rectangle's passageway side inward, full depth" (§4 step 1);
> "the end room ... at the stub's far end" (§4 step 2); "**end room** | ... spanning the strip's `across` plus its two pack sides" (§3);
> "draw each department's x-extent in its range so the row's sum equals the wing length exactly ... filler between departments is given to the departments (never a void" (§6)

**Evidence.** Every extent in the skeleton is fixed by the hull, while every recipe extent is a closed range, and the document
only handles the case "sum of minimums too large". The other direction is not handled anywhere:

- **Stub length.** The row depth is hull-fixed (table above: up to 27 medium, 47 large). The crew-quarters stub is
  `along = [20, 24]` (recipe rev 2 §3.1). On large, "full depth" makes the stub 41-47 long, far outside its range; on
  medium with the mess at the far end it cannot fit at all (finding 2). And "full depth" contradicts step 2: if the stub is
  full depth there is no far end left for the end room.
- **End-room width.** An end room spanning `across + 2 room depths + 2 walls` is at least `6 + 6 + 6 + 2 = 20` wide. The
  mess, the document's own end-room example, is `w = [9, 12]`. Every draw violates the recipe maximum.
- **One depth per pack side.** A pack side has ONE depth (the department's x-extent is one number), but rooms on a side have
  different `w` ranges: heads `w = [6, 6]`, bunks `[6, 8]`, mess `[9, 12]`. Heads and mess can never share a side, and the
  document never says rooms on a side share a depth.
- **Row filler beyond the maxima.** If the sum of the row's departments' MAXIMUM x-demands is below the row length
  (large: 53-59 tiles against, say, crew quarters at most `8 + 8 + 8 + 2 = 26` plus one other department), "filler is given
  to the departments" inflates rooms past their maxima. Only `sum of minimums > length` is named as a reroll.

**Fix.** Name the ELASTIC extents explicitly, per primitive: e.g. (a) the stub's `along` and the pack-side depth absorb slack,
and recipe maxima are soft for those two; (b) an end room is centred on the stub at its recipe width and the two pack sides'
last rooms extend up beside it (or the end room is not full width and the corners are pack rooms); (c) a pack side's depth is
`max(w_min)` over its rooms and smaller rooms are deepened, or rooms are assigned to sides by compatible `w`; (d) state the
reroll rule for `sum of maxima < length` too. Alternatively accept bounded steel voids (finding 17), which removes most of
this at a stroke.

### 2. MAJOR -- the input document's worked example (crew quarters) cannot fit a medium ship, and the layout doc has no worked example

> "the mess at the end of the stub" (§3); the skeleton diagram shows three departments in one row (§2)

**Evidence.** Medium row depth is at most `(62 - 6 - 2) / 2 = 27` (23 with the bypass passageway, ~23 under a hull strip).
Crew quarters need stub `along >= 20` + 1 wall + mess `h >= 9` = **30 > 27**. On the x axis a medium row is 25-30 long; crew
quarters need at least `6 + 6 + 6 + 2 = 20`, a two-stub department `2*6 + 2*6 + 3 = 27`, so a medium row holds ONE department
and the wing two, not the six in the diagram. Large holds about two per row. The recipe doc rev 2 was criticised for exactly
this (prior critique finding 9) and answered it with a feasibility computation; the layout doc, which is where the numbers
actually bite, has none. §6's "a recipe that can NEVER fit is refused at load" would refuse the reference department on medium.

**Fix.** Add a worked example: freighter, medium and large, max_per_team 6, crew quarters + one other department per wing,
with the passageway, slab, row depths, x-demands and the resulting contacts written out. Decide (Erik-visible) whether recipe
ships are large-only in v1 or whether the quarters' ranges shrink.

### 3. BLOCKER -- the flank pair's narrow leg cannot reach the axis band, and the fallback changes the cycle

> "a door down into one department room near the band (that room's back wall), or into the axis section if the strip reaches
> the band and the band's section touches it along its top (then the strip ends at the band wall, which is its end wall: no
> door there)" (§4.2); "The `wide` leg is the passageway itself."

**Evidence.** The strip runs along the HULL, so its top is the hull; a band section lies beyond its END (x), never above it.
The only contact between the strip and a section is the strip's 3-tile end wall, where a 4-wide door cannot be cut
(`_opening_tiles` raises "opening leaves lo..hi", gen_ship.py:627-643; recipe rev 2 §1.2 "a 4-tile door cannot be cut in a
3-tile end wall"). So the "into the axis section" branch is geometrically impossible -- it reintroduces the prior critique's
finding 8. The remaining branch makes the leg's far anchor a DEPARTMENT ROOM, not the band: the cycle realised is
slab - strip - dept room - (its stub) - passageway - slab, which does not contain the band at all, while recipe rev 2 §1.3
defines the flank pair as "two corridors between the same hubs" applied by `edge_to_cycle` between those hubs. Likewise "the wide
leg is the passageway itself" ignores the 1-3 wide-leg rooms the grammar's `edge_to_cycle` created (rev 2 §2 `rooms = [1, 3]`).

**Fix.** Pick a realisable topology and push it back into the recipe doc (a cycle-topology change: Erik in the loop), e.g.:
the narrow leg's B anchor is a band-end LANDING room of row A (a `wide` room that touches the strip's long side and the band
section along its own `along` >= 6) and the landing is the wide leg's last room; or the strip runs along the band's outer
column instead of the hull. Either way, write the mapping "grammar room -> skeleton piece" for both legs.

### 4. MAJOR -- the spawn slab "sized exactly as today" contradicts spawn exits 2-3 and the skeleton diagram

> "**Spawn slab**: ... sized by `_spawn_grid` / `_extent` exactly as today" (§2); "2 = passageway + one department room of a row
> (the slab touches the first department rectangle of each row along the slab's inner wall above / below the passageway); 3 =
> passageway + both rows' first rooms" (§5)

**Evidence.** Today's slab is `_extent(c) + 0..2` wide and `_extent(r) + 0..3` tall (gen_ship.py plan_graph, `sw`, `sh`): 8-10 x
13-16 for six per team, at a drawn `sy0`, with rooms or absorbed slivers above and below it. A door contact needs >= 6
(`DOOR_W + 2`). Exits = 3 needs `6 + 1 + 6 (passageway) + 1 + 6 = 20 > 16`; with the 13-across bypass passageway even exits = 2
needs `13 + 1 + 6 = 20`. Exits = 2 with a 6-across passageway works only if the passageway sits flush with one edge of the slab,
a constraint the document does not state (it draws the passageway's y at "centre +/- jitter" independently). And if the slab
does not span the wing height, the rectangles above and below it (x in the slab's range) are neither rows nor slab: voids or
rooms no recipe asked for. The diagram draws the slab full height; the text sizes it as today. Both cannot hold.

**Fix.** Choose: (a) the slab spans the full wing height (minus a hull strip), formation centred (a feel change: a long thin
spawn, list in §9), or (b) the slab is sized as today and the slab-column remainders are filler rooms of a named kind, with
the exits contact arithmetic written per exits value. Couple the passageway's y draw to the slab (it must lie within the slab's
y-range with >= 1 tile margin, and leave >= 7 tiles of slab on the side of each exit).

### 5. MAJOR -- the axis band's section cuts are drawn blind to the passageway, the bypass and the ring hub

> "**Axis band**: sections along y as today" (§2); "v1 band packing = sections as today, one department per section group" (§12)

**Evidence.** Today the cuts are `_split_1d(rng, iy0, iy1, k, MIN_SECTION)` (gen_ship.py:1019): a multinomial over a minimum of 8,
with no knowledge of anything in the wing. The skeleton needs: the passageway's end wall to face ONE section (a cut inside its
y-range splits the passageway-section contact in two, possibly both < 6); for the bypass a section covering >= 13 + margin
(`MIN_SECTION` is 8); for the ring a hub section of at least `6 + 2*(6 + 1) = 20` (27 with a 6-room ring, side rooms 2*6+1
tall). The zombie room is also unassigned: today the section nearest the centre is tagged (plan_graph), `embed` does
`next(rm for rm in graph.rooms if rm.tags.get("zombie"))` and raises StopIteration if none, and `check_request`'s zombie
capacity (gen_ship.py:914-) is computed from plan_graph's band geometry, which a ring changes.

**Fix.** Specify the band packer as conditioned cuts: first place the passageway's y-range (and the hub's, with its ring
demand) as pinned intervals, then split the remainder with `MIN_SECTION`. Name which room gets `zombie` (a main-cell section
nearest the centre, never the hub) and give the recipe path its own capacity refusal.

### 6. MAJOR -- an airlock between two AXIS rooms cannot be mirror-symmetric (AIRLOCK_W = 5 on an even grid)

> "an `axis` department takes one or more consecutive sections" (§2); "at is drawn inside the shared run with clearance" (§4 step 5)

**Evidence.** All three presets have even `W` (60, 100, 160) and the mirror is `x -> W - 1 - x`; an opening in a horizontal wall
between two self-twin rooms must be centred, which `plan_graph` does as `at = W // 2 - n // 2` (gen_ship.py:1089-1090). That is
symmetric only for even `n`: `DOOR_W = 4` and `GLASS_W = 2` work, `AIRLOCK_W = 5` cannot (the same parity rule
`hull_door_site` enforces for axis hull doors, gen_ship.py:700-704), and the vestibule `_airlock` carves on room `a`'s side is a
5-wide box on one side of the axis. Engineering (axis, `cell = "own"`, rev 2 §4 "engineering the second" cell) touching its fore
and aft `main` sections therefore can use only a bulkhead or a breakable wall there; an airlock fails the symmetry gate on every
attempt and `generate` ends in ShipGenError. Separately, §4 step 5 "draws" `at`; for an axis-axis edge it must not be drawn.

**Fix.** State the parity rule: axis-axis openings are centred, never drawn, and must have even width; an axis-axis cell
crossing is a bulkhead or breakable (or a mirrored PAIR of airlocks off the axis, a multi-edge). The loader refuses an
airlock on an axis-axis edge by name.

### 7. MAJOR -- with an `own`-cell section facing the passageway, the two teams' bots can never meet

> "It joins ... the axis band's facing section by a door, or by the airlock pair when that section is in another cell." (§2)

**Evidence.** In the skeleton the passageway's band-end edge is the wing's only DECLARED link to the band (rows touch the band
through blank walls). If the facing section is engineering (`own`), both wings reach the middle only through airlocks. Recipe
rev 2 §1.3 records the accepted gap: "the bots' A* does not route through an airlock's closed outer door (#142)". `validate`
deliberately walks the outer doors (gen_ship.py:1852-1860), so the gate passes a ship on which a bot match can never make
contact. The v1 match is bots + zombies (#76 scope).

**Fix.** Until #142 lands: the passageway's facing section is in the main cell (engineering is placed on sections the
passageways do not face), or each wing has a second non-airlock route to a main-cell section; add the rule to §2 and a
"bot-walkable between spawns" check (A* with airlock outer doors closed) to the layout's own acceptance.

### 8. MAJOR -- the airlock bypass: right number, incomplete derivation, and a fallback that edits the graph

> "the wall must be >= `AIRLOCK_W` + 2 + `DOOR_W` + 2 (= 13) long ... otherwise the bypass is applied on a
> department-room-to-section wall instead (a room packed at the band end of a row touches the band along its depth). The
> layout tries the passageway first, then the row's band-end rooms, then rejects the application (not the ship)." (§4.3)

**Evidence.**
- 13 is sufficient, but the reason is not the one given. The interior airlock mounts its far pump and far probe ON THE SHARED
  WALL at both ends of the outer door (`far_pump = (line, y0 - 1, ...)`, `far_probe = (line, y1 + 1, ...)`, gen_ship.py
  ~813-815) and `_carve_airlock` rejects a mount that is not steel or hull. A breakable butted against the outer door turns
  that mount tile to wood and rejects every draw. The vestibule's side wall at `(y1 + 1, line - 1)` also faces the first tile
  past the door. The document must say the breakable keeps >= 1 steel tile from the outer door's span.
- Both openings are on one wall; with `at = None` `_opening_tiles` centres BOTH (gen_ship.py:631), so they overlap. The layout
  must set both `at`s, non-overlapping; §4 step 5 does not say so.
- The fallback room needs depth >= `VESTIBULE_D + 1 + FORMATION_FOOTPRINT = 9` perpendicular to the band wall
  (`_airlock`, gen_ship.py:839) and >= 13 along it; a pack-side room is 6-8 deep. It touches the band along its `along` extent
  (y), not "its depth".
- `parallel` is a second opening on an EXISTING A-B edge (rev 2 §0.1). A department-room-to-section airlock is an edge the
  grammar never declared; and "rejects the application (not the ship)" deletes a grammar application after the grammar has run
  its invariants: a graph edit by the layout, against "Never re-enter the grammar (L8)" in §7 and the reproducibility claim
  (rev 2 §3.4: the (recipe hash, seed) reproduces the RESOLVED graph).
- The breakable crosses cells; `CELL_CROSSING_KINDS = ("airlock", "bulkhead")` (gen_ship.py:279), so until rev 2 §5.2's row
  lands `ShipGraph.check()` raises GraphSchemaError, which `generate` does not catch (it catches `_Reject` only): a crash,
  not a reroll. Not in the Costs table.

**Fix.** Write the wall as `1 + AIRLOCK_W + 1 (mount) + 1 + DOOR_W + 1`, both `at`s explicit; the grammar (not the layout)
decides where the bypass goes, from a precondition the layout can guarantee (the passageway across >= 13 when the archetype
applies it); the layout never drops an application -- a miss is `_Reject`. List the `CELL_CROSSING_KINDS` row as a dependency.

### 9. MAJOR -- the layout silently picks anchor rooms and adds edges the grammar never declared

> "2 = passageway + one department room of a row" (§5); "a door down into one department room near the band" (§4.2);
> "then the row's band-end rooms" (§4.3)

**Evidence.** The output contract is the grammar's graph with geometry (§1: "a `ShipGraph` exactly as `embed` consumes it").
In three places the layout chooses a department room and gives it an edge to the slab, the strip or a section. The grammar
cannot have declared those edges, because which room lands at the slab's corner or under the strip's end is decided by the
layout's own order and packing. Consequences: (a) if that department is `cell = "own"`, the new door crosses a cell and
`ShipGraph.check()` raises GraphSchemaError (gen_ship.py:589), which `generate` lets through (crash, no reroll); (b) a bunk
becomes a through-room, which changes the department's walk graph and the pocket / privacy semantics the recipe authored;
(c) the no-triangle and L5 invariants ran on a graph without these edges.

**Fix.** Either the grammar declares the anchor by role (the archetype names an `exit` room per department or a band-end
landing, and the department packer must put that room at the slab / strip corner: a placement constraint, written down), or
the layout is allowed to add edges from a closed list, in main-cell rooms only, and re-runs the §1.5 invariants on its output.
Any violation the layout detects is `_Reject`, never a schema error.

### 10. MAJOR -- the skeleton's rooms are not tied to the grammar's axiom (L3)

> "**Wing passageway**: ONE corridor room ... It is the wing's walk spine." (§2); input "rooms: ... (from the grammar + resolver)" (§1)

**Evidence.** L3: "Ships: axiom = the mandatory set as a pre-connected skeleton". Rev 2's department field `entry = "stub"  # the
room that joins the ship's spine` needs a spine NODE to join. The document creates the passageway, the slab and the band
sections in the layout, but its input is the grammar's rooms; it never says whether those three are grammar nodes. If not, the
grammar's invariants (no triangle, L5, reachability) run on a graph without the spine and every department's entry edge has
no other end. If yes, the document must give the mapping from axiom roles to skeleton pieces.

**Fix.** State: the freighter archetype's axiom IS the skeleton (slab, passageway, band sections as quotient nodes with
fixed roles); the layout maps each axiom role to its primitive and fails loudly on an unknown role. One table, in §2.

### 11. MAJOR -- the retreat loop is placed where rev 2 says it is not, and its legs do not map

> "`two-stub`: the department rectangle gets two parallel stubs, each with its own entry door on the passageway, and the end room
> spans both ... Rooms pack ... between them if the gap allows a room (else the gap is the end room's neck)." (§4.1)

**Evidence.** Rev 2 §3.1: "A retreat loop is a SHIP-level type the archetype recipe applies between departments' entries"; the
quarters are a tree on purpose. §4.1 builds it INSIDE one department rectangle. `edge_to_cycle` on A-B (passageway - the
department's `entry` stub) yields A-C-B and A-D-B with C, D NEW rooms of 1-2 each (rev 2 §1.3, §2); the two-stub has exactly
one room per leg and never says which is the old `entry` (does the authored stub become the end room? then the department's
packed rooms, whose edges are to `stub`, lose their hub). "The end room's neck" is a non-rectangular room -- `Room` is one
rectangle (gen_ship.py `Room.box`) -- so the no-room case is a void, which §6 forbids.

**Fix.** Either align with rev 2 (the loop joins two departments' entries through the passageway: C and D are the entry stubs
of NEIGHBOURING department rectangles and B is a room spanning both rectangles' far ends -- a cross-department end room, which
must be said), or amend rev 2 §3.1 explicitly. Give the leg -> primitive mapping for 1 and 2 rooms per leg, and make the gap
either a packed room or an explicit steel void.

### 12. MAJOR -- the department packer's room roles (pack / end / satellite) have no recipe field and no inference rule

> "the end room (if the recipe marks one: the mess)"; "satellites behind their parents" (§4); "**satellite** | splits a packed
> room's slab along the strip's direction: parent nearest the strip, satellite behind it ... | galley off the mess" (§3);
> "The hub-perimeter feasibility check of the recipe doc (§3.1) is exactly 'the minimum pack of step 3 fits the maximum stub'"

**Evidence.** Rev 2's room keys are `id, shape, required/count/p, width, size, fixture, fuel, light, vent, water` (§3.2):
nothing marks an end room. "Satellite" is defined on a PACKED room, but the example parent (the mess) is the END room. A
satellite needs its side's depth >= `6 + 1 + 6 = 13`, and the side depth is shared (finding 1). "Along the strip's direction"
vs "behind it" reads both ways. The feasibility claim is not "exactly" rev 2's check: rev 2 sums the stub's whole perimeter
including both ends (`2*24 + 2*8 - 6 = 58` against a demand of 45), a necessary condition only; the packer needs a two-bin
packing of side rooms into two sides of `along` each (plus walls), with the end room on the far end, which can fail while
45 <= 58 holds (e.g. the mess on a side: 10 + 3 rooms of 7 = 31 > 24).

**Fix.** Either add a recipe key (`attach = "side" | "end" | "behind:<room>"`) to rev 2's format, or state a deterministic
inference (a room whose only edge is to a non-entry room is that room's satellite; ...). Replace "exactly" with the packer's
own two-bin check, run by the loader at the recipe's minima.

### 13. MAJOR -- the maintenance ring: a reroll rate nobody computed, an unrealisable 8-ring, a hub the wing cannot reach

> "the ring is 2 side rooms (each split into 1-2 along y: that gives 4 or 6 ring rooms ...) ... Band width demand: 2·6 + 2 + hub
> minimum; the archetype's band range must afford it or the loader refuses the ring on that preset." (§4.4)

**Evidence.**
- Demand `12 + 2 + 6 = 20`. `band_w` is (10, 14), (12, 18), (14, 22): small and medium can never host a ring; large only when
  `wm` draws 20 or 22, 2 of its 5 even values. "Afford" is ambiguous (range top or bottom?); if the top, ~60 % of large
  attempts are rejected unless the `wm` draw is conditioned on the ring. The hub is then 6-8 wide: the engine room is a closet
  (feel, not in §9).
- Rev 2 allows `rooms = [4, 8]`; "split into 1-2" realises only 4 or 6. An 8-ring needs sides split into 3.
- Ring numbering (where r_1 is) decides whether the hub's doors go to fore/aft (axis, centred) or to the side pair. Both are
  mirror-safe, but two implementers will build different rings. Whether fore/aft span `wm` or only the hub width is also
  open (either gives 6-tile corner contacts, exactly `DOOR_W + 2`).
- With a ring, the band's outer columns at the hub's y-range are RING SIDE ROOMS. The passageway (or any wing room) facing
  engineering touches a ring room, not the hub: a declared passageway-hub edge is unrealisable on every attempt.
- The side ring rooms are a mirror PAIR inside the band: "the axis band laid out once" must say band rooms can have twins.

**Fix.** Condition the `wm` draw (and the hub section's height, finding 5) on the ring; refuse `rooms` > 6 in v1 or add the
3-split; fix the numbering (r_1 = fore); say that wing-facing edges to a ringed hub land on ring side rooms (grammar side: the
passageway's band anchor is a ring room). List the closet-sized engine room in §9.

### 14. MAJOR -- determinism: the rng split contradicts "never re-enter the grammar", and half the draws have no stated order

> "every draw from the one `rng` in a fixed order (rows, then departments in gradient order, then rooms in recipe order)" (§11);
> "reject the draw (`_Reject`), the attempt rerolls under `MAX_ATTEMPTS`, as today. Never re-enter the grammar (L8)." (§7)

**Evidence.** `generate` builds ONE generator per attempt, `PCG64(SeedSequence([seed, attempt]))` (gen_ship.py:1958), and runs
the whole chain from it. If the grammar draws from "the one rng", every reroll regenerates the GRAPH: re-entering the grammar
is exactly what happens. If the grammar is drawn once, it needs its own stream, and rev 2 §3.4's "(recipe hash, seed)
reproduces the resolved graph" requires that stream to be attempt-independent. Draws with no stated order: the skeleton's own
(insets, `wm`, `k`, cuts, passageway y, slab jitter); department -> row assignment; the "order drawn" of middle departments;
the split of `count` instances across sides; the stub's x-position in its rectangle; every `at`; the hull strip's door room;
the ring numbering. "No set iteration" is right but not sufficient.

**Fix.** Two streams written into §11, e.g. `SeedSequence([seed, 0])` for the grammar (once) and `SeedSequence([seed, 1,
attempt])` for layout + embed (any fixed scheme that leaves `ship_demo`'s recipe-less path untouched); then the full ordered
draw list, one line per draw.

### 15. MAJOR -- rows: assignment, empty rows and depth are unspecified

> "Each row is a 1D sequence of DEPARTMENT RECTANGLES along x, ordered by the gradient" (§2); "v1 rows have equal depth, so no"
> (§12); "at a y the layout draws (centre ± jitter ...)" (§2)

**Evidence.** Nothing says how a wing's departments are split between row A and row B (alternate in recipe order? by
demand? drawn?). An odd department count, or one wing department, leaves a row partly or wholly empty: a void, which §6
forbids, with no filler named. Both rows have a slab end and a band end, so `toward = "spawn"` departments compete for two
slots with no tie rule. "Equal depth" is contradicted by the passageway's y jitter and by the hull strip (row A's departments
under the strip lose `across + 1 = 4`, the last one does not: row A is stepped). Where a stub sits inside its department
rectangle (centred? slab side?) is open too, and it decides which rooms touch the slab or the band.

**Fix.** A row-assignment rule (e.g. first-fit by x-demand in gradient order, ties to row A), a named filler for a short row
(or the steel void of finding 17), drop "equal depth" or drop the jitter, and pin the stub position.

### 16. MAJOR -- an `own`-cell department's entry airlock does not fit the guarantee or the vestibule rule

> "every declared edge is realised as a contact of length >= its opening (`DOOR_W` + 2 ..., `AIRLOCK_W` + 2 ...)" (§1);
> "its passageway end gets the entry door" (§4 step 1)

**Evidence.** A stub meets the passageway at its END wall, so the contact is the stub's `across`, 6-8 for the quarters. An
`own` department's entry must be an airlock (L5; `ShipGraph._check_ducts_and_cells`, gen_ship.py:589): `AIRLOCK_W + 2 = 7 > 6`,
so the document's own guarantee fails on a 6-across stub. Which room is `a` (the vestibule side) is never said for entries: if
the passageway, `_airlock` needs it `>= VESTIBULE_D + 1 + FORMATION_FOOTPRINT = 9` deep across the wall (gen_ship.py:839), so
the passageway's minimum `across` rises to 9 whenever any `own` department hangs off it. And the passageway-end vestibule (the
engineering airlock of §2) carved into a 6-wide passageway spans its full width for 5 tiles: any stub door in that x-range cuts
the vestibule's ring and `_carve_airlock` rejects.

**Fix.** For entries, `a` = the department's stub (depth = `along` >= 9, true for every corridor stub); an `own` department's
stub `across >= 7`; stub entry doors keep >= `VESTIBULE_D + 2` from any passageway-end vestibule.

### 17. MINOR -- "never a void" rests on a wrong reading of embed

> "filler between departments is given to the departments (never a void: a void would be a sealed box with no vents, which
> `validate`'s compartment gate would count as a lost room)" (§6)

**Evidence.** `embed` fills the hull interior with STEEL and stamps air only in room rectangles (gen_ship.py:1545-1550). A gap is
solid steel: no compartment, no vent, invisible to `compartments()` and `_graph_compartment_reasons`. B7 (i), which §3 cites,
says gaps are allowed. A steel void's real costs lie elsewhere (thick walls breach and conduct heat differently; wasted area;
it reads oddly on the map).

**Fix.** Correct the rationale, then decide on the merits: a bounded steel void (e.g. <= 2 tiles thick, or only behind back
rooms) would absorb most of finding 1's slack and finding 15's short rows cheaply.

### 18. MAJOR -- `hull_door` rooms: the layout can place them off the hull, and the miss is a crash, not a reroll

**Evidence.** The cargo hold's `hull_door` tag (vocabulary B.4, E.2 row 5) is checked when the `ShipGraph` is BUILT:
`__post_init__ -> check() -> _check_hull_door -> hull_door_site`, which raises **GraphSchemaError** for a room with no hull run,
a run shorter than the door, or an axis room of the wrong parity (gen_ship.py:521, 691-704: "the geometry is the graph's").
`generate` catches only `_Reject`. plan_graph never tags `hull_door`; under the layout, a hold packed away from the hull, under
a hull strip (finding 3 takes row A's hull contact), or on the axis with an odd run crashes generation. The document never
mentions `hull_door`.

**Fix.** The packer places a `hull_door` room as a back room against the hull with a run >= its width and never under the hull
strip; on the recipe path the hull-door site miss is a `_Reject` (or the layout pre-checks it).

### 19. MINOR -- contract wording, misattributed rules and unacknowledged departures

- "`Edge`s with `at`, `hull`, `mirror_x`" (§1): `Edge` is `a, b, kind, at`; `hull` and `mirror_x` are `ShipGraph` fields. Input
  `shape`, `department`, `width` have no `Room` field (dept and cell are tags); a recipe `vestibule`-shaped room vs `_airlock`'s
  carve (a Room AND a carve?) is undefined.
- The "+ 2 clearance" is presented as the opening rule; `_opening_tiles` has NO clearance (any `at` in `[lo, hi - n + 1]`,
  gen_ship.py:631-633). The `+ 2` is `plan_graph`'s candidate filter (gen_ship.py:1081, applied to windows too) and `_at`'s
  one-tile margin (gen_ship.py:996). Say it is the layout's convention.
- "area is the one failure mode (B7)" (§1): `_airlock` depth, `_vent_site`, `_floor_site`, section alignment and the symmetry
  gate all reject too.
- §7's misses are the wrong ones: a satellite parent is >= `MIN_ROOM = 6 = DOOR_W + 2`, so it never misses a door. The real
  misses are cross-structure contacts (slab-department exits, strip-room, passageway-section, the bypass wall); name those.
- Departures from rev 2 / v0.1 not marked: the fixed skeleton replaces rev 2 §5.1's "slicing subdivision ... with the recipe's
  size hints as weights"; the swap move is dropped (rev 2 §5.1 lists it; §7 gives a reason, not the amendment); §10 redefines
  rev 2 §1.1's "contact graph: rooms that share a wall" as declared edges (correct and necessary -- every ship has undeclared
  physical triangles, e.g. passageway / stub / first packed room -- but an amendment to rev 2 §1.1, §1.5 and v0.1 §9's last
  sentence, not a "nuance"); L9's shape class and the research's "mandatory and prefab rooms placed first" (rev 2 §1.5: "a
  wanted three-room corner is a prefab") are absent; `wish_near` / `wish_far` are ignored without saying who honours them.

**Fix.** Correct the contract sentence and the clearance attribution; add a short "amends" list at the top.

### 20. MINOR -- the Costs table omits pieces the stage cannot ship without

- The GRAMMAR executor (move library, cycle applications, §1.5 invariants, phases) is in no row; only loader + resolver are.
  `generate`'s `recipe=` path "grammar -> resolver -> layout" cannot be built from the listed patches.
- The recipe path needs its own `check_request` (today's is plan_graph's arithmetic, `wing = (interior_w - band_w[0]) // 2 - 1`,
  gen_ship.py:926), and the loader's "can NEVER fit" check needs the skeleton arithmetic: a coupling between the loader patch
  and `ship_layout.py` the table hides.
- `CELL_CROSSING_KINDS` + `breakable` (finding 8) as a prerequisite.
- Lighting: `embed` gives every room ONE static lamp at its centre, range 16 (gen_ship.py:1600-1608). A medium passageway is
  25-30 long, a large one 53-59, the hull strip similar: their ends are unlit. A per-length lamp rule is a stage-2 extension,
  and dark passageway ends are a feel consequence missing from §9.
- "`generate` grows a `recipe=` path ... small": it also needs the rng split (finding 14), schema errors turned into
  `_Reject` on the recipe path (findings 8, 9, 18), and `ShipPlan` carrying the recipe hash.

## What the document gets right (must survive revision)

- Constructive placement over search for the freighter: strips + 1D packs make most declared edges contacts BY CONSTRUCTION,
  the research's "subdivision, not search" done honestly; one bounded repair then reroll is the right failure policy.
- The output contract: the stage emits a `ShipGraph` and never stamps a tile; `embed` / `validate` stay the one stage 2 and the
  one gate set; `plan_graph` kept as the recipe-less producer behind the `ship_demo` byte-identity gate.
- Mirror by reflecting a laid-out half, never by a second draw; repair before reflection.
- Declared edges realised, undeclared touches allowed (§10 first bullet): the only reading under which any layout exists; B7's
  "filled triangle" nuance is stated correctly.
- The narrow leg entered only from its long sides: `validate`'s 3x3 sliding window (gen_ship.py:1860) passes a 3-wide corridor
  exactly, a 4-wide door admits the footprint, and fuel stays refused on it.
- A per-archetype skeleton, freighter only, with the hull as a LIST of rectangles from day one (§8).
- Feel consequences listed separately for Erik's veto; the Systems section's one-layouter rule.
