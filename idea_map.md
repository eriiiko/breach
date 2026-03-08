# Breach — Idea Association Map

How the design documents connect to each other through shared ideas.

## Document Key

| Short Name | File | Focus |
|---|---|---|
| **Architecture** | `00_architecture_overview.tex` | Engine, two-layer arch, grid/tile system, materials, language pipeline, NN training |
| **Gameplay** | `gameplay_ideas_collected.md` | Creatures, rooms, environmental systems, missions, currency, weapons, lore |
| **Graphics** | `graphics_lighting_design.md` | 2D raycasting light, normal maps, shadow stealth, smoke/light interaction, art pipeline |
| **Physics** | `map_and_physics_design.md` | Map class, Laplacian, explosions (wave eq), atmosphere (diffusion), smoke (advection) |
| **Story** | `story_research_watergate_and_princes.md` | Nixon Conspiracy + Princes of the Yen → plot architecture, character templates |
| **Narrative** | `docs/narrative_media_systems_update_2026-03-08.md` | Chase Hughes "6 Layers", news cycle, player as narrative subject, media systems |

---

## Association Diagram

```mermaid
graph LR
    %% Nodes
    ARCH["<b>Architecture</b><br/>Engine, grid, materials,<br/>two-layer arch, NN"]
    GAME["<b>Gameplay</b><br/>Creatures, rooms,<br/>missions, currency"]
    GFX["<b>Graphics</b><br/>Lighting, shadows,<br/>normal maps, art"]
    PHYS["<b>Physics</b><br/>Explosions, atmosphere,<br/>smoke, Laplacian"]
    STORY["<b>Story</b><br/>Watergate + Princes,<br/>plot structure"]
    NARR["<b>Narrative</b><br/>Media systems,<br/>news cycle, Chase Hughes"]

    %% Physics cluster
    PHYS -->|"tile objects, cached arrays,<br/>material system"| ARCH
    PHYS -->|"fire↔oxygen↔decompression,<br/>emergent cascades"| GAME
    PHYS -->|"smoke density drives<br/>light occlusion"| GFX

    %% Graphics cluster
    GFX -->|"engine choice,<br/>tile resolution"| ARCH
    GFX -->|"shadow stealth mechanic,<br/>smoke blocks LOS"| GAME

    %% Lore/narrative cluster
    STORY -->|"Princes, economy, 2087 world,<br/>mission types, lore"| GAME
    STORY -->|"frame-up mechanism,<br/>whistleblower, plot twist"| NARR

    %% Narrative to gameplay
    NARR -->|"social credit, CBD currency,<br/>notifications, media as weapon"| GAME

    %% Architecture to gameplay
    ARCH -->|"document roadmap<br/>references all systems"| GAME
    ARCH -->|"grid = NN input,<br/>headless simulation"| PHYS

    style ARCH fill:#4a90d9,color:#fff,stroke:#2a5f9e
    style GAME fill:#d94a4a,color:#fff,stroke:#9e2a2a
    style GFX fill:#d9a54a,color:#fff,stroke:#9e7a2a
    style PHYS fill:#4ad97a,color:#fff,stroke:#2a9e4a
    style STORY fill:#9a4ad9,color:#fff,stroke:#6a2a9e
    style NARR fill:#d94a9a,color:#fff,stroke:#9e2a6a
```

---

## Idea Clusters (What Connects What)

### 1. Destructible Environment Chain
> Physics ↔ Graphics ↔ Gameplay

The core loop that ties three documents together:
- **Physics** defines *how* walls break (wave equation, pressure gradient damage, wall HP)
- **Graphics** defines *what happens visually* (light map recalculates, shadows shift, light floods through breaches)
- **Gameplay** defines *why it matters* (tactical routing, creature escape, emergent cascades)

Shared concepts: `gas_block` matrix, tile HP, material properties, hull breach

### 2. Smoke as Cross-System Glue
> Physics ↔ Graphics ↔ Gameplay

Smoke appears in three documents with different roles:
- **Physics**: diffusion + advection equations, carried by atmosphere gradient toward breaches
- **Graphics**: semi-transparent occluder, volumetric light shafts, feeds into `_light_block` cache
- **Gameplay**: vision blocker enabling stealth, tactical smoke grenades, fire byproduct

### 3. Material System
> Architecture ↔ Physics ↔ Gameplay

- **Architecture**: defines the data-driven material table (HP, flammable, blocks_light, blocks_gas, blast_resistance)
- **Physics**: uses `gas_block` and `blast_resistance` for propagation and wall destruction
- **Gameplay**: materials determine room behavior (wood burns, glass shatters, hull is tough)

### 4. The Princes / Central Bank
> Story ↔ Narrative ↔ Gameplay

The antagonist system spans three documents:
- **Story**: Princes of the Yen research → behavioral checklist, crisis-as-weapon, hidden credit control
- **Narrative**: Princes' *media apparatus* — news cycles, competing outlets, footage manipulation
- **Gameplay**: CBD currency (inflation mechanic), economy design, mission objectives ("audit the central bank")

### 5. The Frame-Up Plot
> Story ↔ Narrative ↔ Gameplay

The core plot thread:
- **Story**: Nixon Conspiracy structure → 5-phase plot (obvious crime → building case → anomalies → inversion → choice)
- **Narrative**: news system is the *mechanism* of the frame-up, player becomes narrative subject, context stripping
- **Gameplay**: mission types (Watergate break-ins, psy-ops, Kennedy-grade incidents), whistleblower clues

### 6. Grid / Tile Architecture
> Architecture ↔ Physics ↔ Graphics

The foundational data structure:
- **Architecture**: tile objects + cached arrays, 3×3 fine grid, entity list, double buffering
- **Physics**: all equations operate on the same grid, shared Laplacian function
- **Graphics**: tile-based sprites, normal maps match tile dimensions, light map is per-tile

### 7. Creatures ↔ Environmental Systems
> Gameplay ↔ Physics ↔ Graphics

Creatures are designed to interact with every environmental system:
- Blood (liquid layer) attracts predators → **Physics** liquid system
- Decompression sucks swarms through breaches → **Physics** atmosphere
- Smoke affects creature vision → **Graphics** LOS + **Physics** smoke density
- Fire blocks/damages all types → **Physics** fire + **Gameplay** tactical decisions

### 8. Stealth Mechanics
> Graphics ↔ Gameplay

- **Graphics**: shadow stealth = sample light intensity at tile, below threshold = hidden
- **Gameplay**: shoot out lights, use smoke, exploit dark rooms, creatures with enhanced senses (see through smoke?)

---

## Documents With No Direct Link

| Pair | Why |
|---|---|
| **Architecture ↔ Story** | Architecture is pure tech; story is pure narrative. They connect *through* Gameplay (mission structure) and Physics (simulation enables the world the story inhabits). |
| **Architecture ↔ Narrative** | Same — narrative media systems don't touch engine/grid decisions directly. |
| **Graphics ↔ Story** | No direct link. The political stage as "literal theatre" (Narrative doc) is a visual idea that would flow through Graphics eventually, but isn't there yet. |
| **Graphics ↔ Narrative** | The political theatre visual design (Part 5 of Narrative) is a future bridge. |
| **Physics ↔ Story** | No connection — physics doesn't touch plot. |
| **Physics ↔ Narrative** | No connection — though environmental destruction feeds into the news cycle indirectly (player causes events → news reports on them). |

---

## Heaviest Hub: Gameplay

`gameplay_ideas_collected.md` connects to **every other document**. It's the central hub — the place where technical systems, visual design, and narrative all converge into "what the player actually experiences." If any document needs to be split up, it's this one.
