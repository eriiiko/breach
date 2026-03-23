# Campaign Meta-Game — Dynamic Faction System

_Created: 2026-03-23 (from brainstorm session 2026-03-22)_

_Status: Concept phase. Goal/vision document, not implementation-ready._

---

## The Vision

Every campaign is unique. The player exists in a world of competing factions, each
with their own AI, objectives, and resources. Missions aren't a fixed sequence —
they emerge from the faction simulation.

**Core loop:**
1. The meta-game generates available missions based on world state
2. All factions (including the player) choose which missions to pursue
3. Missions play out (player missions are tactical gameplay, AI missions resolve off-screen)
4. World state updates — territory, resources, alliances, intel shift
5. Repeat

---

## Key Mechanics

### Missions as Shared Objectives

Missions are available to all factions, but each faction has different stakes:
- A cargo ship might contain weapons (valuable to military factions), data (valuable
  to intelligence factions), or biosamples (valuable to the grays)
- The player chooses which missions matter to them
- Other factions are pursuing their own objectives simultaneously

### Mission Templates, Not Authored Sequences

**20 mission templates, not 50 authored missions.** Each template defines:
- Environment type (cargo ship, station, research vessel, luxury liner)
- Phase structure (infiltrate → objective → extraction)
- Slots for: enemy type, enemy equipment, environmental hazards, intel items

The meta-state fills these slots based on who controls what:
- Which faction's troops garrison this ship? → determines enemy types
- What equipment tier does that faction have? → determines enemy loadout
- Is the target contested? → determines reinforcement waves
- Any unique items in play? → places them as objectives

### Faction AI

Two levels of AI, both candidates for ML training:

**Tactical AI** (unit-level): Controls enemy units during missions. Select-and-move,
cover usage, flanking, suppression. This is the Civulator-transferable work — trained
via DQN/self-play, same architecture patterns.

**Strategic AI** (faction-level): Chooses which missions to pursue, when to ally/betray,
resource allocation. Simpler — could be rule-based initially, ML-trained later.

**Key insight**: The tactical AI is work we're already doing (Civulator → Breach transfer
via egregore concept nodes). Once unit AI works, faction tournaments are just a wrapper.

### Alliances and Betrayal

- Factions can ally (joint operations, shared intel) or betray (defect mid-mission,
  steal objectives)
- Alliance stability depends on shared interests and trust score
- The player can participate in this — ally with a faction, receive support, but risk
  being sold out
- Betrayal incentive scales with prize value (unique items!)

### Unique Items

**Only one exists in the world.** Creates:
- Real scarcity — not just stats, political weight
- Intel value — knowing where it is becomes a mission objective
- Betrayal incentive — alliances break when the prize is big enough
- Emergent stories — "the faction that controls the Gray Disruptor has dominated
  for three missions, now everyone is gunning for them"

### Randomness Between AI Factions

When two AI factions contest a mission off-screen, the outcome isn't deterministic.
Weighted random based on force strength, equipment, intel advantage — but upsets
happen. This keeps the world dynamic and prevents optimal-play stagnation.

---

## Replayability

The player doesn't need to see every mission template to have a complete campaign.
Each playthrough traverses a different path through the possibility space:
- Different faction alliances form
- Different unique items fall into different hands
- Different missions become available based on world state
- The same template plays differently with different enemies and stakes

**Goal: no two campaigns feel the same**, achieved through emergent faction dynamics
rather than branching authored narrative.

---

## Implementation Phases

### Phase 1: Tactical AI (current — via Civulator)
- Train unit-level combat AI
- Transfer learned patterns to Breach via egregore concept nodes

### Phase 2: One Mission, Full Polish
- Complete Mission 1 (Silent Cargo) with full graphics, weapons, unit types
- Add 1-2 more unit classes (marines, gray escapee)
- This proves the tactical layer works

### Phase 3: Mission Templates
- Abstract Mission 1 into a template
- Create 2-3 more templates (assassination, infiltration, extraction)
- Parameterize enemy composition, equipment, objectives

### Phase 4: Faction Simulation
- Implement strategic AI (rule-based first)
- World state tracker (territory, resources, alliances, unique items)
- Mission generation from world state

### Phase 5: ML Strategic AI
- Train faction-level decision-making
- Tournament matches between faction AIs
- Player as chaos agent in the mix

---

## Relation to Other Docs

- `missions.md` — individual mission concepts that become templates in Phase 3
- `design_v2_turn_and_combat_overhaul.md` — the tactical layer this sits on top of
- `lore_the_femme_fatale.md` — honeypot mechanic as faction intelligence operation
- `lore_the_grays.md` — grays as a faction with their own objectives
- Civulator `NEXT_STEPS.md` — tactical AI training that transfers here
- Egregore concept nodes — the transfer mechanism

---

_This is a GOAL document. Implementation depends on the tactical layer being solid
first. The faction simulation is the reward for getting unit AI right._
