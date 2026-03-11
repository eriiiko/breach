# Breach — Collected Gameplay Ideas
## Creatures, Rooms, Mechanics & Lore Scattered Across Design Conversations

*Compiled from project chat history, Feb 25–Mar 2, 2026*
*Status: Raw idea collection — not all ideas are finalized design decisions*

---

## 1. Creature Types

### 1.1 Mindless Swarms
- **Giant Bees / Wasps** — aggressive insect swarms kept in containment rooms. When released (by breach, explosion, or deliberate player action), they attack anything nearby. Swarm behavior — individually weak, dangerous in numbers. Could be attracted to light or heat.
- **Gene-Manipulated Humanoids** — products of organ harvesting / genetic experimentation programs. Shambling, aggressive, low intelligence. Think "failed experiments" wandering the ship. Zombie-like behavior but with a lore explanation tied to the Princes' mad science.

### 1.2 Predators (Blood/Gore Attracted)
- **Vampires** — fits the satirical theme of elites literally feeding on people. Could be engineered creatures or lore-genuine vampires. Attracted to blood pools (the liquid layer on the grid). Creating gore becomes a tactical decision — blow someone up and the blood might draw something worse.
- **Predator Animals** — generic large predators kept in animal containment. Attracted to gore/blood on the liquid matrix. A blown-up body creates a blood pool → blood attracts predators → predators become a new threat. Emergent chain from the existing systems.

### 1.3 Intelligent Tactical Opponents
- **Genetic Soldiers** — purpose-built combat organisms. These are the "high end" enemies — they use cover, coordinate, flank. They interact with all the same systems players do (affected by pressure, fire, smoke). The AI challenge: making them feel smart without scripted behavior, letting them react to the emergent environment.
- **Human-Animal Hybrids** — unique abilities depending on the animal DNA. Could have enhanced senses (see through smoke?), enhanced speed, climbing ability (ignoring some wall restrictions?). Each hybrid type requires different tactical approaches.

### 1.4 Design Principle for All Creatures
Every AI should be trained as a neural network, just with different reward functions:
Every creature type should interact with the existing environmental systems without special-case code:
- Swarms are affected by decompression (sucked through hull breaches)
- Blood-attracted creatures respond to the liquid matrix
- Fire blocks or damages all creature types
- Smoke affects creature vision just like it affects players
- Intelligent creatures use the same cover/LOS system as player squads


---

## 2. Special Room Types

### 2.1 Animal Containment
Standard rooms where animals are caged. Destroying the containment (shooting cages, explosions, pressure changes) releases them. Creates tactical choices: do you breach through the animal room as a shortcut, risking releasing whatever's inside? Or do you deliberately release them as a distraction against ship defenders?

### 2.2 Alien Specimen Containment (Rare)
High-security containment for alien organisms — an Alien-movie-style encounter. These are rare, high-threat rooms. The alien is extremely dangerous but the containment is also extremely tough. Breaking in (or breaking the containment accidentally with explosions) creates an "oh shit" moment. Could be a mission objective itself: "Retrieve the alien specimen."

### 2.3 Organ Harvesting Rooms
Headless bodies connected to medical equipment — tied to the game's dark satirical lore about elite corruption and China's "organ logistics" industry. These rooms contain:
- Bodies on operating tables hooked to machines
- Valuable organs that could be mission objectives or sellable items
- Medical supplies/equipment that might be useful
- Lore implications — finding these rooms reveals what the ship is really being used for

### 2.4 Reactor Core
Central to every ship. The big "don't shoot here" zone. Too much damage and the entire ship explodes — mission failed for everyone. Creates a natural no-fire zone in the middle of the map, forcing tactical routing around it. Could also be a mission objective: "Sabotage the reactor" (set a timed charge and get out).

### 2.5 Pressure Door Rooms / Bulkhead Sections
Rooms separated by automatic pressure doors that seal when atmosphere drops. These create dynamic chokepoints — a hull breach on one side seals the doors, changing the map's traversability mid-mission. Opening a sealed door between pressurized and depressurized sections creates a rush of atmosphere (and anything in it).

### 2.6 "Establishments" (Satirical)
Rooms where politicians are "absolutely, positively not doing anything unseemly." Mission objective: extract the politician. These are played for dark comedy — luxuriously appointed rooms on otherwise industrial/military ships, with lore implications about what the elites get up to.

### 2.7 Botanical Garden Rooms
Exoctic plants and alien species - stored on reserach ships. Some plants may emit poisonous (invisible but flammable?) gas. (we could have different types of gasses).
---

## 3. Environmental Systems as Gameplay

### 3.1 Blood as a Dynamic Substance
Bodies destroyed by weapons create blood pools on the liquid layer of the grid. Blood functions like other liquids (water, fuel, coolant) but with unique interactions:
- Attracts predatory creatures
- Slippery surface (movement penalty?)
- Gore level increases over time as combat continues
- Creates tactical dilemma: messy combat draws attention

### 3.2 Fire ↔ Oxygen ↔ Decompression Chain
Fire consumes oxygen → local atmosphere drops → decompression can starve fires. Deliberate hull breach to vent fire = valid tactic but costs atmosphere for the whole section. Fire near hull breach gets pulled toward the breach by airflow.

### 3.3 Smoke as Vision Blocker → Stealth
Smoke density above threshold blocks line of sight. Combined with the lighting system's shadow stealth mechanic: characters in dark + smoky areas become nearly invisible. Smoke gets carried by airflow toward hull breaches.

### 3.4 Coolant/Water Flooding
Burst pipes flood rooms. Water extinguishes fire but fills sealed rooms. Water conducts electricity. Flooded rooms slow movement. If pressure doors seal a flooded section, the water has nowhere to go.

### 3.5 Emergent Cascade Examples
These weren't designed — they fall out of the math naturally:
- Grenade blows wall → starts fire → triggers pressure door → seals escape route → forced through flooded corridor → enemy waiting with shotgun
- Explosion breaks hull → atmosphere vents → smoke gets sucked out → fires starve → but everyone without suits suffocates
- Deliberate hull breach to vent fire/smoke = valid tactic
- Blood from combat attracts predators from containment breach two rooms over

---

## 4. Mission Types & Objectives

### 4.1 Core Objectives (from initial pitch)
- **Assassination** — kill a specific target on the ship
- **Theft** — steal a classified object
- **Hostage Rescue** — extract prisoners or captives
- **Extraction** — remove politicians from compromising situations
- **Sabotage** — destroy specific equipment (reactor, communications, etc.)

### 4.2 Lore-Inspired Missions
- **Watergate-style break-ins** — infiltrate, steal documents, get out without being detected
- **Kennedy-grade "unfortunate incidents"** — assassination missions dressed up as something else
- **Psy-ops missions** — plant false evidence, manipulate communications
- **Prison break** — high-security prison ships with experimental subjects (inspired by Abuse game's plot — prisoners subjected to genetic experiments, things go wrong)
- **Audit the central bank** — the meta-plot: missions where you're investigating the Princes themselves

---

## 5. Currency & Economy

### Three currencies with tactical trade-offs:

| Currency | Value Trend | Legal Status | Trackability | Weight |
|----------|-------------|--------------|--------------|--------|
| **Central Bank Dollars (CBD)** | -2% per turn (inflation) | Fully legal | Fully trackable | None |
| **Gold** | Stable | Not legal tender | Untrackable | Heavy — costs inventory/movement |
| **Bitcoin** | Appreciates | Illegal | Digitally trackable between wallets | None, but wallets may be anonymous (rare/expensive) |

Every choice has friction. The "safe" currency robs you slowly, gold is solid but cumbersome, bitcoin is powerful but puts a target on your back.

---

## 6. Weapons Arsenal

From the original pitch: MP5s, shotguns, pistols, Uzis, flamethrowers, laser rifles, sniper rifles, bazookas, grenades, flashbangs, smoke grenades, teargas grenades.

Interaction with environment:
- Light weapons → penetrate thin walls
- Heavy weapons → destroy walls entirely  
- Explosions → can breach hull, risk reactor
- Laser weapons → can start fires
- Flashbangs → interact with lighting system

---

## 7. Lore Elements (Scattered Ideas)

### 7.1 The World of 2087
- **International Central Banking Consortium** — ruled by "Princes" (Machiavelli's playbook as employee handbook)
- **The Expendables Foundation** — the Princes' "charity" with a goat-in-pentagram logo
- **EU** = Eastern Europe + Arab states + Canada (clerical error)
- **WHO** = disease innovation, motto: "You Can't Cure What Doesn't Exist Yet"
- **China** = "organ logistics" industry
- **USA North** = tech oligarch feudalism
- **USA South** = theocratic NASCAR state with nuclear capabilities  
- **France** = Neo-Napoleonic Empire (expanding into Belgium again)
- **Germany & Great Britain** = "historically interesting things" (TBD)

### 7.2 Plot Architecture (from story research)
Inspired by *The Nixon Conspiracy* + *Princes of the Yen*:
- Player starts as law enforcement investigating a "corrupt" president
- Evidence seems damning at first
- Gradually discovers the investigation was pre-cooked by the central bank
- The "crime" was fabricated to remove a president who threatened to audit them
- Twist structure: the "evidence" found early means the opposite of what player thought
- Anonymous whistleblower (like Geoff Shepard) leaves cryptic clues
- Central bank uses crisis-as-weapon playbook (inflate → crash → demand reforms)

### 7.3 The Princesses' Behavioral Checklist
- Control credit through hidden, extralegal mechanisms
- Public face: boring technocrats. Private agenda: ideological transformation
- Genuinely believe they're doing the right thing
- Inner circle groomed over generations
- Most employees oblivious to real agenda
- Can sabotage government recovery without anyone understanding how
- They've done this before in another context
- Elected government gets blamed, not them

---

## 8. Game Title

Working title: **Breach**

Other candidates that came up: Star Chamber, Diplomatic Immunity, Void Ops, Boarders, Breach Protocol, Section 8, Hull Down, Kill Order. Also inspired by game references: XCOM (tactical), Syndicate (dystopia), Abuse (prison experiments), Space Quest (nostalgia).

---

## 9. Ideas Still TBD / Unexplored

- Germany and Great Britain's satirical 2087 identities
- Specific creature stats and AI behaviors
- How currency interacts with mission loadout/equipment
- Multiplayer considerations
- Level editor details (paint materials, preview layers)
- Specific mission scripts and level designs
- Sound design and music direction
- Turn based mechanic - Eriks simultanious turns version worth exploring.

---

## 10. Observations from Physics Prototype (Mar 11, 2026)

*From running the smoke/fire/decompression simulation prototype.*

### Fire Maze Level Concept
- **Level idea:** Fire spreading through a maze. Player must run in, retrieve an item, and get back out. The maze difficulty *increases dynamically* as fire blocks corridors. The path that looks best initially might be blocked by the time you reach it — constant adaptation required.
- This could be a standalone mission type or a consequence of an explosion during any mission.

### O2 Starvation & Fire Extinction Cycle
- When enough of the station is on fire, atmosphere drops globally. Eventually O2 drops below ~0.15 atm and fires start going out on their own.
- **Gameplay dynamic:** Survive the fire phase (find pockets the fire doesn't reach), then when O2 starvation extinguishes the flames, a second phase opens up — new combat, new routes through the burned-out structure.
- Burned-through walls create new paths that didn't exist before.

### Environmental Hazards as Mechanics
- **Poison mist** — same diffusion system as smoke, but damages player. Spreads through corridors, blocked by sealed doors.
- **Tear gas** — already discussed, now confirmed the simulation framework handles it naturally. Same advection by pressure gradients, same wall interactions.
- All hazards ride on the same atmosphere field — a breach sucks *everything* out (smoke, poison, tear gas).

### Decompression Tuning Notes
- 3-tile breach drains a room fast. 1-tile breach is a slow leak. The physics scales naturally with breach width — no special cases needed.
- High D_ATM (~200) gives dramatic fast venting but requires small dt for CFL stability → computationally expensive. Optimization needed for real-time gameplay (this is a known future task, not a current priority).
- Fire near a breach gets starved of O2 and dies. Fire far from a breach keeps burning with remaining atmosphere. Emergent tactical implication: *deliberately breach the hull to fight fire*.