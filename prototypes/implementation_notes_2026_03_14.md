# Implementation Notes — 2026-03-14

## What Was Built

### 1. Wind Test Prototypes
- `wind_test.py` — 3-panel visualization (atmosphere, smoke, wind arrows via quiver plot)
- `wind_test_small.py` — compact 2-panel version
- Demonstrates: atmosphere diffusion, smoke advection by pressure gradient, mid-simulation explosion pressure spike
- Validates: wind-from-gradient mechanism, explosion → atmosphere → wind chain

### 2. Playable Game Prototype (`game.py`)
First playable Pygame prototype of Breach. Features:

**Map:**
- Fine-tile grid (1/3 m tiles, 120×75) with coarse-tile overlay (1m, 40×25)
- Ship layout: hull border, wood interior walls, doors
- Material system: hull (HP 300), wood (HP 60), doors (HP 40)

**Units:**
- 3 marines (Alpha, Bravo, Charlie) with 8-directional sprites
- Click to select, place orders on timeline
- Each unit occupies 3×3 fine tiles

**Order System:**
- T=5 timesteps per turn
- Move & Attack (1), Move w/ Cover (2), Sprint (3) — different speeds
- Grenade (G) — click target, scroll wheel to set detonation delay (0.5T increments)
- Explosive/breaching charge (B) — same UI, smaller blast, destroys walls/doors
- Escape to clear orders, Space/Enter to execute

**Execution Phase:**
- Orders play out in real time with smooth unit interpolation
- Physics run during execution: atmosphere diffusion, smoke advection
- Explosions: destroy walls, spike atmosphere (creates wind), generate smoke
- +/- keys to adjust playback speed

**UI Panel:**
- Mode selection with color coding
- Selected unit info (HP, position, inventory)
- Visual timeline bar with order blocks
- Orders list with timing details
- Controls help text

### 3. Design Discussion Document
`design_discussion_2026_03_14.md` — comprehensive notes from design session covering:
- Topic 1: Wind (gradient-based, validated in prototype)
- Topic 2: Atmosphere as wave medium (scale wave propagation by local atmosphere)
- Topic 3: Fire-wind interaction (wind-biased spreading, combined ignition formula, radiation via lighting raycast)
- Topic 4: Computation cost (fine grid 230× faster than real time in C++, no optimization needed)
- Topic 5: Dynamic resolution (skipped — unnecessary given fine grid performance)
- Topic 6: Simultaneous turns (full turn system design with T=5 timesteps)
- Topic 7: Path to playable demo (Pygame first, production engine later)

## Dependencies Added
- `pygame-ce` 2.5.7 (community edition)

## Files Added
- `game.py` — main game prototype
- `prototypes/wind_test.py` — wind visualization (3-panel)
- `prototypes/wind_test_small.py` — wind visualization (compact)
- `prototypes/wind_test.gif` — wind test output
- `prototypes/wind_test_small.gif` — wind test output (small)
- `prototypes/design_discussion_2026_03_14.md` — design session notes
- `prototypes/implementation_notes_2026_03_14.md` — this file
