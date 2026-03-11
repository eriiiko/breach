"""
Smoke, fire & atmospheric decompression prototype for Breach.

Demonstrates:
- Random maze generation (recursive backtracker, 3-wide corridors)
- Three scalar fields: atmosphere, smoke, fire — all on the same grid
- Shared Laplacian with wall boundaries (Neumann BC)
- Fire spreads to flammable walls, consumes O2, produces smoke
- Hull breach vents atmosphere, starving fires near the breach
- Smoke advected by pressure gradients (wind from decompression)

Physics from map_and_physics_design.md.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
GRID = 41
CORRIDOR_W = 3
CELL_STEP = CORRIDOR_W + 1

MAZE_NX = (GRID - 1) // CELL_STEP
MAZE_NY = (GRID - 1) // CELL_STEP

# Physics
D_ATM = 200.0           # atmosphere diffusion (fast venting to vacuum)
D_SMOKE = 0.4            # smoke diffusion
D_FIRE = .3             # fire spread rate (to neighboring flammable tiles)
ADVECTION_RATE = 25.0    # smoke advection by pressure gradient
dx = 1.0
dt = 0.001              # CFL: dt < dx^2/(4*D) = 0.00125 for D=200
SMOKE_DECAY = 1.0

# Fire
FIRE_O2_THRESHOLD = 0.15   # fire dies below this atmosphere level
FIRE_O2_CONSUMPTION = 0.3  # how much atmosphere fire eats per step
FIRE_SMOKE_EMISSION = 0.8  # how much smoke fire produces per step
FIRE_WALL_DAMAGE = 0.4     # HP damage to wall per step while burning
WALL_HP_WOOD = 60.0        # HP for flammable inner walls

# Simulation
TOTAL_FRAMES = 200
STEPS_PER_FRAME = 80    # more sub-steps to compensate smaller dt
BREACH_FRAME = 30
FIRE_CORNER_FRAME = 0      # far corner fire starts immediately
FIRE_BREACH_FRAME = 30     # near-breach fire starts with the breach

# Smoke sources (kept from before, but fire is the main smoke producer now)
SMOKE_EMISSION = 1.0
SRC1_START, SRC1_END = 0, 40
SRC2_START, SRC2_END = -1, -1  # disabled — fire produces smoke now

# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------
def generate_maze(nx, ny, seed=42):
    rng = np.random.default_rng(seed)
    visited = np.zeros((ny, nx), dtype=bool)
    h_walls = np.ones((ny, nx - 1), dtype=bool)
    v_walls = np.ones((ny - 1, nx), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True

    while stack:
        cy, cx = stack[-1]
        neighbors = []
        for dy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny2, nx2 = cy + dy, cx + ddx
            if 0 <= ny2 < ny and 0 <= nx2 < nx and not visited[ny2, nx2]:
                neighbors.append((ny2, nx2, dy, ddx))
        if neighbors:
            ny2, nx2, dy, ddx = neighbors[rng.integers(len(neighbors))]
            if dy == 0:
                h_walls[cy, min(cx, nx2)] = False
            else:
                v_walls[min(cy, ny2), cx] = False
            visited[ny2, nx2] = True
            stack.append((ny2, nx2))
        else:
            stack.pop()

    # Remove ~40% of remaining walls for open layout
    for row in range(ny):
        for col in range(nx - 1):
            if h_walls[row, col] and rng.random() < 0.4:
                h_walls[row, col] = False
    for row in range(ny - 1):
        for col in range(nx):
            if v_walls[row, col] and rng.random() < 0.4:
                v_walls[row, col] = False

    grid = np.ones((GRID, GRID), dtype=bool)
    for row in range(ny):
        for col in range(nx):
            ty = 1 + row * CELL_STEP
            tx = 1 + col * CELL_STEP
            grid[ty:ty + CORRIDOR_W, tx:tx + CORRIDOR_W] = False

    for row in range(ny):
        for col in range(nx - 1):
            if not h_walls[row, col]:
                ty = 1 + row * CELL_STEP
                tx = 1 + col * CELL_STEP + CORRIDOR_W
                grid[ty:ty + CORRIDOR_W, tx:tx + 1] = False

    for row in range(ny - 1):
        for col in range(nx):
            if not v_walls[row, col]:
                ty = 1 + row * CELL_STEP + CORRIDOR_W
                tx = 1 + col * CELL_STEP
                grid[ty:ty + 1, tx:tx + CORRIDOR_W] = False

    return grid


# ---------------------------------------------------------------------------
# Shared Laplacian (from design doc)
# ---------------------------------------------------------------------------
def compute_laplacian_with_walls(p, wall):
    """Discrete Laplacian with Neumann BC at walls (zero flux)."""
    up    = np.roll(p,  1, axis=0)
    down  = np.roll(p, -1, axis=0)
    left  = np.roll(p,  1, axis=1)
    right = np.roll(p, -1, axis=1)

    wall_up    = np.roll(wall,  1, axis=0)
    wall_down  = np.roll(wall, -1, axis=0)
    wall_left  = np.roll(wall,  1, axis=1)
    wall_right = np.roll(wall, -1, axis=1)

    up    = np.where(wall_up,    p, up)
    down  = np.where(wall_down,  p, down)
    left  = np.where(wall_left,  p, left)
    right = np.where(wall_right, p, right)

    return up + down + left + right - 4.0 * p


# ---------------------------------------------------------------------------
# Physics steps
# ---------------------------------------------------------------------------
def atmosphere_step(atm, wall, is_vacuum, fire, dt_step, D):
    lap = compute_laplacian_with_walls(atm, wall)
    atm_next = atm + D * dt_step * lap

    # Fire consumes oxygen in neighboring air tiles
    for dy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor_fire = np.roll(fire, (dy, ddx), axis=(0, 1))
        atm_next -= FIRE_O2_CONSUMPTION * dt_step * neighbor_fire

    atm_next[wall] = 0.0
    atm_next[is_vacuum] = 0.0
    return np.clip(atm_next, 0.0, 1.0)


def smoke_step(smoke, atm, wall, is_vacuum, fire, dt_step, D_s, adv):
    lap = compute_laplacian_with_walls(smoke, wall)
    smoke_next = smoke + D_s * dt_step * lap

    # Advection: smoke carried by pressure gradient (wind)
    grad_y = (np.roll(atm, -1, axis=0) - np.roll(atm, 1, axis=0)) / 2.0
    grad_x = (np.roll(atm, -1, axis=1) - np.roll(atm, 1, axis=1)) / 2.0
    dsmoke_dy = (np.roll(smoke, -1, axis=0) - np.roll(smoke, 1, axis=0)) / 2.0
    dsmoke_dx = (np.roll(smoke, -1, axis=1) - np.roll(smoke, 1, axis=1)) / 2.0

    # Air flows from high to low pressure: v = -grad(p)
    # Advection: d(smoke)/dt = -v . grad(smoke) = +grad(p) . grad(smoke)
    smoke_next += adv * dt_step * (grad_x * dsmoke_dx + grad_y * dsmoke_dy)

    # Fire produces smoke in neighboring air tiles
    for dy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor_fire = np.roll(fire, (dy, ddx), axis=(0, 1))
        # Only add smoke to non-wall tiles
        smoke_next += FIRE_SMOKE_EMISSION * dt_step * neighbor_fire * (~wall).astype(float)

    smoke_next *= SMOKE_DECAY
    smoke_next[wall] = 0.0
    smoke_next[is_vacuum] = 0.0
    return np.clip(smoke_next, 0.0, 1.0)


def fire_step(fire, atm, wall, flammable, wall_hp, dt_step, D_f):
    """Fire spreads to neighboring flammable walls, needs O2, damages walls."""
    fire_next = fire.copy()

    # Spread: each burning tile ignites neighboring flammable tiles
    # Check direct neighbors AND 2-tile range (radiant heat / embers)
    neighbor_fire = np.zeros_like(fire)
    for dy, ddx in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2),
                     (-1,-1),(-1,1),(1,-1),(1,1)]:
        neighbor_fire += np.roll(fire, (dy, ddx), axis=(0, 1))
    # Flammable tiles that aren't burning yet catch fire from neighbors
    can_ignite = flammable & (fire < 0.01) & (neighbor_fire > 0.1)
    fire_next[can_ignite] += D_f * dt_step * neighbor_fire[can_ignite]

    # Burning tiles grow toward full intensity
    burning = fire_next > 0.01
    fire_next[burning] += 0.5 * dt_step  # slow ramp-up

    # Fire only lives on flammable wall tiles
    fire_next[~flammable] = 0.0

    # Check oxygen: fire needs atmosphere in neighboring air tiles
    neighbor_atm = np.zeros_like(atm)
    count = np.zeros_like(atm)
    for dy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_atm = np.roll(atm, (dy, ddx), axis=(0, 1))
        shifted_wall = np.roll(wall, (dy, ddx), axis=(0, 1))
        is_air = ~shifted_wall
        neighbor_atm += shifted_atm * is_air.astype(float)
        count += is_air.astype(float)
    count = np.maximum(count, 1.0)
    avg_neighbor_atm = neighbor_atm / count

    # Starve fire where oxygen is too low
    fire_next[avg_neighbor_atm < FIRE_O2_THRESHOLD] = 0.0

    # Fire damages walls
    wall_hp_next = wall_hp - FIRE_WALL_DAMAGE * dt_step * fire_next

    # Walls that burn through: become open air
    burned_out = (wall_hp_next <= 0) & flammable & wall
    wall_hp_next[burned_out] = 0.0
    fire_next[burned_out] = 0.0

    fire_next = np.clip(fire_next, 0.0, 1.0)
    wall_hp_next = np.clip(wall_hp_next, 0.0, WALL_HP_WOOD)

    return fire_next, wall_hp_next, burned_out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("Generating maze...")
walls = generate_maze(MAZE_NX, MAZE_NY, seed=42)

# Hull boundary (non-flammable)
walls[0, :] = True
walls[-1, :] = True
walls[:, 0] = True
walls[:, -1] = True

# Flammable grid: all inner walls are flammable, hull is not
flammable = walls.copy()
flammable[0, :] = False
flammable[-1, :] = False
flammable[:, 0] = False
flammable[:, -1] = False

# Wall HP
wall_hp = np.zeros((GRID, GRID), dtype=np.float64)
wall_hp[flammable] = WALL_HP_WOOD

is_vacuum = np.zeros((GRID, GRID), dtype=bool)

# Initialize fields
atmosphere = np.where(walls, 0.0, 1.0)
smoke = np.zeros((GRID, GRID), dtype=np.float64)
fire = np.zeros((GRID, GRID), dtype=np.float64)

# Smoke source near center
center = GRID // 2
smoke_src = None
for off in range(20):
    for dy, ddx in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
        yy, xx = center + dy + off, center + ddx + off
        if 0 < yy < GRID - 1 and 0 < xx < GRID - 1 and not walls[yy, xx]:
            smoke_src = (yy, xx)
            break
    if smoke_src:
        break

print(f"Smoke source 1 (center) at: {smoke_src}")
smoke_src2 = None

# Breach point: find corridor touching left hull wall
breach_tiles = []
for row in range(2, GRID - CORRIDOR_W - 1):
    if all(not walls[row + k, 1] for k in range(CORRIDOR_W)):
        for k in range(CORRIDOR_W):
            breach_tiles.append((row + k, 0))
        break

if not breach_tiles:
    for col in range(2, GRID - CORRIDOR_W - 1):
        if all(not walls[GRID - 2, col + k] for k in range(CORRIDOR_W)):
            for k in range(CORRIDOR_W):
                breach_tiles.append((GRID - 1, col + k))
            break

print(f"Breach tiles: {breach_tiles}")

# Place second smoke source ~8 tiles inward from breach
if breach_tiles:
    br_y = breach_tiles[len(breach_tiles) // 2][0]
    br_x = breach_tiles[0][1]
    for offset in range(8, 25):
        candidate = (br_y, br_x + offset)
        if 0 < candidate[0] < GRID - 1 and 0 < candidate[1] < GRID - 1:
            if not walls[candidate[0], candidate[1]]:
                smoke_src2 = candidate
                break
    print(f"Smoke source 2 (near breach) at: {smoke_src2}")

# --- Fire ignition points ---
# Fire 1: near the breach — find cluster of flammable wall tiles close to breach
fire_breach_tiles = []
if breach_tiles:
    br_y = breach_tiles[len(breach_tiles) // 2][0]
    br_x = breach_tiles[0][1]
    for offset in range(2, 15):
        for dy in range(-3, 4):
            fy, fx = br_y + dy, br_x + offset
            if 0 < fy < GRID - 1 and 0 < fx < GRID - 1 and flammable[fy, fx]:
                fire_breach_tiles.append((fy, fx))
        if len(fire_breach_tiles) >= 3:
            break
print(f"Fire near breach tiles: {fire_breach_tiles}")

# Fire 2: far corner (bottom-right) — find cluster of flammable wall tiles
fire_corner_tiles = []
for dist in range(3, 25):
    for dy in range(-3, 4):
        for ddx in range(-3, 4):
            fy, fx = GRID - 2 - dist + dy, GRID - 2 - dist + ddx
            if 0 < fy < GRID - 1 and 0 < fx < GRID - 1 and flammable[fy, fx]:
                fire_corner_tiles.append((fy, fx))
    if len(fire_corner_tiles) >= 3:
        break
print(f"Fire far corner tiles: {fire_corner_tiles}")

breach_active = False

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
print("Setting up animation...")
fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=90)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

img = ax.imshow(np.zeros((GRID, GRID, 3)), origin="upper", interpolation="nearest")

ax.plot(smoke_src[1], smoke_src[0], "o", color="yellow", markersize=5,
        markeredgecolor="white", markeredgewidth=0.5, label="Smoke src", zorder=5)
if smoke_src2:
    ax.plot(smoke_src2[1], smoke_src2[0], "o", color="orange", markersize=5,
            markeredgecolor="white", markeredgewidth=0.5, label="Smoke src 2", zorder=5)
if fire_corner_tiles:
    fy, fx = fire_corner_tiles[0]
    ax.plot(fx, fy, "^", color="red", markersize=7,
            markeredgecolor="white", markeredgewidth=0.5, label="Fire (corner)", zorder=5)
if fire_breach_tiles:
    fy, fx = fire_breach_tiles[0]
    ax.plot(fx, fy, "^", color="orangered", markersize=7,
            markeredgecolor="white", markeredgewidth=0.5, label="Fire (breach)", zorder=5)

breach_marker, = ax.plot([], [], "X", color="cyan", markersize=10,
                         markeredgewidth=2, label="Breach", zorder=5)

ax.set_xlim(-0.5, GRID - 0.5)
ax.set_ylim(GRID - 0.5, -0.5)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper right", fontsize=6, facecolor="black", edgecolor="gray",
          labelcolor="white")
title = ax.set_title("", fontsize=9, color="white", pad=8)
fig.tight_layout()


def update(frame):
    global atmosphere, smoke, fire, wall_hp, walls, flammable
    global breach_active, is_vacuum

    # --- Events ---
    # Breach
    if frame == BREACH_FRAME and not breach_active:
        breach_active = True
        for by, bx in breach_tiles:
            walls[by, bx] = False
            flammable[by, bx] = False
            is_vacuum[by, bx] = True
            atmosphere[by, bx] = 0.0
        mid = breach_tiles[len(breach_tiles) // 2]
        breach_marker.set_data([mid[1]], [mid[0]])
        print(f"  BREACH at frame {frame}!")

    # Ignite fire near breach
    if frame == FIRE_BREACH_FRAME:
        for fy, fx in fire_breach_tiles:
            if flammable[fy, fx]:
                fire[fy, fx] = 1.0
        if fire_breach_tiles:
            print(f"  FIRE near breach ignited at frame {frame}!")

    # Ignite fire in far corner
    if frame == FIRE_CORNER_FRAME:
        for fy, fx in fire_corner_tiles:
            if flammable[fy, fx]:
                fire[fy, fx] = 1.0
        if fire_corner_tiles:
            print(f"  FIRE in far corner ignited at frame {frame}!")

    # --- Sub-stepping ---
    for _ in range(STEPS_PER_FRAME):
        # Atmosphere (with fire consuming O2)
        atmosphere = atmosphere_step(atmosphere, walls, is_vacuum, fire, dt, D_ATM)

        # Fire spread + wall damage
        fire, wall_hp, burned_out = fire_step(
            fire, atmosphere, walls, flammable, wall_hp, dt, D_FIRE
        )

        # Walls that burned through become open air
        if np.any(burned_out):
            walls[burned_out] = False
            flammable[burned_out] = False
            atmosphere[burned_out] = atmosphere[
                np.roll(burned_out, 1, axis=0) |
                np.roll(burned_out, -1, axis=0) |
                np.roll(burned_out, 1, axis=1) |
                np.roll(burned_out, -1, axis=1)
            ].mean() if np.any(~walls) else 0.5

        # Smoke sources (burst windows)
        sources = [
            (smoke_src,  SRC1_START, SRC1_END),
            (smoke_src2, SRC2_START, SRC2_END),
        ]
        for src, t_start, t_end in sources:
            if src is None or not (t_start <= frame <= t_end):
                continue
            sy, sx = src
            for dy in range(-1, 2):
                for ddx in range(-1, 2):
                    ny_, nx_ = sy + dy, sx + ddx
                    if 0 <= ny_ < GRID and 0 <= nx_ < GRID and not walls[ny_, nx_]:
                        s = SMOKE_EMISSION if (dy == 0 and ddx == 0) else SMOKE_EMISSION * 0.3
                        smoke[ny_, nx_] = min(smoke[ny_, nx_] + s * dt, 1.0)

        # Smoke (with fire producing smoke)
        smoke = smoke_step(smoke, atmosphere, walls, is_vacuum, fire, dt, D_SMOKE,
                           ADVECTION_RATE)

    # --- Composite RGB image ---
    rgb = np.zeros((GRID, GRID, 3))

    # Walls: flammable = warm brown, hull = dark gray
    hull = walls & ~flammable
    wood = walls & flammable
    rgb[hull, :] = 0.2
    # Wood walls colored by HP (darker as damaged)
    hp_frac = np.zeros((GRID, GRID))
    hp_frac[wood] = wall_hp[wood] / WALL_HP_WOOD
    rgb[wood, 0] = 0.18 + 0.12 * hp_frac[wood]   # brownish
    rgb[wood, 1] = 0.12 + 0.08 * hp_frac[wood]
    rgb[wood, 2] = 0.06 + 0.04 * hp_frac[wood]

    # Atmosphere: blue tint
    open_cells = ~walls & ~is_vacuum
    atm_vals = atmosphere[open_cells]
    rgb[open_cells, 0] += atm_vals * 0.03
    rgb[open_cells, 1] += atm_vals * 0.06
    rgb[open_cells, 2] += atm_vals * 0.35

    # Fire: bright yellow/orange on burning walls
    burning = fire > 0.05
    rgb[burning, 0] = np.clip(0.4 + fire[burning] * 0.6, 0, 1)    # bright orange
    rgb[burning, 1] = np.clip(0.15 + fire[burning] * 0.45, 0, 1)   # yellow component
    rgb[burning, 2] = np.clip(fire[burning] * 0.05, 0, 1)           # minimal blue

    # Smoke: red/orange overlay on air tiles
    smoke_air = smoke * (~walls).astype(float)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] + smoke_air * 0.85, 0, 1)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] + smoke_air * 0.2, 0, 1)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] + smoke_air * 0.05, 0, 1)

    # Breach tiles: cyan glow
    for by, bx in breach_tiles:
        if is_vacuum[by, bx]:
            rgb[by, bx] = [0.0, 0.4, 0.5]

    img.set_data(np.clip(rgb, 0, 1))

    interior = ~walls & ~is_vacuum
    avg_atm = atmosphere[interior].mean() if np.any(interior) else 0
    n_burning = int(np.sum(fire > 0.05))
    status = "PRE-BREACH" if not breach_active else "DECOMPRESSING"
    title.set_text(f"Frame {frame}/{TOTAL_FRAMES}  |  {status}  |  "
                   f"Pressure: {avg_atm:.2f} atm  |  Tiles on fire: {n_burning}")

    if frame % 40 == 0:
        print(f"  Frame {frame}/{TOTAL_FRAMES}, atm={avg_atm:.3f}, "
              f"fire_tiles={n_burning}")

    return [img, breach_marker, title]


# ---------------------------------------------------------------------------
# Run and save
# ---------------------------------------------------------------------------
print(f"Grid: {GRID}x{GRID}, {TOTAL_FRAMES} frames, "
      f"{STEPS_PER_FRAME} sub-steps/frame")
print(f"CFL check: dt={dt}, dx^2/(4*D)={dx**2/(4*D_ATM):.4f} -> "
      f"{'STABLE' if dt < dx**2/(4*D_ATM) else 'UNSTABLE!'}")

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True, interval=50)

output_path = "C:/Users/steen/projects/breach/prototypes/smoke_sim.gif"
print(f"Saving to {output_path} ...")
anim.save(output_path, writer=PillowWriter(fps=20))
print("Done!")
plt.close()
