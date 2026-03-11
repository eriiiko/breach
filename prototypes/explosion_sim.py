"""
Explosion wave equation prototype for Breach.

Full chain: grenade detonation → shockwave propagation (wave equation) →
wall destruction → hull breach → atmospheric decompression → smoke venting.

All four systems on the same grid, same Laplacian.
Physics from map_and_physics_design.md.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
GRID = 41
CORRIDOR_W = 3
CELL_STEP = CORRIDOR_W + 1
MAZE_NX = (GRID - 1) // CELL_STEP
MAZE_NY = (GRID - 1) // CELL_STEP

# ---------------------------------------------------------------------------
# Wave equation parameters (from design doc)
# ---------------------------------------------------------------------------
WAVE_C = 343.0          # speed of sound (m/s)
WAVE_DT = 0.001         # wave timestep (seconds)
WAVE_DX = 1.0           # 1 tile = 1 meter
WAVE_R = (WAVE_C * WAVE_DT / WAVE_DX) ** 2  # must be <= 0.5
NUM_TRAVERSALS = 3
WAVE_MAX_STEPS = int((GRID * NUM_TRAVERSALS) / (WAVE_C * WAVE_DT))
WAVE_FRAMES = 40        # animation frames dedicated to showing the wave

# Explosion
EXPLOSION_STRENGTH = 15.0  # initial pressure spike (big grenade)
GRENADE_FRAME = 10         # when the grenade goes off

# Material HP for wave damage
HP_HULL = 200.0
HP_WOOD = 60.0
DAMAGE_SCALE = 25.0        # multiplier: gradient → HP damage

# Reflection coefficients
REFLECT_HULL = 0.95
REFLECT_WOOD = 0.50

# ---------------------------------------------------------------------------
# Atmosphere / smoke parameters (post-explosion phase)
# ---------------------------------------------------------------------------
D_ATM = 200.0
D_SMOKE = 2.0
ADVECTION_RATE = 25.0
ATM_DT = 0.001
ATM_STEPS_PER_FRAME = 80
SMOKE_EMISSION_RATE = 0.5  # smoke from burning debris at blast site

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
TOTAL_FRAMES = 200
# Frames 0-9: calm station
# Frame 10: GRENADE → wave sim runs, shown over frames 10-49
# Frame 50+: atmosphere/smoke phase

ATM_PHASE_START = GRENADE_FRAME + WAVE_FRAMES  # frame 50

# ---------------------------------------------------------------------------
# Maze generation (same as smoke_sim.py)
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
# Laplacian with partial reflection (for wave equation)
# ---------------------------------------------------------------------------
def compute_laplacian_wave(p, wall, reflect):
    """Laplacian with per-tile reflection coefficients at walls."""
    up    = np.roll(p,  1, axis=0)
    down  = np.roll(p, -1, axis=0)
    left  = np.roll(p,  1, axis=1)
    right = np.roll(p, -1, axis=1)

    wall_up    = np.roll(wall,  1, axis=0)
    wall_down  = np.roll(wall, -1, axis=0)
    wall_left  = np.roll(wall,  1, axis=1)
    wall_right = np.roll(wall, -1, axis=1)

    # At wall boundaries: reflected_value = reflect_coeff * p_center
    up    = np.where(wall_up,    reflect * p, up)
    down  = np.where(wall_down,  reflect * p, down)
    left  = np.where(wall_left,  reflect * p, left)
    right = np.where(wall_right, reflect * p, right)

    return up + down + left + right - 4.0 * p


def compute_laplacian_with_walls(p, wall):
    """Standard Laplacian with Neumann BC (for atmosphere/smoke)."""
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
# Run the full wave simulation and store snapshots
# ---------------------------------------------------------------------------
def run_explosion(grenade_pos, walls, is_hull, wall_hp):
    """
    Run wave equation to completion. Returns:
    - snapshots: list of (pressure_field, walls_state) at intervals
    - wall_hp: updated HP after blast damage
    - destroyed: boolean mask of walls destroyed by the blast
    """
    gy, gx = grenade_pos

    # Reflection coefficient grid
    reflect = np.ones((GRID, GRID))
    reflect[is_hull] = REFLECT_HULL
    reflect[walls & ~is_hull] = REFLECT_WOOD

    # Wave fields: p_now, p_prev (both zero initially)
    p_now = np.zeros((GRID, GRID), dtype=np.float64)
    p_prev = np.zeros((GRID, GRID), dtype=np.float64)

    # Initial pressure spike at grenade position (3x3 area)
    for dy in range(-1, 2):
        for ddx in range(-1, 2):
            ny_, nx_ = gy + dy, gx + ddx
            if 0 <= ny_ < GRID and 0 <= nx_ < GRID and not walls[ny_, nx_]:
                dist = abs(dy) + abs(ddx)
                p_now[ny_, nx_] = EXPLOSION_STRENGTH * (1.0 if dist == 0 else 0.5)

    # Track peak gradient for damage
    peak_gradient = np.zeros((GRID, GRID), dtype=np.float64)

    # Store snapshots for animation
    steps_per_snapshot = max(1, WAVE_MAX_STEPS // WAVE_FRAMES)
    snapshots = []

    print(f"  Running wave equation: {WAVE_MAX_STEPS} steps, r={WAVE_R:.4f}, "
          f"snapshot every {steps_per_snapshot} steps")

    walls_live = walls.copy()
    wall_hp_live = wall_hp.copy()
    destroyed_total = np.zeros((GRID, GRID), dtype=bool)

    for step in range(WAVE_MAX_STEPS):
        lap = compute_laplacian_wave(p_now, walls_live, reflect)

        # Wave equation: p_next = 2*p_now - p_prev + r*laplacian
        p_next = 2.0 * p_now - p_prev + WAVE_R * lap

        # Pressure is zero inside walls (walls don't carry the wave)
        p_next[walls_live] = 0.0

        # Vacuum boundary: pressure = 0 at edges (open to space)
        p_next[0, :] = 0.0
        p_next[-1, :] = 0.0
        p_next[:, 0] = 0.0
        p_next[:, -1] = 0.0

        # Track peak gradient (for damage)
        grad_x = (np.roll(p_now, -1, axis=1) - np.roll(p_now, 1, axis=1)) / (2 * WAVE_DX)
        grad_y = (np.roll(p_now, -1, axis=0) - np.roll(p_now, 1, axis=0)) / (2 * WAVE_DX)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        peak_gradient = np.maximum(peak_gradient, grad_mag)

        # Mid-simulation wall destruction
        damage = DAMAGE_SCALE * peak_gradient
        newly_destroyed = (damage > wall_hp_live) & walls_live
        # Hull is much tougher — only destroyed right near the blast
        newly_destroyed &= ~is_hull | (damage > HP_HULL)

        if np.any(newly_destroyed):
            walls_live[newly_destroyed] = False
            destroyed_total |= newly_destroyed
            wall_hp_live[newly_destroyed] = 0.0
            # Update reflection grid
            reflect[newly_destroyed] = 0.0

        # Shift time
        p_prev = p_now
        p_now = p_next

        # Store snapshot
        if step % steps_per_snapshot == 0:
            snapshots.append((
                p_now.copy(),
                walls_live.copy(),
                peak_gradient.copy()
            ))

    print(f"  Wave done. Walls destroyed: {int(destroyed_total.sum())}")
    return snapshots, wall_hp_live, destroyed_total


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("Generating maze...")
walls = generate_maze(MAZE_NX, MAZE_NY, seed=42)

# Hull boundary
walls[0, :] = True
walls[-1, :] = True
walls[:, 0] = True
walls[:, -1] = True

is_hull = np.zeros((GRID, GRID), dtype=bool)
is_hull[0, :] = True
is_hull[-1, :] = True
is_hull[:, 0] = True
is_hull[:, -1] = True

# Wall HP
wall_hp = np.zeros((GRID, GRID), dtype=np.float64)
wall_hp[is_hull] = HP_HULL
wall_hp[walls & ~is_hull] = HP_WOOD

is_vacuum = np.zeros((GRID, GRID), dtype=bool)

atmosphere = np.where(walls, 0.0, 1.0)
smoke = np.zeros((GRID, GRID), dtype=np.float64)

# Place grenade: RIGHT next to the left hull wall
# Find an air tile 1 tile from the hull
grenade_pos = None
for row in range(2, GRID - 2):
    if not walls[row, 1]:
        grenade_pos = (row, 1)
        break
# Fallback: slightly further in
if grenade_pos is None:
    for offset in range(2, 10):
        for row in range(2, GRID - 2):
            if not walls[row, offset]:
                grenade_pos = (row, offset)
                break
        if grenade_pos:
            break

print(f"Grenade position: {grenade_pos}")
print(f"Wave CFL check: r={WAVE_R:.4f} -> {'STABLE' if WAVE_R <= 0.5 else 'UNSTABLE!'}")

# ---------------------------------------------------------------------------
# Run explosion (pre-compute all wave snapshots)
# ---------------------------------------------------------------------------
print(f"\nDetonation! Running wave simulation...")
wave_snapshots, wall_hp, destroyed = run_explosion(
    grenade_pos, walls.copy(), is_hull.copy(), wall_hp.copy()
)

# Apply destruction to the main state
walls_post = walls.copy()
walls_post[destroyed] = False

# Any destroyed hull tiles become vacuum
vacuum_from_blast = destroyed & is_hull
is_vacuum_post = is_vacuum.copy()
is_vacuum_post[vacuum_from_blast] = True

# Also open vacuum outside destroyed hull (the edge of the map IS space)
# Any tile at row 0, row -1, col 0, col -1 that lost its wall is vacuum
for y in range(GRID):
    for x in range(GRID):
        if destroyed[y, x] and (y == 0 or y == GRID-1 or x == 0 or x == GRID-1):
            is_vacuum_post[y, x] = True

# Count breach tiles for display
breach_tiles_blast = list(zip(*np.where(vacuum_from_blast)))
print(f"Hull breach tiles from explosion: {len(breach_tiles_blast)}")
print(f"Total walls destroyed: {int(destroyed.sum())}")

# Update atmosphere for destroyed walls
atmosphere_post = np.where(walls_post, 0.0, atmosphere)
atmosphere_post[is_vacuum_post] = 0.0
# Newly opened tiles get neighbor pressure
for y, x in zip(*np.where(destroyed & ~is_hull)):
    neighbors = []
    for dy, ddx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny_, nx_ = y+dy, x+ddx
        if 0 <= ny_ < GRID and 0 <= nx_ < GRID and not walls_post[ny_, nx_]:
            neighbors.append(atmosphere_post[ny_, nx_])
    atmosphere_post[y, x] = np.mean(neighbors) if neighbors else 0.5

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
print("\nSetting up animation...")
fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=90)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

img = ax.imshow(np.zeros((GRID, GRID, 3)), origin="upper", interpolation="nearest")

# Grenade marker
ax.plot(grenade_pos[1], grenade_pos[0], "*", color="yellow", markersize=12,
        markeredgecolor="white", markeredgewidth=0.5, label="Grenade", zorder=5)

ax.set_xlim(-0.5, GRID - 0.5)
ax.set_ylim(GRID - 0.5, -0.5)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper right", fontsize=6, facecolor="black", edgecolor="gray",
          labelcolor="white")
title = ax.set_title("", fontsize=9, color="white", pad=8)
fig.tight_layout()

# State for atmosphere phase
atm_state = {
    'walls': walls_post.copy(),
    'atmosphere': atmosphere_post.copy(),
    'smoke': smoke.copy(),
    'is_vacuum': is_vacuum_post.copy(),
}


def update(frame):
    rgb = np.zeros((GRID, GRID, 3))

    if frame < GRENADE_FRAME:
        # --- Pre-explosion: calm station ---
        rgb[walls, :] = 0.2
        # Wood walls brown
        wood = walls & ~is_hull
        rgb[wood, 0] = 0.30
        rgb[wood, 1] = 0.20
        rgb[wood, 2] = 0.10
        # Hull gray
        rgb[is_hull, :] = 0.25

        # Atmosphere blue
        open_cells = ~walls
        atm_v = atmosphere[open_cells]
        rgb[open_cells, 0] += atm_v * 0.03
        rgb[open_cells, 1] += atm_v * 0.06
        rgb[open_cells, 2] += atm_v * 0.35

        title.set_text(f"Frame {frame}/{TOTAL_FRAMES}  |  CALM  |  "
                       f"Pressure: 1.00 atm")

    elif frame < ATM_PHASE_START:
        # --- Wave phase: show shockwave expanding ---
        wave_idx = frame - GRENADE_FRAME
        snap_idx = min(wave_idx, len(wave_snapshots) - 1)
        pressure, walls_snap, peak_grad = wave_snapshots[snap_idx]

        # Walls (some may be destroyed by this snapshot)
        rgb[walls_snap, :] = 0.2
        wood_snap = walls_snap & ~is_hull
        rgb[wood_snap, 0] = 0.30
        rgb[wood_snap, 1] = 0.20
        rgb[wood_snap, 2] = 0.10
        rgb[is_hull & walls_snap, :] = 0.25

        # Destroyed walls: dark red glow (rubble)
        destroyed_so_far = walls & ~walls_snap
        rgb[destroyed_so_far, 0] = 0.5
        rgb[destroyed_so_far, 1] = 0.1
        rgb[destroyed_so_far, 2] = 0.0

        # Atmosphere blue (still full, wave is milliseconds)
        open_snap = ~walls_snap & ~is_vacuum
        rgb[open_snap, 0] += 0.03
        rgb[open_snap, 1] += 0.06
        rgb[open_snap, 2] += 0.35

        # Shockwave: pressure as white/yellow overlay
        # Positive pressure = bright white, negative = blue
        abs_p = np.abs(pressure)
        max_p = max(abs_p.max(), 0.01)
        norm_p = abs_p / max_p

        pos_mask = (pressure > 0.05) & ~walls_snap
        neg_mask = (pressure < -0.05) & ~walls_snap

        # Positive pressure: white-yellow flash
        rgb[pos_mask, 0] = np.clip(rgb[pos_mask, 0] + norm_p[pos_mask] * 1.0, 0, 1)
        rgb[pos_mask, 1] = np.clip(rgb[pos_mask, 1] + norm_p[pos_mask] * 0.9, 0, 1)
        rgb[pos_mask, 2] = np.clip(rgb[pos_mask, 2] + norm_p[pos_mask] * 0.5, 0, 1)

        # Negative pressure (rarefaction): blue tint
        rgb[neg_mask, 0] = np.clip(rgb[neg_mask, 0] - norm_p[neg_mask] * 0.1, 0, 1)
        rgb[neg_mask, 1] = np.clip(rgb[neg_mask, 1] + norm_p[neg_mask] * 0.1, 0, 1)
        rgb[neg_mask, 2] = np.clip(rgb[neg_mask, 2] + norm_p[neg_mask] * 0.7, 0, 1)

        n_destroyed = int(destroyed_so_far.sum())
        title.set_text(f"Frame {frame}/{TOTAL_FRAMES}  |  SHOCKWAVE  |  "
                       f"Peak pressure: {abs_p.max():.1f}  |  "
                       f"Walls destroyed: {n_destroyed}")

    else:
        # --- Atmosphere phase: decompression + smoke ---
        w = atm_state['walls']
        atm = atm_state['atmosphere']
        smk = atm_state['smoke']
        vac = atm_state['is_vacuum']

        for _ in range(ATM_STEPS_PER_FRAME):
            # Atmosphere diffusion
            lap = compute_laplacian_with_walls(atm, w)
            atm = atm + D_ATM * ATM_DT * lap
            atm[w] = 0.0
            atm[vac] = 0.0
            atm = np.clip(atm, 0.0, 1.0)

            # Smoke from blast debris (at destroyed wall locations)
            debris_smoke = destroyed & ~is_hull & ~w
            smk[debris_smoke] = np.clip(
                smk[debris_smoke] + SMOKE_EMISSION_RATE * ATM_DT, 0, 1
            )

            # Smoke diffusion + advection
            lap_s = compute_laplacian_with_walls(smk, w)
            smk = smk + D_SMOKE * ATM_DT * lap_s

            grad_y = (np.roll(atm, -1, axis=0) - np.roll(atm, 1, axis=0)) / 2.0
            grad_x = (np.roll(atm, -1, axis=1) - np.roll(atm, 1, axis=1)) / 2.0
            ds_dy = (np.roll(smk, -1, axis=0) - np.roll(smk, 1, axis=0)) / 2.0
            ds_dx = (np.roll(smk, -1, axis=1) - np.roll(smk, 1, axis=1)) / 2.0
            smk += ADVECTION_RATE * ATM_DT * (grad_x * ds_dx + grad_y * ds_dy)

            smk[w] = 0.0
            smk[vac] = 0.0
            smk = np.clip(smk, 0.0, 1.0)

        atm_state['atmosphere'] = atm
        atm_state['smoke'] = smk

        # --- Render ---
        # Walls
        hull_live = is_hull & w
        wood_live = w & ~is_hull
        rgb[hull_live, :] = 0.25
        hp_frac = np.zeros((GRID, GRID))
        hp_frac[wood_live] = wall_hp[wood_live] / HP_WOOD
        rgb[wood_live, 0] = 0.18 + 0.12 * hp_frac[wood_live]
        rgb[wood_live, 1] = 0.12 + 0.08 * hp_frac[wood_live]
        rgb[wood_live, 2] = 0.06 + 0.04 * hp_frac[wood_live]

        # Destroyed walls: dark rubble
        rubble = destroyed & ~w & ~vac
        rgb[rubble, 0] = 0.15
        rgb[rubble, 1] = 0.08
        rgb[rubble, 2] = 0.02

        # Vacuum: black with subtle cyan edge
        rgb[vac, :] = 0.0
        # Cyan border around vacuum
        for dy, ddx in [(-1,0),(1,0),(0,-1),(0,1)]:
            vac_neighbor = np.roll(vac, (dy, ddx), axis=(0, 1))
            edge = vac_neighbor & ~vac & ~w
            rgb[edge, 1] = np.clip(rgb[edge, 1] + 0.15, 0, 1)
            rgb[edge, 2] = np.clip(rgb[edge, 2] + 0.25, 0, 1)

        # Atmosphere blue
        open_cells = ~w & ~vac
        atm_v = atm[open_cells]
        rgb[open_cells, 0] += atm_v * 0.03
        rgb[open_cells, 1] += atm_v * 0.06
        rgb[open_cells, 2] += atm_v * 0.35

        # Smoke
        smoke_vis = smk * (~w).astype(float)
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + smoke_vis * 0.7, 0, 1)
        rgb[:, :, 1] = np.clip(rgb[:, :, 1] + smoke_vis * 0.2, 0, 1)
        rgb[:, :, 2] = np.clip(rgb[:, :, 2] + smoke_vis * 0.05, 0, 1)

        interior = ~w & ~vac
        avg_atm = atm[interior].mean() if np.any(interior) else 0
        title.set_text(f"Frame {frame}/{TOTAL_FRAMES}  |  DECOMPRESSING  |  "
                       f"Pressure: {avg_atm:.2f} atm  |  "
                       f"Breach: {len(breach_tiles_blast)} tiles")

        if frame % 40 == 0:
            print(f"  Frame {frame}/{TOTAL_FRAMES}, atm={avg_atm:.3f}")

    img.set_data(np.clip(rgb, 0, 1))
    return [img, title]


# ---------------------------------------------------------------------------
# Run and save
# ---------------------------------------------------------------------------
print(f"\nGrid: {GRID}x{GRID}, {TOTAL_FRAMES} frames")
print(f"Wave phase: frames {GRENADE_FRAME}-{ATM_PHASE_START-1} "
      f"({len(wave_snapshots)} snapshots)")
print(f"Atm phase: frames {ATM_PHASE_START}-{TOTAL_FRAMES-1}")

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True, interval=50)

output_path = "C:/Users/steen/projects/breach/prototypes/explosion_sim.gif"
print(f"Saving to {output_path} ...")
anim.save(output_path, writer=PillowWriter(fps=20))
print("Done!")
plt.close()
