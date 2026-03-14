"""
Wind test prototype for Breach.

Demonstrates:
- Simple room (no maze) with atmosphere diffusion
- Smoke patches advected by pressure gradient ("wind")
- Mid-simulation explosion: atmosphere spike creates blast wind
- Hull breach on one side pulling air/smoke out

Setup:
- 40x25 room, walls around edges
- Small hull breach (3 tiles) on the left wall
- Three smoke patches placed at known positions
- Frame 60: explosion spike at center-right (atmosphere -> 10.0)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
WIDTH = 50
HEIGHT = 30

# Physics
D_ATM = 200.0           # atmosphere diffusion
D_SMOKE = 0.4           # smoke diffusion (slow natural spread)
ADVECTION_RATE = 25.0   # smoke advection by pressure gradient
dx = 1.0
dt = 0.001              # CFL: dt < dx^2/(4*D) = 0.00125 for D=200

# Simulation
TOTAL_FRAMES = 200
STEPS_PER_FRAME = 80
EXPLOSION_FRAME = 60
EXPLOSION_PRESSURE = 10.0  # atmosphere spike at detonation point

# ---------------------------------------------------------------------------
# Shared Laplacian (Neumann BC at walls)
# ---------------------------------------------------------------------------
def compute_laplacian_with_walls(p, wall):
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
def atmosphere_step(atm, wall, is_vacuum, dt_step):
    lap = compute_laplacian_with_walls(atm, wall)
    atm_next = atm + D_ATM * dt_step * lap
    atm_next[wall] = 0.0
    atm_next[is_vacuum] = 0.0
    return np.clip(atm_next, 0.0, 20.0)  # allow overpressure from explosions


def smoke_step(smoke, atm, wall, is_vacuum, dt_step):
    lap = compute_laplacian_with_walls(smoke, wall)
    smoke_next = smoke + D_SMOKE * dt_step * lap

    # Advection: smoke carried by pressure gradient (wind)
    grad_y = (np.roll(atm, -1, axis=0) - np.roll(atm, 1, axis=0)) / 2.0
    grad_x = (np.roll(atm, -1, axis=1) - np.roll(atm, 1, axis=1)) / 2.0
    dsmoke_dy = (np.roll(smoke, -1, axis=0) - np.roll(smoke, 1, axis=0)) / 2.0
    dsmoke_dx = (np.roll(smoke, -1, axis=1) - np.roll(smoke, 1, axis=1)) / 2.0

    smoke_next += ADVECTION_RATE * dt_step * (grad_x * dsmoke_dx + grad_y * dsmoke_dy)

    smoke_next[wall] = 0.0
    smoke_next[is_vacuum] = 0.0
    return np.clip(smoke_next, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("Setting up wind test...")

# Walls: border around the room
walls = np.zeros((HEIGHT, WIDTH), dtype=bool)
walls[0, :] = True    # top hull
walls[-1, :] = True   # bottom hull
walls[:, 0] = True    # left hull
walls[:, -1] = True   # right hull

is_vacuum = np.zeros((HEIGHT, WIDTH), dtype=bool)

# Hull breach: 3 tiles on left wall, centered vertically
breach_center_y = HEIGHT // 2
for dy in range(-1, 2):
    by = breach_center_y + dy
    walls[by, 0] = False
    is_vacuum[by, 0] = True

# Atmosphere: 1.0 inside, 0.0 at vacuum
atmosphere = np.where(walls, 0.0, 1.0)
atmosphere[is_vacuum] = 0.0

# Smoke patches at three locations
smoke = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

# Patch 1: near the breach (left side)
smoke[breach_center_y-2:breach_center_y+3, 5:9] = 0.8

# Patch 2: center of room
smoke[HEIGHT//2-2:HEIGHT//2+3, WIDTH//2-2:WIDTH//2+2] = 0.8

# Patch 3: right side
smoke[HEIGHT//2-1:HEIGHT//2+2, WIDTH-10:WIDTH-7] = 0.8

# Explosion point (center-right area)
explosion_y = HEIGHT // 2
explosion_x = WIDTH * 3 // 4

print(f"Grid: {WIDTH}x{HEIGHT}")
print(f"Breach: left wall at y={breach_center_y}")
print(f"Explosion at frame {EXPLOSION_FRAME}: tile ({explosion_x}, {explosion_y}), "
      f"pressure spike to {EXPLOSION_PRESSURE}")

explosion_happened = False

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=90)
fig.patch.set_facecolor("black")

titles_text = ["Atmosphere (pressure)", "Smoke density", "Wind (pressure gradient)"]
imgs = []

for i, ax in enumerate(axes):
    ax.set_facecolor("black")
    if i < 2:
        im = ax.imshow(np.zeros((HEIGHT, WIDTH, 3)), origin="upper",
                       interpolation="nearest")
    else:
        # Wind field: quiver plot
        im = ax.imshow(np.zeros((HEIGHT, WIDTH, 3)), origin="upper",
                       interpolation="nearest", alpha=0.3)
    imgs.append(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles_text[i], color="white", fontsize=10)

# Quiver for wind arrows (subsample for readability)
skip = 2
Y, X = np.mgrid[0:HEIGHT:skip, 0:WIDTH:skip]
quiver = axes[2].quiver(X, Y, np.zeros_like(X, dtype=float),
                         np.zeros_like(Y, dtype=float),
                         color="cyan", scale=30, width=0.004, headwidth=4,
                         zorder=5)

# Mark explosion point
for ax in axes:
    ax.plot(explosion_x, explosion_y, 'x', color='red', markersize=8,
            markeredgewidth=2, zorder=10, alpha=0.5)

suptitle = fig.suptitle("", color="white", fontsize=11, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])


def update(frame):
    global atmosphere, smoke, explosion_happened

    # --- Explosion event ---
    if frame == EXPLOSION_FRAME and not explosion_happened:
        explosion_happened = True
        # Spike atmosphere at detonation point and small neighborhood
        atmosphere[explosion_y, explosion_x] = EXPLOSION_PRESSURE
        for dy in range(-1, 2):
            for ddx in range(-1, 2):
                if dy == 0 and ddx == 0:
                    continue
                ny, nx = explosion_y + dy, explosion_x + ddx
                if 0 < ny < HEIGHT-1 and 0 < nx < WIDTH-1 and not walls[ny, nx]:
                    atmosphere[ny, nx] = EXPLOSION_PRESSURE * 0.5
        print(f"  EXPLOSION at frame {frame}! Pressure spike to {EXPLOSION_PRESSURE}")

    # --- Sub-stepping ---
    for _ in range(STEPS_PER_FRAME):
        atmosphere = atmosphere_step(atmosphere, walls, is_vacuum, dt)
        smoke = smoke_step(smoke, atmosphere, walls, is_vacuum, dt)

    # --- Compute wind field for visualization ---
    grad_y = (np.roll(atmosphere, -1, axis=0) - np.roll(atmosphere, 1, axis=0)) / 2.0
    grad_x = (np.roll(atmosphere, -1, axis=1) - np.roll(atmosphere, 1, axis=1)) / 2.0
    # Wind = -grad(p)
    wind_x = -grad_x
    wind_y = -grad_y
    wind_x[walls] = 0
    wind_y[walls] = 0
    wind_mag = np.sqrt(wind_x**2 + wind_y**2)

    # --- Render atmosphere ---
    rgb_atm = np.zeros((HEIGHT, WIDTH, 3))
    # Walls
    rgb_atm[walls, :] = 0.25
    # Vacuum
    rgb_atm[is_vacuum] = [0.0, 0.3, 0.4]
    # Atmosphere: blue intensity
    interior = ~walls & ~is_vacuum
    atm_norm = np.clip(atmosphere / max(atmosphere.max(), 1.0), 0, 1)
    rgb_atm[interior, 0] = atm_norm[interior] * 0.15
    rgb_atm[interior, 1] = atm_norm[interior] * 0.25
    rgb_atm[interior, 2] = atm_norm[interior] * 0.8
    # Overpressure glow (yellow-white for atmosphere > 1.0)
    overpressure = atmosphere > 1.05
    if np.any(overpressure):
        op_norm = np.clip((atmosphere[overpressure] - 1.0) / (EXPLOSION_PRESSURE - 1.0), 0, 1)
        rgb_atm[overpressure, 0] = 0.3 + op_norm * 0.7
        rgb_atm[overpressure, 1] = 0.3 + op_norm * 0.5
        rgb_atm[overpressure, 2] = 0.3 + op_norm * 0.2
    imgs[0].set_data(np.clip(rgb_atm, 0, 1))

    # --- Render smoke ---
    rgb_smoke = np.zeros((HEIGHT, WIDTH, 3))
    rgb_smoke[walls, :] = 0.25
    rgb_smoke[is_vacuum] = [0.0, 0.3, 0.4]
    smoke_vis = smoke * (~walls & ~is_vacuum).astype(float)
    rgb_smoke[:, :, 0] = np.clip(rgb_smoke[:, :, 0] + smoke_vis * 0.9, 0, 1)
    rgb_smoke[:, :, 1] = np.clip(rgb_smoke[:, :, 1] + smoke_vis * 0.4, 0, 1)
    rgb_smoke[:, :, 2] = np.clip(rgb_smoke[:, :, 2] + smoke_vis * 0.1, 0, 1)
    imgs[1].set_data(np.clip(rgb_smoke, 0, 1))

    # --- Render wind field background ---
    rgb_wind = np.zeros((HEIGHT, WIDTH, 3))
    rgb_wind[walls, :] = 0.25
    rgb_wind[is_vacuum] = [0.0, 0.3, 0.4]
    # Wind magnitude as brightness
    wm_norm = np.clip(wind_mag / max(wind_mag.max(), 0.01), 0, 1)
    rgb_wind[interior, 0] = wm_norm[interior] * 0.1
    rgb_wind[interior, 1] = wm_norm[interior] * 0.5
    rgb_wind[interior, 2] = wm_norm[interior] * 0.3
    imgs[2].set_data(np.clip(rgb_wind, 0, 1))

    # Update quiver arrows
    wx_sub = wind_x[::skip, ::skip]
    wy_sub = wind_y[::skip, ::skip]
    quiver.set_UVC(wx_sub, wy_sub)

    # Status
    interior_mask = ~walls & ~is_vacuum
    avg_atm = atmosphere[interior_mask].mean() if np.any(interior_mask) else 0
    max_atm = atmosphere[interior_mask].max() if np.any(interior_mask) else 0
    max_wind = wind_mag[interior_mask].max() if np.any(interior_mask) else 0
    total_smoke = smoke[interior_mask].sum()

    status = "PRE-EXPLOSION" if not explosion_happened else "POST-EXPLOSION"
    suptitle.set_text(
        f"Frame {frame}/{TOTAL_FRAMES}  |  {status}  |  "
        f"Avg P: {avg_atm:.3f}  Max P: {max_atm:.3f}  |  "
        f"Max wind: {max_wind:.3f}  |  Smoke: {total_smoke:.1f}"
    )

    if frame % 20 == 0:
        print(f"  Frame {frame}/{TOTAL_FRAMES}, avg_atm={avg_atm:.3f}, "
              f"max_atm={max_atm:.3f}, max_wind={max_wind:.4f}, smoke={total_smoke:.1f}")

    return imgs + [quiver, suptitle]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Running: {TOTAL_FRAMES} frames, {STEPS_PER_FRAME} sub-steps/frame")
print(f"CFL check: dt={dt}, dx^2/(4*D)={dx**2/(4*D_ATM):.4f} -> "
      f"{'STABLE' if dt < dx**2/(4*D_ATM) else 'UNSTABLE!'}")

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True, interval=50)

output_path = "C:/Users/steen/projects/breach/prototypes/wind_test.gif"
print(f"Saving to {output_path} ...")
anim.save(output_path, writer=PillowWriter(fps=20))
print("Done!")
plt.close()
