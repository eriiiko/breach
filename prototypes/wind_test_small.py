"""
Wind test prototype (compact version for viewing).
Same physics as wind_test.py but fewer frames and lower resolution for smaller GIF.
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
D_ATM = 200.0
D_SMOKE = 0.4
ADVECTION_RATE = 25.0
dx = 1.0
dt = 0.001

# Simulation - fewer frames for smaller GIF
TOTAL_FRAMES = 120
STEPS_PER_FRAME = 80
EXPLOSION_FRAME = 40
EXPLOSION_PRESSURE = 10.0

# ---------------------------------------------------------------------------
# Shared Laplacian
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

def atmosphere_step(atm, wall, is_vacuum, dt_step):
    lap = compute_laplacian_with_walls(atm, wall)
    atm_next = atm + D_ATM * dt_step * lap
    atm_next[wall] = 0.0
    atm_next[is_vacuum] = 0.0
    return np.clip(atm_next, 0.0, 20.0)

def smoke_step(smoke, atm, wall, is_vacuum, dt_step):
    lap = compute_laplacian_with_walls(smoke, wall)
    smoke_next = smoke + D_SMOKE * dt_step * lap
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
walls = np.zeros((HEIGHT, WIDTH), dtype=bool)
walls[0, :] = True
walls[-1, :] = True
walls[:, 0] = True
walls[:, -1] = True

is_vacuum = np.zeros((HEIGHT, WIDTH), dtype=bool)
breach_center_y = HEIGHT // 2
for dy in range(-1, 2):
    walls[breach_center_y + dy, 0] = False
    is_vacuum[breach_center_y + dy, 0] = True

atmosphere = np.where(walls, 0.0, 1.0)
atmosphere[is_vacuum] = 0.0

smoke = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
smoke[breach_center_y-2:breach_center_y+3, 5:9] = 0.8
smoke[HEIGHT//2-2:HEIGHT//2+3, WIDTH//2-2:WIDTH//2+2] = 0.8
smoke[HEIGHT//2-1:HEIGHT//2+2, WIDTH-10:WIDTH-7] = 0.8

explosion_y = HEIGHT // 2
explosion_x = WIDTH * 3 // 4
explosion_happened = False

# ---------------------------------------------------------------------------
# Visualization - 2 panels only, lower DPI
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=70)
fig.patch.set_facecolor("black")

for ax in axes:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])

img_atm = axes[0].imshow(np.zeros((HEIGHT, WIDTH, 3)), origin="upper",
                          interpolation="nearest")
axes[0].set_title("Atmosphere", color="white", fontsize=10)

img_smoke = axes[1].imshow(np.zeros((HEIGHT, WIDTH, 3)), origin="upper",
                            interpolation="nearest")
axes[1].set_title("Smoke", color="white", fontsize=10)

# Mark explosion point
for ax in axes:
    ax.plot(explosion_x, explosion_y, 'x', color='red', markersize=8,
            markeredgewidth=2, zorder=10, alpha=0.5)
    # Mark breach
    ax.plot(0, breach_center_y, '>', color='cyan', markersize=8, zorder=10)

suptitle = fig.suptitle("", color="white", fontsize=10, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])


def update(frame):
    global atmosphere, smoke, explosion_happened

    if frame == EXPLOSION_FRAME and not explosion_happened:
        explosion_happened = True
        atmosphere[explosion_y, explosion_x] = EXPLOSION_PRESSURE
        for dy in range(-1, 2):
            for ddx in range(-1, 2):
                if dy == 0 and ddx == 0:
                    continue
                ny, nx = explosion_y + dy, explosion_x + ddx
                if 0 < ny < HEIGHT-1 and 0 < nx < WIDTH-1 and not walls[ny, nx]:
                    atmosphere[ny, nx] = EXPLOSION_PRESSURE * 0.5

    for _ in range(STEPS_PER_FRAME):
        atmosphere = atmosphere_step(atmosphere, walls, is_vacuum, dt)
        smoke = smoke_step(smoke, atmosphere, walls, is_vacuum, dt)

    # Atmosphere panel
    rgb_atm = np.zeros((HEIGHT, WIDTH, 3))
    rgb_atm[walls, :] = 0.25
    rgb_atm[is_vacuum] = [0.0, 0.3, 0.4]
    interior = ~walls & ~is_vacuum
    atm_norm = np.clip(atmosphere / max(atmosphere.max(), 1.0), 0, 1)
    rgb_atm[interior, 0] = atm_norm[interior] * 0.15
    rgb_atm[interior, 1] = atm_norm[interior] * 0.25
    rgb_atm[interior, 2] = atm_norm[interior] * 0.8
    overpressure = atmosphere > 1.05
    if np.any(overpressure):
        op_norm = np.clip((atmosphere[overpressure] - 1.0) / (EXPLOSION_PRESSURE - 1.0), 0, 1)
        rgb_atm[overpressure, 0] = 0.3 + op_norm * 0.7
        rgb_atm[overpressure, 1] = 0.3 + op_norm * 0.5
        rgb_atm[overpressure, 2] = 0.3 + op_norm * 0.2
    img_atm.set_data(np.clip(rgb_atm, 0, 1))

    # Smoke panel
    rgb_smoke = np.zeros((HEIGHT, WIDTH, 3))
    rgb_smoke[walls, :] = 0.25
    rgb_smoke[is_vacuum] = [0.0, 0.3, 0.4]
    smoke_vis = smoke * (~walls & ~is_vacuum).astype(float)
    rgb_smoke[:, :, 0] = np.clip(rgb_smoke[:, :, 0] + smoke_vis * 0.9, 0, 1)
    rgb_smoke[:, :, 1] = np.clip(rgb_smoke[:, :, 1] + smoke_vis * 0.4, 0, 1)
    rgb_smoke[:, :, 2] = np.clip(rgb_smoke[:, :, 2] + smoke_vis * 0.1, 0, 1)
    img_smoke.set_data(np.clip(rgb_smoke, 0, 1))

    interior_mask = ~walls & ~is_vacuum
    avg_atm = atmosphere[interior_mask].mean()
    max_atm = atmosphere[interior_mask].max()
    total_smoke = smoke[interior_mask].sum()

    status = "PRE-EXPLOSION" if not explosion_happened else "POST-EXPLOSION"
    suptitle.set_text(
        f"Frame {frame}/{TOTAL_FRAMES}  |  {status}  |  "
        f"Avg P: {avg_atm:.3f}  Max P: {max_atm:.3f}  |  Smoke: {total_smoke:.1f}")

    if frame % 20 == 0:
        print(f"  Frame {frame}, avg_atm={avg_atm:.3f}, max_atm={max_atm:.3f}, "
              f"smoke={total_smoke:.1f}")

    return [img_atm, img_smoke, suptitle]

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True, interval=50)
output_path = "C:/Users/steen/projects/breach/prototypes/wind_test_small.gif"
print(f"Saving to {output_path} ...")
anim.save(output_path, writer=PillowWriter(fps=20))
print("Done!")
plt.close()
