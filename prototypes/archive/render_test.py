"""Render a grenade explosion sequence and save as frames + GIF."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import math
from config import CFG

# Minimal physics reproduction from game.py
FINE_W = CFG.display.fine_w
FINE_H = CFG.display.fine_h
COARSE = CFG.display.coarse
MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR = 0, 1, 2, 3

def build_map():
    m = np.zeros((FINE_H, FINE_W), dtype=np.int8)
    m[:] = MAT_HULL
    hull_t = COARSE
    m[hull_t:FINE_H - hull_t, hull_t:FINE_W - hull_t] = MAT_AIR
    wall_x = 15 * COARSE
    for y in range(hull_t, FINE_H - hull_t):
        m[y, wall_x] = MAT_WOOD
    for door_cy in [10, 18]:
        door_y = door_cy * COARSE
        for dy in range(COARSE):
            m[door_y + dy, wall_x] = MAT_DOOR
    return m

material = build_map()
is_wall = np.isin(material, [MAT_HULL, MAT_WOOD])
is_vacuum = np.zeros_like(is_wall)
is_vacuum[0:COARSE, :] = True
is_vacuum[FINE_H - COARSE:, :] = True
is_vacuum[:, 0:COARSE] = True
is_vacuum[:, FINE_W - COARSE:] = True

atmosphere = np.where(is_wall | is_vacuum, 0.0, 1.0).astype(np.float32)
wave_p = np.zeros((FINE_H, FINE_W), dtype=np.float32)
wave_v = np.zeros((FINE_H, FINE_W), dtype=np.float32)
smoke = np.zeros((FINE_H, FINE_W), dtype=np.float32)

# Create some smoke first (as if from a previous explosion)
smoke_cx, smoke_cy = 30, 37  # center of left room
for dy in range(-8, 9):
    for dx in range(-8, 9):
        ny, nx = smoke_cy + dy, smoke_cx + dx
        if 0 <= ny < FINE_H and 0 <= nx < FINE_W and not is_wall[ny, nx]:
            dist = math.sqrt(dy**2 + dx**2)
            if dist < 8:
                smoke[ny, nx] = 0.8 * (1 - dist/8)

# Detonate a grenade nearby to push the smoke
det_fx, det_fy = 20, 37
pressure = 10.0
radius = 6
for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
        ny, nx = det_fy + dy, det_fx + dx
        if 0 <= ny < FINE_H and 0 <= nx < FINE_W:
            dist = math.sqrt(dy**2 + dx**2)
            if dist <= radius:
                falloff = 1.0 - (dist / radius)
                if not is_wall[ny, nx] and not is_vacuum[ny, nx]:
                    wave_p[ny, nx] += pressure * falloff
                    atmosphere[ny, nx] += pressure * falloff * 0.3
                if dist <= radius * 0.4:
                    smoke[ny, nx] = 0.0

# Physics parameters (same as game.py)
c_grid = 300.0
c_squared = c_grid * c_grid
dt = 0.70 / c_grid
damping = min(400.0, 0.8 / dt)
transfer_rate = 30.0
advection_rate = CFG.physics.advection_rate
wave_advection_rate = 25000.0

print(f"c_grid={c_grid}, dt={dt:.6f}, damping={damping:.1f}")
print(f"CFL check: c*dt = {c_grid * dt:.4f} (must be < 0.707)")
print(f"Damping check: damping*dt = {damping * dt:.4f} (must be < 2)")
print(f"Initial wave_p max: {wave_p.max():.2f}")
print(f"Initial smoke max: {smoke.max():.2f}")
print(f"Initial atmosphere max: {atmosphere.max():.2f}")

# Run simulation and capture frames
try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False
    print("No PIL - will use matplotlib instead")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

frames = []
n_frames = 80
substeps_per_frame = 8

for frame in range(n_frames):
    # Physics substeps
    for _ in range(substeps_per_frame):
        # Wave equation
        up = np.roll(wave_p, 1, axis=0)
        down = np.roll(wave_p, -1, axis=0)
        left = np.roll(wave_p, 1, axis=1)
        right = np.roll(wave_p, -1, axis=1)
        wall_up = np.roll(is_wall, 1, axis=0)
        wall_down = np.roll(is_wall, -1, axis=0)
        wall_left = np.roll(is_wall, 1, axis=1)
        wall_right = np.roll(is_wall, -1, axis=1)
        up = np.where(wall_up, wave_p, up)
        down = np.where(wall_down, wave_p, down)
        left = np.where(wall_left, wave_p, left)
        right = np.where(wall_right, wave_p, right)
        lap = up + down + left + right - 4.0 * wave_p
        wave_v += (c_squared * lap - damping * wave_v) * dt
        wave_p += wave_v * dt
        wave_p[is_wall] = 0.0
        wave_p[is_vacuum] = 0.0
        atmosphere += wave_p * transfer_rate * dt
        atmosphere[is_wall] = 0.0
        atmosphere[is_vacuum] = 0.0
        np.clip(atmosphere, 0.0, 20.0, out=atmosphere)

        # Smoke advection
        ds_dy = (np.roll(smoke, -1, axis=0) - np.roll(smoke, 1, axis=0)) / 2.0
        ds_dx = (np.roll(smoke, -1, axis=1) - np.roll(smoke, 1, axis=1)) / 2.0
        a_grad_y = (np.roll(atmosphere, -1, axis=0) - np.roll(atmosphere, 1, axis=0)) / 2.0
        a_grad_x = (np.roll(atmosphere, -1, axis=1) - np.roll(atmosphere, 1, axis=1)) / 2.0
        smoke += advection_rate * dt * (a_grad_x * ds_dx + a_grad_y * ds_dy)
        w_grad_y = (np.roll(wave_p, -1, axis=0) - np.roll(wave_p, 1, axis=0)) / 2.0
        w_grad_x = (np.roll(wave_p, -1, axis=1) - np.roll(wave_p, 1, axis=1)) / 2.0
        smoke += wave_advection_rate * dt * (w_grad_x * ds_dx + w_grad_y * ds_dy)
        # Smoke diffusion
        s_lap = (np.roll(smoke, 1, axis=0) + np.roll(smoke, -1, axis=0) +
                 np.roll(smoke, 1, axis=1) + np.roll(smoke, -1, axis=1) - 4.0 * smoke)
        smoke += 0.4 * dt * s_lap
        smoke[is_wall] = 0.0
        smoke[is_vacuum] = 0.0
        np.clip(smoke, 0.0, 1.0, out=smoke)

    # Render frame
    total = atmosphere + wave_p
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pressure
    ax = axes[0]
    p_display = np.where(is_wall, np.nan, total)
    im = ax.imshow(p_display, cmap='hot', vmin=0.5, vmax=3.0, origin='upper')
    ax.set_title(f'Pressure (frame {frame})\nmax={total[~is_wall].max():.2f}')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Wave
    ax = axes[1]
    w_display = np.where(is_wall, np.nan, wave_p)
    im = ax.imshow(w_display, cmap='RdBu_r', vmin=-2, vmax=2, origin='upper')
    ax.set_title(f'Wave field\nmax={abs(wave_p).max():.2f}')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Smoke
    ax = axes[2]
    s_display = np.where(is_wall, np.nan, smoke)
    im = ax.imshow(s_display, cmap='gray_r', vmin=0, vmax=1, origin='upper')
    ax.set_title(f'Smoke\nmax={smoke.max():.3f}')
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    fname = f'c:/Users/steen/projects/breach/prototypes/frame_{frame:03d}.png'
    plt.savefig(fname, dpi=80)
    plt.close()

    if frame % 10 == 0:
        print(f"Frame {frame}: wave_max={abs(wave_p).max():.3f}, "
              f"atm_max={atmosphere.max():.3f}, smoke_max={smoke.max():.4f}")

# Make GIF
if has_pil:
    imgs = []
    for i in range(n_frames):
        imgs.append(Image.open(f'c:/Users/steen/projects/breach/prototypes/frame_{i:03d}.png'))
    imgs[0].save('c:/Users/steen/projects/breach/prototypes/explosion_test.gif',
                 save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print("Saved explosion_test.gif")
else:
    print("Install Pillow for GIF: pip install Pillow")

print("Done!")
