"""
Tilted ship flooding: water flows through corridors on a sloped floor.

Think Titanic — the ship is tilting, water enters from one end and flows
downhill through rooms and corridors.

Features:
  - Sloped terrain (ship tilt)
  - Multiple connected rooms with doorways
  - Water source at the high end (breach in hull)
  - Watch water flow downhill through the ship
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81
DX = 0.33  # 1/3 meter per tile (3 tiles per meter, same as game fine grid)
DAMPING = 0.5


class PipeModel:
    def __init__(self, h, terrain, walls):
        self.h = h.copy()
        self.terrain = terrain
        self.walls = walls
        self.vx = np.zeros_like(h)
        self.vy = np.zeros_like(h)

    def step(self, dt):
        h, vx, vy = self.h, self.vx, self.vy
        surface = self.terrain + h

        gx = np.zeros_like(surface)
        gy = np.zeros_like(surface)
        gx[:, 1:-1] = (surface[:, 2:] - surface[:, :-2]) / (2 * DX)
        gy[1:-1, :] = (surface[2:, :] - surface[:-2, :]) / (2 * DX)

        vx += dt * (-G * gx - DAMPING * vx)
        vy += dt * (-G * gy - DAMPING * vy)
        vx[self.walls] = 0.0
        vy[self.walls] = 0.0

        def flux_x(v, hf):
            v_face = 0.5 * (v[:, :-1] + v[:, 1:])
            h_face = np.where(v_face > 0, hf[:, :-1], hf[:, 1:])
            f = v_face * h_face
            f[self.walls[:, :-1] | self.walls[:, 1:]] = 0.0
            return f

        def flux_y(v, hf):
            v_face = 0.5 * (v[:-1, :] + v[1:, :])
            h_face = np.where(v_face > 0, hf[:-1, :], hf[1:, :])
            f = v_face * h_face
            f[self.walls[:-1, :] | self.walls[1:, :]] = 0.0
            return f

        Fx = flux_x(vx, h)
        Fy = flux_y(vy, h)

        div = np.zeros_like(h)
        div[:, 1:-1] += (Fx[:, 1:] - Fx[:, :-1]) / DX
        div[1:-1, :] += (Fy[1:, :] - Fy[:-1, :]) / DX

        h -= dt * div
        h[h < 0] = 0.0
        h[self.walls] = 0.0
        self.h, self.vx, self.vy = h, vx, vy


def run_tilted_ship():
    # Ship interior: 60 wide x 24 tall (20m x 8m at 3 tiles/m)
    W, H = 60, 24
    terrain = np.zeros((H, W))
    walls = np.zeros((H, W), dtype=bool)

    # Border (hull)
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True

    # Ship tilt: slopes from left (high) to right (low)
    # 5 degree tilt over 20m = about 1.7m height difference
    tilt_deg = 5.0
    for j in range(W):
        x_meters = j * DX
        terrain[:, j] += np.tan(np.radians(tilt_deg)) * (W * DX - x_meters)

    # Rooms: internal walls with doorways
    # Room dividers (vertical walls running top to bottom with gaps)
    dividers = [
        (15, slice(1, 8), slice(10, H - 1)),   # wall at col 15, door at rows 8-10
        (30, slice(1, 10), slice(14, H - 1)),   # wall at col 30, door at rows 10-14
        (45, slice(1, 6), slice(8, 16), slice(20, H - 1)),  # col 45, two doors
    ]

    for col, *segments in dividers:
        for seg in segments:
            walls[seg, col] = True

    # Some furniture (raised terrain, not walls — water flows over if deep enough)
    # Tables in room 1
    terrain[4:7, 5:9] = 0.15
    # Crates in room 2
    terrain[15:18, 20:23] = 0.2
    walls[15:18, 20:23] = True
    # Equipment in room 3
    terrain[5:8, 35:38] = 0.1
    # Barrels in room 4
    for r, c in [(10, 50), (11, 50), (10, 51), (11, 51)]:
        terrain[r, c] = 0.25
        walls[r, c] = True

    # Set wall terrain high
    terrain[walls] = np.maximum(terrain[walls], 2.0)

    # Water: starts empty, source at left side (hull breach)
    h = np.zeros((H, W))
    source_tiles = [(r, 2) for r in range(3, 8)]  # breach in hull, left side

    sim = PipeModel(h, terrain, walls)

    # --- Visualization ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: terrain heightmap
    terrain_vis = terrain.copy()
    terrain_vis[walls] = terrain_vis.max()
    im_terrain = axes[0].imshow(terrain_vis, origin='upper', cmap='terrain',
                                 vmin=terrain[~walls].min() - 0.1,
                                 vmax=terrain[~walls].max() + 0.5,
                                 interpolation='bilinear', aspect='equal')
    axes[0].set_title(f'Terrain (ship tilted {tilt_deg} deg — left is high, right is low)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im_terrain, ax=axes[0], label='Height (m)', shrink=0.6)

    # Bottom: water depth
    im_water = axes[1].imshow(sim.h, origin='upper', cmap='Blues', vmin=0, vmax=0.3,
                               interpolation='bilinear', aspect='equal')
    # Wall overlay
    wall_vis = np.ma.masked_where(~walls, np.ones_like(walls, dtype=float))
    axes[1].imshow(wall_vis, origin='upper', cmap='Greys', vmin=0, vmax=1,
                    interpolation='nearest', alpha=0.7)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im_water, ax=axes[1], label='Water depth (m)', shrink=0.6)

    title = fig.suptitle('Tilted Ship Flooding', fontsize=13)
    plt.tight_layout()

    sim_time = 0.0
    dt = 0.01  # smaller dt for finer grid
    steps_per_frame = 8

    def update(frame):
        nonlocal sim_time

        for _ in range(steps_per_frame):
            # Water source: hull breach constantly lets water in
            for r, c in source_tiles:
                sim.h[r, c] = max(sim.h[r, c], 0.3)

            sim.step(dt)
            sim_time += dt

        im_water.set_data(sim.h)
        vol = sim.h.sum() * DX * DX
        wet_tiles = (sim.h > 0.001).sum()
        title.set_text(
            f'Tilted Ship Flooding — t={sim_time:.1f}s  |  '
            f'water vol={vol:.2f} m3  |  wet tiles={wet_tiles}'
        )
        return [im_water, title]

    anim = FuncAnimation(fig, update, frames=1500, interval=33, blit=False)
    plt.show()


if __name__ == '__main__':
    print("=== Tilted Ship Flooding ===")
    print("Water enters from hull breach on the left (high end)")
    print("Watch it flow downhill through rooms and corridors")
    run_tilted_ship()
