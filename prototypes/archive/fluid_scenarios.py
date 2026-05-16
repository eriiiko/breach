"""
Fluid scenarios: dam break (aquarium burst) and maze flooding.

Two windows shown sequentially:
  1. Aquarium burst: tall water column behind a wall, wall removed at t=2s
  2. Maze flood: water source at entrance, watch it find its way through
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81
DX = 1.0
DAMPING = 0.3


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


# =====================================================================
# Scenario 1: Aquarium Burst
# =====================================================================

def run_aquarium():
    N = 30
    terrain = np.zeros((N, N))
    walls = np.zeros((N, N), dtype=bool)

    # Border walls
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True
    terrain[walls] = 1.0

    # Aquarium: thick glass wall at column 8
    dam_row = slice(1, N - 1)
    dam_col = 8
    walls[dam_row, dam_col] = True
    terrain[dam_row, dam_col] = 1.0

    # Water: deep pool behind the dam (left side)
    h = np.zeros((N, N))
    h[1:-1, 1:dam_col] = 0.5  # 50cm of water - tall aquarium

    # Some furniture on the right side (obstacles for water to flow around)
    for r, c in [(10, 15), (10, 16), (11, 15), (11, 16),   # table
                 (20, 12), (20, 13), (21, 12), (21, 13),   # desk
                 (5, 20), (5, 21), (6, 20), (6, 21)]:      # crate
        walls[r, c] = True
        terrain[r, c] = 0.3

    sim = PipeModel(h, terrain, walls)
    dam_broken = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Terrain view
    terrain_vis = terrain.copy()
    terrain_vis[walls] = 0.8
    axes[0].imshow(terrain_vis, origin='upper', cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title('Terrain + walls')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im = axes[1].imshow(sim.h, origin='upper', cmap='Blues', vmin=0, vmax=0.5,
                         interpolation='bilinear')
    # Overlay walls
    wall_vis = np.ma.masked_where(~walls, np.ones_like(walls, dtype=float))
    axes[1].imshow(wall_vis, origin='upper', cmap='Greys', vmin=0, vmax=1,
                    interpolation='nearest', alpha=0.7)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    title = fig.suptitle('Aquarium Burst — waiting...', fontsize=13)
    plt.tight_layout()

    sim_time = 0.0
    dt = 0.02
    steps_per_frame = 5

    def update(frame):
        nonlocal sim_time, dam_broken

        for _ in range(steps_per_frame):
            # Break the dam at t=2s
            if sim_time >= 2.0 and not dam_broken:
                sim.walls[dam_row, dam_col] = False
                sim.terrain[dam_row, dam_col] = 0.0
                dam_broken = True

            sim.step(dt)
            sim_time += dt

        im.set_data(sim.h)
        vol = sim.h.sum() * DX * DX

        if not dam_broken:
            title.set_text(f'Aquarium Burst — t={sim_time:.1f}s (dam breaks at t=2.0s)  |  vol={vol:.3f}')
        else:
            title.set_text(f'Aquarium Burst — t={sim_time:.1f}s  DAM BROKEN!  |  vol={vol:.3f}')

        return [im, title]

    anim = FuncAnimation(fig, update, frames=600, interval=33, blit=False)
    plt.show()


# =====================================================================
# Scenario 2: Maze Flooding
# =====================================================================

def run_maze():
    N = 30
    terrain = np.zeros((N, N))
    walls = np.zeros((N, N), dtype=bool)

    # Border
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True
    terrain[walls] = 1.0

    # Maze walls: corridors that connect, water can flow through the whole maze
    maze_walls = [
        # Vertical walls with gaps (doors)
        (slice(2, 10), 6),     # left wall, gap at row 10-12
        (slice(13, 22), 6),    # continues below gap
        (slice(2, 8), 12),     # middle-left wall, gap at row 8-10
        (slice(11, 20), 12),   # continues below gap
        (slice(22, 28), 12),   # bottom section
        (slice(2, 14), 18),    # middle-right wall, gap at row 14-16
        (slice(17, 28), 18),   # continues below gap
        (slice(5, 15), 24),    # right wall, gap at row 15-17
        (slice(18, 26), 24),   # continues below gap
        # Horizontal walls with gaps
        (8, slice(1, 5)),       # top corridor divider
        (16, slice(7, 11)),     # middle horizontal
        (10, slice(13, 17)),    # upper-middle
        (22, slice(7, 17)),     # lower horizontal
        (14, slice(19, 23)),    # right area
        (24, slice(19, 28)),    # bottom-right
    ]
    for r, c in maze_walls:
        walls[r, c] = True
        terrain[r, c] = 1.0

    # Water: starts in top-left room
    h = np.zeros((N, N))
    h[1:5, 1:5] = 0.4

    # Continuous water source (simulates a broken pipe)
    source_r, source_c = 2, 2

    sim = PipeModel(h, terrain, walls)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Maze layout
    maze_vis = terrain.copy()
    maze_vis[walls] = 0.8
    axes[0].imshow(maze_vis, origin='upper', cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title('Maze layout')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im = axes[1].imshow(sim.h, origin='upper', cmap='Blues', vmin=0, vmax=0.4,
                         interpolation='bilinear')
    wall_vis = np.ma.masked_where(~walls, np.ones_like(walls, dtype=float))
    axes[1].imshow(wall_vis, origin='upper', cmap='Greys', vmin=0, vmax=1,
                    interpolation='nearest', alpha=0.7)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    title = fig.suptitle('Maze Flooding', fontsize=13)
    plt.tight_layout()

    sim_time = 0.0
    dt = 0.02
    steps_per_frame = 5

    def update(frame):
        nonlocal sim_time

        for _ in range(steps_per_frame):
            # Continuous water source (broken pipe)
            sim.h[source_r, source_c] = max(sim.h[source_r, source_c], 0.4)

            sim.step(dt)
            sim_time += dt

        im.set_data(sim.h)
        vol = sim.h.sum() * DX * DX
        title.set_text(f'Maze Flooding — t={sim_time:.1f}s  |  vol={vol:.3f}')

        return [im, title]

    anim = FuncAnimation(fig, update, frames=900, interval=33, blit=False)
    plt.show()


# =====================================================================
# Run both
# =====================================================================

if __name__ == '__main__':
    print("=== Scenario 1: Aquarium Burst ===")
    print("Close the window to proceed to Scenario 2")
    run_aquarium()

    print("\n=== Scenario 2: Maze Flooding ===")
    run_maze()

    print("\nDone!")
