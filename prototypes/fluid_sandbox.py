"""
Small shallow water sandbox — watch fluid behavior on a tiny grid.

20x20 tiles, heightmap with obstacles, water dropped from one corner.
Runs both pipe model and shallow water side by side for comparison.
No tilting — just gravity and terrain.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Grid ---
N = 20
DX = 1.0
G = 9.81
DT = 0.01
STEPS_PER_FRAME = 5
DAMPING = 0.5

# --- Terrain heightmap ---
terrain = np.zeros((N, N))

# Walls around the edge
walls = np.zeros((N, N), dtype=bool)
walls[0, :] = True
walls[-1, :] = True
walls[:, 0] = True
walls[:, -1] = True
terrain[walls] = 1.0

# A ramp in the middle (sloped terrain)
for i in range(5, 10):
    for j in range(5, 15):
        terrain[i, j] = 0.05 * (i - 5)  # slopes upward

# A wall/obstacle
terrain[12, 5:10] = 0.5
walls[12, 5:10] = True

# A pit
terrain[15:18, 12:16] = -0.1


# --- Initial water ---
def make_water():
    h = np.zeros((N, N))
    h[2:5, 2:5] = 0.3  # pool in top-left corner
    h[walls] = 0.0
    return h


# --- Pipe model (from fluid_test.py) ---
class PipeModel:
    def __init__(self):
        self.h = make_water()
        self.vx = np.zeros_like(self.h)
        self.vy = np.zeros_like(self.h)

    def step(self, dt):
        h, vx, vy = self.h, self.vx, self.vy
        surface = terrain + h

        # Surface gradient
        gx = np.zeros_like(surface)
        gy = np.zeros_like(surface)
        gx[:, 1:-1] = (surface[:, 2:] - surface[:, :-2]) / (2 * DX)
        gy[1:-1, :] = (surface[2:, :] - surface[:-2, :]) / (2 * DX)

        # Velocity: accelerate downhill, damp
        vx += dt * (-G * gx - DAMPING * vx)
        vy += dt * (-G * gy - DAMPING * vy)
        vx[walls] = 0.0
        vy[walls] = 0.0

        # Upwind flux
        def flux_x(v, hf):
            v_face = 0.5 * (v[:, :-1] + v[:, 1:])
            h_face = np.where(v_face > 0, hf[:, :-1], hf[:, 1:])
            f = v_face * h_face
            f[walls[:, :-1] | walls[:, 1:]] = 0.0
            return f

        def flux_y(v, hf):
            v_face = 0.5 * (v[:-1, :] + v[1:, :])
            h_face = np.where(v_face > 0, hf[:-1, :], hf[1:, :])
            f = v_face * h_face
            f[walls[:-1, :] | walls[1:, :]] = 0.0
            return f

        Fx = flux_x(vx, h)
        Fy = flux_y(vy, h)

        div = np.zeros_like(h)
        div[:, 1:-1] += (Fx[:, 1:] - Fx[:, :-1]) / DX
        div[1:-1, :] += (Fy[1:, :] - Fy[:-1, :]) / DX

        h -= dt * div
        h[h < 0] = 0.0
        h[walls] = 0.0
        self.h, self.vx, self.vy = h, vx, vy


# --- Shallow Water Equations ---
class ShallowWater:
    def __init__(self):
        self.h = make_water()
        self.hu = np.zeros_like(self.h)
        self.hv = np.zeros_like(self.h)

    def step(self, dt):
        # CFL substeps
        max_speed = np.sqrt(G * max(self.h.max(), 0.001))
        dt_cfl = 0.4 * DX / max(max_speed, 0.1)
        n_sub = max(1, int(np.ceil(dt / dt_cfl)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._substep(sub_dt)

    def _substep(self, dt):
        h, hu, hv = self.h, self.hu, self.hv
        eps = 1e-6
        h_safe = np.maximum(h, eps)
        u = np.where(h > eps, hu / h_safe, 0.0)
        v = np.where(h > eps, hv / h_safe, 0.0)

        # Surface = terrain + h
        surface = terrain + h

        # Gradients
        def gx(f):
            g = np.zeros_like(f)
            g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * DX)
            return g

        def gy(f):
            g = np.zeros_like(f)
            g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * DX)
            return g

        # Source: gravity on surface slope
        sx = -G * h * gx(surface)
        sy = -G * h * gy(surface)

        # Lax-Friedrichs flux
        def lf_x(q, fq):
            fl = fq[:, :-1]
            fr = fq[:, 1:]
            ql = q[:, :-1]
            qr = q[:, 1:]
            c = np.maximum(
                np.abs(u[:, :-1]) + np.sqrt(G * np.maximum(h[:, :-1], 0)),
                np.abs(u[:, 1:]) + np.sqrt(G * np.maximum(h[:, 1:], 0))
            )
            F = 0.5 * (fl + fr) - 0.5 * c * (qr - ql)
            F[walls[:, :-1] | walls[:, 1:]] = 0.0
            return F

        def lf_y(q, fq):
            fu = fq[:-1, :]
            fd = fq[1:, :]
            qu = q[:-1, :]
            qd = q[1:, :]
            c = np.maximum(
                np.abs(v[:-1, :]) + np.sqrt(G * np.maximum(h[:-1, :], 0)),
                np.abs(v[1:, :]) + np.sqrt(G * np.maximum(h[1:, :], 0))
            )
            F = 0.5 * (fu + fd) - 0.5 * c * (qd - qu)
            F[walls[:-1, :] | walls[1:, :]] = 0.0
            return F

        # Mass flux
        Fxh = lf_x(h, hu)
        Fyh = lf_y(h, hv)

        # Momentum flux
        Fxhu = lf_x(hu, hu * u + 0.5 * G * h * h)
        Fyhu = lf_y(hu, hu * v)
        Fxhv = lf_x(hv, hv * u)
        Fyhv = lf_y(hv, hv * v + 0.5 * G * h * h)

        def div(Fx, Fy):
            d = np.zeros((N, N))
            d[:, 1:-1] += (Fx[:, 1:] - Fx[:, :-1]) / DX
            d[1:-1, :] += (Fy[1:, :] - Fy[:-1, :]) / DX
            return d

        h -= dt * div(Fxh, Fyh)
        hu -= dt * div(Fxhu, Fyhu) - dt * sx - dt * DAMPING * hu
        hv -= dt * div(Fxhv, Fyhv) - dt * sy - dt * DAMPING * hv

        h[h < 0] = 0.0
        h[walls] = 0.0
        hu[walls] = 0.0
        hv[walls] = 0.0
        self.h, self.hu, self.hv = h, hu, hv


# --- Visualization ---
pipe = PipeModel()
shallow = ShallowWater()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Terrain
axes[0].imshow(terrain, origin='upper', cmap='terrain', vmin=-0.2, vmax=0.6)
axes[0].set_title('Terrain heightmap')

# Pipe model
im_pipe = axes[1].imshow(pipe.h, origin='upper', cmap='Blues', vmin=0, vmax=0.3,
                          interpolation='bilinear')
axes[1].set_title('Pipe model')

# Shallow water
im_sw = axes[2].imshow(shallow.h, origin='upper', cmap='Blues', vmin=0, vmax=0.3,
                        interpolation='bilinear')
axes[2].set_title('Shallow water equations')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

suptitle = fig.suptitle('t = 0.0 s', fontsize=12)
plt.tight_layout()

sim_time = 0.0

def update(frame):
    global sim_time
    for _ in range(STEPS_PER_FRAME):
        pipe.step(DT)
        shallow.step(DT)
        sim_time += DT

    im_pipe.set_data(pipe.h)
    im_sw.set_data(shallow.h)

    pipe_vol = pipe.h.sum() * DX * DX
    sw_vol = shallow.h.sum() * DX * DX
    suptitle.set_text(f't = {sim_time:.1f}s  |  Pipe vol: {pipe_vol:.3f}  |  SW vol: {sw_vol:.3f}')

    return [im_pipe, im_sw, suptitle]

anim = FuncAnimation(fig, update, frames=600, interval=33, blit=False)
plt.show()
