"""
Fluid sloshing prototype for Breach.

Compares two models side by side:
  Left:  Pipe + damped velocity (simple, cheap)
  Right: Shallow water equations (physically accurate)

Setup:
  - Ship: 30m x 8m (90x24 tiles at 1/3m)
  - Water initially pooled in the left third
  - Ship rocks slowly around Y-axis (tilt_x oscillates)
  - Terrain: flat floor with slight random bumps
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Grid (1/3 meter tiles)
# ---------------------------------------------------------------------------
DX = 1.0 / 3.0          # tile size in meters
W_TILES = 90             # 30m / (1/3)
H_TILES = 24             # 8m / (1/3)
G = 9.81                 # gravity m/s²

# Tilt
TILT_AMPLITUDE = 3.0     # degrees
TILT_PERIOD = 12.0       # seconds per full oscillation (slow slosh)
TILT_OMEGA = 2.0 * np.pi / TILT_PERIOD

# Simulation
DT = 0.02                # 20ms per step (fluid is slow, big dt is fine)
STEPS_PER_FRAME = 4
FPS = 30
TOTAL_FRAMES = 450       # 15 seconds — see full slosh cycle
DAMPING = 1.0            # velocity damping (1/s)

# Walls: border around the ship
walls = np.zeros((H_TILES, W_TILES), dtype=bool)
walls[0, :] = True
walls[-1, :] = True
walls[:, 0] = True
walls[:, -1] = True

# Terrain: flat with small random bumps (furniture, floor texture)
np.random.seed(42)
terrain = np.random.uniform(0.0, 0.01, (H_TILES, W_TILES))
terrain[walls] = 0.5  # walls are tall

# Ship center (for tilt calculation)
cx = W_TILES / 2.0
cy = H_TILES / 2.0

# ---------------------------------------------------------------------------
# Helper: compute tilt offset for each tile
# ---------------------------------------------------------------------------
def compute_tilt(tilt_x_deg, tilt_y_deg=0.0):
    """Returns height offset per tile from ship tilt."""
    tilt_x_rad = np.radians(tilt_x_deg)
    tilt_y_rad = np.radians(tilt_y_deg)
    xs = (np.arange(W_TILES) - cx) * DX  # position in meters from center
    ys = (np.arange(H_TILES) - cy) * DX
    XX, YY = np.meshgrid(xs, ys)
    return np.tan(tilt_x_rad) * XX + np.tan(tilt_y_rad) * YY

# ---------------------------------------------------------------------------
# Initial conditions: water pooled in left third of ship
# ---------------------------------------------------------------------------
def make_initial_water():
    h = np.zeros((H_TILES, W_TILES), dtype=np.float64)
    h[1:-1, 1:W_TILES//3] = 0.05  # 5cm of water
    h[walls] = 0.0
    return h

# ---------------------------------------------------------------------------
# Model 1: Pipe + Damped Velocity
# ---------------------------------------------------------------------------
class PipeModel:
    def __init__(self):
        self.h = make_initial_water()        # water depth
        self.vx = np.zeros_like(self.h)      # x-velocity
        self.vy = np.zeros_like(self.h)      # y-velocity

    def step(self, tilt_offset, dt):
        h, vx, vy = self.h, self.vx, self.vy

        # Surface level = terrain + tilt + water depth
        surface = terrain + tilt_offset + h

        # Surface gradient (central difference, Neumann at walls)
        # grad points downhill for flow
        def grad_x(f):
            g = np.zeros_like(f)
            g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * DX)
            return g
        def grad_y(f):
            g = np.zeros_like(f)
            g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * DX)
            return g

        gx = grad_x(surface)
        gy = grad_y(surface)

        # Velocity update: accelerate downhill, damp
        vx += dt * (-G * gx - DAMPING * vx)
        vy += dt * (-G * gy - DAMPING * vy)

        # Zero velocity on walls
        vx[walls] = 0.0
        vy[walls] = 0.0

        # Flux: water moves in velocity direction (upwind)
        # flux_x = vx * h (at cell faces, upwind selection)
        def flux_x_upwind(v, h_field):
            f = np.zeros_like(v)
            # Right face of cell (i, j) is between (i,j) and (i,j+1)
            # Use h from upwind side
            v_face = 0.5 * (v[:, :-1] + v[:, 1:])
            h_left = h_field[:, :-1]
            h_right = h_field[:, 1:]
            h_face = np.where(v_face > 0, h_left, h_right)
            flux = v_face * h_face
            # Don't flow through walls
            wall_face = walls[:, :-1] | walls[:, 1:]
            flux[wall_face] = 0.0
            return flux

        def flux_y_upwind(v, h_field):
            f = np.zeros_like(v)
            v_face = 0.5 * (v[:-1, :] + v[1:, :])
            h_up = h_field[:-1, :]
            h_down = h_field[1:, :]
            h_face = np.where(v_face > 0, h_up, h_down)
            flux = v_face * h_face
            wall_face = walls[:-1, :] | walls[1:, :]
            flux[wall_face] = 0.0
            return flux

        Fx = flux_x_upwind(vx, h)
        Fy = flux_y_upwind(vy, h)

        # Divergence of flux: dh/dt = -div(flux)
        div_F = np.zeros_like(h)
        div_F[:, 1:-1] += (Fx[:, 1:] - Fx[:, :-1]) / DX
        div_F[1:-1, :] += (Fy[1:, :] - Fy[:-1, :]) / DX

        h -= dt * div_F

        # Enforce: no negative water, no water on walls
        h[h < 0] = 0.0
        h[walls] = 0.0

        self.h = h

# ---------------------------------------------------------------------------
# Model 2: Shallow Water Equations (simplified)
# ---------------------------------------------------------------------------
class ShallowWater:
    def __init__(self):
        self.h = make_initial_water()
        self.hu = np.zeros_like(self.h)  # momentum x
        self.hv = np.zeros_like(self.h)  # momentum y

    def step(self, tilt_offset, dt):
        # Substep for CFL stability: dt_cfl = dx / max(|u| + sqrt(g*h))
        max_speed = np.sqrt(G * np.maximum(self.h.max(), 0.001))
        dt_cfl = 0.4 * DX / max(max_speed, 0.1)
        n_sub = max(1, int(np.ceil(dt / dt_cfl)))
        sub_dt = dt / n_sub

        for _ in range(n_sub):
            self._substep(tilt_offset, sub_dt)

    def _substep(self, tilt_offset, dt):
        h, hu, hv = self.h, self.hu, self.hv

        # Total terrain (static + tilt)
        B = terrain + tilt_offset

        # Velocity from momentum (safe division)
        eps = 1e-6
        h_safe = np.maximum(h, eps)
        u = hu / h_safe
        v = hv / h_safe
        u[h < eps] = 0.0
        v[h < eps] = 0.0

        def grad_x(f):
            g = np.zeros_like(f)
            g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * DX)
            return g
        def grad_y(f):
            g = np.zeros_like(f)
            g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * DX)
            return g

        # Shallow water: source term = -g*h * grad(B + h)
        surface = B + h
        sx = -G * h * grad_x(surface)
        sy = -G * h * grad_y(surface)

        # Flux divergence for mass: -div(hu, hv)
        # Using Lax-Friedrichs for stability
        def lax_friedrichs_x(q, flux_q):
            """Lax-Friedrichs flux in x-direction."""
            f_left = flux_q[:, :-1]
            f_right = flux_q[:, 1:]
            q_left = q[:, :-1]
            q_right = q[:, 1:]
            # Max wave speed
            c_max = np.maximum(np.abs(u[:, :-1]) + np.sqrt(G * np.maximum(h[:, :-1], 0)),
                               np.abs(u[:, 1:]) + np.sqrt(G * np.maximum(h[:, 1:], 0)))
            # Numerical flux
            F = 0.5 * (f_left + f_right) - 0.5 * c_max * (q_right - q_left)
            # Block walls
            wall_face = walls[:, :-1] | walls[:, 1:]
            F[wall_face] = 0.0
            return F

        def lax_friedrichs_y(q, flux_q):
            f_up = flux_q[:-1, :]
            f_down = flux_q[1:, :]
            q_up = q[:-1, :]
            q_down = q[1:, :]
            c_max = np.maximum(np.abs(v[:-1, :]) + np.sqrt(G * np.maximum(h[:-1, :], 0)),
                               np.abs(v[1:, :]) + np.sqrt(G * np.maximum(h[1:, :], 0)))
            F = 0.5 * (f_up + f_down) - 0.5 * c_max * (q_down - q_up)
            wall_face = walls[:-1, :] | walls[1:, :]
            F[wall_face] = 0.0
            return F

        # Mass fluxes
        Fx_h = lax_friedrichs_x(h, hu)
        Fy_h = lax_friedrichs_y(h, hv)

        # Momentum fluxes
        Fx_hu = lax_friedrichs_x(hu, hu * u + 0.5 * G * h * h)
        Fy_hu = lax_friedrichs_y(hu, hu * v)
        Fx_hv = lax_friedrichs_x(hv, hv * u)
        Fy_hv = lax_friedrichs_y(hv, hv * v + 0.5 * G * h * h)

        # Update with flux divergence + source
        def div_flux(F_x, F_y):
            d = np.zeros((H_TILES, W_TILES))
            d[:, 1:-1] += (F_x[:, 1:] - F_x[:, :-1]) / DX
            d[1:-1, :] += (F_y[1:, :] - F_y[:-1, :]) / DX
            return d

        h  -= dt * div_flux(Fx_h, Fy_h)
        hu -= dt * div_flux(Fx_hu, Fy_hu) - dt * sx - dt * DAMPING * hu
        hv -= dt * div_flux(Fx_hv, Fy_hv) - dt * sy - dt * DAMPING * hv

        # Enforce
        h[h < 0] = 0.0
        h[walls] = 0.0
        hu[walls] = 0.0
        hv[walls] = 0.0

        self.h, self.hu, self.hv = h, hu, hv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("Setting up fluid sloshing test...")
pipe = PipeModel()
shallow = ShallowWater()

print(f"Grid: {W_TILES}x{H_TILES} tiles ({W_TILES*DX:.0f}m x {H_TILES*DX:.0f}m)")
print(f"Tilt: ±{TILT_AMPLITUDE}° with period {TILT_PERIOD}s")
print(f"Initial water: {pipe.h.sum() * DX * DX:.3f} m³")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
fig.patch.set_facecolor("black")

titles = ["Pipe + Damped Velocity", "Shallow Water Equations"]
imgs = []
for i, ax in enumerate(axes):
    ax.set_facecolor("black")
    im = ax.imshow(np.zeros((H_TILES, W_TILES)), origin="upper",
                   cmap="Blues", vmin=0, vmax=0.08, interpolation="bilinear")
    ax.set_title(titles[i], color="white", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw walls
    wall_overlay = np.ma.masked_where(~walls, np.ones_like(walls, dtype=float))
    ax.imshow(wall_overlay, origin="upper", cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest", alpha=0.5)
    imgs.append(im)

suptitle = fig.suptitle("", color="white", fontsize=11, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])

sim_time = 0.0

def update(frame):
    global sim_time

    for _ in range(STEPS_PER_FRAME):
        # Tilt oscillates
        tilt_x = TILT_AMPLITUDE * np.sin(TILT_OMEGA * sim_time)
        tilt_offset = compute_tilt(tilt_x)

        pipe.step(tilt_offset, DT)
        shallow.step(tilt_offset, DT)
        sim_time += DT

    tilt_now = TILT_AMPLITUDE * np.sin(TILT_OMEGA * sim_time)

    imgs[0].set_data(pipe.h)
    imgs[1].set_data(shallow.h)

    pipe_vol = pipe.h.sum() * DX * DX
    shallow_vol = shallow.h.sum() * DX * DX

    suptitle.set_text(
        f"t={sim_time:.1f}s  |  Tilt: {tilt_now:+.1f}°  |  "
        f"Pipe water: {pipe_vol:.4f}m³  |  Shallow water: {shallow_vol:.4f}m³"
    )

    if frame % 30 == 0:
        print(f"  Frame {frame}/{TOTAL_FRAMES}, t={sim_time:.1f}s, "
              f"tilt={tilt_now:+.1f}°, pipe={pipe_vol:.4f}, shallow={shallow_vol:.4f}")

    return imgs + [suptitle]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Running: {TOTAL_FRAMES} frames at {FPS} fps")
anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True, interval=1000//FPS)

output_path = "C:/Users/steen/projects/breach/prototypes/fluid_test.gif"
print(f"Saving to {output_path} ...")
anim.save(output_path, writer=PillowWriter(fps=FPS))
print("Done!")
plt.close()
