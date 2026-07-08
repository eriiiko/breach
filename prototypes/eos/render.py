"""
State -> RGB frame, and GIF assembly, for the EOS Phase-1.2 visual prototype.

Compositing (back to front): open space (dark) / solid (flat color) / vacuum
(darker shade) as the base, then on open-air tiles only: a blue water tint,
a brighter/whiter smoke overlay proportional to density, and a red->orange
->yellow fire-color addition where hot. Optional sparse velocity arrows on
top. Simple and legible -- colors are cosmetic, tune later (PLAN.md).
"""

import numpy as np
import imageio.v2 as imageio

from state import State, AMBIENT_T

# --- palette (0-255 float, composited then cast to uint8) ------------------
COLOR_BG = np.array([12, 12, 16], dtype=np.float32)        # open space
COLOR_SOLID = np.array([95, 95, 105], dtype=np.float32)    # walls
COLOR_VACUUM = np.array([2, 2, 8], dtype=np.float32)       # darker than open space
COLOR_DOOR = np.array([150, 110, 40], dtype=np.float32)    # amber
COLOR_SMOKE = np.array([235, 235, 245], dtype=np.float32)  # bright/white overlay
COLOR_WATER = np.array([30, 90, 200], dtype=np.float32)    # blue tint
COLOR_ARROW = np.array([80, 220, 255], dtype=np.uint8)     # velocity arrows

# --- fire gradient stops (red -> orange -> yellow) --------------------------
_FIRE_RED = np.array([200, 0, 0], dtype=np.float32)
_FIRE_ORANGE = np.array([255, 120, 0], dtype=np.float32)
_FIRE_YELLOW = np.array([255, 230, 60], dtype=np.float32)


def _fire_color(T: np.ndarray, T_ambient: float, T_hot: float) -> np.ndarray:
    """Red -> orange -> yellow as excess temperature increases, faded in by
    excess so ambient tiles get zero contribution. Returns (H, W, 3) float32
    to be ADDED on top of the base frame."""
    excess = np.clip((T - T_ambient) / (T_hot - T_ambient), 0.0, 1.0)
    t_lo = np.clip(excess * 2.0, 0.0, 1.0)          # red -> orange
    t_hi = np.clip(excess * 2.0 - 1.0, 0.0, 1.0)    # orange -> yellow
    color = (_FIRE_RED[None, None, :] * (1 - t_lo)[..., None]
             + _FIRE_ORANGE[None, None, :] * t_lo[..., None])
    color = color * (1 - t_hi)[..., None] + _FIRE_YELLOW[None, None, :] * t_hi[..., None]
    return color * excess[..., None]


def _draw_line(canvas: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               color: np.ndarray, n: int = 12) -> None:
    """Crude vectorized line: sample n points along the segment and stamp
    those pixels. Good enough for short sparse debug arrows."""
    h, w = canvas.shape[:2]
    t = np.linspace(0.0, 1.0, n)
    xs = np.clip((x0 + (x1 - x0) * t).astype(np.int32), 0, w - 1)
    ys = np.clip((y0 + (y1 - y0) * t).astype(np.int32), 0, h - 1)
    canvas[ys, xs] = color


def _draw_velocity(frame: np.ndarray, state: State, px_per_tile: int, stride: int,
                    ref_speed: float = 2.0) -> None:
    """Sparse downsampled quiver: one short line per sampled open-air tile,
    length/brightness-independent of grid resolution."""
    max_px = 1.4 * px_per_tile
    air = state.open_air
    for row in range(0, state.height, stride):
        for col in range(0, state.width, stride):
            if not air[row, col]:
                continue
            vx, vy = float(state.vx[row, col]), float(state.vy[row, col])
            speed = (vx * vx + vy * vy) ** 0.5
            if speed < 1e-3:
                continue
            length = min(max_px, max_px * speed / ref_speed)
            ux, uy = vx / speed, vy / speed
            cx = (col + 0.5) * px_per_tile
            cy = (row + 0.5) * px_per_tile
            _draw_line(frame, cx, cy, cx + ux * length, cy + uy * length, COLOR_ARROW)


def render_frame(state: State, *, px_per_tile: int = 4, show_velocity: bool = True,
                  velocity_stride: int = 8, smoke_ref: float = 1.0,
                  water_ref: float = 0.5, T_hot: float = 1600.0) -> np.ndarray:
    """Composite one (H*px_per_tile, W*px_per_tile, 3) uint8 frame from `state`."""
    rgb = np.empty((state.height, state.width, 3), dtype=np.float32)
    rgb[:] = COLOR_BG
    rgb[state.vacuum] = COLOR_VACUUM
    rgb[state.solid] = COLOR_SOLID

    closed_door = state.door & state.solid
    open_door = state.door & ~state.solid
    rgb[closed_door] = COLOR_DOOR
    rgb[open_door] = 0.7 * rgb[open_door] + 0.3 * COLOR_DOOR   # faint doorway outline

    open_air = state.open_air   # smoke/water/fire only ever render where gas can live
    water_frac = np.clip(state.water_depth / water_ref, 0.0, 1.0)
    rgb += COLOR_WATER * water_frac[..., None] * open_air[..., None]

    smoke_frac = np.clip(state.smoke / smoke_ref, 0.0, 1.0)
    rgb += COLOR_SMOKE * smoke_frac[..., None] * open_air[..., None]

    rgb += _fire_color(state.T, AMBIENT_T, T_hot) * open_air[..., None]

    frame = np.clip(rgb, 0, 255).astype(np.uint8)
    frame = np.repeat(np.repeat(frame, px_per_tile, axis=0), px_per_tile, axis=1)

    if show_velocity:
        _draw_velocity(frame, state, px_per_tile, velocity_stride)

    return frame


def make_gif(frames: list[np.ndarray], path, fps: int = 20) -> None:
    """Assemble frames -> .gif."""
    imageio.mimsave(str(path), frames, fps=fps)
