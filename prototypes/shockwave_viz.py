"""
Shockwave Visualization Tool

Real-time interactive pressure field viewer at full fine-tile resolution.
Click anywhere to detonate an explosion. Adjust visualization with keyboard.

Controls:
  Left click:  Detonate grenade at cursor
  Right click: Detonate big explosion at cursor
  R:           Reset atmosphere
  Space:       Pause/resume physics
  +/-:         Adjust physics speed
  1-5:         Change color scheme
  S:           Save current frame as PNG
  ESC:         Quit
"""

import pygame
import numpy as np
import math
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CFG

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
FINE_W = CFG.display.fine_w    # 120
FINE_H = CFG.display.fine_h    # 75
COARSE = CFG.display.coarse
SCALE = 8   # pixels per fine tile (larger = more detail visible)

SCREEN_W = FINE_W * SCALE
SCREEN_H = FINE_H * SCALE + 80  # extra space for info bar

# Material IDs
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3


def build_map():
    """Build the ship map (same as game.py)."""
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

    wall_y = 8 * COARSE
    for x in range(hull_t, wall_x):
        m[wall_y, x] = MAT_WOOD
    door_x = 8 * COARSE
    for dx in range(COARSE):
        m[wall_y, door_x + dx] = MAT_DOOR

    wall_y2 = 12 * COARSE
    for x in range(wall_x + 1, FINE_W - hull_t):
        m[wall_y2, x] = MAT_WOOD
    door_x2 = 25 * COARSE
    for dx in range(COARSE):
        m[wall_y2, door_x2 + dx] = MAT_DOOR

    wall_y3 = 6 * COARSE
    for x in range(wall_x + 1, 30 * COARSE):
        m[wall_y3, x] = MAT_WOOD
    door_x3 = 22 * COARSE
    for dx in range(COARSE):
        m[wall_y3, door_x3 + dx] = MAT_DOOR

    wall_x2 = 30 * COARSE
    for y in range(hull_t, wall_y2):
        m[y, wall_x2] = MAT_WOOD
    door_y3 = 4 * COARSE
    for dy in range(COARSE):
        m[door_y3 + dy, wall_x2] = MAT_DOOR

    m[0:hull_t, :] = MAT_HULL
    m[FINE_H - hull_t:, :] = MAT_HULL
    m[:, 0:hull_t] = MAT_HULL
    m[:, FINE_W - hull_t:] = MAT_HULL
    return m


# ---------------------------------------------------------------------------
# Physics (simplified from game)
# ---------------------------------------------------------------------------
class AtmosphereField:
    def __init__(self, material):
        self.material = material
        self.is_wall = np.isin(material, [MAT_HULL, MAT_WOOD])
        self.is_vacuum = np.zeros_like(self.is_wall)
        self.is_vacuum[0:COARSE, :] = True
        self.is_vacuum[FINE_H - COARSE:, :] = True
        self.is_vacuum[:, 0:COARSE] = True
        self.is_vacuum[:, FINE_W - COARSE:] = True
        self.reset()

    def reset(self):
        self.atmosphere = np.where(
            self.is_wall | self.is_vacuum, 0.0, 1.0
        ).astype(np.float32)
        self.peak_pressure = 1.0

    def detonate(self, fy, fx, pressure=10.0, radius=6):
        """Deposit pressure at explosion site."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < FINE_H and 0 <= nx < FINE_W:
                    dist = math.sqrt(dy * dy + dx * dx)
                    if dist <= radius:
                        falloff = 1.0 - (dist / radius)
                        if not self.is_wall[ny, nx] and not self.is_vacuum[ny, nx]:
                            self.atmosphere[ny, nx] += pressure * falloff

    def step(self, n_substeps=5, dt=0.001):
        for _ in range(n_substeps):
            up = np.roll(self.atmosphere, 1, axis=0)
            down = np.roll(self.atmosphere, -1, axis=0)
            left = np.roll(self.atmosphere, 1, axis=1)
            right = np.roll(self.atmosphere, -1, axis=1)
            wall_up = np.roll(self.is_wall, 1, axis=0)
            wall_down = np.roll(self.is_wall, -1, axis=0)
            wall_left = np.roll(self.is_wall, 1, axis=1)
            wall_right = np.roll(self.is_wall, -1, axis=1)
            up = np.where(wall_up, self.atmosphere, up)
            down = np.where(wall_down, self.atmosphere, down)
            left = np.where(wall_left, self.atmosphere, left)
            right = np.where(wall_right, self.atmosphere, right)
            lap = up + down + left + right - 4.0 * self.atmosphere
            self.atmosphere += CFG.physics.d_atm * dt * lap
            self.atmosphere[self.is_wall] = 0.0
            self.atmosphere[self.is_vacuum] = 0.0
            np.clip(self.atmosphere, 0.0, 20.0, out=self.atmosphere)

        self.peak_pressure = self.atmosphere.max()


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------
def color_scheme_fire(val):
    """Fire: black -> red -> orange -> yellow -> white."""
    excess = val - 1.0
    if excess <= 0:
        if val < 0.9:
            # Underpressure: blue
            t = min(1.0, (1.0 - val) * 3)
            return (int(30 * (1-t)), int(30 * (1-t)), int(60 + 195 * t), 255)
        return None  # transparent (normal pressure)
    t = min(1.0, excess / 5.0)
    if t < 0.25:
        s = t / 0.25
        return (int(255 * s), 0, 0, 255)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return (255, int(140 * s), 0, 255)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return (255, int(140 + 115 * s), int(80 * s), 255)
    else:
        s = (t - 0.75) / 0.25
        return (255, 255, int(80 + 175 * s), 255)


def color_scheme_thermal(val):
    """Thermal camera: blue -> cyan -> green -> yellow -> red -> white."""
    excess = val - 1.0
    if excess <= 0:
        if val < 0.9:
            t = min(1.0, (1.0 - val) * 3)
            return (0, 0, int(100 + 155 * t), 255)
        return None
    t = min(1.0, excess / 5.0)
    if t < 0.2:
        s = t / 0.2
        return (0, int(100 * s), int(200 + 55 * s), 255)
    elif t < 0.4:
        s = (t - 0.2) / 0.2
        return (0, int(100 + 155 * s), int(255 - 155 * s), 255)
    elif t < 0.6:
        s = (t - 0.4) / 0.2
        return (int(255 * s), 255, int(100 - 100 * s), 255)
    elif t < 0.8:
        s = (t - 0.6) / 0.2
        return (255, int(255 - 200 * s), 0, 255)
    else:
        s = (t - 0.8) / 0.2
        return (255, int(55 + 200 * s), int(200 * s), 255)


def color_scheme_electric(val):
    """Electric: dark blue -> cyan -> white."""
    excess = val - 1.0
    if excess <= 0:
        if val < 0.9:
            t = min(1.0, (1.0 - val) * 3)
            return (int(40 * t), 0, int(80 * t), 255)
        return None
    t = min(1.0, excess / 5.0)
    if t < 0.3:
        s = t / 0.3
        return (0, int(50 * s), int(150 + 105 * s), 255)
    elif t < 0.6:
        s = (t - 0.3) / 0.3
        return (int(50 * s), int(50 + 200 * s), 255, 255)
    else:
        s = (t - 0.6) / 0.4
        return (int(50 + 205 * s), 255, 255, 255)


def color_scheme_monochrome(val):
    """Simple white intensity — clean and readable."""
    excess = val - 1.0
    if excess <= 0:
        if val < 0.9:
            t = min(1.0, (1.0 - val) * 3)
            return (int(40 * t), int(30 * t), int(80 * t), 255)
        return None
    t = min(1.0, excess / 5.0)
    v = int(255 * t)
    return (v, v, v, 255)


def color_scheme_ocean(val):
    """Ocean: deep blue -> teal -> white foam."""
    excess = val - 1.0
    if excess <= 0:
        if val < 0.9:
            t = min(1.0, (1.0 - val) * 3)
            return (0, int(20 * t), int(60 + 100 * t), 255)
        return None
    t = min(1.0, excess / 5.0)
    if t < 0.3:
        s = t / 0.3
        return (0, int(80 * s), int(120 + 80 * s), 255)
    elif t < 0.6:
        s = (t - 0.3) / 0.3
        return (int(60 * s), int(80 + 120 * s), int(200 + 55 * s), 255)
    else:
        s = (t - 0.6) / 0.4
        return (int(60 + 195 * s), int(200 + 55 * s), 255, 255)


SCHEMES = {
    1: ("Fire", color_scheme_fire),
    2: ("Thermal", color_scheme_thermal),
    3: ("Electric", color_scheme_electric),
    4: ("Monochrome", color_scheme_monochrome),
    5: ("Ocean", color_scheme_ocean),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Shockwave Visualization — Click to Detonate")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    font_big = pygame.font.SysFont("consolas", 18, bold=True)

    material = build_map()
    field = AtmosphereField(material)

    # Pre-render map background (walls, doors, floor)
    map_bg = pygame.Surface((FINE_W * SCALE, FINE_H * SCALE))
    wall_colors = {
        MAT_AIR: (20, 25, 35),
        MAT_HULL: (50, 50, 55),
        MAT_WOOD: (45, 35, 20),
        MAT_DOOR: (35, 50, 35),
    }
    for fy in range(FINE_H):
        for fx in range(FINE_W):
            color = wall_colors.get(material[fy, fx], (20, 25, 35))
            pygame.draw.rect(map_bg, color,
                             (fx * SCALE, fy * SCALE, SCALE, SCALE))

    scheme_id = 1
    paused = False
    physics_speed = 10  # substeps per frame
    frame_count = 0
    save_count = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    field.reset()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    physics_speed = min(50, physics_speed + 2)
                elif event.key == pygame.K_MINUS:
                    physics_speed = max(1, physics_speed - 2)
                elif event.key == pygame.K_s:
                    fname = f"shockwave_frame_{save_count:04d}.png"
                    pygame.image.save(screen, fname)
                    print(f"Saved {fname}")
                    save_count += 1
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                   pygame.K_4, pygame.K_5):
                    scheme_id = event.key - pygame.K_0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if my < FINE_H * SCALE:
                    fx = mx // SCALE
                    fy = my // SCALE
                    if event.button == 1:
                        field.detonate(fy, fx, pressure=10.0, radius=6)
                    elif event.button == 3:
                        field.detonate(fy, fx, pressure=20.0, radius=10)

        # Physics
        if not paused:
            field.step(n_substeps=physics_speed)
            frame_count += 1

        # Draw map background
        screen.blit(map_bg, (0, 0))

        # Draw atmosphere overlay at fine tile resolution
        scheme_name, color_fn = SCHEMES[scheme_id]
        atm = field.atmosphere

        # Build overlay surface
        overlay = pygame.Surface((FINE_W, FINE_H), pygame.SRCALPHA)
        for fy in range(FINE_H):
            for fx in range(FINE_W):
                if field.is_wall[fy, fx] or field.is_vacuum[fy, fx]:
                    continue
                val = atm[fy, fx]
                c = color_fn(val)
                if c is not None:
                    overlay.set_at((fx, fy), c)

        scaled = pygame.transform.scale(overlay, (FINE_W * SCALE, FINE_H * SCALE))
        screen.blit(scaled, (0, 0))

        # Info bar
        info_y = FINE_H * SCALE + 5
        pygame.draw.rect(screen, (15, 15, 20),
                         (0, FINE_H * SCALE, SCREEN_W, 80))

        texts = [
            f"Scheme: {scheme_id} {scheme_name} (1-5 to change)",
            f"Peak: {field.peak_pressure:.2f} atm | "
            f"Speed: {physics_speed} substeps/frame | "
            f"{'PAUSED' if paused else 'RUNNING'}",
            "LClick: grenade | RClick: big bomb | R: reset | "
            "Space: pause | S: save | +/-: speed",
        ]
        for i, t in enumerate(texts):
            color = (200, 200, 210) if i > 0 else (0, 200, 255)
            surf = font.render(t, True, color)
            screen.blit(surf, (10, info_y + i * 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
