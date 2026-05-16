"""
Raylib test: top-down explosion with particles and shockwave ring.
Click anywhere to trigger an explosion.
"""
import pyray as rl
import random
import math

# --- Config ---
W, H = 1200, 800
TILE_SIZE = 10
MAX_PARTICLES = 2000

# --- Particle ---
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'r', 'g', 'b']
    def __init__(self, x, y, vx, vy, life, size, r, g, b):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.max_life = life
        self.size = size
        self.r, self.g, self.b = r, g, b

# --- Shockwave ring ---
class Shockwave:
    __slots__ = ['x', 'y', 'radius', 'max_radius', 'life', 'max_life']
    def __init__(self, x, y, max_radius=200, life=0.6):
        self.x, self.y = x, y
        self.radius = 0
        self.max_radius = max_radius
        self.life = life
        self.max_life = life

particles = []
shockwaves = []

def spawn_explosion(x, y):
    """Spawn particles + shockwave at position."""
    # Fire core (bright, fast, short-lived)
    for _ in range(80):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(50, 300)
        life = random.uniform(0.2, 0.6)
        size = random.uniform(3, 8)
        particles.append(Particle(
            x, y,
            math.cos(angle) * speed, math.sin(angle) * speed,
            life, size,
            255, random.randint(180, 255), random.randint(0, 80)
        ))

    # Smoke (slower, longer-lived, gray)
    for _ in range(60):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(20, 120)
        life = random.uniform(0.8, 2.5)
        size = random.uniform(4, 12)
        gray = random.randint(60, 140)
        particles.append(Particle(
            x, y,
            math.cos(angle) * speed, math.sin(angle) * speed,
            life, size,
            gray, gray, gray
        ))

    # Sparks (fast, tiny, bright)
    for _ in range(40):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(200, 600)
        life = random.uniform(0.1, 0.4)
        size = random.uniform(1, 3)
        particles.append(Particle(
            x, y,
            math.cos(angle) * speed, math.sin(angle) * speed,
            life, size,
            255, 255, random.randint(150, 255)
        ))

    # Shockwave
    shockwaves.append(Shockwave(x, y, max_radius=250, life=0.5))

def draw_floor():
    """Draw a simple dark grid floor."""
    for ty in range(0, H, TILE_SIZE):
        for tx in range(0, W, TILE_SIZE):
            shade = 25 + ((tx // TILE_SIZE + ty // TILE_SIZE) % 2) * 8
            rl.draw_rectangle(tx, ty, TILE_SIZE, TILE_SIZE, rl.Color(shade, shade, shade + 5, 255))

# --- Main ---
rl.init_window(W, H, "Breach - Raylib Explosion Test")
rl.set_target_fps(60)

while not rl.window_should_close():
    dt = rl.get_frame_time()

    # Click to explode
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
        mx, my = rl.get_mouse_x(), rl.get_mouse_y()
        spawn_explosion(mx, my)

    # Update particles
    alive = []
    for p in particles:
        p.life -= dt
        if p.life <= 0:
            continue
        p.x += p.vx * dt
        p.y += p.vy * dt
        # Slow down
        p.vx *= 0.97
        p.vy *= 0.97
        alive.append(p)
    particles = alive

    # Update shockwaves
    alive_sw = []
    for sw in shockwaves:
        sw.life -= dt
        if sw.life <= 0:
            continue
        t = 1.0 - sw.life / sw.max_life
        sw.radius = t * sw.max_radius
        alive_sw.append(sw)
    shockwaves = alive_sw

    # Draw
    rl.begin_drawing()
    rl.clear_background(rl.Color(20, 20, 25, 255))
    draw_floor()

    # Draw shockwave rings
    for sw in shockwaves:
        alpha = int(255 * (sw.life / sw.max_life) * 0.6)
        thickness = max(1, 3 * (sw.life / sw.max_life))
        rl.draw_ring(
            rl.Vector2(sw.x, sw.y),
            sw.radius - thickness, sw.radius + thickness,
            0, 360, 64,
            rl.Color(200, 220, 255, alpha)
        )

    # Draw particles (additive-ish: draw brightest last)
    for p in particles:
        t = p.life / p.max_life
        alpha = int(255 * t)
        size = p.size * (0.5 + 0.5 * t)
        rl.draw_circle(int(p.x), int(p.y), size,
                       rl.Color(p.r, p.g, p.b, alpha))

    # Flash at explosion center (brief white flash)
    for sw in shockwaves:
        if sw.life > sw.max_life * 0.7:
            flash_t = (sw.life - sw.max_life * 0.7) / (sw.max_life * 0.3)
            flash_alpha = int(200 * flash_t)
            rl.draw_circle(int(sw.x), int(sw.y), 30 * flash_t,
                           rl.Color(255, 255, 220, flash_alpha))

    rl.draw_text("Click to explode!", 10, 10, 20, rl.Color(200, 200, 200, 255))
    rl.draw_fps(W - 100, 10)
    rl.end_drawing()

rl.close_window()
