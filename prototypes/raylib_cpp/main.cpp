/*
 * Breach Raylib Prototype — Top-down explosion with particles,
 * shockwave ring, smoke, and a tile grid. Pure C++, GPU-accelerated.
 *
 * Click to explode. Hold right-click to spawn fire.
 */
#include "raylib.h"
#include "raymath.h"
#include <vector>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
constexpr int SCREEN_W = 1200;
constexpr int SCREEN_H = 800;
constexpr int TILE_PX  = 5;
constexpr int GRID_W   = SCREEN_W / TILE_PX;
constexpr int GRID_H   = SCREEN_H / TILE_PX;

// ---------------------------------------------------------------------------
// Random helpers
// ---------------------------------------------------------------------------
static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}
static int randi(int lo, int hi) {
    return lo + rand() % (hi - lo + 1);
}

// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------
struct Particle {
    Vector2 pos;
    Vector2 vel;
    float life, max_life;
    float size;
    unsigned char r, g, b;
};

// ---------------------------------------------------------------------------
// Shockwave
// ---------------------------------------------------------------------------
struct Shockwave {
    Vector2 center;
    float radius, max_radius;
    float life, max_life;
};

// ---------------------------------------------------------------------------
// Tile grid (simple atmosphere + fire for demo)
// ---------------------------------------------------------------------------
static float atmosphere[GRID_H][GRID_W];
static float fire[GRID_H][GRID_W];
static bool  walls[GRID_H][GRID_W];

static void init_grid() {
    for (int y = 0; y < GRID_H; y++)
        for (int x = 0; x < GRID_W; x++) {
            atmosphere[y][x] = 1.0f;
            fire[y][x] = 0.0f;
            walls[y][x] = false;
        }
    // Border walls
    for (int x = 0; x < GRID_W; x++) {
        walls[0][x] = walls[GRID_H-1][x] = true;
    }
    for (int y = 0; y < GRID_H; y++) {
        walls[y][0] = walls[y][GRID_W-1] = true;
    }
    // A few interior walls
    for (int y = 20; y < 40; y++) walls[y][40] = true;
    for (int x = 40; x < 60; x++) walls[20][x] = true;
    for (int y = 40; y < 60; y++) walls[y][80] = true;
}

static void step_atmosphere(float dt) {
    // Simple diffusion
    static float tmp[GRID_H][GRID_W];
    float D = 50.0f;
    float diff_dt = 0.004f;
    int steps = (int)(dt / diff_dt) + 1;
    float actual_dt = dt / steps;

    for (int s = 0; s < steps; s++) {
        for (int y = 1; y < GRID_H-1; y++)
            for (int x = 1; x < GRID_W-1; x++) {
                if (walls[y][x]) { tmp[y][x] = 1.0f; continue; }
                float p = atmosphere[y][x];
                float up    = walls[y-1][x] ? p : atmosphere[y-1][x];
                float down  = walls[y+1][x] ? p : atmosphere[y+1][x];
                float left  = walls[y][x-1] ? p : atmosphere[y][x-1];
                float right = walls[y][x+1] ? p : atmosphere[y][x+1];
                float lap = up + down + left + right - 4.0f * p;
                tmp[y][x] = p + D * actual_dt * lap;
            }
        for (int y = 1; y < GRID_H-1; y++)
            for (int x = 1; x < GRID_W-1; x++)
                atmosphere[y][x] = tmp[y][x];
    }
}

static void step_fire(float dt) {
    for (int y = 1; y < GRID_H-1; y++)
        for (int x = 1; x < GRID_W-1; x++) {
            if (fire[y][x] <= 0.0f) continue;
            // Consume O2
            atmosphere[y][x] -= 0.3f * dt * fire[y][x];
            // Burn down
            fire[y][x] -= 0.15f * dt;
            if (fire[y][x] < 0.0f) fire[y][x] = 0.0f;
            // Die without O2
            if (atmosphere[y][x] < 0.3f) fire[y][x] *= 0.9f;
            // Spread slowly
            if (fire[y][x] > 0.5f && randf(0,1) < 0.01f * dt) {
                int ny = y + randi(-1,1), nx = x + randi(-1,1);
                if (ny > 0 && ny < GRID_H-1 && nx > 0 && nx < GRID_W-1 && !walls[ny][nx])
                    fire[ny][nx] = fmaxf(fire[ny][nx], 0.3f);
            }
        }
}

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static std::vector<Particle> particles;
static std::vector<Shockwave> shockwaves;

static void spawn_explosion(float x, float y) {
    int gx = (int)(x / TILE_PX);
    int gy = (int)(y / TILE_PX);

    // Deposit pressure
    for (int dy = -4; dy <= 4; dy++)
        for (int dx = -4; dx <= 4; dx++) {
            int tx = gx + dx, ty = gy + dy;
            if (tx < 0 || tx >= GRID_W || ty < 0 || ty >= GRID_H) continue;
            if (walls[ty][tx]) continue;
            float dist = sqrtf((float)(dx*dx + dy*dy));
            if (dist > 4.0f) continue;
            float falloff = 1.0f - dist / 4.0f;
            atmosphere[ty][tx] += 3.0f * falloff;
            fire[ty][tx] = fminf(fire[ty][tx] + 0.8f * falloff, 1.0f);
        }

    // Fire core particles
    for (int i = 0; i < 200; i++) {
        float angle = randf(0, 2*PI);
        float speed = randf(80, 500);
        float life  = randf(0.2f, 0.8f);
        Particle p;
        p.pos = {x, y};
        p.vel = {cosf(angle)*speed, sinf(angle)*speed};
        p.life = p.max_life = life;
        p.size = randf(3, 10);
        p.r = 255; p.g = (unsigned char)randi(150, 255); p.b = (unsigned char)randi(0, 60);
        particles.push_back(p);
    }

    // Smoke
    for (int i = 0; i < 120; i++) {
        float angle = randf(0, 2*PI);
        float speed = randf(30, 180);
        float life  = randf(1.0f, 3.5f);
        Particle p;
        p.pos = {x, y};
        p.vel = {cosf(angle)*speed, sinf(angle)*speed};
        p.life = p.max_life = life;
        p.size = randf(5, 16);
        unsigned char gray = (unsigned char)randi(40, 120);
        p.r = gray; p.g = gray; p.b = gray;
        particles.push_back(p);
    }

    // Sparks
    for (int i = 0; i < 80; i++) {
        float angle = randf(0, 2*PI);
        float speed = randf(300, 900);
        float life  = randf(0.1f, 0.5f);
        Particle p;
        p.pos = {x, y};
        p.vel = {cosf(angle)*speed, sinf(angle)*speed};
        p.life = p.max_life = life;
        p.size = randf(1, 3);
        p.r = 255; p.g = 255; p.b = (unsigned char)randi(150, 255);
        particles.push_back(p);
    }

    // Shockwave
    Shockwave sw;
    sw.center = {x, y};
    sw.radius = 0;
    sw.max_radius = 300;
    sw.life = sw.max_life = 0.6f;
    shockwaves.push_back(sw);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    InitWindow(SCREEN_W, SCREEN_H, "Breach — Raylib C++ Prototype");
    SetTargetFPS(60);

    init_grid();

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // Input
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            spawn_explosion((float)GetMouseX(), (float)GetMouseY());
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            int gx = GetMouseX() / TILE_PX;
            int gy = GetMouseY() / TILE_PX;
            if (gx > 0 && gx < GRID_W-1 && gy > 0 && gy < GRID_H-1 && !walls[gy][gx])
                fire[gy][gx] = 1.0f;
        }

        // Physics
        step_atmosphere(dt);
        step_fire(dt);

        // Update particles
        for (auto& p : particles) {
            p.life -= dt;
            p.pos.x += p.vel.x * dt;
            p.pos.y += p.vel.y * dt;
            p.vel.x *= 0.96f;
            p.vel.y *= 0.96f;
        }
        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                           [](const Particle& p) { return p.life <= 0; }),
            particles.end());

        // Update shockwaves
        for (auto& sw : shockwaves) {
            sw.life -= dt;
            float t = 1.0f - sw.life / sw.max_life;
            sw.radius = t * sw.max_radius;
        }
        shockwaves.erase(
            std::remove_if(shockwaves.begin(), shockwaves.end(),
                           [](const Shockwave& s) { return s.life <= 0; }),
            shockwaves.end());

        // --- Draw ---
        BeginDrawing();
        ClearBackground({15, 15, 20, 255});

        // Draw atmosphere + fire as colored tiles
        for (int y = 0; y < GRID_H; y++) {
            for (int x = 0; x < GRID_W; x++) {
                int px = x * TILE_PX, py = y * TILE_PX;

                if (walls[y][x]) {
                    DrawRectangle(px, py, TILE_PX, TILE_PX, {60, 65, 75, 255});
                    continue;
                }

                // Atmosphere color: blue-ish tint for pressure
                float atm = atmosphere[y][x];
                float excess = atm - 1.0f;
                unsigned char base = 22;
                unsigned char r = base, g = base, b = base + 5;

                if (excess > 0.02f) {
                    // Overpressure: yellow-orange
                    float t = fminf(excess / 2.0f, 1.0f);
                    r = (unsigned char)(base + t * 200);
                    g = (unsigned char)(base + t * 140);
                    b = base;
                } else if (excess < -0.02f) {
                    // Underpressure: blue-purple
                    float t = fminf(-excess / 0.5f, 1.0f);
                    r = base;
                    g = base;
                    b = (unsigned char)(base + t * 180);
                }

                // Fire overlay
                float f = fire[y][x];
                if (f > 0.01f) {
                    r = (unsigned char)fminf(255, r + f * 255);
                    g = (unsigned char)fminf(255, g + f * 120);
                    b = (unsigned char)fminf(255, b + f * 20);
                }

                DrawRectangle(px, py, TILE_PX, TILE_PX, {r, g, b, 255});
            }
        }

        // Shockwave rings (additive blend)
        BeginBlendMode(BLEND_ADDITIVE);
        for (auto& sw : shockwaves) {
            float alpha_f = (sw.life / sw.max_life) * 0.7f;
            unsigned char alpha = (unsigned char)(255 * alpha_f);
            float thick = fmaxf(2.0f, 6.0f * (sw.life / sw.max_life));
            DrawRing({sw.center.x, sw.center.y},
                     sw.radius - thick, sw.radius + thick,
                     0, 360, 64, {180, 210, 255, alpha});
        }

        // Particles with additive blending (fire glows!)
        for (auto& p : particles) {
            float t = p.life / p.max_life;
            unsigned char alpha = (unsigned char)(255 * t);
            float size = p.size * (0.3f + 0.7f * t);
            DrawCircle((int)p.pos.x, (int)p.pos.y, size,
                       {p.r, p.g, p.b, alpha});
        }

        // Flash
        for (auto& sw : shockwaves) {
            if (sw.life > sw.max_life * 0.6f) {
                float ft = (sw.life - sw.max_life * 0.6f) / (sw.max_life * 0.4f);
                unsigned char fa = (unsigned char)(255 * ft);
                DrawCircle((int)sw.center.x, (int)sw.center.y, 40 * ft,
                           {255, 255, 200, fa});
            }
        }
        EndBlendMode();

        // HUD
        DrawText("Left click: EXPLODE    Right click: FIRE", 10, 10, 20, {200, 200, 200, 255});
        DrawText(TextFormat("Particles: %d", (int)particles.size()), 10, 35, 16, {150, 150, 150, 255});
        DrawFPS(SCREEN_W - 100, 10);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
