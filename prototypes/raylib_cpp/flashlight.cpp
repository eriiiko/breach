// Flashlight prototype: dark ship with cursor-driven light reveal
// No shader needed — uses RenderTexture as a light mask
// Mouse wheel = flashlight radius, arrow keys = scroll

#include "raylib.h"
#include "../../cpp/src/raycaster.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

int main() {
    Image shipImg = LoadImage("../../art/ships/chatgptSpaceShip1.png");
    if (shipImg.data == NULL) {
        printf("ERROR: Could not load ship image!\n");
        return 1;
    }

    int imgW = shipImg.width;
    int imgH = shipImg.height;
    int winW = imgW < 1400 ? imgW : 1400;
    int winH = imgH < 900 ? imgH : 900;

    InitWindow(winW, winH, "Breach - Flashlight Prototype");
    SetTargetFPS(60);

    // Lit texture (original)
    Texture2D litTex = LoadTextureFromImage(shipImg);

    // Dark texture (desaturated + very dark)
    Image darkImg = ImageCopy(shipImg);
    for (int y = 0; y < darkImg.height; y++) {
        for (int x = 0; x < darkImg.width; x++) {
            Color c = GetImageColor(shipImg, x, y);
            float gray = 0.299f * c.r + 0.587f * c.g + 0.114f * c.b;
            float r = (gray + (c.r - gray) * 0.15f) * 0.08f;
            float g = (gray + (c.g - gray) * 0.15f) * 0.08f;
            float b = (gray + (c.b - gray) * 0.15f) * 0.08f;
            Color dc;
            dc.r = (unsigned char)(r < 0 ? 0 : (r > 255 ? 255 : r));
            dc.g = (unsigned char)(g < 0 ? 0 : (g > 255 ? 255 : g));
            dc.b = (unsigned char)(b < 0 ? 0 : (b > 255 ? 255 : b));
            dc.a = 255;
            ImageDrawPixel(&darkImg, x, y, dc);
        }
    }
    Texture2D darkTex = LoadTextureFromImage(darkImg);
    UnloadImage(darkImg);

    // --- Emissive sprites (sharp screen pixels) ---
    Image maskImg = LoadImage("../../art/ships/chatgptSpaceShip1-emissive_mask.png");
    Texture2D emissiveTex;
    if (maskImg.data != NULL) {
        printf("Loaded emissive mask\n");
        // Where mask has content, show original ship color boosted
        Image emImg = GenImageColor(imgW, imgH, Color{0, 0, 0, 0});
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW; x++) {
                Color m = GetImageColor(maskImg, x, y);
                float intensity = (float)m.a / 255.0f;
                if (intensity < 0.01f) continue;
                Color c = GetImageColor(shipImg, x, y);
                // Tint toward screen blue: crush red, keep green moderate, boost blue
                Color ec;
                ec.r = (unsigned char)(c.r * 0.3f);
                ec.g = (unsigned char)(c.g * 0.6f);
                ec.b = (unsigned char)(c.b * 1.5f > 255 ? 255 : c.b * 1.5f);
                ec.a = (unsigned char)(intensity * 255);
                ImageDrawPixel(&emImg, x, y, ec);
            }
        }
        emissiveTex = LoadTextureFromImage(emImg);
        UnloadImage(emImg);
        UnloadImage(maskImg);
    } else {
        printf("No emissive mask found\n");
        emissiveTex = LoadTextureFromImage(GenImageColor(imgW, imgH, Color{0, 0, 0, 0}));
    }

    // --- Emissive lightmap (blurred halos, reveals lit image tinted blue) ---
    Image lmapImg = LoadImage("../../art/ships/chatgptSpaceShip1-emissive_bloom_v2.png");
    Texture2D lightmapTex;
    bool hasLightmap = false;
    if (lmapImg.data != NULL) {
        printf("Loaded emissive lightmap\n");
        hasLightmap = true;
        // Load as-is — white pixels with alpha gradient
        // We'll tint it blue at draw time
        lightmapTex = LoadTextureFromImage(lmapImg);
        UnloadImage(lmapImg);
    } else {
        printf("No lightmap found\n");
        lightmapTex = LoadTextureFromImage(GenImageColor(1, 1, Color{0, 0, 0, 0}));
    }

    // Keep a CPU copy for raycasted light masking
    Image shipImgCopy = ImageCopy(shipImg);
    UnloadImage(shipImg);

    // Render texture for the lit layer (we draw the lit image here, then mask it)
    RenderTexture2D litRT = LoadRenderTexture(winW, winH);

    float flashRadius = 120.0f;
    float offX = (imgW > winW) ? (imgW - winW) / 2.0f : 0;
    float offY = (imgH > winH) ? (imgH - winH) / 2.0f : 0;
    bool showHelp = true;
    bool showEmissive = true;
    bool showGlow = true;
    bool showFlashlight = true;

    // Live-tweakable bloom parameters
    float bloomR = 80, bloomG = 120, bloomB = 255;
    float bloomAlpha = 200;
    int editChannel = 0; // 0=none, 1=R, 2=G, 3=B, 4=Alpha

    // --- Generate wall map from ship image ---
    // Load the ship image again for wall detection
    Image wallSrcImg = LoadImage("../../art/ships/chatgptSpaceShip1.png");
    // vector<bool> is special in C++, use vector<char> for .data() access
    std::vector<char> wallMapRaw(imgW * imgH, 0);
    std::vector<float> smokeField(imgW * imgH, 0.0f); // no smoke for now
    std::vector<float> lightMap(imgW * imgH, 0.0f);
    if (wallSrcImg.data != NULL) {
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW; x++) {
                Color c = GetImageColor(wallSrcImg, x, y);
                float brightness = (c.r + c.g + c.b) / 3.0f;
                // Dark pixels = walls/hull, bright pixels = floors/objects
                // Also treat pure black (outside ship) as wall
                bool isWall = (brightness < 45) || (c.r < 30 && c.g < 30 && c.b < 30);
                wallMapRaw[y * imgW + x] = isWall ? 1 : 0;
            }
        }
        UnloadImage(wallSrcImg);
        printf("Wall map generated: %dx%d\n", imgW, imgH);
    }

    // Convert to bool array for raycaster
    bool* wallPtr = new bool[imgW * imgH];
    for (int i = 0; i < imgW * imgH; i++) wallPtr[i] = wallMapRaw[i] != 0;

    Raycaster raycaster;
    raycaster.smoke_absorption = 0.0f;
    bool useRaycast = true; // toggle with R

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // Scroll
        float spd = 300.0f * dt;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) offX += spd;
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) offX -= spd;
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) offY += spd;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) offY -= spd;
        if (offX < 0) offX = 0;
        if (offY < 0) offY = 0;
        if (imgW > winW && offX > imgW - winW) offX = (float)(imgW - winW);
        if (imgH > winH && offY > imgH - winH) offY = (float)(imgH - winH);

        // Flashlight radius
        flashRadius += GetMouseWheelMove() * 15.0f;
        if (flashRadius < 30) flashRadius = 30;
        if (flashRadius > 500) flashRadius = 500;

        Vector2 mouse = GetMousePosition();

        // Toggles
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_E)) showEmissive = !showEmissive;
        if (IsKeyPressed(KEY_G)) showGlow = !showGlow;
        if (IsKeyPressed(KEY_F)) showFlashlight = !showFlashlight;
        if (IsKeyPressed(KEY_R)) useRaycast = !useRaycast;

        // Bloom color editing: hold key + scroll
        float wheel = GetMouseWheelMove();
        if (IsKeyDown(KEY_ONE)) { bloomR += wheel * 10; if (bloomR < 0) bloomR = 0; if (bloomR > 255) bloomR = 255; editChannel = 1; }
        else if (IsKeyDown(KEY_TWO)) { bloomG += wheel * 10; if (bloomG < 0) bloomG = 0; if (bloomG > 255) bloomG = 255; editChannel = 2; }
        else if (IsKeyDown(KEY_THREE)) { bloomB += wheel * 10; if (bloomB < 0) bloomB = 0; if (bloomB > 255) bloomB = 255; editChannel = 3; }
        else if (IsKeyDown(KEY_FOUR)) { bloomAlpha += wheel * 10; if (bloomAlpha < 0) bloomAlpha = 0; if (bloomAlpha > 255) bloomAlpha = 255; editChannel = 4; }
        else { editChannel = 0; }

        // Print values when P is pressed (so you can copy them)
        if (IsKeyPressed(KEY_P)) {
            printf("Bloom color: {%d, %d, %d, %d}\n", (int)bloomR, (int)bloomG, (int)bloomB, (int)bloomAlpha);
        }

        // --- Render the lit layer into the render texture with a circular mask ---
        BeginTextureMode(litRT);
        ClearBackground(BLANK);

        // Draw a white gradient circle (this will be our alpha mask)
        // Using multiple concentric circles for smooth falloff
        int steps = 80;
        for (int i = steps; i >= 0; i--) {
            float t = (float)i / steps;
            float r = flashRadius * t;
            // Smooth falloff: bright in center, fading outward
            float intensity = 1.0f - t;
            intensity = intensity * intensity; // quadratic falloff for softer edge
            unsigned char alpha = (unsigned char)(intensity * 255);
            Color col = {alpha, alpha, alpha, 255};
            DrawCircleV(mouse, r, col);
        }

        EndTextureMode();

        // --- Draw everything ---
        BeginDrawing();
        ClearBackground(BLACK);

        Rectangle srcRect = {offX, offY, (float)winW, (float)winH};
        Rectangle dstRect = {0, 0, (float)winW, (float)winH};

        // 1. Draw dark layer as base
        DrawTexturePro(darkTex, srcRect, dstRect, Vector2{0, 0}, 0, WHITE);

        // 3. Flashlight — raycasted or circular
        if (showFlashlight) {
            if (useRaycast) {
                // Cast rays from cursor position in image space
                std::memset(lightMap.data(), 0, imgW * imgH * sizeof(float));
                float imgMX = mouse.x + offX;
                float imgMY = mouse.y + offY;
                LightSource flashlight;
                flashlight.x = imgMX;
                flashlight.y = imgMY;
                flashlight.max_range = flashRadius;
                flashlight.intensity = 1.5f;
                flashlight.angle_spread = 2.0f * 3.14159265f;
                flashlight.falloff = Falloff::UNIFORM;
                flashlight.jitter = 0.01f;
                raycaster.cast_source(flashlight, lightMap.data(), smokeField.data(),
                    wallPtr, imgH, imgW);

                // Draw lit texture where rays reach, using scissor regions
                // Group lit pixels into horizontal spans for efficiency
                int startX = (int)offX, startY = (int)offY;
                int endY = std::min(startY + winH, imgH);
                int endX = std::min(startX + winW, imgW);

                for (int iy = startY; iy < endY; iy++) {
                    int sy = iy - startY;
                    // Find spans of lit pixels in this row
                    int spanStart = -1;
                    float spanMaxLight = 0;
                    for (int ix = startX; ix <= endX; ix++) {
                        float lv = (ix < endX) ? lightMap[iy * imgW + ix] : 0;
                        if (lv > 0.01f) {
                            if (spanStart < 0) { spanStart = ix - startX; spanMaxLight = 0; }
                            if (lv > spanMaxLight) spanMaxLight = lv;
                        } else if (spanStart >= 0) {
                            // End of span — draw lit texture for this span
                            if (spanMaxLight > 1.0f) spanMaxLight = 1.0f;
                            unsigned char alpha = (unsigned char)(spanMaxLight * 255);
                            int spanW = (ix - startX) - spanStart;
                            BeginScissorMode(spanStart, sy, spanW, 1);
                            DrawTexturePro(litTex, srcRect, dstRect, Vector2{0, 0}, 0,
                                Color{255, 255, 255, alpha});
                            EndScissorMode();
                            spanStart = -1;
                        }
                    }
                }
            } else {
                // Fallback: circular rings (old method)
                int rings = 40;
                for (int i = 0; i < rings; i++) {
                    float t = (float)i / rings;
                    float outerR = flashRadius * (1.0f - t);
                    float intensity = t * t;
                    unsigned char alpha = (unsigned char)(intensity * 255);
                    BeginScissorMode(
                        (int)(mouse.x - outerR), (int)(mouse.y - outerR),
                        (int)(outerR * 2), (int)(outerR * 2)
                    );
                    DrawTexturePro(litTex, srcRect, dstRect, Vector2{0, 0}, 0, Color{255, 255, 255, alpha});
                    EndScissorMode();
                }
            }
        }

        // 4. Draw emissive mask (sharp screens, reveals original ship colors)
        if (showEmissive)
            DrawTexturePro(emissiveTex, srcRect, dstRect, Vector2{0, 0}, 0, WHITE);

        // 5. Blue glow from lightmap (blurred halos, additive blue)
        if (showGlow && hasLightmap) {
            BeginBlendMode(BLEND_ADDITIVE);
            DrawTexturePro(lightmapTex, srcRect, dstRect, Vector2{0, 0}, 0,
                Color{(unsigned char)bloomR, (unsigned char)bloomG, (unsigned char)bloomB, (unsigned char)bloomAlpha});
            EndBlendMode();
        }

        // HUD
        DrawFPS(winW - 100, 10);
        DrawText(TextFormat("E:%s G:%s F:%s",
            showEmissive ? "ON" : "off", showGlow ? "ON" : "off",
            showFlashlight ? "ON" : "off"), 10, 10, 20, GREEN);
        // Show bloom color with channel highlight
        DrawText(TextFormat("Bloom: R=%d G=%d B=%d A=%d",
            (int)bloomR, (int)bloomG, (int)bloomB, (int)bloomAlpha), 10, 32, 16,
            editChannel == 1 ? RED : editChannel == 2 ? GREEN : editChannel == 3 ? BLUE :
            editChannel == 4 ? YELLOW : LIGHTGRAY);
        // Color preview swatch
        DrawRectangle(350, 30, 30, 20, Color{(unsigned char)bloomR, (unsigned char)bloomG, (unsigned char)bloomB, 255});

        if (showHelp) {
            int hy = winH - 80;
            DrawRectangle(0, hy - 5, 420, 150, Color{0, 0, 0, 180});
            DrawText("Mouse wheel: flashlight radius", 10, hy, 14, LIGHTGRAY);
            DrawText("Arrow keys / WASD: scroll", 10, hy + 18, 14, LIGHTGRAY);
            DrawText("E: emissive, G: blue glow, F: flashlight", 10, hy + 36, 14, LIGHTGRAY);
            DrawText("Hold 1/2/3/4 + scroll: bloom R/G/B/Alpha", 10, hy + 54, 14, YELLOW);
            DrawText("P: print bloom values to console", 10, hy + 72, 14, LIGHTGRAY);
            DrawText("H: toggle help", 10, hy + 90, 14, LIGHTGRAY);
        }

        EndDrawing();
    }

    UnloadTexture(litTex);
    UnloadTexture(darkTex);
    UnloadTexture(emissiveTex);
    UnloadTexture(lightmapTex);
    UnloadRenderTexture(litRT);
    CloseWindow();
    return 0;
}
