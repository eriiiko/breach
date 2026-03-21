// Flashlight prototype: dark ship with cursor-driven light reveal
// No shader needed — uses RenderTexture as a light mask
// Mouse wheel = flashlight radius, arrow keys = scroll

#include "raylib.h"
#include <cstdio>

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
                Color ec;
                ec.r = (unsigned char)(c.r * 1.3f > 255 ? 255 : c.r * 1.3f);
                ec.g = (unsigned char)(c.g * 1.3f > 255 ? 255 : c.g * 1.3f);
                ec.b = (unsigned char)(c.b * 1.3f > 255 ? 255 : c.b * 1.3f);
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
    Image lmapImg = LoadImage("../../art/ships/chatgptSpaceShip1-emissive_bloom.png");
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

        // 3. Draw the lit image, but multiply its alpha by the light mask
        // Approach: draw lit image with the render texture as alpha
        // We'll use BLEND_MULTIPLIED: lit * mask

        // First draw the lit image into position
        // Use the light mask's luminance as alpha for the lit layer
        // Simple approach: draw lit layer, then use the mask to cut it

        // Actually, the simplest Raylib approach:
        // Draw the lit texture tinted by each circle ring (radial light)
        if (showFlashlight) {
            int rings = 40;
            for (int i = 0; i < rings; i++) {
                float t = (float)i / rings;
                float outerR = flashRadius * (1.0f - t);
                float innerR = flashRadius * (1.0f - (float)(i + 1) / rings);
                if (innerR < 0) innerR = 0;

                // Alpha fades toward edge
                float intensity = t; // brighter toward center
                intensity = intensity * intensity;
                unsigned char alpha = (unsigned char)(intensity * 255);

                // Use scissor mode to only draw within this ring
                float cx = mouse.x;
                float cy = mouse.y;

                // Draw a quad covering the ring area, with the lit texture
                // This is hacky but works without shaders
                BeginScissorMode(
                    (int)(cx - outerR), (int)(cy - outerR),
                    (int)(outerR * 2), (int)(outerR * 2)
                );

                Color tint = {255, 255, 255, alpha};
                DrawTexturePro(litTex, srcRect, dstRect, Vector2{0, 0}, 0, tint);

                EndScissorMode();
            }
        }

        // 4. Draw emissive mask (sharp screens, reveals original ship colors)
        if (showEmissive)
            DrawTexturePro(emissiveTex, srcRect, dstRect, Vector2{0, 0}, 0, WHITE);

        // 5. Blue glow from lightmap (blurred halos, additive blue)
        if (showGlow && hasLightmap) {
            BeginBlendMode(BLEND_ADDITIVE);
            DrawTexturePro(lightmapTex, srcRect, dstRect, Vector2{0, 0}, 0, Color{80, 120, 255, 200});
            EndBlendMode();
        }

        // HUD
        DrawFPS(winW - 100, 10);
        DrawText(TextFormat("R:%.0f E:%s G:%s F:%s", flashRadius,
            showEmissive ? "ON" : "off", showGlow ? "ON" : "off",
            showFlashlight ? "ON" : "off"), 10, 10, 20, GREEN);

        if (showHelp) {
            int hy = winH - 80;
            DrawRectangle(0, hy - 5, 350, 110, Color{0, 0, 0, 180});
            DrawText("Mouse wheel: flashlight radius", 10, hy, 14, LIGHTGRAY);
            DrawText("Arrow keys / WASD: scroll", 10, hy + 18, 14, LIGHTGRAY);
            DrawText("E: emissive, G: blue glow, F: flashlight", 10, hy + 36, 14, LIGHTGRAY);
            DrawText("H: toggle help", 10, hy + 54, 14, LIGHTGRAY);
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
