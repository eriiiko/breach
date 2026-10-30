#version 330
// fire_vfx_debug.fs -- the flame VFX's DEBUG views (G3 F1/F2; dev only).
// Drawn in the DISPLAY stage after the resolve, like the other debug
// colormaps: its colours are display values, alpha-blended over the picture.
//   u_mode 1 = the SOURCE texture through the cover rule (where flame is
//              injected), coloured by T_src through the blackbody GPU LUT,
//              opacity by the source rate s;
//   u_mode 2 = the flame density R (grey, opacity by R / u_r_vis);
//   u_mode 3 = the flame temperature T_vfx = RT / R through the blackbody
//              LUT's chroma, brightness from u_t_lo..u_t_hi, opacity by R
//              (as mode 2, so the faint cooled tail fades out);
//   u_mode 4 = the velocity u: hue = direction, value = |u| / u_u_vis;
//   u_mode 5 = the pressure p: red > 0, blue < 0, |p| / u_p_vis.
// The quad covers the whole world with fragTexCoord (0,0) at the north-west
// corner, so a fragment's position in tiles is fragTexCoord * u_grid.

// @include fire_vfx_common.glsl

in vec2 fragTexCoord;

uniform int u_mode;
uniform sampler2D u_flame;    // the flame field (cells, m per tile)
uniform sampler2D u_vel;      // the velocity field (xy = u, z = p)
uniform sampler2D u_lut;      // blackbody GPU LUT: rgb chroma, a intensity
uniform vec2 u_lut_k;         // its Kelvin domain (floor, ceil)
uniform float u_lut_n;        // its size
uniform float u_strip;
uniform float u_s_vis;        // s that draws fully opaque (mode 1)
uniform float u_r_vis;        // R that draws fully opaque (modes 2, 3)
uniform float u_u_vis;        // |u| (tiles/s) at full value (mode 4)
uniform float u_p_vis;        // |p| at full colour (mode 5)
uniform float u_t_lo;         // T_vfx (K) drawn black (mode 3)
uniform float u_t_hi;         // T_vfx (K) drawn at full brightness

out vec4 finalColor;

vec3 lut_chroma(float kelvin) {
    float x = clamp((kelvin - u_lut_k.x) / (u_lut_k.y - u_lut_k.x), 0.0, 1.0);
    return texture(u_lut, vec2((x * (u_lut_n - 1.0) + 0.5) / u_lut_n, 0.5)).rgb;
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main() {
    vec2 p = fragTexCoord * u_grid;              // tiles, north-west origin
    ivec2 c = ivec2(floor(p * u_m));
    if (u_mode == 1) {
        ivec2 t = ivec2(floor(p));
        vec4 a = tile_fetch(u_src_a, t);
        if (!source_covers(int(a.w + 0.5), fract(p), u_strip) || a.x <= 0.0) {
            finalColor = vec4(0.0);
            return;
        }
        finalColor = vec4(lut_chroma(a.z),
                          0.35 + 0.5 * clamp(a.x / u_s_vis, 0.0, 1.0));
    } else if (u_mode == 2 || u_mode == 3) {
        vec4 v = texelFetch(u_flame, c, 0);
        if (v.x <= 0.0) {
            finalColor = vec4(0.0);
            return;
        }
        float op = 0.9 * clamp(v.x / u_r_vis, 0.0, 1.0);
        if (u_mode == 2) {
            finalColor = vec4(vec3(1.0), op);
        } else {
            float T = v.y / v.x;
            float b = clamp((T - u_t_lo) / (u_t_hi - u_t_lo), 0.0, 1.0);
            finalColor = vec4(lut_chroma(T) * (0.15 + 0.85 * b), op);
        }
    } else if (u_mode == 4) {
        vec4 v = texelFetch(u_vel, c, 0);
        float sp = length(v.xy);
        if (sp <= 1e-4) {
            finalColor = vec4(0.0);
            return;
        }
        float hue = atan(v.y, v.x) / TWO_PI + 0.5;
        float val = clamp(sp / u_u_vis, 0.0, 1.0);
        finalColor = vec4(hsv2rgb(vec3(hue, 0.85, 0.25 + 0.75 * val)),
                          0.25 + 0.6 * val);
    } else {
        float pr = texelFetch(u_vel, c, 0).z;
        float a = clamp(abs(pr) / u_p_vis, 0.0, 1.0);
        if (a <= 1e-3) {
            finalColor = vec4(0.0);
            return;
        }
        finalColor = vec4(pr > 0.0 ? vec3(1.0, 0.25, 0.15)
                                   : vec3(0.2, 0.45, 1.0), 0.15 + 0.75 * a);
    }
}
