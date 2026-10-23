#version 330
// fire_vfx_debug.fs -- the flame VFX's DEBUG view (G3 F1; dev only). Drawn
// in the DISPLAY stage after the resolve, like the other debug colormaps:
// its colours are display values, alpha-blended over the picture.
//   u_mode 1 = the SOURCE texture through the cover rule (where flame is
//              injected), coloured by T_src through the blackbody GPU LUT,
//              opacity by the source rate s;
//   u_mode 2 = the flame density R the passes produced, coloured by its
//              T_vfx = RT / R through the same LUT.
// The quad covers the whole world with fragTexCoord (0,0) at the north-west
// corner, so a fragment's position in tiles is fragTexCoord * u_grid.

// @include fire_vfx_common.glsl

in vec2 fragTexCoord;

uniform int u_mode;
uniform sampler2D u_flame;    // the flame field (cells, m per tile)
uniform sampler2D u_lut;      // blackbody GPU LUT: rgb chroma, a intensity
uniform vec2 u_lut_k;         // its Kelvin domain (floor, ceil)
uniform float u_lut_n;        // its size
uniform float u_strip;
uniform float u_s_vis;        // s that draws fully opaque (mode 1)
uniform float u_r_vis;        // R that draws fully opaque (mode 2)

out vec4 finalColor;

vec3 lut_chroma(float kelvin) {
    float x = clamp((kelvin - u_lut_k.x) / (u_lut_k.y - u_lut_k.x), 0.0, 1.0);
    return texture(u_lut, vec2((x * (u_lut_n - 1.0) + 0.5) / u_lut_n, 0.5)).rgb;
}

void main() {
    vec2 p = fragTexCoord * u_grid;              // tiles, north-west origin
    if (u_mode == 1) {
        ivec2 t = ivec2(floor(p));
        vec4 a = tile_fetch(u_src_a, t);
        if (!source_covers(int(a.w + 0.5), fract(p), u_strip) || a.x <= 0.0) {
            finalColor = vec4(0.0);
            return;
        }
        finalColor = vec4(lut_chroma(a.z),
                          0.35 + 0.5 * clamp(a.x / u_s_vis, 0.0, 1.0));
    } else {
        ivec2 c = ivec2(floor(p * u_m));
        vec4 v = texelFetch(u_flame, c, 0);
        if (v.x <= 0.0) {
            finalColor = vec4(0.0);
            return;
        }
        finalColor = vec4(lut_chroma(v.y / v.x),
                          0.85 * clamp(v.x / u_r_vis, 0.0, 1.0));
    }
}
