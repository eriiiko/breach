#version 330
// fire_vfx_inject.fs -- flame VFX stage 4b (G3, design §3.3, §3.4 step 4):
// add s * puff * dt of flame density at T_src, premultiplied:
// R += s puff dt, RT += s puff dt T_src -- a mix, so it never exceeds T_src.
// The source is the PER-TILE texture (sampled nearest, §13.14) through the
// cover rule: a whole tile (permeable fuel) or strips along faces (solid
// fuel next door). Spec: renderer/fire_vfx_ref.py inject + rasterize_sources
// + puff_factor. Target: the flame field (x = R, y = R*T_vfx in K, zw = 0).

// @include fire_vfx_common.glsl

uniform sampler2D u_self;     // the flame field, front buffer
uniform float u_t;            // the noise clock at this substep, s
uniform float u_dt;           // s
uniform float u_strip;        // solid fuel's strip depth, of a tile
uniform float u_puff_amp;
uniform float u_puff_wobble;  // rad

out vec4 finalColor;

void main() {
    ivec2 c = cell_here();
    vec4 v = texelFetch(u_self, c, 0);
    ivec2 t = tile_of(c);
    vec4 a = tile_fetch(u_src_a, t);
    if (source_covers(int(a.w + 0.5), tile_frac(c), u_strip)) {
        vec4 b = tile_fetch(u_src_b, t);
        float add = a.x * puff_factor(b.x, b.y, u_t, u_puff_amp,
                                      u_puff_wobble) * u_dt;
        v.x += add;
        v.y += add * a.z;
    }
    finalColor = v;
}
