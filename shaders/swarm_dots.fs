#version 330
// swarm_dots.fs -- THE swarm dot view (arc #63 P4, renderer/swarm_dots.py).
// RENDER-ONLY, LINEAR HDR into the scene RT, UNLIT (an emissive constant: a
// diagnostic stays visible in the dark), PREMULTIPLIED ALPHA (blend ONE,
// ONE_MINUS_SRC_ALPHA). The one tone-map (world_resolve.fs) does the rest.
//
// A sprite is a disc of radius u_radius_px at its centre plus, when the tip
// differs from the centre, a thin line (half-width u_tick_half_px) from the
// centre to the tip: the unit's heading. Antialiased over one pixel.

in vec2 v_px;
flat in vec2 v_centre;
flat in vec2 v_tip;
flat in vec4 v_rgba;

uniform float u_radius_px;
uniform float u_tick_half_px;

out vec4 finalColor;

void main() {
    float d = length(v_px - v_centre) - u_radius_px;
    vec2 seg = v_tip - v_centre;
    float len2 = dot(seg, seg);
    if (len2 > 1e-6) {
        float t = clamp(dot(v_px - v_centre, seg) / len2, 0.0, 1.0);
        d = min(d, length(v_px - (v_centre + t * seg)) - u_tick_half_px);
    }
    float cover = clamp(0.5 - d, 0.0, 1.0);
    float a = clamp(v_rgba.a, 0.0, 1.0) * cover;
    if (a <= 0.0) discard;
    finalColor = vec4(v_rgba.rgb * a, a);
}
