#version 330
// particle_debris.fs -- THE particle primitive's DEBRIS stage (graphics G4
// P4, renderer/particles.py ParticleDraw.draw_debris). RENDER-ONLY, LINEAR
// HDR into the scene RT, PREMULTIPLIED ALPHA (blend ONE, ONE_MINUS_SRC_ALPHA
// on RGB and alpha) -- a dark chip must darken the floor, which an additive
// draw never can. The one tone-map (world_resolve.fs) does the rest.
//
// A chip is lit the props' way: the light arriving at its tile is read from
// light_tex_a/b through the lit-3D block (renderer/lit3d.py
// _FIELD_SAMPLE_GLSL, spliced in below with _COMMON_GLSL: the smoke's cast
// shadow and the light curve included), so a chip in the blast's smoke
// shadow is dark. Colour = albedo x (ambient + exposure x light): a flat
// thing on the floor, no N.L.

in vec2 v_local;
in float v_half;
in float v_alpha;
in vec2 v_world_px;

uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: A = dir.y
uniform vec2  u_world_px;     // (world_px_w, world_px_h) for the field UV
uniform float u_light_z;      // the lit-3D block's grazing key (unused: flat)
uniform vec3  u_ambient;      // the ship's ambient floor
uniform float u_light_gain;   // the ship's render exposure
uniform vec3  u_albedo;       // [render.debris] albedo, linear

out vec4 finalColor;

// <COMMON>

void main() {
    // the chip lies on the floor plane: the lit-3D block's Y-up world frame
    vec3 fragWorldPos = vec3(v_world_px.x, 0.0, v_world_px.y);
    vec3 fragWorldNormal = vec3(0.0, 1.0, 0.0);
// <FIELD_SAMPLE>

    vec3 light_rgb = incoming_rgb * u_light_gain;
    light_rgb *= lightCurveGain(light_rgb);
    vec3 albedo = lightCurveAlbedo(u_albedo, light_rgb);
    vec3 lit = albedo * (u_ambient + light_rgb);

    // the square's edge, antialiased over one pixel
    vec2 e = v_half - abs(v_local);
    float cover = clamp(min(e.x, e.y) + 0.5, 0.0, 1.0);
    float a = clamp(v_alpha, 0.0, 1.0) * cover;
    finalColor = vec4(lit * a, a);
}
