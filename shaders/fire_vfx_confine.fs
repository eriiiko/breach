#version 330
// fire_vfx_confine.fs -- flame VFX pass "confine" (G3 F2, design §3.4 step 2,
// §13.2) = twin confine: vorticity confinement (Fedkiw, Stam & Jensen 2001)
// plus curl noise (Bridson, Hourihan & Nordenstam 2007) scaled by the local
// fire intensity, both accelerations applied over dt; solid cells at rest.
//   w      = (d uy/dx - d ux/dy), central, neighbours beyond the box
//            replicated (the twin's 'replicate');
//   N      = grad|w| / |grad|w||;  f_conf = eps h (N x w) = eps h (Ny w, -Nx w)
//   f_curl = curl_amp * I_local * curl(psi), psi = sum_k a_k cos(k.x + phi_k(t))
// The noise phases phi_k(t) = (omega_k t + phase_k) mod 2 pi come from the
// CPU in float64 every substep (renderer/fire_vfx_ref.curl_mode_phases), so
// the float32 argument stays small however long the game runs.
// Target: the velocity field (xy = u, z = p, w = rhs pass through).

// @include fire_vfx_common.glsl

const int MAX_MODES = 32;

uniform sampler2D u_vel;
uniform float u_dt;
uniform float u_vort_eps;
uniform float u_curl_amp;           // tiles/s^2 at local I = 1
uniform vec4 u_modes[MAX_MODES];    // kx, ky (rad/tile), amplitude, 0
uniform float u_mode_phase[MAX_MODES];
uniform int u_n_modes;

out vec4 finalColor;

vec2 U(ivec2 c) { return texelFetch(u_vel, c, 0).xy; }

float vort(ivec2 c) {
    float h2 = 2.0 / u_m;
    return ((U(nbr(c, D_E)).y - U(nbr(c, D_W)).y)
            - (U(nbr(c, D_S)).x - U(nbr(c, D_N)).x)) / h2;
}

void main() {
    ivec2 c = cell_here();
    find_box(c);
    vec4 v = texelFetch(u_vel, c, 0);
    if (cell_solid(c)) {
        finalColor = vec4(0.0, 0.0, v.z, v.w);
        return;
    }
    float hc = 1.0 / u_m;
    float w = vort(c);
    float gx = (abs(vort(nbr(c, D_E))) - abs(vort(nbr(c, D_W)))) / (2.0 * hc);
    float gy = (abs(vort(nbr(c, D_S))) - abs(vort(nbr(c, D_N)))) / (2.0 * hc);
    float gn = sqrt(gx * gx + gy * gy) + 1e-9;
    vec2 f = u_vort_eps * hc * vec2((gy / gn) * w, -(gx / gn) * w);
    vec2 p = cell_centre_tiles(c);
    vec2 cn = vec2(0.0);
    for (int k = 0; k < MAX_MODES; ++k) {
        if (k >= u_n_modes) break;
        vec4 md = u_modes[k];
        float cc = md.z * cos(md.x * p.x + md.y * p.y + u_mode_phase[k]);
        cn += vec2(md.y * cc, -md.x * cc);
    }
    float i_loc = tile_bilinear(u_wind, p).z;
    f += u_curl_amp * i_loc * cn;
    finalColor = vec4(v.xy + u_dt * f, v.z, v.w);
}
