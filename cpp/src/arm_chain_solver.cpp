// ============================================================================
// ArmChainSolver (#106 P6c) -- see arm_chain_solver.h for the contract, the
// technique credits (Macklin 2016 XPBD, Macklin 2019 Small steps, Bishop 1975,
// Yekutieli 2005 I/II, Macklin 2014 unified particles) and the rule: this file
// TRANSCRIBES docs/octopus_integer_reference/chain_ref_q.py `World.tick`,
// function for function, rounding for rounding; tests/test_octopus_arm_chain_
// solver.py holds it there at tolerance 0. Each function below names the
// reference function it transcribes. Pure integer: the kit's
// shr_round_away / div_round_away / mul_shr_away (sra / dra / msa below),
// sqrt_q16, atan2_q16, sin_q16, cos_q16, floordiv_q. A signed left shift is
// written as a multiply (`shl`): Python's `x << s` on a negative x.
// ============================================================================
#include "arm_chain_solver.h"

#include <vector>

#include "fixed_point.h"

namespace {
#include "activation_table.inc"   // ACT_TABLE_X_MAX, ACT_TABLE_STEP_SHIFT, ACT_TABLE_SAMPLES, ACT_TABLE_DATA

constexpr bool act_table_monotone() {
    for (int i = 1; i < ACT_TABLE_SAMPLES; ++i) {
        if (ACT_TABLE_DATA[i] < ACT_TABLE_DATA[i - 1]) return false;
    }
    return true;
}
}  // namespace

static_assert(ACT_TABLE_X_MAX == 8 && ACT_TABLE_STEP_SHIFT == 10,
              "activation_table.inc's grid differs from the lookup law's "
              "(clamp to +-8, index = a shift by 10): regenerate with "
              "tools/gen_activation_table.py --write");
static_assert(ACT_TABLE_SAMPLES == ((2 * ACT_TABLE_X_MAX) << (16 - ACT_TABLE_STEP_SHIFT)) + 1,
              "the activation table must sample [-8, 8] at step 1/64, both ends");
static_assert(sizeof(ACT_TABLE_DATA) == sizeof(int32_t) * ACT_TABLE_SAMPLES,
              "the activation table must hold ACT_TABLE_SAMPLES int32");
static_assert(ACT_TABLE_DATA[0] == 0 && ACT_TABLE_DATA[ACT_TABLE_SAMPLES - 1] == 65536,
              "the activation table is S(x) in Q16: 0 at x = -8, 1.0 at x = +8");
static_assert(act_table_monotone(), "S(x) = (1 + tanh x) / 2 is non-decreasing");

namespace arm_chain {
namespace {

namespace fp = fixedpoint;

constexpr int CHAIN_SHIFT = fp::CHAIN_SHIFT;              // 24
constexpr int U_SHIFT = 30;
constexpr int64_t U_ONE = (int64_t)1 << U_SHIFT;
constexpr int FINE_BITS = 8;
constexpr int DIST_J_SHIFT = 40;
constexpr int ATTACH_PRE = 38;
constexpr int ATTACH_J_SHIFT = CHAIN_SHIFT + ATTACH_PRE - 32;   // 30
constexpr int KW_PRE_SHIFT = 48;
constexpr int G_SHIFT = 16;
constexpr int G_PRE = G_SHIFT + 32 - CHAIN_SHIFT;               // 24
constexpr int DLAM_PRE = 8;
constexpr int64_t D_MAX_Q16 = 205887;                           // the kit's quantize(pi)
constexpr int64_t OPC_FLOOR = 16;
constexpr int64_t N0_MIN = 1073742;
constexpr int64_t ALT_X_LIMIT = 966367642;
constexpr int64_t PI_Q24 = 52707179;
constexpr int64_t TWO_PI_Q24 = 2 * PI_Q24;
constexpr int64_t WAVE_CLOCK_SPAN = 4;
constexpr int64_t FADE_BACK_NUM = 2, FADE_BACK_DEN = 5;
constexpr int64_t FORCE_SAT_Q8 = (int64_t)1 << 30;
constexpr int64_t ACT_X_MAX_Q16 = (int64_t)ACT_TABLE_X_MAX << 16;

static_assert(FINE_BITS == fp::CHAIN_FINE_BITS,
              "the chain's fine corrections are CHAIN_FINE_BITS below a count");
static_assert(ATTACH_J_SHIFT == 30, "the attachment impulse is Q30 kg m");

inline int64_t sra(int64_t x, int s) { return fp::shr_round_away(x, s); }
inline int64_t dra(int64_t n, int64_t d) { return fp::div_round_away(n, d); }
inline int64_t msa(int64_t a, int64_t b, int s) { return fp::mul_shr_away(a, b, s); }
inline int64_t isqrt(int64_t x) { return (int64_t)fp::sqrt_q16(x); }
inline int64_t shl(int64_t x, int s) { return x * ((int64_t)1 << s); }

struct V3 { int64_t x = 0, y = 0, z = 0; };

// _move: a FINE (Q32) correction through the node's remainder.
inline void move(int64_t* P, int64_t* R, int64_t fx, int64_t fy, int64_t fz) {
    int64_t t = R[0] + fx;
    int64_t i = sra(t, FINE_BITS);
    P[0] += i;
    R[0] = t - shl(i, FINE_BITS);
    t = R[1] + fy;
    i = sra(t, FINE_BITS);
    P[1] += i;
    R[1] = t - shl(i, FINE_BITS);
    t = R[2] + fz;
    i = sra(t, FINE_BITS);
    P[2] += i;
    R[2] = t - shl(i, FINE_BITS);
}

// _ip: a Q24 vector along a Q30 unit vector.
inline int64_t ip(const int64_t* d, const V3& e) {
    return sra(d[0] * e.x + d[1] * e.y + d[2] * e.z, U_SHIFT);
}

// _reject: v - (v . t) t.
inline V3 reject(const V3& v, const V3& t) {
    const int64_t s = sra(v.x * t.x + v.y * t.y + v.z * t.z, U_SHIFT);
    return V3{v.x - sra(s * t.x, U_SHIFT), v.y - sra(s * t.y, U_SHIFT),
              v.z - sra(s * t.z, U_SHIFT)};
}

inline V3 cross_q30(const V3& a, const V3& b) {
    return V3{sra(a.y * b.z - a.z * b.y, U_SHIFT), sra(a.z * b.x - a.x * b.z, U_SHIFT),
              sra(a.x * b.y - a.y * b.x, U_SHIFT)};
}

// _rebuild: the PBD velocity rebuild with both remainders carried; `lo`
// clamps the rebuilt part at 0 from below (floor / low-side wall), `hi` from
// above (high-side wall).
inline int64_t rebuild(int64_t v_pred, int64_t dp, int64_t res, int64_t disp, int64_t inv_dt,
                       bool lo, bool hi) {
    int64_t pbd = inv_dt * dp + sra(inv_dt * res, FINE_BITS);
    if (lo && pbd < 0) pbd = 0;
    if (hi && pbd > 0) pbd = 0;
    return pbd + (v_pred - inv_dt * disp);
}

// _friction: position-based Coulomb on the two tangential axes a0, a1.
inline void friction(int64_t* P, int64_t* R, const int64_t* Pp, int64_t mu_q16, int64_t pen_f,
                     int a0, int a1) {
    const int64_t dx = shl(P[a0] - Pp[a0], FINE_BITS) + R[a0];
    const int64_t dy = shl(P[a1] - Pp[a1], FINE_BITS) + R[a1];
    const int64_t s = isqrt(dx * dx + dy * dy);
    if (s == 0) return;
    const int64_t ratio = dra(shl(mu_q16 * pen_f, 14), s);
    int64_t keep = U_ONE - ratio;
    if (keep >= U_ONE) return;
    if (keep < 0) keep = 0;
    const int ax[2] = {a0, a1};
    const int64_t dd[2] = {dx, dy};
    for (int k = 0; k < 2; ++k) {
        const int c = ax[k];
        const int64_t t = keep ? msa(dd[k], keep, U_SHIFT) : 0;
        const int64_t i = sra(t, FINE_BITS);
        P[c] = Pp[c] + i;
        R[c] = t - shl(i, FINE_BITS);
    }
}

// _force_q8: |J| / dt^2, Q8 N, components saturated.
inline int64_t force_q8(const int64_t J[3], int64_t inv_dt2) {
    int64_t s = 0;
    for (int c = 0; c < 3; ++c) {
        int64_t f = sra(J[c] * inv_dt2, ATTACH_J_SHIFT - 8);
        if (f > FORCE_SAT_Q8) f = FORCE_SAT_Q8;
        else if (f < -FORCE_SAT_Q8) f = -FORCE_SAT_Q8;
        s += f * f;
    }
    return isqrt(s);
}

// Topology: the static blocked view (None in the reference = has_topo 0).
struct Topo {
    bool on = false;
    const uint8_t* grid = nullptr;
    int h = 0, w = 0;
    int64_t T = 1, ox = 0, oy = 0;
    const int32_t* rects = nullptr;
    int U = 0;
    bool blocked(int64_t ix, int64_t iy) const {
        if (ix < 0 || iy < 0 || iy >= h || ix >= w) return true;
        return grid[iy * (int64_t)w + ix] != 0;
    }
    int64_t tile_x(int64_t x) const { return fp::floordiv_q(x - ox, T); }
    int64_t tile_y(int64_t y) const { return fp::floordiv_q(y - oy, T); }
};

struct Touch { bool lx = false, hx = false, ly = false, hy = false, any = false; };

// _wall_touch
inline Touch wall_touch(const Topo& t, const int64_t* q) {
    Touch r;
    const int64_t ix = t.tile_x(q[0]), iy = t.tile_y(q[1]);
    const bool inside = t.blocked(ix, iy);
    r.lx = t.blocked(t.tile_x(q[0] - 1), iy);
    r.hx = t.blocked(t.tile_x(q[0] + 1), iy);
    r.ly = t.blocked(ix, t.tile_y(q[1] - 1));
    r.hy = t.blocked(ix, t.tile_y(q[1] + 1));
    r.any = inside || r.lx || r.hx || r.ly || r.hy;
    return r;
}

// The working state of one environment, int64 scratch (sized once per call).
struct Body {
    int64_t x[3] = {0, 0, 0};
    int64_t v[3] = {0, 0, 0};
    int64_t th = 0, om = 0;
    int64_t res[3] = {0, 0, 0};
    int64_t res_th = 0;
    int64_t w = 0, winvI = 0, cb = 0, vref = 0, u = 0, y = 0, mu = 0, coll_r = 0;
    int drag_mode = 0;
    bool gravity = true, floor = true;
    uint8_t contact = 0;
    int64_t prev_x[3] = {0, 0, 0};
    int64_t prev_th = 0;
    int64_t disp[3] = {0, 0, 0};
    int64_t dth = 0;
};

struct Row {
    int n = 0, nseg = 0, nb = 0, n_sub = 0, n_pass = 0;
    int64_t inv_dt = 0, inv_dt2 = 0, L = 0, k_dist = 0, gamma = 0, onepg = 0, g_dv = 0;
    int64_t plane_min = 1;
    int atan2_shift = 0;
    const int64_t* rn = nullptr;
    int64_t s_node(int i) const { return rn[i * RN_N + RN_S_NODE_Q24]; }
    int64_t w(int i) const { return rn[i * RN_N + RN_W_Q32]; }
    int64_t k_per(int i) const { return rn[i * RN_N + RN_K_PER_Q16]; }
    int64_t k_tan(int i) const { return rn[i * RN_N + RN_K_TAN_Q16]; }
    int64_t rest0(int i) const { return rn[i * RN_N + RN_REST0_Q24]; }
    int64_t k_act(int j) const { return rn[j * RN_N + RN_K_ACT_Q16]; }
    int64_t k_pas(int j) const { return rn[j * RN_N + RN_K_PAS_Q16]; }
};

struct Env {
    const Row* row = nullptr;
    int A = 0, B = 0, G = 0;
    bool floor_on = true;
    Topo topo;
    // per arm, per node (A x n x 3 unless noted)
    std::vector<int64_t> p, v, res, pprev, disp, anchor_pos;
    std::vector<int64_t> w;                 // A x n
    std::vector<int64_t> r_dist;            // A x nseg
    std::vector<int64_t> c_per, c_tan;      // A x n
    std::vector<int64_t> act, k_sub;        // A x nb
    std::vector<int64_t> rest;              // A x nseg
    std::vector<int64_t> target;            // A x nb x 2
    std::vector<int64_t> root_local;        // A x 2 x 3
    std::vector<uint8_t> pinned, anchored, popped, contact;   // A x n
    std::vector<int64_t> anchor_force;      // A x n
    std::vector<int64_t> clock;             // A
    std::vector<const int64_t*> cfg;        // A rows of AC_N
    std::vector<Body> bodies;
    std::vector<int64_t> grabs;             // G x GR_N
    std::vector<int64_t> grab_force;        // G
    // frames scratch (one arm at a time): nb x (t, nA, nB)
    std::vector<V3> ft, fa, fb;

    int64_t* P(int a, int i) { return &p[((size_t)a * row->n + i) * 3]; }
    int64_t* Vv(int a, int i) { return &v[((size_t)a * row->n + i) * 3]; }
    int64_t* Rr(int a, int i) { return &res[((size_t)a * row->n + i) * 3]; }
    int64_t* PP(int a, int i) { return &pprev[((size_t)a * row->n + i) * 3]; }
    int64_t* D(int a, int i) { return &disp[((size_t)a * row->n + i) * 3]; }
    size_t ni(int a, int i) const { return (size_t)a * row->n + i; }
};

// ---- 1. activation (World._activation) ---------------------------------------
void activation(Env& e, int a) {
    const Row& row = *e.row;
    const int64_t* cfg = e.cfg[a];
    int64_t* act = &e.act[(size_t)a * row.nb];
    if (cfg[AC_WAVE_ON]) {
        const int64_t tact = cfg[AC_WAVE_TACT_SUB];
        const int64_t span = WAVE_CLOCK_SPAN * tact;
        int64_t ne = e.clock[a];
        ne = ne < -span ? -span : (ne > span ? span : ne);
        const int64_t den = cfg[AC_WAVE_W_Q24] * tact;
        const int64_t front = row.L * ne;
        const int64_t back = dra(FADE_BACK_NUM * row.L, FADE_BACK_DEN) * tact;
        for (int j = 0; j < row.nb; ++j) {
            const int64_t sj = row.s_node(j + 1) * tact;
            const int64_t x = dra(shl(front - sj, 16), den);
            int64_t aj = sra(cfg[AC_WAVE_A_CA_Q16] * activation_s_q16(x), 16);
            if (!cfg[AC_WAVE_HOLD]) {
                const int64_t x2 = dra(shl(sj - front + back, 16), den);
                aj = sra(aj * activation_s_q16(x2), 16);
            }
            act[j] = aj;
        }
    }
    int64_t* k = &e.k_sub[(size_t)a * row.nb];
    for (int j = 0; j < row.nb; ++j) k[j] = row.k_pas(j) + sra(act[j] * row.k_act(j), 16);
}

// ---- 2. drag (World._drag) -------------------------------------------------------
void drag(Env& e, int a) {
    const Row& row = *e.row;
    const int n = row.n;
    const int64_t* cfg = e.cfg[a];
    const int mode = (int)cfg[AC_DRAG_MODE];
    for (int i = 0; i < n; ++i) {
        if (e.w[e.ni(a, i)] == 0) continue;
        int64_t* vi = e.Vv(a, i);
        if (mode == DRAG_UNIFORM) {
            const int64_t den = U_ONE + cfg[AC_U_Q30];
            vi[0] = dra(shl(vi[0], U_SHIFT), den);
            vi[1] = dra(shl(vi[1], U_SHIFT), den);
            vi[2] = dra(shl(vi[2], U_SHIFT), den);
            continue;
        }
        const int lo = i > 0 ? i - 1 : 0;
        const int hi = i < n - 1 ? i + 1 : n - 1;
        const int64_t* ph = e.P(a, hi);
        const int64_t* pl = e.P(a, lo);
        int64_t tx = ph[0] - pl[0], ty = ph[1] - pl[1], tz = ph[2] - pl[2];
        const int64_t ln = isqrt(tx * tx + ty * ty + tz * tz);
        if (ln) {
            tx = dra(shl(tx, U_SHIFT), ln);
            ty = dra(shl(ty, U_SHIFT), ln);
            tz = dra(shl(tz, U_SHIFT), ln);
        } else {
            tx = ty = tz = 0;
        }
        const int64_t vt = sra(vi[0] * tx + vi[1] * ty + vi[2] * tz, U_SHIFT);
        const int64_t vtx = sra(vt * tx, U_SHIFT), vty = sra(vt * ty, U_SHIFT),
                      vtz = sra(vt * tz, U_SHIFT);
        const int64_t nx = vi[0] - vtx, ny = vi[1] - vty, nz = vi[2] - vtz;
        int64_t xt, xn;
        const size_t k = e.ni(a, i);
        if (mode == DRAG_QUADRATIC) {
            const int64_t sn = isqrt(nx * nx + ny * ny + nz * nz);
            xt = sra(e.c_tan[k] * (vt >= 0 ? vt : -vt), 26);
            xn = sra(e.c_per[k] * sn, 26);
        } else {
            xt = xn = sra(e.c_per[k] * cfg[AC_VREF_Q24], 26);
        }
        if (xt == 0 && xn == 0) continue;
        const int64_t vt2 = dra(shl(vt, U_SHIFT), U_ONE + xt);
        const int64_t dn = U_ONE + xn;
        vi[0] = sra(vt2 * tx, U_SHIFT) + dra(shl(nx, U_SHIFT), dn);
        vi[1] = sra(vt2 * ty, U_SHIFT) + dra(shl(ny, U_SHIFT), dn);
        vi[2] = sra(vt2 * tz, U_SHIFT) + dra(shl(nz, U_SHIFT), dn);
    }
}

// ---- 6. bodies (World._predict_bodies) ----------------------------------------------
void predict_bodies(Env& e) {
    const Row& row = *e.row;
    for (Body& b : e.bodies) {
        for (int c = 0; c < 3; ++c) b.prev_x[c] = b.x[c];
        b.prev_th = b.th;
        b.res[0] = b.res[1] = b.res[2] = 0;
        b.res_th = 0;
        int64_t* v = b.v;
        int64_t x;
        if (b.drag_mode == DRAG_UNIFORM) {
            x = b.u;
        } else if (b.drag_mode == DRAG_LINEAR) {
            x = sra(b.cb * b.vref, 26);
        } else {
            x = sra(b.cb * isqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 26);
        }
        if (x) {
            const int64_t den = U_ONE + x;
            v[0] = dra(shl(v[0], U_SHIFT), den);
            v[1] = dra(shl(v[1], U_SHIFT), den);
            v[2] = dra(shl(v[2], U_SHIFT), den);
        }
        if (b.gravity && row.g_dv) v[2] -= row.g_dv;
        if (b.y) b.om = dra(shl(b.om, U_SHIFT), U_ONE + b.y);
        for (int c = 0; c < 3; ++c) b.disp[c] = dra(v[c], row.inv_dt);
        b.dth = dra(b.om, row.inv_dt);
        for (int c = 0; c < 3; ++c) b.x[c] += b.disp[c];
        b.th += b.dth;
    }
}

// ---- 7a. distance (World._project_distance) ---------------------------------------------
void project_distance(Env& e, int a) {
    const Row& row = *e.row;
    const int64_t k = row.k_dist;
    const int64_t* rest = &e.rest[(size_t)a * row.nseg];
    const int64_t* rd = &e.r_dist[(size_t)a * row.nseg];
    for (int i = 0; i < row.nseg; ++i) {
        int64_t* P = e.P(a, i);
        int64_t* Q = e.P(a, i + 1);
        const int64_t dx = Q[0] - P[0], dy = Q[1] - P[1], dz = Q[2] - P[2];
        const int64_t ln = isqrt(dx * dx + dy * dy + dz * dz);
        if (ln == 0) continue;
        const int64_t C = ln - rest[i];
        if (C == 0) continue;
        const int64_t cx = dra(C * dx, ln), cy = dra(C * dy, ln), cz = dra(C * dz, ln);
        const int64_t r = rd[i];
        const int64_t jx = dra(k * cx, r), jy = dra(k * cy, r), jz = dra(k * cz, r);
        const int64_t wa = e.w[e.ni(a, i)], wb = e.w[e.ni(a, i + 1)];
        if (wa) move(P, e.Rr(a, i), msa(wa, jx, DIST_J_SHIFT), msa(wa, jy, DIST_J_SHIFT),
                     msa(wa, jz, DIST_J_SHIFT));
        if (wb) move(Q, e.Rr(a, i + 1), -msa(wb, jx, DIST_J_SHIFT), -msa(wb, jy, DIST_J_SHIFT),
                     -msa(wb, jz, DIST_J_SHIFT));
    }
}

// ---- 7b. the Bishop frame (World._frames) ------------------------------------------------
void frames(Env& e, int a) {
    const Row& row = *e.row;
    const int nb = row.nb;
    const int64_t* cfg = e.cfg[a];
    for (int j = 0; j < nb; ++j) {
        const int64_t* P = e.P(a, j);
        const int64_t* Q = e.P(a, j + 1);
        const int64_t dx = Q[0] - P[0], dy = Q[1] - P[1], dz = Q[2] - P[2];
        const int64_t ln = isqrt(dx * dx + dy * dy + dz * dz);
        e.ft[j] = ln ? V3{dra(shl(dx, U_SHIFT), ln), dra(shl(dy, U_SHIFT), ln),
                          dra(shl(dz, U_SHIFT), ln)}
                     : V3{0, 0, 0};
    }
    const V3 t0 = e.ft[0];
    const V3 ref{cfg[AC_ROOT_REF_X_Q30], cfg[AC_ROOT_REF_Y_Q30], cfg[AC_ROOT_REF_Z_Q30]};
    V3 n0 = reject(ref, t0);
    int64_t nn = isqrt(n0.x * n0.x + n0.y * n0.y + n0.z * n0.z);
    if (nn < N0_MIN) {
        const int64_t ax = t0.x < 0 ? -t0.x : t0.x;
        const V3 alt = ax < ALT_X_LIMIT ? V3{U_ONE, 0, 0} : V3{0, U_ONE, 0};
        n0 = reject(alt, t0);
        nn = isqrt(n0.x * n0.x + n0.y * n0.y + n0.z * n0.z);
    }
    e.fa[0] = V3{dra(shl(n0.x, U_SHIFT), nn), dra(shl(n0.y, U_SHIFT), nn),
                 dra(shl(n0.z, U_SHIFT), nn)};
    for (int j = 0; j < nb - 1; ++j) {
        const V3& tj = e.ft[j];
        const V3& tk = e.ft[j + 1];
        const V3 n = e.fa[j];
        const int64_t c = sra(tj.x * tk.x + tj.y * tk.y + tj.z * tk.z, U_SHIFT);
        int64_t opc = U_ONE + c;
        if (opc < OPC_FLOOR) opc = OPC_FLOOR;
        const int64_t nt = sra(n.x * tk.x + n.y * tk.y + n.z * tk.z, U_SHIFT);
        const int64_t f = dra(shl(nt, U_SHIFT), opc);
        V3 m{n.x - msa(f, tj.x + tk.x, U_SHIFT), n.y - msa(f, tj.y + tk.y, U_SHIFT),
             n.z - msa(f, tj.z + tk.z, U_SHIFT)};
        m = reject(m, tk);
        const int64_t mm = isqrt(m.x * m.x + m.y * m.y + m.z * m.z);
        e.fa[j + 1] = mm ? V3{dra(shl(m.x, U_SHIFT), mm), dra(shl(m.y, U_SHIFT), mm),
                              dra(shl(m.z, U_SHIFT), mm)}
                         : n;
    }
    for (int j = 0; j < nb; ++j) e.fb[j] = cross_q30(e.fa[j], e.ft[j]);
}

// ---- 7c. bends (World._project_bend) --------------------------------------------------------
void project_bend(Env& e, int a) {
    const Row& row = *e.row;
    const int ds = 32 - FINE_BITS;
    const int sh = row.atan2_shift;
    const int64_t gam = row.gamma, onepg = row.onepg, inv_dt2 = row.inv_dt2;
    const int64_t pmin = row.plane_min;
    const int64_t* kk = &e.k_sub[(size_t)a * row.nb];
    const int64_t* tgt = &e.target[(size_t)a * row.nb * 2];
    for (int j = 0; j < row.nb; ++j) {
        const int64_t w0 = e.w[e.ni(a, j)], w1 = e.w[e.ni(a, j + 1)], w2 = e.w[e.ni(a, j + 2)];
        if (w0 == 0 && w1 == 0 && w2 == 0) continue;
        const V3 e1 = e.ft[j];
        const int64_t kj = kk[j];
        for (int plane = 0; plane < 2; ++plane) {
            const V3 eX = plane == 0 ? e.fa[j] : e.fb[j];
            int64_t* P0 = e.P(a, j);
            int64_t* P1 = e.P(a, j + 1);
            int64_t* P2 = e.P(a, j + 2);
            const int64_t d1[3] = {P1[0] - P0[0], P1[1] - P0[1], P1[2] - P0[2]};
            const int64_t d2[3] = {P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]};
            const int64_t a1x = ip(d1, e1), a1y = ip(d1, eX);
            const int64_t a2x = ip(d2, e1), a2y = ip(d2, eX);
            const int64_t n1 = sra(a1x * a1x + a1y * a1y, 16);
            const int64_t n2 = sra(a2x * a2x + a2y * a2y, 16);
            if (n1 < pmin || n2 < pmin) continue;   // the in-plane guard
            const int64_t cross = a1x * a2y - a1y * a2x;
            const int64_t dot = a1x * a2x + a1y * a2y;
            const int64_t yn = sra(cross, sh), xn = sra(dot, sh);
            const int64_t C = (int64_t)fp::atan2_q16((int32_t)yn, (int32_t)xn) - tgt[j * 2 + plane];
            const int64_t g0x = dra(shl(-a1y, G_PRE), n1), g0y = dra(shl(a1x, G_PRE), n1);
            const int64_t g2x = dra(shl(-a2y, G_PRE), n2), g2y = dra(shl(a2x, G_PRE), n2);
            const int64_t g1x = -(g0x + g2x), g1y = -(g0y + g2y);
            int64_t W = 0;
            if (w0) W += msa(w0, g0x * g0x + g0y * g0y, 32);
            if (w1) W += msa(w1, g1x * g1x + g1y * g1y, 32);
            if (w2) W += msa(w2, g2x * g2x + g2y * g2y, 32);
            const int64_t kW = msa(kj, W, 32);
            const int64_t r = sra(kW * onepg, 32) + inv_dt2;
            const int64_t* Q0 = e.PP(a, j);
            const int64_t* Q1 = e.PP(a, j + 1);
            const int64_t* Q2 = e.PP(a, j + 2);
            const int64_t q0[3] = {P0[0] - Q0[0], P0[1] - Q0[1], P0[2] - Q0[2]};
            const int64_t q1[3] = {P1[0] - Q1[0], P1[1] - Q1[1], P1[2] - Q1[2]};
            const int64_t q2[3] = {P2[0] - Q2[0], P2[1] - Q2[1], P2[2] - Q2[2]};
            const int64_t S = g0x * ip(q0, e1) + g0y * ip(q0, eX) + g1x * ip(q1, e1) +
                              g1y * ip(q1, eX) + g2x * ip(q2, e1) + g2y * ip(q2, eX);
            int64_t D = sra(S * gam, 40);
            if (D > D_MAX_Q16) D = D_MAX_Q16;
            else if (D < -D_MAX_Q16) D = -D_MAX_Q16;
            const int64_t E = C + D;
            const int64_t kE = kj * E;
            const int64_t dlam = -dra(shl(kE, DLAM_PRE), r);
            if (dlam == 0) continue;
            const int64_t ws[3] = {w0, w1, w2};
            const int64_t gxs[3] = {g0x, g1x, g2x};
            const int64_t gys[3] = {g0y, g1y, g2y};
            int64_t* Ps[3] = {P0, P1, P2};
            for (int m = 0; m < 3; ++m) {
                if (ws[m] == 0) continue;
                const int64_t wd = msa(ws[m], dlam, 32);
                const int64_t dax = msa(wd, gxs[m], ds), day = msa(wd, gys[m], ds);
                move(Ps[m], e.Rr(a, j + m), sra(dax * e1.x + day * eX.x, U_SHIFT),
                     sra(dax * e1.y + day * eX.y, U_SHIFT), sra(dax * e1.z + day * eX.z, U_SHIFT));
            }
        }
    }
}

// ---- 7e-f. attachment (World._attach); true + J when it acted ---------------------------------
bool attach(Env& e, int a, int k, Body& b, const int64_t* L, int64_t J[3]) {
    const int64_t th16 = sra(b.th, CHAIN_SHIFT - 16);
    const int64_t c = fp::cos_q16((int32_t)th16), s = fp::sin_q16((int32_t)th16);
    const int64_t rx = sra(c * L[0] - s * L[1], 16);
    const int64_t ry = sra(s * L[0] + c * L[1], 16);
    const int64_t rz = L[2];
    int64_t* P = e.P(a, k);
    const int64_t Cx = P[0] - (b.x[0] + rx), Cy = P[1] - (b.x[1] + ry), Cz = P[2] - (b.x[2] + rz);
    const int64_t cn = isqrt(Cx * Cx + Cy * Cy + Cz * Cz);
    if (cn == 0) return false;
    const int64_t nx = dra(shl(Cx, U_SHIFT), cn), ny = dra(shl(Cy, U_SHIFT), cn);
    const int64_t rxn = sra(rx * ny - ry * nx, U_SHIFT);
    const int64_t wn = e.w[e.ni(a, k)];
    const int64_t wrot = msa(msa(rxn, rxn, CHAIN_SHIFT), b.winvI, CHAIN_SHIFT);
    const int64_t W = wn + b.w + wrot;
    const int64_t jx = dra(shl(Cx, ATTACH_PRE), W), jy = dra(shl(Cy, ATTACH_PRE), W),
                  jz = dra(shl(Cz, ATTACH_PRE), W);
    const int js = ATTACH_J_SHIFT;
    if (wn) move(P, e.Rr(a, k), -msa(wn, jx, js), -msa(wn, jy, js), -msa(wn, jz, js));
    move(b.x, b.res, msa(b.w, jx, js), msa(b.w, jy, js), msa(b.w, jz, js));
    const int64_t tq = rx * jy - ry * jx;
    if (tq) {
        const int64_t t = b.res_th + msa(tq, b.winvI, CHAIN_SHIFT + ATTACH_J_SHIFT);
        const int64_t i = sra(t, FINE_BITS);
        b.th += i;
        b.res_th = t - shl(i, FINE_BITS);
    }
    J[0] = jx;
    J[1] = jy;
    J[2] = jz;
    return true;
}

// ---- 7d-f. anchors, roots, grabs (World._project_extra) ----------------------------------------
void project_extra(Env& e) {
    const Row& row = *e.row;
    for (int a = 0; a < e.A; ++a) {
        const int64_t grip = e.cfg[a][AC_GRIP_Q8];
        for (int i = 0; i < row.n; ++i) {
            const size_t k = e.ni(a, i);
            if (!e.anchored[k]) continue;
            const int64_t* A = &e.anchor_pos[k * 3];
            int64_t* P = e.P(a, i);
            const int64_t dx = A[0] - P[0], dy = A[1] - P[1], dz = A[2] - P[2];
            const int64_t dist = isqrt(dx * dx + dy * dy + dz * dz);
            int64_t force = 0;
            if (e.w[k]) {
                force = shl(dist * row.inv_dt2, 16) / e.w[k];   // >= 0: C++ / is the floor
                if (force > e.anchor_force[k]) e.anchor_force[k] = force;
            }
            if (grip >= 0) {
                const bool pop = e.w[k] == 0 ? dist > 0 : force > grip;
                if (pop) {
                    e.anchored[k] = 0;
                    e.popped[k] = 1;
                    continue;
                }
            }
            P[0] = A[0];
            P[1] = A[1];
            P[2] = A[2];
            int64_t* R = e.Rr(a, i);
            R[0] = R[1] = R[2] = 0;
        }
    }
    int64_t J[3];
    for (int bi = 0; bi < e.B; ++bi) {
        for (int a = 0; a < e.A; ++a) {
            if (e.cfg[a][AC_BODY] != bi) continue;
            for (int k = 0; k < 2; ++k) attach(e, a, k, e.bodies[bi], &e.root_local[((size_t)a * 2 + k) * 3], J);
        }
    }
    for (int g = 0; g < e.G; ++g) {
        const int64_t* gr = &e.grabs[(size_t)g * GR_N];
        if (gr[GR_ARM] < 0) continue;
        const int64_t L[3] = {gr[GR_LX], gr[GR_LY], gr[GR_LZ]};
        if (attach(e, (int)gr[GR_ARM], (int)gr[GR_NODE], e.bodies[(size_t)gr[GR_BODY]], L, J)) {
            const int64_t f = force_q8(J, row.inv_dt2);
            if (f > e.grab_force[g]) e.grab_force[g] = f;
        }
    }
}

// ---- 7g. floor (World._project_floor, World._floor_bodies) --------------------------------------
void project_floor(Env& e, int a) {
    const int64_t mu = e.cfg[a][AC_MU_Q16];
    for (int i = 0; i < e.row->n; ++i) {
        int64_t* P = e.P(a, i);
        if (P[2] >= 0 || e.pinned[e.ni(a, i)]) continue;
        int64_t* R = e.Rr(a, i);
        const int64_t pen = -(shl(P[2], FINE_BITS) + R[2]);
        P[2] = 0;
        R[2] = 0;
        if (mu > 0) friction(P, R, e.PP(a, i), mu, pen, 0, 1);
    }
}

void floor_bodies(Env& e) {
    for (Body& b : e.bodies) {
        if (!b.floor || b.x[2] >= 0) continue;
        const int64_t pen = -(shl(b.x[2], FINE_BITS) + b.res[2]);
        b.x[2] = 0;
        b.res[2] = 0;
        if (b.mu > 0) friction(b.x, b.res, b.prev_x, b.mu, pen, 0, 1);
    }
}

// ---- 7h. static topology (World._project_walls, World._collide_bodies) ------------------------
void project_walls(Env& e, int a) {
    const Topo& t = e.topo;
    const int64_t T = t.T, ox = t.ox, oy = t.oy;
    const int64_t mu = e.cfg[a][AC_MU_Q16];
    for (int i = 0; i < e.row->n; ++i) {
        if (e.pinned[e.ni(a, i)]) continue;
        int64_t* P = e.P(a, i);
        const int64_t ix = t.tile_x(P[0]), iy = t.tile_y(P[1]);
        if (!t.blocked(ix, iy)) continue;
        int64_t best_d = 0, best_face = 0;
        int best_ax = -1;
        const int axs[4] = {0, 0, 1, 1};
        const int64_t nxs[4] = {ix - 1, ix + 1, ix, ix};
        const int64_t nys[4] = {iy, iy, iy - 1, iy + 1};
        const int64_t faces[4] = {ox + ix * T - 1, ox + (ix + 1) * T, oy + iy * T - 1,
                                  oy + (iy + 1) * T};
        for (int c = 0; c < 4; ++c) {
            if (t.blocked(nxs[c], nys[c])) continue;
            const int ax = axs[c];
            const int64_t face = faces[c];
            const int64_t d = face < P[ax] ? P[ax] - face : face - P[ax];
            if (best_ax < 0 || d < best_d) {
                best_d = d;
                best_ax = ax;
                best_face = face;
            }
        }
        int64_t* R = e.Rr(a, i);
        if (best_ax < 0) {
            const int64_t* pp = e.PP(a, i);
            P[0] = pp[0];
            P[1] = pp[1];
            R[0] = R[1] = 0;
            continue;
        }
        int64_t pen = shl(best_face, FINE_BITS) - (shl(P[best_ax], FINE_BITS) + R[best_ax]);
        if (pen < 0) pen = -pen;
        P[best_ax] = best_face;
        R[best_ax] = 0;
        if (mu > 0) friction(P, R, e.PP(a, i), mu, pen, 1 - best_ax, 2);
    }
}

// _push_disc
bool push_disc(Body& b, int64_t r, int64_t X0, int64_t Y0, int64_t X1, int64_t Y1) {
    const int64_t cx = b.x[0], cy = b.x[1];
    const int64_t qx = cx < X0 ? X0 : (cx > X1 ? X1 : cx);
    const int64_t qy = cy < Y0 ? Y0 : (cy > Y1 ? Y1 : cy);
    const int64_t dx = cx - qx, dy = cy - qy;
    const int64_t d2 = dx * dx + dy * dy;
    if (d2 >= r * r) return false;
    int64_t fx, fy;
    if (d2 > 0) {
        const int64_t d = isqrt(d2);
        const int64_t push = r - d;
        fx = dra(shl(dx * push, FINE_BITS), d);
        fy = dra(shl(dy * push, FINE_BITS), d);
    } else {
        int64_t best_d = cx - X0, best_s = -1;
        int best_ax = 0;
        const int64_t dds[3] = {X1 - cx, cy - Y0, Y1 - cy};
        const int64_t sgs[3] = {1, -1, 1};
        const int axs[3] = {0, 1, 1};
        for (int c = 0; c < 3; ++c) {
            if (dds[c] < best_d) {
                best_d = dds[c];
                best_s = sgs[c];
                best_ax = axs[c];
            }
        }
        const int64_t mag = best_s * shl(best_d + r, FINE_BITS);
        fx = best_ax == 0 ? mag : 0;
        fy = best_ax == 0 ? 0 : mag;
    }
    move(b.x, b.res, fx, fy, 0);
    return true;
}

void collide_bodies(Env& e) {
    const Topo& t = e.topo;
    const int64_t T = t.T, ox = t.ox, oy = t.oy;
    for (Body& b : e.bodies) {
        const int64_t r = b.coll_r;
        if (r <= 0) continue;
        const int64_t x0 = t.tile_x(b.x[0] - r), x1 = t.tile_x(b.x[0] + r);
        const int64_t y0 = t.tile_y(b.x[1] - r), y1 = t.tile_y(b.x[1] + r);
        for (int64_t iy = y0; iy <= y1; ++iy) {
            for (int64_t ix = x0; ix <= x1; ++ix) {
                if (t.blocked(ix, iy) &&
                    push_disc(b, r, ox + ix * T, oy + iy * T, ox + (ix + 1) * T, oy + (iy + 1) * T)) {
                    b.contact |= CONTACT_WALL;
                }
            }
        }
        for (int u = 0; u < t.U; ++u) {
            const int32_t* rc = &t.rects[(size_t)u * 4];
            const int64_t tx0 = rc[0], ty0 = rc[1], tx1 = rc[2], ty1 = rc[3];
            if (tx1 <= tx0 || ty1 <= ty0) continue;
            if (push_disc(b, r, ox + tx0 * T, oy + ty0 * T, ox + tx1 * T, oy + ty1 * T)) {
                b.contact |= CONTACT_UNIT;
            }
        }
    }
}

// ---- one substep (World.substep) ---------------------------------------------------------------
void substep(Env& e) {
    const Row& row = *e.row;
    const int n = row.n;
    for (int a = 0; a < e.A; ++a) {
        activation(e, a);
        drag(e, a);
        const int64_t gdv = row.g_dv;
        for (int i = 0; i < n; ++i) {
            const size_t k = e.ni(a, i);
            int64_t* vi = e.Vv(a, i);
            if (e.w[k] > 0 && gdv) vi[2] -= gdv;
            if (e.pinned[k] || e.anchored[k]) vi[0] = vi[1] = vi[2] = 0;
        }
    }
    for (int a = 0; a < e.A; ++a) {
        for (int i = 0; i < n; ++i) {
            int64_t* q = e.P(a, i);
            int64_t* qp = e.PP(a, i);
            int64_t* R = e.Rr(a, i);
            int64_t* d = e.D(a, i);
            const int64_t* vi = e.Vv(a, i);
            for (int c = 0; c < 3; ++c) {
                qp[c] = q[c];
                R[c] = 0;
                d[c] = dra(vi[c], row.inv_dt);
                q[c] += d[c];
            }
        }
    }
    predict_bodies(e);
    for (int pass = 0; pass < row.n_pass; ++pass) {
        for (int a = 0; a < e.A; ++a) {
            project_distance(e, a);
            frames(e, a);
            project_bend(e, a);
        }
        project_extra(e);
        if (e.floor_on) {
            for (int a = 0; a < e.A; ++a) project_floor(e, a);
            floor_bodies(e);
        }
        if (e.topo.on) {
            for (int a = 0; a < e.A; ++a) project_walls(e, a);
            collide_bodies(e);
        }
    }
    const int64_t inv_dt = row.inv_dt;
    for (int a = 0; a < e.A; ++a) {
        for (int i = 0; i < n; ++i) {
            const size_t k = e.ni(a, i);
            const int64_t* q = e.P(a, i);
            const bool on_floor = e.floor_on && q[2] <= 0;
            const Touch touch = e.topo.on ? wall_touch(e.topo, q) : Touch{};
            if (on_floor) e.contact[k] |= CONTACT_FLOOR;
            if (touch.any) e.contact[k] |= CONTACT_WALL;
            int64_t* v = e.Vv(a, i);
            if (e.pinned[k] || e.anchored[k]) {
                v[0] = v[1] = v[2] = 0;
                continue;
            }
            const int64_t* qp = e.PP(a, i);
            const int64_t* di = e.D(a, i);
            const int64_t* rr = e.Rr(a, i);
            v[0] = rebuild(v[0], q[0] - qp[0], rr[0], di[0], inv_dt, touch.lx, touch.hx);
            v[1] = rebuild(v[1], q[1] - qp[1], rr[1], di[1], inv_dt, touch.ly, touch.hy);
            v[2] = rebuild(v[2], q[2] - qp[2], rr[2], di[2], inv_dt, on_floor, false);
        }
    }
    for (Body& b : e.bodies) {
        const bool on = e.floor_on && b.floor && b.x[2] <= 0;
        if (on) b.contact |= CONTACT_FLOOR;
        for (int c = 0; c < 3; ++c) {
            b.v[c] = rebuild(b.v[c], b.x[c] - b.prev_x[c], b.res[c], b.disp[c], inv_dt,
                             c == 2 && on, false);
        }
        b.om = rebuild(b.om, b.th - b.prev_th, b.res_th, b.dth, inv_dt, false, false);
        if (b.th > PI_Q24) b.th -= TWO_PI_Q24;
        else if (b.th <= -PI_Q24) b.th += TWO_PI_Q24;
    }
    for (int a = 0; a < e.A; ++a) e.clock[a] += 1;
}

}  // namespace

// The reference's activation_s_q16, on the compiled table.
int64_t activation_s_q16(int64_t x_q16) {
    if (x_q16 < -ACT_X_MAX_Q16) x_q16 = -ACT_X_MAX_Q16;
    else if (x_q16 > ACT_X_MAX_Q16) x_q16 = ACT_X_MAX_Q16;
    const int64_t u = x_q16 + ACT_X_MAX_Q16;
    int64_t i = u >> ACT_TABLE_STEP_SHIFT;
    const int64_t last = ACT_TABLE_SAMPLES - 2;
    if (i > last) i = last;
    const int64_t f = u - (i << ACT_TABLE_STEP_SHIFT);
    const int64_t t0 = ACT_TABLE_DATA[i];
    return t0 + sra((ACT_TABLE_DATA[i + 1] - t0) * f, ACT_TABLE_STEP_SHIFT);
}

int activation_table_samples() { return ACT_TABLE_SAMPLES; }

void tick(const ArmChainView& vw) {
    Row row;
    row.n = (int)vw.row[ROW_N_NODES];
    row.nseg = row.n - 1;
    row.nb = row.n - 2;
    row.n_sub = (int)vw.row[ROW_N_SUB];
    row.n_pass = (int)vw.row[ROW_N_PASS];
    row.inv_dt = vw.row[ROW_INV_DT];
    row.inv_dt2 = vw.row[ROW_INV_DT2];
    row.L = vw.row[ROW_L_Q24];
    row.k_dist = vw.row[ROW_K_DIST_Q16];
    row.gamma = vw.row[ROW_GAMMA_Q16];
    row.onepg = vw.row[ROW_ONEPG_Q16];
    row.atan2_shift = (int)vw.row[ROW_ATAN2_SHIFT];
    row.g_dv = vw.row[ROW_G_DV_Q24];
    row.plane_min = vw.row[ROW_PLANE_MIN_Q32];
    row.rn = vw.row_node;
    const int n = row.n, A = vw.A, B = vw.B, G = vw.G;

    Env e;
    e.row = &row;
    e.A = A;
    e.B = B;
    e.G = G;
    const size_t an = (size_t)A * n;
    e.p.resize(an * 3);
    e.v.resize(an * 3);
    e.res.resize(an * 3);
    e.pprev.resize(an * 3);
    e.disp.resize(an * 3);
    e.anchor_pos.resize(an * 3);
    e.w.resize(an);
    e.c_per.resize(an);
    e.c_tan.resize(an);
    e.r_dist.resize((size_t)A * row.nseg);
    e.rest.resize((size_t)A * row.nseg);
    e.act.resize((size_t)A * row.nb);
    e.k_sub.resize((size_t)A * row.nb);
    e.target.resize((size_t)A * row.nb * 2);
    e.root_local.resize((size_t)A * 6);
    e.pinned.resize(an);
    e.anchored.resize(an);
    e.popped.resize(an);
    e.contact.resize(an);
    e.anchor_force.resize(an);
    e.clock.resize(A);
    e.cfg.resize(A);
    e.bodies.resize(B);
    e.grabs.resize((size_t)G * GR_N);
    e.grab_force.resize(G);
    e.ft.resize(row.nb > 0 ? row.nb : 1);
    e.fa.resize(row.nb > 0 ? row.nb : 1);
    e.fb.resize(row.nb > 0 ? row.nb : 1);

    for (int env = 0; env < vw.N; ++env) {
        const int64_t* wd = &vw.world[(size_t)env * WD_N];
        e.floor_on = wd[WD_FLOOR_ON] != 0;
        e.topo = Topo{};
        e.topo.on = wd[WD_HAS_TOPO] != 0;
        if (e.topo.on) {
            e.topo.grid = &vw.blocked[(size_t)env * vw.h * vw.w];
            e.topo.h = vw.h;
            e.topo.w = vw.w;
            e.topo.T = wd[WD_TILE_Q24];
            e.topo.ox = wd[WD_ORIGIN_X_Q24];
            e.topo.oy = wd[WD_ORIGIN_Y_Q24];
            e.topo.rects = &vw.unit_rects[(size_t)env * vw.U * 4];
            e.topo.U = vw.U;
        }
        // ---- load: buffers -> int64 scratch; the per-arm constants (World.add_arm)
        for (int a = 0; a < A; ++a) {
            const size_t ea = (size_t)env * A + a;
            e.cfg[a] = &vw.arm_cfg[ea * AC_N];
            e.clock[a] = vw.wave_clock[ea];
            for (int i = 0; i < n; ++i) {
                const size_t k = e.ni(a, i);
                const size_t src = ea * n + i;
                for (int c = 0; c < 3; ++c) {
                    e.p[k * 3 + c] = vw.p[src * 3 + c];
                    e.v[k * 3 + c] = vw.v[src * 3 + c];
                    e.anchor_pos[k * 3 + c] = vw.anchor_pos[src * 3 + c];
                }
                e.pinned[k] = vw.pinned[src] ? 1 : 0;
                e.anchored[k] = vw.anchored[src] ? 1 : 0;
                e.w[k] = e.pinned[k] ? 0 : row.w(i);
                e.c_per[k] = dra(row.k_per(i) * e.w[k], shl(row.inv_dt, 16));
                e.c_tan[k] = dra(row.k_tan(i) * e.w[k], shl(row.inv_dt, 16));
                e.popped[k] = 0;      // World.tick: outputs reset at the tick's start
                e.contact[k] = 0;
                e.anchor_force[k] = 0;
            }
            for (int i = 0; i < row.nseg; ++i) {
                const size_t k = (size_t)a * row.nseg + i;
                e.r_dist[k] = msa(row.k_dist, e.w[e.ni(a, i)] + e.w[e.ni(a, i + 1)], KW_PRE_SHIFT) +
                              row.inv_dt2;
                e.rest[k] = sra(row.rest0(i) * (int64_t)vw.rest_scale[ea * row.nseg + i], 16);
            }
            for (int j = 0; j < row.nb; ++j) {
                const size_t k = (size_t)a * row.nb + j;
                e.act[k] = vw.act[ea * row.nb + j];
                e.target[k * 2] = vw.bend_target[(ea * row.nb + j) * 2];
                e.target[k * 2 + 1] = vw.bend_target[(ea * row.nb + j) * 2 + 1];
            }
            for (int c = 0; c < 6; ++c) e.root_local[(size_t)a * 6 + c] = vw.root_local[ea * 6 + c];
        }
        for (int bi = 0; bi < B; ++bi) {
            Body& b = e.bodies[bi];
            b = Body{};
            const int32_t* bs = &vw.body[((size_t)env * B + bi) * BS_N];
            const int64_t* bc = &vw.body_cfg[((size_t)env * B + bi) * BC_N];
            for (int c = 0; c < 3; ++c) {
                b.x[c] = bs[BS_X + c];
                b.v[c] = bs[BS_VX + c];
            }
            b.th = bs[BS_TH];
            b.om = bs[BS_OM];
            b.w = bc[BC_W_Q32];
            b.winvI = bc[BC_WINVI_Q32];
            b.cb = bc[BC_CB_Q32];
            b.drag_mode = (int)bc[BC_DRAG_MODE];
            b.vref = bc[BC_VREF_Q24];
            b.u = bc[BC_U_Q30];
            b.y = bc[BC_Y_Q30];
            b.mu = bc[BC_MU_Q16];
            b.gravity = bc[BC_GRAVITY] != 0;
            b.floor = bc[BC_FLOOR] != 0;
            b.coll_r = bc[BC_COLL_R_Q24];
            b.contact = 0;
        }
        for (int g = 0; g < G; ++g) {
            for (int c = 0; c < GR_N; ++c) e.grabs[(size_t)g * GR_N + c] = vw.grabs[((size_t)env * G + g) * GR_N + c];
            e.grab_force[g] = 0;
        }

        for (int s = 0; s < row.n_sub; ++s) substep(e);

        // ---- store: scratch -> buffers
        for (int a = 0; a < A; ++a) {
            const size_t ea = (size_t)env * A + a;
            vw.wave_clock[ea] = e.clock[a];
            for (int i = 0; i < n; ++i) {
                const size_t k = e.ni(a, i);
                const size_t dst = ea * n + i;
                for (int c = 0; c < 3; ++c) {
                    vw.p[dst * 3 + c] = (int32_t)e.p[k * 3 + c];
                    vw.v[dst * 3 + c] = (int32_t)e.v[k * 3 + c];
                }
                vw.anchored[dst] = e.anchored[k];
                vw.popped[dst] = e.popped[k];
                vw.contact[dst] = e.contact[k];
                vw.anchor_force[dst] = e.anchor_force[k];
            }
            for (int j = 0; j < row.nb; ++j) vw.act[ea * row.nb + j] = (int32_t)e.act[(size_t)a * row.nb + j];
        }
        for (int bi = 0; bi < B; ++bi) {
            const Body& b = e.bodies[bi];
            int32_t* bs = &vw.body[((size_t)env * B + bi) * BS_N];
            for (int c = 0; c < 3; ++c) {
                bs[BS_X + c] = (int32_t)b.x[c];
                bs[BS_VX + c] = (int32_t)b.v[c];
            }
            bs[BS_TH] = (int32_t)b.th;
            bs[BS_OM] = (int32_t)b.om;
            vw.body_contact[(size_t)env * B + bi] = b.contact;
        }
        for (int g = 0; g < G; ++g) vw.grab_force[(size_t)env * G + g] = e.grab_force[g];
    }
}

}  // namespace arm_chain
