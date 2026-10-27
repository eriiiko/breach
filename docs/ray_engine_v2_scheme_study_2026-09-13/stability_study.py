"""The stability rail, and what the literature already solved (2026-09-13).

Erik: "stability rail - oh, this is a real blocker - can we search online for solutions
in papers? can we cap it, it doesnt need to be totally physical correct ... i plan on
having plasma bullets at close to max temperature too."

THE PROBLEM.  The material update is explicit: a cell loses a*E_deg[T] of heat per tick
and E_deg goes as T^4, so the step is monotone only while

        4 x (per-tick radiative loss)  <  T

At the shipped rad_scale that holds to about T = 1800 game and fails hard above it: at
T_MAX_PHYS = 16000 game the cell overshoots by a factor of 864 in ONE tick.  Plasma is
exactly the regime that breaks it, and no calibration constant fixes it -- the constant
that would be stable at T_MAX is 432x smaller than the shipped one, which makes every
radiative rate 432x slower and the game unplayable.

THE LITERATURE SOLVED THIS IN 1971.  Fleck & Cummings' Implicit Monte Carlo introduces
the FLECK FACTOR: time-average the emission over the step as a linear combination of its
start- and end-of-step values, solve for it, and what falls out is that the emission is
simply SCALED by

        f = 1 / (1 + alpha * beta * sigma * c * dt),     beta = 4aT^3 / c_v

(Wollaber, "Four Decades of Implicit Monte Carlo", eqs. 19-21; alpha in [0.5, 1], and
alpha >= 0.5 is what makes it unconditionally stable -- alpha = 1 is used in practice to
suppress temporal oscillation.)  The group `beta*sigma*c*dt` is dimensionless and is
EXACTLY our `4 x per-tick loss / T`.  So in our variables the whole remedy is

        f = 1 / (1 + 4 * loss_per_tick / T)
        emitted <- f * emitted

WHY IT FITS US ALMOST SUSPICIOUSLY WELL
  - It is one multiply and one divide per cell per tick, computed BEFORE the sweep, so
    the sweep's exact-integer conservation is untouched: the stream simply carries a
    smaller source.  Nothing leaks, so it needs no counter in the arc-54 books.
  - It is NOT a no-op at fire temperatures -- f is 0.96 at ignition and 0.69 at 1263
    game with alpha = 0.5 -- but the correction is in the ACCURATE direction.  Measured
    against the analytic solution, cooling a 1263-game cell for half a second: the
    current explicit scheme lands 5.8% low, alpha = 0.5 lands 2.2% high.  So it is not
    a stability tax paid out of fidelity; at alpha = 0.5 it is strictly better than what
    ships today AND unconditionally stable at plasma.
  - It is monotone and positivity-preserving at every temperature, including plasma.
  - What it costs is physical fidelity in the stiff regime, which is the trade Erik
    already said he would take.  IMC recovers that fidelity by turning the damped
    fraction into effective scattering; we do not need to, because we are not trying to
    get the plasma's radiation field right, only to keep it from exploding.

This file measures all of it against the real constants.
"""
from __future__ import annotations
import math

RAD_SCALE = 5.1427e-5
K_AMB = 293.0
ONE = 65536.0
TICK_HZ = 24.0
A = 0.5          # furniture-class absorptivity
HIS = 3          # thermal_mass shift


def E_deg(T):
    return RAD_SCALE * (K_AMB + max(T, 0.0)) ** 4


def loss_per_tick(T):
    """The explicit per-tick radiative temperature loss, game units."""
    return A * E_deg(T) / (2.0 ** HIS) / ONE


def fleck(T, alpha=1.0):
    """Fleck & Cummings 1971: f = 1/(1 + alpha*beta*sigma*c*dt), beta = 4aT^3/c_v.

    THE DENOMINATOR IS THE ABSOLUTE TEMPERATURE, NOT OUR GAME UNITS.  beta is defined
    with T in Kelvin, so the dimensionless group is 4 * loss_per_tick / (T_game + 293).
    Using the game value instead -- which is a delta above ambient -- makes the group
    diverge as T_game -> 0 and drives f to zero at ambient, damping the entire map.
    (I made exactly that error first; it is the kind of thing that hides because it
    only misbehaves where nothing interesting is happening.)"""
    return 1.0 / (1.0 + alpha * 4.0 * loss_per_tick(T) / (max(T, 0.0) + K_AMB))


def exact_cool(T0, seconds):
    """Analytic reference.  dT/dt = -c (T+K_AMB)^4 with c = A*RAD_SCALE*24/(8*65536)
    per second, integrated exactly:  (T+K)^-3 = (T0+K)^-3 + 3*c*t."""
    c = A * RAD_SCALE * TICK_HZ / (2.0 ** HIS) / ONE
    k0 = T0 + K_AMB
    return (k0 ** -3 + 3.0 * c * seconds) ** (-1.0 / 3.0) - K_AMB


def march(T0, seconds, mode, alpha=1.0):
    """Integrate the cooling of one cell.  mode: 'explicit' or 'fleck'."""
    T = float(T0)
    for _ in range(int(seconds * TICK_HZ)):
        f = 1.0 if mode == "explicit" else fleck(T, alpha)
        T = T - f * loss_per_tick(T)
        if not math.isfinite(T) or T < -1e9:
            return float("nan")
    return T


print(__doc__.split("This file measures")[0])
print("=" * 78)
print("1. WHERE THE EXPLICIT UPDATE BREAKS, and what the Fleck factor does there")
print("=" * 78)
print(f"{'T (game)':>10}{'K':>8}{'loss/tick':>13}{'4*loss/T':>11}{'verdict':>14}"
      f"{'Fleck f':>10}")
for T in (280, 900, 1263, 1800, 2500, 5000, 16000, 60000):
    L = loss_per_tick(T)
    r = 4.0 * L / T
    verdict = "ok" if r < 1 else ("marginal" if r < 1.2 else "OVERSHOOTS")
    print(f"{T:10d}{K_AMB+T:8.0f}{L:13.2f}{r:11.2f}{verdict:>14}{fleck(T):10.4f}")

print()
print("=" * 78)
print("2. ONE CELL COOLING FROM PLASMA, integrated with the real constants")
print("=" * 78)
print(f"{'start T':>9} {'after':>7} | {'explicit':>14}{'Fleck alpha=1':>16}"
      f"{'exact':>12}")
for T0 in (1263.0, 5000.0, 16000.0, 60000.0):
    for secs in (0.5, 2.0, 10.0):
        e = march(T0, secs, "explicit")
        f = march(T0, secs, "fleck")
        x = exact_cool(T0, secs)
        es = "BLEW UP" if not math.isfinite(e) or e < -1e6 else f"{e:14.1f}"
        print(f"{T0:9.0f} {secs:6.1f}s | {es:>14}{f:16.1f}{x:12.1f}")
    print()

print("=" * 78)
print("3. IS IT A NO-OP WHERE IT SHOULD BE?  (a receiver near ignition)")
print("=" * 78)
for T in (0, 50, 280, 573, 900):
    print(f"   T = {T:5d} game:  f = {fleck(T):.6f}   "
          f"(1.000000 means the update is untouched)")

print()
print("=" * 78)
print("4. ACCURACY COST IN THE STIFF REGIME -- the trade being bought")
print("=" * 78)
print(f"{'start T':>9}{'after 1 s: Fleck':>19}{'exact':>10}{'error':>10}")
for T0 in (1263.0, 2500.0, 5000.0, 16000.0, 60000.0):
    f = march(T0, 1.0, "fleck")
    x = exact_cool(T0, 1.0)
    print(f"{T0:9.0f}{f:19.1f}{x:10.1f}{(f-x)/max(x,1e-9)*100:9.1f}%")
print()
print("   The damping makes a plasma cell cool SLOWER than physics, never faster, so")
print("   it errs toward a lingering hot spot rather than toward energy appearing from")
print("   nowhere -- and it is bounded: the cell can never lose more than it has.")

print()
print("=" * 78)
print("5. THE alpha DIAL.  The literature allows alpha in [0.5, 1]; alpha >= 0.5 is")
print("   what makes it unconditionally stable, alpha = 1 suppresses oscillation.")
print("   Lower alpha damps less, so it is both more accurate and closer to explicit.")
print("=" * 78)
print(f"{'T (game)':>10}" + "".join(f"{'a='+str(a):>10}" for a in (0.5, 0.75, 1.0)))
for T in (280, 573, 900, 1263, 1800, 16000):
    print(f"{T:10d}" + "".join(f"{fleck(T, a):10.4f}" for a in (0.5, 0.75, 1.0)))
print()
print(f"   cooling from 1263 game for 0.5 s   (exact = {exact_cool(1263.0, 0.5):.1f})")
for a in (0.5, 0.75, 1.0):
    v = march(1263.0, 0.5, "fleck", alpha=a)
    x = exact_cool(1263.0, 0.5)
    print(f"     alpha {a:4.2f}:  {v:8.1f}   ({(v-x)/x*100:+5.1f}% vs exact)")
print(f"     explicit :  {march(1263.0, 0.5, 'explicit'):8.1f}   "
      f"({(march(1263.0, 0.5, 'explicit') - exact_cool(1263.0, 0.5))/exact_cool(1263.0, 0.5)*100:+5.1f}% vs exact)")
print()
print("   and the same dial at plasma, 16000 game for 0.5 s:")
for a in (0.5, 0.75, 1.0):
    print(f"     alpha {a:4.2f}:  {march(16000.0, 0.5, 'fleck', alpha=a):8.1f}")


# ---------------------------------------------------------------------------
# 6. THE TWO PATHOLOGIES WOLLABER'S REVIEW WARNS ABOUT.
#
# Reading the review in full turned up two known IMC deficiencies that a
# stability measurement alone would have missed:
#
#   MAXIMUM PRINCIPLE VIOLATIONS -- "arguably the most serious deficiency of the
#   IMC equations ... interior temperatures can exceed boundary temperatures, even
#   in the absence of internal sources" (sec 4.2.2).  The mechanism, in our terms:
#   we damp a cell's EMISSION by f but not its ABSORPTION, so its equilibrium sits
#   where Phi = f*E_deg[T] instead of Phi = E_deg[T], i.e. HOTTER by (1/f)^(1/4).
#
#   And because f itself falls as T rises, the bias feeds back.  Asymptotically
#   f ~ T^-3, so f*E_deg ~ T, which means the equilibrium stops going as Phi^(1/4)
#   and goes LINEARLY in Phi -- a much steeper response.  Whether that matters
#   depends on where our fluxes actually sit, which is what this measures.
#
#   TELEPORTATION ERROR (sec 5.2) -- "photons entering one side of a zone are
#   completely absorbed very near the boundary.  The scoring of energy deposition,
#   however, is done as if the absorption had taken place evenly across the entire
#   zone.  On the next time step, the corresponding spontaneous emission ... occurs
#   too early on the far side of the zone and energy is transported across the
#   problem too quickly."  That is EXACTLY our through-wall result in
#   contact_faces_study.py, and it has a name and a known cure (linear sub-cell
#   temperature; a histogram field like ours is documented as inadequate).
# ---------------------------------------------------------------------------

def equilibrium(G, T_src, alpha=None, iters=400):
    """Where a receiver settles under a held source, with and without damping.

    G is the geometric factor: the fluence arriving per unit of source emission.
    alpha=None means no Fleck damping (the true answer, T = (G)^(1/4) * K_src)."""
    T = 0.0
    phi = G * E_deg(T_src)
    for _ in range(iters * 24):
        f = 1.0 if alpha is None else fleck(T, alpha)
        gain = A * phi / (2.0 ** HIS) / ONE
        loss = f * loss_per_tick(T)
        T = max(T + gain - loss, 0.0)
    return T


print()
print("=" * 78)
print("6. THE EQUILIBRIUM BIAS, and can a receiver exceed its SOURCE?")
print("   (Wollaber sec 4.2.2: maximum principle violations)")
print("=" * 78)
print(f"{'source T':>10}{'G':>8} | {'true T_eq':>11}{'alpha=0.5':>11}{'alpha=1':>10}"
      f"{'  bias':>9}   exceeds source?")
for T_src in (1263.0, 2500.0, 5000.0, 16000.0):
    for G in (0.25, 0.05):
        t0 = equilibrium(G, T_src, None)
        t5 = equilibrium(G, T_src, 0.5)
        t1 = equilibrium(G, T_src, 1.0)
        flag = "  *** YES ***" if t5 > T_src else "   no"
        print(f"{T_src:10.0f}{G:8.2f} | {t0:11.1f}{t5:11.1f}{t1:10.1f}"
              f"{t5/max(t0,1e-9):8.2f}x{flag}")
print()
print("   G = 0.25 is roughly an adjacent tile, G = 0.05 a few tiles away.")


# ---------------------------------------------------------------------------
# 7. THE COMPANION THE FLECK FACTOR NEEDS: a maximum-principle clamp.
#
# Section 6 shows emission-only damping is not enough.  The reason is structural:
# we damp the LOSS but not the GAIN, so equilibrium moves from Phi = E_deg[T] to
# Phi = f*E_deg[T].  And since f falls as T^-3 at high temperature, f*E_deg goes
# as T rather than T^4 -- the equilibrium stops being a fourth root of the flux
# and becomes LINEAR in it.  Next to a plasma source that is a million game units.
#
# Real IMC does not have this problem because it scales ABSORPTION by f as well,
# turning the undamped fraction into effective scattering.  We cannot copy that
# cheaply: scattering couples the ordinates and would cost a second sweep.
#
# The cheap, honest alternative is to enforce the maximum principle directly.  A
# body cannot be hotter than a black body in equilibrium with the radiation it is
# absorbing, so after the material update:
#
#         if E_deg[T] > Phi:   T <- E_deg_inverse(Phi)
#
# It is local, it is one inverse lookup against a table we already bake, it binds
# ONLY where the bias appears, and the energy it removes is booked to a named rail
# exactly like eos_solver's e_energy_floor_sum.  This is the "cap" Erik asked about,
# and it happens to be the exact statement of the property being violated.
# ---------------------------------------------------------------------------

def E_inv(E):
    """Invert E_deg by bisection (the shipped engine has a baked 4000-bucket table)."""
    lo, hi = 0.0, 1e7
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if E_deg(mid) < E:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equilibrium_clamped(G, T_src, alpha=0.5, iters=400):
    T = 0.0
    phi = G * E_deg(T_src)
    cap = E_inv(phi)                      # the maximum-principle temperature
    railed = 0.0
    for _ in range(iters * 24):
        gain = A * phi / (2.0 ** HIS) / ONE
        loss = fleck(T, alpha) * loss_per_tick(T)
        T = max(T + gain - loss, 0.0)
        if T > cap:
            railed += T - cap
            T = cap
    return T, cap, railed


print()
print("=" * 78)
print("7. WITH THE MAXIMUM-PRINCIPLE CLAMP")
print("=" * 78)
print(f"{'source T':>10}{'G':>8} | {'true T_eq':>11}{'Fleck only':>12}"
      f"{'Fleck+clamp':>13}   verdict")
for T_src in (1263.0, 2500.0, 5000.0, 16000.0, 60000.0):
    for G in (0.25, 0.05):
        t_true = E_inv(G * E_deg(T_src))          # the exact equilibrium, no scheme
        t_f = equilibrium(G, T_src, 0.5)
        t_c, cap, railed = equilibrium_clamped(G, T_src, 0.5)
        ok = "ok" if t_c <= T_src * 1.001 else "STILL VIOLATES"
        print(f"{T_src:10.0f}{G:8.2f} | {t_true:11.1f}{t_f:12.1f}{t_c:13.1f}   {ok}")
print()
print("   'true T_eq' is E_deg inverted on the arriving fluence -- the exact answer,")
print("   computed without any scheme.  The clamp reproduces it, because the clamp IS")
print("   that statement.  It binds only where the Fleck bias appears: at 1263 game")
print("   and below it never fires at all.")
