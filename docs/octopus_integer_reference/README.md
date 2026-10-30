# The arm chain's integer reference (#106 P6b, contacts P6c)

`chain_ref_q.py` is the octopus arm chain in integers: the executable
specification the C++ `ArmChainSolver` (`cpp/src/arm_chain_solver.*`, P6c) is
held to at tol 0 (`tests/test_octopus_arm_chain_solver.py`). It transcribes
the float 3-D chain P5z validated (`prototypes/octopus_motion/arm_chain3d.py` +
`arm_bank3d.py`) in the currencies of `src/simulation/creature_fixed.py`, by
design section 6 of `docs/octopus_p6_sim_port_design_2026-10-30.md`. Pure Python
ints; nothing in `src/` or `cpp/` imports it. A change to the chain's arithmetic
lands HERE first, then the gates, then the C++.

    C:/Users/steen/anaconda3/python.exe docs/octopus_integer_reference/chain_ref_q_gates.py          # ~10 min, writes report_p6c.md
    C:/Users/steen/anaconda3/python.exe docs/octopus_integer_reference/chain_ref_q_gates.py --fast   # seconds (what CI runs)

G3 and G4 replay P5z's float traces: regenerate them first with
`prototypes/octopus_motion/gates3d.py` (~5 min; `out/traces_p5z.npz` is gitignored).

| file | what |
|---|---|
| `chain_ref_q.py` | `ChainRow` (the row in integers, from a `CreatureDef`), `World` (arms, bodies, grabs; `tick()`), `replay(trace)`; `CHECK = True` measures every int64 site |
| `chain_ref_q_gates.py` | the seven gates (P1's seven on ints, dead arm, no slide + its floor-rounding pair, float vs int, determinism, headroom, cost) |
| `report_p6b.md` | P6b's full run (kept as is) |
| `report_p6c.md` | P6c's full gate run (the eight gates, generated); the tol-0 table is in tests/test_octopus_arm_chain_solver.py's docstring and the P6c run log |
| `src/simulation/fixedpoint_twins.py` | the kit's bit-exact Python twins + the chain's round-half-away kit (`tests/test_fixedpoint_twins.py`) |
| `cpp/src/activation_table.inc` | S(x) = (1 + tanh x)/2, 1025 Q16 samples (`tools/gen_activation_table.py`) |

## What P6c must mirror (the arithmetic decisions)

The module docstring states the currencies and the substep order; every constant
is a named module-level literal. The decisions that are easy to get wrong:

1. **Rounding.** Every signed correction and narrow is round-half-away-from-zero
   (`shr_round_away`, `div_round_away`, `mul_shr_away` -- the last on the exact
   128-bit product). Floors only in `sqrt_q16` and the anchor-force comparison.
   `tests/test_octopus_chain_reference.py` pins it with an exact mirror test.
2. **Velocity rebuild.** `v = 384 (p - p_prev) + sra(384 r, 8)` [its z part
   clamped >= 0 on the floor] `+ (v_pred - 384 disp)`: the predict's rounding
   remainder and the corrections' FINE remainder `r` are carried, never erased.
3. **Fine corrections.** Every constraint correction is formed in Q32 (8 bits
   finer than a position) and applied through `_move`: the node moves by the
   rounded whole counts, the remainder (|r| <= half a count) waits in `r`.
   Without it the passive tip's corrections and a heavy body's attachment
   share fall into a dead band (measured: a dead arm's slide 77 mm off the
   float; with it 15 mm).
4. **Distance, impulse form**: `r = msa(k, w_i + w_j, 48) + 147456` per segment
   at arm build; per constraint `c = dra(C d, |d|)` (Q24), the shared IMPULSE
   `J = dra(k c, r)` (Q40 kg m), and each node moves by `+-msa(w, J, 40)` (Q32
   fine). Momentum is then exact to the 1e-10 of `m w = 1`. (P6a's per-node
   factor `f = k w / r` rounded each node's share separately, 1e-7 relative,
   and slid the free animal 1.1 mm in 60 s; the impulse form: 0.04 mm.)
5. **Bend**: in-plane coordinates `sra(d . e, 30)`; atan2 on
   `sra(cross, atan2_shift)`, `sra(dot, atan2_shift)`; `g = dra(perp(a) << 24, sra(|a|^2, 16))`
   (Q16 1/m), `g1 = -(g0 + g2)` exactly; `W = sum msa(w, |g|^2, 32)`;
   `r = sra(msa(k, W, 32) (1 + gamma)_q16, 32) + 147456`; `D = sra(S gamma_q16, 40)`
   clamped to +-pi; `dlam = -dra((k (C + D)) << 8, r)` (Q40, pre-shift 8);
   per node `wd = msa(w, dlam, 32)`, in-plane `msa(wd, g, 24)` (Q32), lifted
   `sra(dax e1 + day eX, 30)`.
6. **Bishop frame**: `t = dra(d << 30, |d|)`; root normal by rejection of the
   body's up (fallback below 1e-3: x, or y when |t0_x| >= 0.9); transport
   `f = dra(sra(n . t', 30) << 30, max(2^30 + sra(t . t', 30), 16))`,
   `n' = n - msa(f, t + t', 30)`, Gram-Schmidt, normalise; `nB = sra(nA x t, 30)`.
7. **Drag**: rates `c = dra(K_q16 w, 384 << 16)` (Q32 s/m) at arm build;
   `x = sra(c |v_part|, 26)` (Q30); `v_part' = dra(v_part << 30, 2^30 + x)`.
8. **Attachment, impulse form**: rotation by `cos_q16 / sin_q16` of
   `sra(th_q24, 8)`; `W = w_n + w_b + msa(msa(rxn, rxn, 24), 1/I, 24)` with
   `rxn` from the unit `n = dra(C << 30, |C|)`; the shared IMPULSE
   `J = dra(C << 38, W)` (Q30 kg m, needs |C| < 1 m); node `-msa(w_n, J, 30)`,
   body `+msa(w_b, J, 30)`, yaw `msa(rx J_y - ry J_x, 1/I, 54)` (Q32 rad), each
   through its fine remainder. The BODY's remainders are reset every substep
   like the nodes' (a stale body remainder re-injected itself every substep:
   the crawl replay's body ended 215 mm off the float; reset, 5 mm).
9. **Heading** is Q24 rad in the body state (not Q16): one Q16 count of yaw
   is 7 um at the crown's rim, 118 chain counts.
10. **Activation**: `x = dra((L n_e - s_j tact) << 16, w tact)` at the per-arm
    elapsed-substep clock (saturating at +-4 t_act), then the table lookup.
11. **Static-topology contacts (P6c)** -- `Topology` (`GameMap.solid` as a grid,
    the tile, the origin, other units' footprint rects). A node in a blocked
    tile leaves through the open-neighbour face of least penetration (-x, +x,
    -y, +y on ties) to `tile_low - 1` / `tile_high`, its fine remainder zeroed on
    that axis, friction on the two tangential axes (`_friction(..., ax)`); at the
    rebuild a node AT a face has that velocity component clamped (`_wall_touch`,
    `_rebuild(..., at_high_wall)`). A body with `coll_r_q24 > 0` is a disc pushed
    out of the blocked tiles under its bounding box (row-major) and then the
    footprint rects (`_push_disc`). The contact report is positional, read at
    the end of every substep. Nothing touches units from the nodes.
12. **The bend's in-plane guard (P6c)** -- a plane whose in-plane segment
    length squared is below `ChainRow.plane_min_q32` (the door's l_min^2) is
    skipped: W ~ 1/|a|^2 reached 63 bits on a tip folded against a corner.
13. **Outputs (P6c)** -- `Arm.contact`, `Arm.popped`, `Arm.anchor_force`
    (`((dist 1/dt^2) << 16) // w`, Q8 N, always), `Body.contact`,
    `World.grab_force` (`_force_q8`: components narrowed by 22, saturated at
    2^30, floor isqrt); all reset at the tick's start. The C++ keeps the wave
    clock per arm as `sub_count - t0_sub`.
