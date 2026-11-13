#pragma once
// SIGHT FOR MANY TILES (#195 patch 1c; docs/rl_agent_design_2026-11-11.md
// section 3.4 / 3.7 and its section 10 rule "Sight for many tiles").
//
// THE one sight test, run in bulk: `GameMap.has_los` (src/simulation/
// gamemap.py), step for step. A line is the integer Bresenham walk from the
// start tile to the end tile with the error term `err = dx - dy`, both axis
// steps decided from the SAME `e2 = 2 * err` (the x step first, then the y
// step), and:
//   - the walk returns "clear" the moment it stands on the END tile, so the
//     end tile itself never blocks (a unit looking at a wall sees the wall);
//   - every tile before it, the START tile included, blocks when it is in
//     the map and `solid` is set, or `screen` is set (the sight screen: a
//     gas-open material whose row says `blocks_sight`, the grille);
//   - a tile OUTSIDE the map never blocks (the walk goes on through it);
//   - `physical` is the other question has_los answers (does what travels
//     with the air get through): the screen is not read, `solid` alone stops
//     it.
// Cover that blocks sight (vision.ray_clear's second half) is NOT here: it is
// a handful of float rectangles, applied in Python over this routine's answer.
//
// Two entries over the one walk: `rays` (an array of start/end tile pairs ->
// clear yes / no each) and `window` (one tile to every tile of a rectangle, or
// every tile of the rectangle to the one tile: the walk is not symmetric, so
// the direction is an argument, never an assumption). Every window cell is its
// own independent walk, so a window equals the many-rays answer for the same
// pairs by construction.
//
// Pure integer (int64 error term, so any tile coordinate an int32 can hold is
// safe), no float, no allocation, no engine state: a host call in both
// modules, like the horde field. Its pure-Python reference is
// `simulation.vision.sight_lines_reference`; the tests hold the two, and
// has_los, equal. Bresenham 1965, "Algorithm for computer control of a
// digital plotter", IBM Systems Journal 4(1).

#include <cstdint>

namespace sight {

// One line, has_los's walk. `solid` (h*w, nonzero = solid) and `screen`
// (h*w, nonzero = sight screen; ignored when `physical`, may then be null).
bool line_clear(const uint8_t* solid, const uint8_t* screen, int h, int w,
                bool physical, int32_t y0, int32_t x0, int32_t y1, int32_t x1);

// `n` lines: `pairs` is 4*n int32 (y0, x0, y1, x1, ...), `out` n bytes
// (1 = clear, 0 = blocked).
void rays(const uint8_t* solid, const uint8_t* screen, int h, int w,
          bool physical, const int32_t* pairs, int64_t n, uint8_t* out);

// A window: the wh x ww rectangle whose top-left tile is (wy, wx), possibly
// reaching outside the map. `out` is wh*ww bytes, row-major, cell (j, i) the
// line between the tile (y, x) and the rectangle's tile (wy + j, wx + i):
// FROM (y, x) to it when `to_window` is false, FROM it to (y, x) when true.
void window(const uint8_t* solid, const uint8_t* screen, int h, int w,
            bool physical, int32_t y, int32_t x, int32_t wy, int32_t wx,
            int32_t wh, int32_t ww, bool to_window, uint8_t* out);

}  // namespace sight
