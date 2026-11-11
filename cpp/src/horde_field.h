#pragma once
// The HORDE FIELD (gameplay pack v1 G9p; the residency design's P3d, pulled
// forward by Erik's ruling of 2026-11-10; docs/residency_batching_design_
// 2026-11-07.md §4.9, docs/bots_zombies_speeds_design_2026-11-10.md §3).
//
// One integer BFS distance field over the ANCHOR grid of a footprint `fp`:
// cell (y, x) is the top-left tile of an fp x fp body, and it is PASSABLE
// when every tile of that block is in bounds and `tile_ok` (a tile a body
// may overlap: the mobility table's `mobility > 0`, plus whatever the caller
// adds -- a bashable closed door -- or removes -- a burning or hot tile).
// The flood is multi-source from `seeds` (anchor cells, row y then column x)
// and 4-connected; a seed is labelled 0 whether or not it is passable (a
// prey standing where no body of this footprint fits is still a goal), and
// the flood enters only passable cells. Every cell it does not reach holds
// UNREACHED.
//
// Pure integer, no float, no allocation besides the frontier: the distances
// are unique, so the result does not depend on the order of the seeds (a
// unit permutation leaves the field unchanged). The host reference of the
// horde law (src/simulation/horde.py); its device kernel is residency P7d
// (one block per (env, field), int16 distances in shared memory).
//
// Standard multi-source breadth-first search (Moore 1959, "The shortest path
// through a maze"; Lee 1961's wave expansion on a grid).

#include <cstdint>

namespace horde {

// The value of a cell the flood never reached (no path: the zombie waits).
constexpr int32_t UNREACHED = 0x7FFFFFFF;

// Flood `dist` (h*w, row-major) from `n_seeds` anchor cells `seeds_yx`
// (2*n_seeds ints: y0, x0, y1, x1, ...; out-of-bounds seeds are skipped)
// over the anchor cells whose fp x fp block lies in bounds and on `tile_ok`
// (h*w, nonzero = ok). Returns the number of cells reached (seeds included).
int64_t flood(const uint8_t* tile_ok, int h, int w, int fp,
              const int32_t* seeds_yx, int n_seeds, int32_t* dist);

}  // namespace horde
