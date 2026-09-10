"""Larva density study on the unhcr_vessel level.

How many larvae is "a lot"? Loads the real ship level through level_loader,
derives the walkable floor mask the way GameMap.is_passable does (mobility > 0:
air + open door; hull is solid; SPACE is vacuum), decomposes the floor into rooms
(4-connected components of the AIR tiles, doors excluded so they split rooms),
and renders N larvae to scale on the diffuse art.

Run from the breach repo root with the Anaconda python:
    C:/Users/steen/anaconda3/python.exe <this file>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\steen\projects\breach")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import EllipseCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from collections import deque  # noqa: E402


class ndimage:  # scipy.ndimage stand-in: base scipy is NumPy-1 built and will not import
    """4-connected labelling + erosion in pure numpy (grid is 50x120, BFS is instant)."""

    @staticmethod
    def label(mask, structure=None):
        mask = np.asarray(mask, dtype=bool)
        hh, ww = mask.shape
        lab = np.zeros(mask.shape, dtype=np.int32)
        n = 0
        for sy, sx in zip(*np.nonzero(mask)):
            if lab[sy, sx]:
                continue
            n += 1
            lab[sy, sx] = n
            q = deque([(sy, sx)])
            while q:
                y, x = q.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < hh and 0 <= nx < ww and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        q.append((ny, nx))
        return lab, n

    @staticmethod
    def binary_erosion(mask, structure=None):
        m = np.asarray(mask, dtype=bool)
        p = np.pad(m, 1, constant_values=False)
        return m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]

import level_loader  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR  # noqa: E402

# ---------------------------------------------------------------- parameters
LEVEL = "unhcr_vessel"
LARVA_LEN_M = 0.60          # garden larva, nose to tail
LARVA_WID_M = 0.15
PACKED_TILES_PER_LARVA = 1.5  # "packed" room: one larva per 1.5 floor tiles
MIN_ROOM_TILES = 9          # a "room" is >= 1 m^2 of floor (drops door-sill slivers)
N_UNIFORM = [250, 500, 1000, 2000, 5000]
N_INVERSE = [250, 500, 1000, 2000, 5000, 10000]
SEED = 20260909

# Colour: a single series (larvae) on dark ship art -> one warm hue, thin dark edge.
LARVA_FILL = "#FFB020"
LARVA_EDGE = "#3A2400"
ROOM_OUTLINE = "#4FC3F7"
INK = "#F2F2F2"
PANEL_BG = "#111111"

# ---------------------------------------------------------------- level load
lvl = level_loader.load(LEVEL, levels_dir=str(REPO / "levels"))
material, is_vacuum = level_loader.materials_from_tilemap(lvl.tilemap, lvl.version)
h, w = lvl.tilemap.shape
tile_m = float(lvl.tile_size_m)
tile_m2 = tile_m * tile_m

# Walkable exactly as GameMap.is_passable sees it: mobility > 0. In config.toml
# air and door carry mobility 1000, hull 0. Vacuum tiles are air-material but
# open space, not floor.
walkable = ((material == MAT_AIR) | (material == MAT_DOOR)) & (~is_vacuum)
floor_air = (material == MAT_AIR) & (~is_vacuum)     # rooms proper (doors split)
n_walk = int(walkable.sum())
n_door = int(((material == MAT_DOOR) & ~is_vacuum).sum())
area_m2 = n_walk * tile_m2

art = Image.open(lvl.diffuse_path).convert("RGB")
art_w, art_h = art.size
px_per_tile = (art_w / w, art_h / h)
art_np = np.asarray(art)

# ---------------------------------------------------------------- rooms
four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
lab_all, n_comp_all = ndimage.label(walkable, structure=four)
sizes_all = np.bincount(lab_all.ravel())[1:]

lab_door, n_door_split = ndimage.label(floor_air, structure=four)
sizes_door_split = np.sort(np.bincount(lab_door.ravel())[1:])[::-1]

# (b) Structural sections: erode 3 tiles so any passage <= 6 wide breaks, label the
# cores, then flood-assign every walkable tile to the nearest core (BFS in tiles).
core = walkable.copy()
for _ in range(3):
    core = ndimage.binary_erosion(core, structure=four)
lab_sec, n_sec = ndimage.label(core, structure=four)
lab_sec = lab_sec.astype(np.int32)
frontier = deque(zip(*np.nonzero(lab_sec)))
while frontier:
    y, x = frontier.popleft()
    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= ny < h and 0 <= nx < w and walkable[ny, nx] and not lab_sec[ny, nx]:
            lab_sec[ny, nx] = lab_sec[y, x]
            frontier.append((ny, nx))
sizes_sec = np.sort(np.bincount(lab_sec.ravel())[1:])[::-1]

# (c) ART rooms: the diffuse art draws interior partitions (cabins, mess, galley,
# hydroponics bays, cargo hold) that the physics tilemap does NOT wall. These are
# the rooms a player sees, so "one room full" is measured against them. Rectangles
# read off a gridded render of the art by eye (tile coords, half-open, +-2 tiles),
# intersected with the walkable mask. Corridors are not rooms.
ART_ROOMS = {
    "bridge":            (19, 3, 31, 17),
    "cabin fwd-port":    (8, 28, 14, 44),
    "cabin fwd-port 2":  (14, 28, 20, 44),
    "mess":              (27, 28, 44, 44),
    "cabin aft-port":    (6, 45, 19, 58),
    "galley":            (28, 45, 44, 58),
    "hydroponics port":  (6, 63, 22, 82),
    "hydroponics stbd":  (28, 63, 44, 82),
    "cargo hold":        (5, 86, 45, 117),
}
lab_room = np.zeros((h, w), dtype=np.int32)
rooms = []          # (id, tiles) largest first
room_names = {}
for i, (name, (x0, y0, x1, y1)) in enumerate(ART_ROOMS.items(), 1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    m &= walkable & (lab_room == 0)
    lab_room[m] = i
    rooms.append((i, int(m.sum())))
    room_names[i] = name
rooms.sort(key=lambda t: -t[1])
room_sizes = np.array([s for _, s in rooms])
n_room_raw = len(rooms)
slivers = 0
sliver_tiles = 0

median_tiles = float(np.median(room_sizes))
median_room_id, median_room_tiles = min(rooms, key=lambda t: abs(t[1] - median_tiles))
largest_room_id, largest_room_tiles = rooms[0]

# Erosion-1 heuristic (the one the brief suggested), kept for the record:
eroded = ndimage.binary_erosion(walkable, structure=four)
lab_er, n_er = ndimage.label(eroded, structure=four)
sizes_er = np.bincount(lab_er.ravel())[1:]
sizes_er = sizes_er[sizes_er >= MIN_ROOM_TILES]


def packed(tiles: float) -> int:
    return int(round(tiles / PACKED_TILES_PER_LARVA))


pack_median = packed(median_tiles)
pack_largest = packed(largest_room_tiles)
pack_ship = packed(n_walk)

# ---------------------------------------------------------------- placement
rng = np.random.default_rng(SEED)
larva_len_t = LARVA_LEN_M / tile_m
larva_wid_t = LARVA_WID_M / tile_m


def place(mask: np.ndarray, n: int) -> np.ndarray:
    """n larva centres (x, y, heading_deg) uniformly at random over mask tiles."""
    ys, xs = np.nonzero(mask)
    pick = rng.integers(0, len(ys), size=n)
    x = xs[pick] + rng.random(n)
    y = ys[pick] + rng.random(n)
    ang = rng.uniform(0.0, 360.0, size=n)
    return np.column_stack([x, y, ang])


def draw_larvae(ax, pts: np.ndarray, lw: float = 0.25):
    """Ellipses in DATA units (tiles) so they scale with the map, never fat markers."""
    coll = EllipseCollection(
        widths=np.full(len(pts), larva_len_t),
        heights=np.full(len(pts), larva_wid_t),
        angles=pts[:, 2],
        units="xy", offsets=pts[:, :2], offset_transform=ax.transData,
        facecolors=LARVA_FILL, edgecolors=LARVA_EDGE, linewidths=lw, alpha=0.95,
    )
    ax.add_collection(coll)


def ship_axes(ax, title: str | None = None):
    ax.imshow(art_np, extent=(0, w, h, 0), interpolation="lanczos", zorder=0)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=6, loc="left")


def caption(n: int) -> str:
    frac = n / pack_ship
    return (f"N = {n:,}   |   {n / area_m2:.2f} /m²   |   "
            f"{100 * frac:.0f}% of ship packed   |   {n / pack_median:.1f} median rooms full")


def room_outline(ax, room_id: int, color=ROOM_OUTLINE, lw=1.0):
    m = lab_room == room_id
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        # draw only exterior edges
        if y == 0 or not m[y - 1, x]:
            ax.plot([x, x + 1], [y, y], color=color, lw=lw, zorder=3)
        if y == h - 1 or not m[y + 1, x]:
            ax.plot([x, x + 1], [y + 1, y + 1], color=color, lw=lw, zorder=3)
        if x == 0 or not m[y, x - 1]:
            ax.plot([x, x], [y, y + 1], color=color, lw=lw, zorder=3)
        if x == w - 1 or not m[y, x + 1]:
            ax.plot([x + 1, x + 1], [y, y + 1], color=color, lw=lw, zorder=3)


plt.rcParams.update({
    "figure.facecolor": PANEL_BG, "axes.facecolor": PANEL_BG,
    "text.color": INK, "font.family": "DejaVu Sans", "font.size": 10,
    "savefig.facecolor": PANEL_BG,
})

# ---------------------------------------------------------------- figures 1: uniform
uniform_pts = {n: place(walkable, n) for n in N_UNIFORM}

for n, pts in uniform_pts.items():
    fig, ax = plt.subplots(figsize=(5.5, 12.5))
    ship_axes(ax)
    draw_larvae(ax, pts, lw=0.3)
    fig.suptitle(f"{lvl.name}: {n:,} larvae, uniform over walkable floor",
                 color=INK, fontsize=12, x=0.02, ha="left", y=0.985)
    ax.set_title(caption(n), color="#BBBBBB", fontsize=9, loc="left", pad=4)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / f"density_{n}.png", dpi=200)
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 14))
for ax, n in zip(axes.ravel(), N_UNIFORM):
    ship_axes(ax, f"N = {n:,}   ({n / area_m2:.2f} /m², {100 * n / pack_ship:.0f}% packed)")
    draw_larvae(ax, uniform_pts[n], lw=0.2)
ax = axes.ravel()[-1]
ax.axis("off")
lines = [
    f"{lvl.name}",
    f"grid {w}×{h} tiles, tile {tile_m} m",
    f"walkable floor {n_walk:,} tiles = {area_m2:.0f} m²",
    f"art rooms: {len(rooms)}",
    f"median room {median_tiles:.0f} tiles = {median_tiles * tile_m2:.1f} m²",
    f"largest room {largest_room_tiles:,} tiles = {largest_room_tiles * tile_m2:.0f} m²",
    "",
    f"larva {LARVA_LEN_M} × {LARVA_WID_M} m  (drawn to scale)",
    f"packed = 1 larva / {PACKED_TILES_PER_LARVA} tiles ({PACKED_TILES_PER_LARVA * tile_m2:.3f} m²)",
    f"  median room packed: {pack_median:,}",
    f"  largest room packed: {pack_largest:,}",
    f"  whole ship packed: {pack_ship:,}",
]
ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=11,
        family="DejaVu Sans Mono", transform=ax.transAxes)
fig.suptitle("How many larvae is a lot?  Uniform placement over the ship floor",
             fontsize=14, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "density_grid.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: packed median room
room_mask = lab_room == median_room_id
pts = place(room_mask, pack_median)
ys, xs = np.nonzero(room_mask)
pad = 6
x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
fig, (ax_ctx, ax_zoom) = plt.subplots(
    1, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [1, 2.2]})
ship_axes(ax_ctx, "where it is")
ax_ctx.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=ROOM_OUTLINE, lw=1.2, zorder=4))
draw_larvae(ax_ctx, pts, lw=0.2)
ship_axes(ax_zoom)
ax_zoom.set_xlim(x0, x1)
ax_zoom.set_ylim(y1, y0)
room_outline(ax_zoom, median_room_id, lw=1.5)
draw_larvae(ax_zoom, pts, lw=0.6)
# 1 m scale bar
ax_zoom.plot([x0 + 1, x0 + 1 + 1 / tile_m], [y1 - 1.2, y1 - 1.2], color=INK, lw=2.5)
ax_zoom.text(x0 + 1, y1 - 1.7, "1 m", color=INK, fontsize=9)
ax_zoom.set_title(
    f"median room = {room_names[median_room_id]} ({median_room_tiles} tiles = {median_room_tiles * tile_m2:.1f} m²) "
    f"packed: {pack_median} larvae  (1 per {PACKED_TILES_PER_LARVA} tiles)",
    color=INK, fontsize=11, loc="left")
fig.suptitle("What a 'full' room looks like", fontsize=14, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "room_packed.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 3: infestation, 3 largest rooms
N_INF = 1000
top3 = rooms[:3]
cap3 = sum(packed(s) for _, s in top3)
alloc = []
share = np.array([s for _, s in top3], dtype=float)
share /= share.sum()
counts = np.floor(share * min(N_INF, cap3)).astype(int)
counts[0] += min(N_INF, cap3) - counts.sum()
overflow = N_INF - counts.sum()
inf_pts = []
for (rid, s), c in zip(top3, counts):
    c = min(c, packed(s))
    inf_pts.append(place(lab_room == rid, c))
inf_pts = np.vstack(inf_pts)
fig, ax = plt.subplots(figsize=(5.5, 12.5))
ship_axes(ax)
for rid, s in top3:
    room_outline(ax, rid, lw=0.8)
draw_larvae(ax, inf_pts, lw=0.3)
fig.suptitle(f"{lvl.name}: infestation, {int(counts.sum()):,} larvae, 3 largest rooms",
             color=INK, fontsize=12, x=0.02, ha="left", y=0.985)
ax.set_title(
    "\n".join(f"{room_names[rid]} ({s * tile_m2:.0f} m²): {c} larvae = {100 * c / packed(s):.0f}% packed"
              for (rid, s), c in zip(top3, counts))
    + (f"   [{overflow} did not fit]" if overflow > 0 else ""),
    color="#BBBBBB", fontsize=8.5, loc="left", pad=4)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(OUT / "infestation_3rooms_1000.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 4: room map + size histogram
fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(13, 9),
                                      gridspec_kw={"width_ratios": [1, 1.6]})
ship_axes(ax_map, f"art rooms: {len(rooms)}  (label = m²)")
for rid, sz in rooms:
    room_outline(ax_map, rid, lw=0.6)
    ys, xs = np.nonzero(lab_room == rid)
    ax_map.text(xs.mean(), ys.mean(), f"{sz * tile_m2:.0f}", color=INK, fontsize=6,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="#000000AA", ec="none"))
ax_hist.set_facecolor(PANEL_BG)
ax_hist.bar(np.arange(len(room_sizes)), room_sizes * tile_m2, color=ROOM_OUTLINE, width=0.8)
ax_hist.set_xticks(np.arange(len(rooms)))
ax_hist.set_xticklabels([room_names[rid] for rid, _ in rooms], rotation=35, ha="right", fontsize=8)
ax_hist.axhline(median_tiles * tile_m2, color=LARVA_FILL, lw=1.2)
ax_hist.text(len(room_sizes) - 0.5, median_tiles * tile_m2 + 1,
             f"median {median_tiles * tile_m2:.1f} m²", color=LARVA_FILL, ha="right", fontsize=9)
ax_hist.set_xlabel("", color=INK)
ax_hist.set_ylabel("floor area (m²)", color=INK)
ax_hist.tick_params(colors=INK)
for s in ("top", "right"):
    ax_hist.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax_hist.spines[s].set_color("#555555")
ax_hist.grid(axis="y", color="#333333", lw=0.5)
ax_hist.set_title("art-room floor area (rectangles read off the art, ±2 tiles)", color=INK, loc="left")
fig.suptitle("Rooms of the UNHCR Vessel as drawn in the art (the physics tilemap does not wall them)",
             fontsize=14, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "rooms.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- numbers -> markdown
L = []
L.append(f"## Level: {lvl.name} (`levels/{LEVEL}/`)\n")
L.append(f"- grid: {w} wide x {h} tall tiles; tile_size_m = {tile_m} -> {w * tile_m:.1f} m x {h * tile_m:.1f} m")
L.append(f"- art: {art_w} x {art_h} px -> {px_per_tile[0]:.0f} x {px_per_tile[1]:.0f} px per tile")
L.append(f"- tile codes: hull {int((material == 1).sum())}, door {n_door}, air (floor) {int(floor_air.sum())}, SPACE (vacuum) {int(is_vacuum.sum())}")
L.append(f"- walkable (mobility > 0: air + open door, not vacuum): **{n_walk:,} tiles = {area_m2:.1f} m²**")
L.append(f"- walkable connected components (4-conn, doors included): {n_comp_all} "
         f"(sizes: {', '.join(str(int(s)) for s in sorted(sizes_all, reverse=True)[:8])}"
         f"{', ...' if n_comp_all > 8 else ''})\n")
L.append("## Rooms\n")
L.append("Three decompositions, because the physics tilemap is much coarser than the art:\n")
L.append(f"1. **Door-split** (air tiles, doors excluded, 4-conn): {n_door_split} components, sizes "
         f"{', '.join(str(int(x)) for x in sizes_door_split)}. The `DDD` door rows at y=42/45/75/95 stand in open "
         f"floor (no wall beside them), so only the bridge door row (y=18) actually splits anything.")
L.append(f"2. **Erode-1 heuristic** (the brief's suggestion): {len(sizes_er)} components >= {MIN_ROOM_TILES} tiles, sizes "
         f"{', '.join(str(int(x)) for x in sorted(sizes_er, reverse=True))} -- still does not split the hall.")
L.append(f"3. **Structural sections** (erode 3 tiles -> cores -> flood-assign every floor tile): {n_sec} sections, sizes "
         f"{', '.join(str(int(x)) for x in sizes_sec)} tiles = "
         f"{', '.join(f'{x * tile_m2:.0f}' for x in sizes_sec)} m². The 5-wide throat at rows 80-84 is the only real chokepoint.")
L.append(f"4. **Art rooms** (USED for the room numbers below): {len(rooms)} rectangles read off a gridded render of "
         f"`diffuse.png` by eye (+-2 tiles), intersected with the walkable mask. The art draws cabin / mess / galley / "
         f"hydroponics / cargo partitions that `tilemap.csv` does not wall -- a caveat for any room-based gameplay, "
         f"not just larvae.\n")
L.append(f"- art rooms: **{len(rooms)}**; median {median_tiles:.0f} tiles = {median_tiles * tile_m2:.1f} m²; "
         f"largest ({room_names[largest_room_id]}) {largest_room_tiles:,} tiles = {largest_room_tiles * tile_m2:.1f} m²; "
         f"smallest ({room_names[rooms[-1][0]]}) {room_sizes.min()} tiles = {room_sizes.min() * tile_m2:.1f} m²\n")
L.append("| rank | room | rect (x0,y0,x1,y1) | tiles | m² | packed larvae (1 / 1.5 tiles) |")
L.append("|---:|---|---|---:|---:|---:|")
for i, (rid, sz) in enumerate(rooms, 1):
    L.append(f"| {i} | {room_names[rid]} | {ART_ROOMS[room_names[rid]]} | {sz} | {sz * tile_m2:.1f} | {packed(sz)} |")
L.append("")
L.append("## Larva footprint and 'packed'\n")
L.append(f"- larva {LARVA_LEN_M} m x {LARVA_WID_M} m = {larva_len_t:.2f} x {larva_wid_t:.2f} tiles "
         f"(body area {np.pi / 4 * LARVA_LEN_M * LARVA_WID_M:.3f} m² as an ellipse)")
L.append(f"- packed = 1 larva per {PACKED_TILES_PER_LARVA} tiles = 1 per {PACKED_TILES_PER_LARVA * tile_m2:.3f} m² "
         f"= {1 / (PACKED_TILES_PER_LARVA * tile_m2):.1f} larvae/m² "
         f"(bodies cover {100 * (np.pi / 4 * LARVA_LEN_M * LARVA_WID_M) / (PACKED_TILES_PER_LARVA * tile_m2):.0f}% of the floor)\n")
L.append("| target | tiles | m² | larvae to pack |")
L.append("|---|---:|---:|---:|")
L.append(f"| median art room ({room_names[median_room_id]}) | {median_tiles:.0f} | {median_tiles * tile_m2:.1f} | **{pack_median}** |")
L.append(f"| largest art room ({room_names[largest_room_id]}) | {largest_room_tiles} | {largest_room_tiles * tile_m2:.1f} | **{pack_largest}** |")
L.append(f"| whole ship floor | {n_walk:,} | {area_m2:.1f} | **{pack_ship:,}** |")
L.append("")
L.append("## Inverse: what N larvae means on this ship\n")
L.append("| N | larvae / m² | tiles per larva | fraction of ship packed | median rooms full | largest rooms full |")
L.append("|---:|---:|---:|---:|---:|---:|")
for n in N_INVERSE:
    L.append(f"| {n:,} | {n / area_m2:.2f} | {n_walk / n:.1f} | {100 * n / pack_ship:.0f}% | "
             f"{n / pack_median:.1f} | {n / pack_largest:.2f} |")
L.append("")
L.append(f"Infestation figure: {int(counts.sum()):,} larvae into the 3 largest art rooms "
         f"(capacity {cap3:,} packed): " +
         ", ".join(f"{room_names[rid]} ({s * tile_m2:.0f} m²) gets {c} ({100 * c / packed(s):.0f}% packed)"
                   for (rid, s), c in zip(top3, counts)) +
         (f"; {overflow} did not fit" if overflow > 0 else "") + ".")
(OUT / "_numbers.md").write_text("\n".join(L), encoding="utf-8")
print("written", OUT / "_numbers.md")
