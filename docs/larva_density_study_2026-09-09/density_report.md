# Larva density on the UNHCR Vessel -- how many is a lot?

Study for the fauna arc (#63) capacity decision. Real level `levels/unhcr_vessel/`
loaded through `level_loader.load`; walkable = `mobility > 0` exactly as
`GameMap.is_passable` sees it (air + open door, hull solid, SPACE vacuum).
Script: `larva_density.py` (this folder, seed 20260909, rerunnable).

## The ship

| quantity | value |
|---|---|
| grid | 50 wide x 120 tall tiles |
| tile_size_m | 0.333 (ship 16.7 m x 40.0 m) |
| art | 1000 x 2400 px = 20 px / tile |
| tile codes | hull 613, door 21, air 3768, SPACE 1598 |
| walkable floor | **3,789 tiles = 420 m²** |

## Larva footprint and "packed"

- larva 0.60 x 0.15 m = 1.80 x 0.45 tiles (ellipse body 0.071 m²)
- packed = 1 larva / 1.5 tiles = 1 / 0.166 m² = **6.0 larvae per m²**; bodies then
  cover 42 % of the floor (see `room_packed.png` -- dense, still crawlable)

| target | tiles | m² | larvae to pack |
|---|---:|---:|---:|
| median art room (galley) | 208 | 23.1 | **139** |
| largest art room (cargo hold) | 1,178 | 130.6 | **785** |
| whole ship floor | 3,789 | 420.2 | **2,526** |

## Inverse: what N larvae means on this ship

| N | larvae / m² | floor tiles per larva | fraction of ship packed | median rooms (23 m²) full | cargo holds full |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.60 | 15.2 | 10 % | 1.8 | 0.32 |
| 500 | 1.19 | 7.6 | 20 % | 3.6 | 0.64 |
| 1,000 | 2.38 | 3.8 | 40 % | 7.2 | 1.27 |
| 2,000 | 4.76 | 1.9 | 79 % | 14.4 | 2.55 |
| 5,000 | 11.90 | 0.8 | 198 % (does not fit) | 36.0 | 6.37 |
| 10,000 | 23.80 | 0.4 | 396 % (does not fit) | 71.9 | 12.74 |

The ship's hard ceiling is ~2,500 larvae; above that they must stack.

## Rooms

The physics tilemap is much coarser than the art, so the room question has
three answers (all in `larva_density.py`):

1. **Door-split components** (air tiles, doors excluded, 4-conn): **2** --
   3,600 + 168 tiles. Only the bridge door row (y=18) spans wall to wall; the
   `DDD` rows at y=42/45/75/95 stand in open floor with no wall beside them and
   split nothing.
2. **Erode-1 heuristic** (the brief's suggestion): still 2 components
   (3,125 + 120). The hall is too wide for a one-tile erosion to cut.
3. **Structural sections** (erode 3 tiles -> cores -> flood-assign every floor
   tile): **4** -- 2,197 / 1,302 / 168 / 122 tiles = 244 / 144 / 19 / 14 m². The
   5-tile-wide throat at rows 80-84 is the only real chokepoint.
4. **Art rooms** (used for the room numbers above): the diffuse art draws cabin /
   mess / galley / hydroponics / cargo partitions that `tilemap.csv` does not
   wall. Nine rectangles read off a gridded render of the art
   (`_art_grid.png`, +-2 tiles), intersected with the walkable mask:

| rank | room | rect (x0,y0,x1,y1) | tiles | m² | packed larvae |
|---:|---|---|---:|---:|---:|
| 1 | cargo hold | (5, 86, 45, 117) | 1,178 | 130.6 | 785 |
| 2 | hydroponics port | (6, 63, 22, 82) | 272 | 30.2 | 181 |
| 3 | hydroponics stbd | (28, 63, 44, 82) | 272 | 30.2 | 181 |
| 4 | mess | (27, 28, 44, 44) | 241 | 26.7 | 161 |
| 5 | galley | (28, 45, 44, 58) | 208 | 23.1 | 139 |
| 6 | cabin aft-port | (6, 45, 19, 58) | 169 | 18.7 | 113 |
| 7 | bridge | (19, 3, 31, 17) | 168 | 18.6 | 112 |
| 8 | cabin fwd-port 2 | (14, 28, 20, 44) | 90 | 10.0 | 60 |
| 9 | cabin fwd-port | (8, 28, 14, 44) | 79 | 8.8 | 53 |

Distribution: one 130 m² hall, five rooms of 19-30 m², three cabins of 9-19 m².
Median 23 m² (208 tiles); a typical room fills at **~110-180 larvae**, a cabin
at ~55, the cargo hold at ~785.

Infestation figure (`infestation_3rooms_1000.png`): 1,000 larvae in the cargo
hold + both hydroponics bays = 87 % packed in each (capacity 1,147).

## Reading

1. **250 is sparse**: 0.6 /m², one larva per 15 tiles -- a scattering you walk
   between; even the cabins hold only a handful.
2. **500 reads as "present everywhere" but not threatening** (1.2 /m²); it fills
   ~3.5 typical rooms if clustered, or thinly coats the whole ship if uniform.
3. **1,000 is the infestation threshold**: uniform it is 40 % of the ship's
   capacity (every room visibly crawling, ~1 per 4 tiles); clustered it packs the
   cargo hold and both hydroponics bays to 87 % -- the "nest" picture.
4. **2,000 is the ceiling of what fits** (79 % packed, bodies covering a third of
   every floor); 2,500 is wall-to-wall; 5,000+ cannot exist on this hull without
   stacking or a much bigger level.
5. So Erik's gut (~1k tops) matches the geometry: **1,000 fills the three
   largest rooms of this ship; a 2,000-2,500 cap is the hard physical limit,
   and only a larger level (or multi-floor) would use more.**

## Caveats

- Anaconda base `scipy` is compiled against NumPy 1.x and does not import; the
  script uses a pure-numpy 4-conn labeller + erosion instead (grid is 6,000
  tiles, instant). No environment was changed. No conda env named `data` exists
  on this machine; the repo's documented interpreter
  `C:/Users/steen/anaconda3/python.exe` was used (`docs/dev_setup.md`).
- Art rooms are hand-measured rectangles, +-2 tiles (~+-10 % on a cabin, ~+-3 %
  on a bay). Good enough for the capacity call, not for level data.
- Finding worth its own issue: the art's interior partitions are not in the
  physics tilemap, so gas, fire, LOS and pathing see one open hall from row 42
  to 118. That affects every room-based mechanic, not just larvae.
- Larva placement is uniform over walkable tiles (furniture in the art is not in
  the tilemap either, so larvae are drawn over crates/tables).

## Files

| file | what |
|---|---|
| `density_grid.png` | 2x3: N = 250 / 500 / 1000 / 2000 / 5000 uniform + numbers panel |
| `density_250.png` ... `density_5000.png` | one panel each, full resolution |
| `room_packed.png` | the median art room (galley) at packed density, zoomed, 1 m bar |
| `infestation_3rooms_1000.png` | 1,000 larvae clustered into the 3 largest art rooms |
| `rooms.png` | art-room outlines on the ship + size bar chart |
| `_art_grid.png` | gridded art the room rectangles were read from |
| `_numbers.md` | raw numbers dump from the script |
| `larva_density.py` | the study; rerun to regenerate everything |
