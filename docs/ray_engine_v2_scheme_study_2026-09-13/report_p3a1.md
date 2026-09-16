# P3a-1 — the live radiation planes widen to int64

Issue #12, ray engine v2, patch P3a-1. Branch `12-p3a1-widen-live-rad-planes`
off `fire-12`, worktree `C:\Users\steen\projects\breach-p3a1`.

**Written incrementally, section by section, committed as it is written.** If
this report ends mid-sentence, the work it describes above that point is on the
branch and the rest was never started.

---

## 0. What this patch is, in one paragraph

`gmap.rad_net`, `gmap.rad_amb` and `gmap.rad_flux` — the three LIVE radiation
planes, the ones the old `cast_fire_heat` writes every tick and the temperature
fold reads — go `np.int32 -> np.int64`, and every C++ signature, binding and
kernel along their path widens with them **in the same commit**. Nothing else
changes. The old cast still writes the same integers; the fold still reads
whatever it reads today. **No golden moves.** The flip of the fold onto the
sweep's shadow planes is P3a-2 and is feel-gated.

The patch exists because a widening that is *incomplete* is worse than no
widening at all: pybind11's default `forcecast` would let a missed binding
silently hand C++ a truncated int32 **copy** of an int64 plane — fire heat
vanishing with no error anywhere. So the whole path moves at once, and every
plane argument of every binding touched gets `.noconvert()` (design row 36) so
a stale caller raises `TypeError` instead.

## 1. Status

- [x] Report skeleton committed (this file), before any code change.
- [ ] Surface inventory verified against the tree (section 2).
- [ ] D1 — the widening (section 3).
- [ ] D2 — the loudness test (section 4).
- [ ] Gate: CPU build + suite (section 5).
- [ ] Gate: CUDA build + suite (section 5).
- [ ] Findings (section 6).

## 2. The inventory — every 32-bit surface on the live planes

(filled once verified against the tree.)

## 3. D1 — the widening

(filled as the change lands.)

## 4. D2 — the test that proves it is loud

(filled as the test lands.)

## 5. The gate

(filled with the measured counts.)

## 6. Findings

(filled as they are found.)
