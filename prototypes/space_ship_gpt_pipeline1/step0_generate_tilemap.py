"""Step 0: Generate tilemap CSV for alien ship layout."""
import numpy as np
import matplotlib.pyplot as plt

H, W = 120, 50
grid = np.zeros((H, W), dtype=int)

# --- Outer hull (tapered with diagonals) ---
for r in range(H):
    taper = max(0, 10 - r // 4)
    left = 5 + taper
    right = W - 6 - taper

    if left >= right:
        left, right = 10, W - 10

    grid[r, left] = 1
    grid[r, right] = 1

    # fill interior with floor
    grid[r, left + 1:right] = 2

# --- Horizontal sealing (top/bottom caps) ---
grid[0, :] = 1
grid[-1, :] = 1

# --- Main corridor (6 tiles wide) ---
grid[:, 22:28] = 2

# --- Cockpit ---
grid[2:18, 18:32] = 7
grid[2:18, 18] = 1
grid[2:18, 31] = 1
grid[2, 18:32] = 1
grid[17, 18:32] = 1

# Door (3-wide)
grid[18, 23:26] = 3

# --- Crew quarters ---
grid[22:42, 10:40] = 2
grid[22:42, 10] = 1
grid[22:42, 39] = 1
grid[22, 10:40] = 1
grid[41, 10:40] = 1

# Dining
grid[26:36, 20:30] = 2

# Locker rooms
grid[22:32, 10:20] = 8
grid[32:42, 30:40] = 8

# Door
grid[42, 23:26] = 3

# --- Lab ---
grid[45:75, 10:40] = 2
grid[50:60, 14:24] = 4
grid[60:70, 26:36] = 4

# Big door (6-wide)
grid[45, 22:28] = 3

# --- Plants ---
grid[78:95, 14:36] = 5
grid[75, 23:26] = 3

# --- Storage ---
grid[98:118, 10:40] = 6
grid[95, 22:28] = 3

# --- Save CSV ---
np.savetxt("tilemap.csv", grid, fmt="%d", delimiter=",")

# --- Debug image ---
plt.imshow(grid, interpolation="nearest")
plt.colorbar()
plt.title("Tilemap Debug")
plt.savefig("debug.png", dpi=150)
plt.close()

print(f"Saved tilemap.csv ({H}x{W}) and debug.png")
