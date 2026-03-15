"""Visualize circles under the alternating 1-2 diagonal cost metric.

For each grid tile, compute distance from center using:
- Cardinal step: cost 1
- Diagonal step: alternating cost 1, 2, 1, 2...

Then draw "circles" (all tiles at distance <= r) for radii 1, 2, 3, 4.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def alternating_diagonal_distance(dx, dy):
    """Distance from (0,0) to (dx,dy) using alternating 1-2 diagonal cost."""
    dx, dy = abs(dx), abs(dy)
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    # Alternating 1, 2, 1, 2... for diagonal steps
    # pairs of 2 diag steps cost 3, so:
    full_pairs = diag // 2
    remainder = diag % 2
    diag_cost = full_pairs * 3 + remainder  # each pair costs 1+2=3, leftover costs 1
    return straight + diag_cost


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
radii = [1, 2, 3, 4]
grid_size = 11  # -5 to +5
center = grid_size // 2

for ax, radius in zip(axes.flat, radii):
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Radius = {radius}', fontsize=14, fontweight='bold')

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='#333333', linewidth=0.5)
        ax.axvline(i - 0.5, color='#333333', linewidth=0.5)

    # Color tiles by distance
    for x in range(grid_size):
        for y in range(grid_size):
            dx = x - center
            dy = y - center
            dist = alternating_diagonal_distance(dx, dy)

            if dist <= radius:
                # Filled tile - color by distance
                intensity = 1.0 - (dist / (radius + 1))
                color = (0.2, 0.4 + 0.4 * intensity, 0.8)
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                         facecolor=color, edgecolor='#555555',
                                         linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x, y, f'{dist}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
            elif dist <= radius + 2:
                # Show distance outside circle too
                ax.text(x, y, f'{dist}', ha='center', va='center',
                        fontsize=6, color='#666666')

    # Draw a true circle for comparison
    circle = plt.Circle((center, center), radius, fill=False,
                         edgecolor='red', linewidth=2, linestyle='--',
                         label='True circle')
    ax.add_patch(circle)
    ax.legend(loc='upper right', fontsize=9)

    # Count tiles
    n_tiles = sum(1 for x in range(grid_size) for y in range(grid_size)
                  if alternating_diagonal_distance(x - center, y - center) <= radius)
    true_area = np.pi * radius**2
    ax.set_xlabel(f'{n_tiles} tiles (true circle area: {true_area:.1f})', fontsize=10)

plt.suptitle('Alternating 1-2 Diagonal Cost: Grid Circles vs True Circles',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('c:/Users/steen/projects/breach/prototypes/diagonal_circles.png', dpi=150)
plt.show()
print("Saved to prototypes/diagonal_circles.png")
