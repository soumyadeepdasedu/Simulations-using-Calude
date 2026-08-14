"""
2D Rosette Simulation
======================
Generates a classic rose/rosette curve using the polar equation:
    r = cos(k * theta)
    x = r * cos(theta)
    y = r * sin(theta)

k controls the number of petals:
    - if k is odd  -> k petals
    - if k is even -> 2k petals

Animates the rosette "blooming" (drawing itself) and slowly rotating.

Requires: numpy, matplotlib
Install with: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Rosette parameters (tweak these)
# ----------------------------
K = 7                 # petal frequency -> controls number of petals
N_POINTS = 1000        # resolution of the curve
N_LAYERS = 25          # nested layers for a fuller rosette
LAYER_SPACING = 0.03   # radial spacing between layers
ROTATION_SPEED = 1.5   # degrees per animation frame


def generate_rosette_layer(scale=1.0, phase=0.0):
    """Generate one layer of the 2D rosette curve."""
    theta = np.linspace(0, 2 * np.pi, N_POINTS)
    r = scale * np.cos(K * theta)
    x = r * np.cos(theta + phase)
    y = r * np.sin(theta + phase)
    return x, y


def build_full_rosette():
    """Stack multiple scaled layers to build a fuller rosette shape."""
    layers = []
    for i in range(N_LAYERS):
        scale = 1.0 - i * LAYER_SPACING
        if scale <= 0:
            break
        layers.append(generate_rosette_layer(scale=scale))
    return layers


def main():
    layers = build_full_rosette()

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    lines = []
    cmap = plt.get_cmap("plasma")
    for i in range(len(layers)):
        color = cmap(i / max(len(layers) - 1, 1))
        (line,) = ax.plot([], [], color=color, linewidth=1.3, alpha=0.85)
        lines.append(line)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"2D Rosette (k={K})", color="white", fontsize=13)

    def update(frame):
        angle = np.radians(frame * ROTATION_SPEED)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        for line, (x, y) in zip(lines, layers):
            # Rotate each layer's points around the origin
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            line.set_data(xr, yr)

        return lines

    anim = FuncAnimation(fig, update, frames=240, interval=40, blit=False)

    plt.tight_layout()
    plt.show()

    # To save as a gif instead of / in addition to showing it live, uncomment:
    # anim.save("rosette_2d.gif", writer="pillow", fps=25)


if __name__ == "__main__":
    main()
