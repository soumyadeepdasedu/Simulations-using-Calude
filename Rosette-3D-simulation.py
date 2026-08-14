"""
3D Rosette Simulation
======================
Generates a rose/rosette curve using parametric equations, then lifts it
into 3D by adding a sinusoidal z-component and animates it rotating in space.

A classic 2D rose curve is:
    r = cos(k * theta)
    x = r * cos(theta)
    y = r * sin(theta)

Here we extend it to 3D by:
    z = A * sin(m * theta)   -> gives the petals a rippling, "bloom" motion
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.animation import FuncAnimation


# ----------------------------
# Rosette parameters (tweak these)
# ----------------------------
K = 7          # petal frequency -> controls number of petals (odd k -> k petals, even k -> 2k petals)
M = 3          # z-ripple frequency -> controls how many "waves" the petals have in 3D
Z_AMPLITUDE = 0.4   # how much the petals rise/fall in z
N_POINTS = 2000     # resolution of the curve
N_LAYERS = 40        # number of nested rosette layers (for a fuller 3D bloom)
LAYER_SPACING = 0.015  # radial spacing between layers
ROTATION_SPEED = 2.0   # degrees per animation frame


def generate_rosette_layer(scale=1.0, z_offset=0.0):
    """Generate one layer of the 3D rosette curve."""
    theta = np.linspace(0, 2 * np.pi, N_POINTS)
    r = scale * np.cos(K * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = Z_AMPLITUDE * np.sin(M * theta) * scale + z_offset
    return x, y, z


def build_full_rosette():
    """Stack multiple scaled layers to build a fuller 3D rosette shape."""
    layers = []
    for i in range(N_LAYERS):
        scale = 1.0 - i * LAYER_SPACING
        if scale <= 0:
            break
        z_offset = i * 0.01
        layers.append(generate_rosette_layer(scale=scale, z_offset=z_offset))
    return layers


def main():
    layers = build_full_rosette()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    lines = []
    cmap = plt.get_cmap("plasma")
    for i, (x, y, z) in enumerate(layers):
        color = cmap(i / max(len(layers) - 1, 1))
        (line,) = ax.plot(x, y, z, color=color, linewidth=1.2, alpha=0.85)
        lines.append(line)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_axis_off()
    ax.set_title(f"3D Rosette (k={K}, m={M})", color="white", fontsize=13)

    def update(frame):
        # Rotate the camera around the rosette
        ax.view_init(elev=25, azim=frame * ROTATION_SPEED)

        # Slowly pulse the z-amplitude for a "breathing" bloom effect
        pulse = 1.0 + 0.15 * np.sin(np.radians(frame * 4))
        for line, (x, y, z) in zip(lines, layers):
            line.set_data(x, y)
            line.set_3d_properties(z * pulse)

        return lines

    anim = FuncAnimation(fig, update, frames=180, interval=40, blit=False)

    plt.tight_layout()
    plt.show()

    # To save as a gif instead of / in addition to showing it live, uncomment:
    # anim.save("rosette_3d.gif", writer="pillow", fps=25)


if __name__ == "__main__":
    main()