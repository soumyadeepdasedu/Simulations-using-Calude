"""
Electron Simulation in an Atom (Bohr Model)
=============================================
Simulates electrons orbiting a nucleus in discrete shells, like the classic
Bohr model of the atom. Each shell can hold a configurable number of
electrons, all orbiting at their own radius and speed (inner shells orbit
faster, just like real orbital mechanics / Bohr's model implies).

Requires: numpy, matplotlib
Install with: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Atom configuration (tweak these)
# ----------------------------
# Electron shell configuration, e.g. Sodium (Na, Z=11) = [2, 8, 1]
# Some examples:
#   Hydrogen (H):  [1]
#   Carbon (C):    [2, 4]
#   Neon (Ne):     [2, 8]
#   Sodium (Na):   [2, 8, 1]
#   Argon (Ar):    [2, 8, 8]
SHELL_CONFIG = [2, 8, 1]     # electrons per shell
ELEMENT_NAME = "Sodium (Na)"

SHELL_SPACING = 1.0          # radius spacing between shells
BASE_SPEED = 2.5             # base angular speed (deg/frame), inner shells go faster
NUCLEUS_SIZE = 400           # marker size for the nucleus
ELECTRON_SIZE = 90           # marker size for electrons
TRAIL_LENGTH = 25            # how many past positions to show as a fading trail


def build_shells():
    """Assign each electron a shell radius, starting angle, and orbital speed."""
    electrons = []  # list of dicts: shell, radius, angle0, speed
    for shell_idx, count in enumerate(SHELL_CONFIG):
        radius = SHELL_SPACING * (shell_idx + 1)
        # Inner shells orbit faster than outer shells (rough 1/r relationship)
        speed = BASE_SPEED / (shell_idx + 1) ** 0.5
        for e in range(count):
            angle0 = (360.0 / count) * e if count > 0 else 0
            electrons.append({
                "shell": shell_idx,
                "radius": radius,
                "angle0": angle0,
                "speed": speed,
                "trail_x": [],
                "trail_y": [],
            })
    return electrons


def main():
    electrons = build_shells()
    n_shells = len(SHELL_CONFIG)
    max_radius = SHELL_SPACING * n_shells

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Draw static shell orbit paths
    theta_full = np.linspace(0, 2 * np.pi, 200)
    for shell_idx in range(n_shells):
        r = SHELL_SPACING * (shell_idx + 1)
        ax.plot(r * np.cos(theta_full), r * np.sin(theta_full),
                 color="gray", linewidth=0.6, alpha=0.4, linestyle="--")

    # Nucleus at the center
    ax.scatter([0], [0], s=NUCLEUS_SIZE, color="orangered", zorder=5,
               edgecolors="yellow", linewidths=1.5)

    # Electron markers + trail lines
    cmap = plt.get_cmap("cool")
    electron_dots = []
    electron_trails = []
    for e in electrons:
        color = cmap(e["shell"] / max(n_shells - 1, 1))
        dot = ax.scatter([], [], s=ELECTRON_SIZE, color=color, zorder=6,
                          edgecolors="white", linewidths=0.5)
        (trail,) = ax.plot([], [], color=color, linewidth=1.0, alpha=0.5)
        electron_dots.append(dot)
        electron_trails.append(trail)

    lim = max_radius + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Electron Simulation — {ELEMENT_NAME}\n"
                 f"Shell config: {SHELL_CONFIG}", color="white", fontsize=12)

    def update(frame):
        artists = []
        for e, dot, trail in zip(electrons, electron_dots, electron_trails):
            angle = np.radians(e["angle0"] + frame * e["speed"])
            x = e["radius"] * np.cos(angle)
            y = e["radius"] * np.sin(angle)

            e["trail_x"].append(x)
            e["trail_y"].append(y)
            if len(e["trail_x"]) > TRAIL_LENGTH:
                e["trail_x"].pop(0)
                e["trail_y"].pop(0)

            dot.set_offsets([[x, y]])
            trail.set_data(e["trail_x"], e["trail_y"])
            artists.extend([dot, trail])
        return artists

    anim = FuncAnimation(fig, update, frames=360, interval=30, blit=False)

    plt.tight_layout()
    plt.show()

    # To save as a gif instead of / in addition to showing it live, uncomment:
    # anim.save("atom_simulation.gif", writer="pillow", fps=30)


if __name__ == "__main__":
    main()
