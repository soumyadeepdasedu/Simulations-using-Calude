"""
3D Monte Carlo Simulation
===========================
Classic Monte Carlo method extended to 3D: estimate the volume of a sphere
by randomly scattering points inside a cube and checking what fraction
land inside the sphere.

Math:
    Cube volume   = (2r)^3
    Sphere volume = (4/3) * pi * r^3
    fraction inside sphere ≈ sphere_volume / cube_volume

    => estimated_sphere_volume = fraction_inside * cube_volume

As more points are sampled, the Monte Carlo estimate converges to the true
value (4/3 * pi * r^3). This script animates the sampling process live,
showing points landing inside (hit) vs outside (miss) the sphere in 3D,
plus a running convergence plot of the volume/pi estimate.

Requires: numpy, matplotlib
Install with: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation


# ----------------------------
# Simulation parameters (tweak these)
# ----------------------------
RADIUS = 1.0             # sphere radius
POINTS_PER_FRAME = 60    # how many new random points to add each animation frame
N_FRAMES = 200           # total animation frames -> total points = N_FRAMES * POINTS_PER_FRAME
SEED = None              # set an integer for reproducible results


def main():
    rng = np.random.default_rng(SEED)

    true_sphere_volume = (4 / 3) * np.pi * RADIUS**3
    cube_volume = (2 * RADIUS) ** 3

    # Storage for all sampled points and running stats
    all_points = np.empty((0, 3))
    all_inside = np.empty((0,), dtype=bool)
    running_estimates = []
    running_counts = []

    fig = plt.figure(figsize=(13, 6))
    fig.patch.set_facecolor("black")

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_facecolor("black")

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_facecolor("black")

    # Draw a wireframe sphere for reference
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    sx = RADIUS * np.cos(u) * np.sin(v)
    sy = RADIUS * np.sin(u) * np.sin(v)
    sz = RADIUS * np.cos(v)
    ax3d.plot_wireframe(sx, sy, sz, color="cyan", linewidth=0.4, alpha=0.4)

    hits_scatter = ax3d.scatter([], [], [], color="lime", s=6, alpha=0.7, label="Inside sphere")
    miss_scatter = ax3d.scatter([], [], [], color="red", s=6, alpha=0.35, label="Outside sphere")

    lim = RADIUS * 1.1
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.set_axis_off()
    ax3d.set_title("Monte Carlo Sampling in 3D", color="white", fontsize=12)
    ax3d.legend(loc="upper right", facecolor="black", labelcolor="white", fontsize=8)

    # Convergence line plot
    (est_line,) = ax2d.plot([], [], color="orange", linewidth=1.5, label="MC estimate")
    ax2d.axhline(true_sphere_volume, color="cyan", linestyle="--", linewidth=1,
                 label=f"True volume = {true_sphere_volume:.4f}")
    ax2d.set_xlim(0, N_FRAMES * POINTS_PER_FRAME)
    ax2d.set_ylim(0, cube_volume)
    ax2d.set_xlabel("Number of sampled points", color="white")
    ax2d.set_ylabel("Estimated sphere volume", color="white")
    ax2d.set_title("Convergence of the Estimate", color="white", fontsize=12)
    ax2d.tick_params(colors="white")
    for spine in ax2d.spines.values():
        spine.set_color("white")
    ax2d.legend(loc="upper right", facecolor="black", labelcolor="white", fontsize=8)

    def update(frame):
        nonlocal all_points, all_inside

        # Sample new random points uniformly inside the bounding cube
        new_points = rng.uniform(-RADIUS, RADIUS, size=(POINTS_PER_FRAME, 3))
        new_inside = np.linalg.norm(new_points, axis=1) <= RADIUS

        all_points = np.vstack([all_points, new_points])
        all_inside = np.concatenate([all_inside, new_inside])

        # Update 3D scatter
        inside_pts = all_points[all_inside]
        outside_pts = all_points[~all_inside]
        hits_scatter._offsets3d = (inside_pts[:, 0], inside_pts[:, 1], inside_pts[:, 2])
        miss_scatter._offsets3d = (outside_pts[:, 0], outside_pts[:, 1], outside_pts[:, 2])

        # Update running volume estimate
        n_total = len(all_points)
        fraction_inside = all_inside.sum() / n_total
        estimate = fraction_inside * cube_volume
        running_estimates.append(estimate)
        running_counts.append(n_total)

        est_line.set_data(running_counts, running_estimates)

        ax3d.set_title(
            f"Monte Carlo Sampling in 3D  (n={n_total})", color="white", fontsize=12
        )
        error_pct = abs(estimate - true_sphere_volume) / true_sphere_volume * 100
        ax2d.set_title(
            f"Convergence — estimate={estimate:.4f}, error={error_pct:.2f}%",
            color="white", fontsize=11
        )

        return hits_scatter, miss_scatter, est_line

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=40, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()

    # To save as a gif instead of / in addition to showing it live, uncomment:
    # anim.save("monte_carlo_3d.gif", writer="pillow", fps=25)

    # Final printed summary
    n_total = len(all_points)
    fraction_inside = all_inside.sum() / n_total
    final_estimate = fraction_inside * cube_volume
    print(f"Points sampled : {n_total}")
    print(f"True volume    : {true_sphere_volume:.6f}")
    print(f"MC estimate    : {final_estimate:.6f}")
    print(f"Error          : {abs(final_estimate - true_sphere_volume) / true_sphere_volume * 100:.3f}%")


if __name__ == "__main__":
    main()
