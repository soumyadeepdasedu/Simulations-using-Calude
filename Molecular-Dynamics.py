"""
Molecular Dynamics Simulation (2D, Lennard-Jones fluid)
---------------------------------------------------------
Simulates N particles interacting via the Lennard-Jones potential:

    U(r) = 4*epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]

Integration is done with the Velocity Verlet algorithm, which is
the standard choice for MD because it's time-reversible and
conserves energy well over long runs.

Particles are confined to a square box with reflective walls
(they bounce elastically off the edges). The simulation runs live
in a matplotlib window as the script executes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Simulation parameters
# ----------------------------
N = 40                  # number of particles
box_size = 10.0          # simulation box side length
dt = 0.005                # integration time step
epsilon = 1.0              # LJ well depth
sigma = 0.5                # LJ particle diameter scale
mass = 1.0                  # particle mass
r_cutoff = 2.5 * sigma        # interaction cutoff (for speed)
temperature_init = 1.0          # sets initial random velocity scale

rng = np.random.default_rng(seed=42)

# ----------------------------
# Initialize positions on a grid (avoids overlaps), then jitter
# ----------------------------
grid_n = int(np.ceil(np.sqrt(N)))
spacing = box_size / grid_n
xs, ys = np.meshgrid(
    (np.arange(grid_n) + 0.5) * spacing,
    (np.arange(grid_n) + 0.5) * spacing
)
positions = np.column_stack([xs.ravel(), ys.ravel()])[:N]
positions += rng.uniform(-0.1, 0.1, positions.shape)

# Initial velocities: random, then zero the net momentum so the
# whole system doesn't drift.
velocities = rng.normal(0, np.sqrt(temperature_init), (N, 2))
velocities -= velocities.mean(axis=0)

# ----------------------------
# Force calculation (Lennard-Jones, pairwise, with cutoff)
# ----------------------------
def compute_forces(pos):
    forces = np.zeros_like(pos)
    potential_energy = 0.0
    for i in range(N):
        diff = pos[i] - pos          # vector from all particles to i
        diff[i] = np.inf              # skip self-interaction
        dist2 = np.sum(diff**2, axis=1)
        mask = dist2 < r_cutoff**2
        mask[i] = False

        r2 = dist2[mask]
        r6 = (sigma**2 / r2)**3
        r12 = r6**2

        # LJ force magnitude / r  (so we can multiply directly by diff)
        f_over_r = 24 * epsilon * (2 * r12 - r6) / r2
        forces[i] += np.sum((f_over_r[:, None]) * diff[mask], axis=0)

        potential_energy += np.sum(4 * epsilon * (r12 - r6))

    return forces, potential_energy / 2  # /2 to avoid double counting

forces, _ = compute_forces(positions)

# ----------------------------
# Figure setup
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect("equal")
ax.set_title("2D Molecular Dynamics — Lennard-Jones Fluid")
ax.set_xlabel("x")
ax.set_ylabel("y")

scatter = ax.scatter(positions[:, 0], positions[:, 1],
                      s=200 * sigma, c=np.linalg.norm(velocities, axis=1),
                      cmap="plasma", edgecolors="k", linewidths=0.5)
time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                     fontsize=10, family="monospace")
cbar = plt.colorbar(scatter, ax=ax, label="Speed")

def reflect_walls(pos, vel):
    """Elastic reflection off the box edges."""
    for dim in range(2):
        too_low = pos[:, dim] < 0
        too_high = pos[:, dim] > box_size
        vel[too_low | too_high, dim] *= -1
        pos[too_low, dim] = -pos[too_low, dim]
        pos[too_high, dim] = 2 * box_size - pos[too_high, dim]
    return pos, vel

sim_time = 0.0
steps_per_frame = 4  # sub-steps between redraws, for smoother/faster dynamics

def step():
    """One Velocity Verlet integration step."""
    global positions, velocities, forces, sim_time
    positions_new = positions + velocities * dt + 0.5 * (forces / mass) * dt**2
    positions_new, velocities = reflect_walls(positions_new, velocities)
    forces_new, pe = compute_forces(positions_new)
    velocities_new = velocities + 0.5 * (forces + forces_new) / mass * dt

    positions[:] = positions_new
    velocities[:] = velocities_new
    forces[:] = forces_new
    sim_time += dt
    return pe

def animate(frame):
    pe = 0.0
    for _ in range(steps_per_frame):
        pe = step()

    speeds = np.linalg.norm(velocities, axis=1)
    ke = 0.5 * mass * np.sum(speeds**2)

    scatter.set_offsets(positions)
    scatter.set_array(speeds)
    time_text.set_text(
        f"t = {sim_time:6.2f}\nKE = {ke:6.2f}\nPE = {pe:6.2f}\nE  = {ke + pe:6.2f}"
    )
    return scatter, time_text

anim = FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
