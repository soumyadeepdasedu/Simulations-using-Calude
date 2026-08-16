"""
3D Electromagnetic Wave Propagation
------------------------------------
Simulates a plane EM wave travelling along the z-axis:
    E(z, t) = E0 * sin(kz - wt)  x-hat   (electric field, oscillates in x)
    B(z, t) = B0 * sin(kz - wt)  y-hat   (magnetic field, oscillates in y)

E and B are perpendicular to each other and to the direction of
propagation (z), and oscillate in phase, as required by Maxwell's
equations for a plane wave in free space.

Produces an animated GIF showing the two fields propagating together.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# ----------------------------
# Wave parameters
# ----------------------------
E0 = 1.0                 # Electric field amplitude
B0 = 1.0                 # Magnetic field amplitude
wavelength = 2.0          # meters
k = 2 * np.pi / wavelength    # wave number
frequency = 1.0            # Hz
omega = 2 * np.pi * frequency  # angular frequency

z = np.linspace(0, 4 * wavelength, 400)   # propagation axis
n_frames = 100
t_vals = np.linspace(0, 2 / frequency, n_frames)  # two full periods

# ----------------------------
# Figure setup
# ----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

def init():
    ax.set_xlim(0, z.max())
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("z  (direction of propagation)")
    ax.set_ylabel("x  (Electric field, E)")
    ax.set_zlabel("y  (Magnetic field, B)")
    ax.set_title("Plane EM Wave Propagation")
    return []

# Line objects: E field (in x), B field (in y), and the propagation axis
E_line, = ax.plot([], [], [], color="crimson", lw=2, label="E field")
B_line, = ax.plot([], [], [], color="royalblue", lw=2, label="B field")
axis_line, = ax.plot(z, np.zeros_like(z), np.zeros_like(z),
                      color="gray", lw=1, ls="--", alpha=0.6)

# A handful of field vectors (like little arrows) sampled along z,
# drawn as vertical/horizontal segments to suggest the vector field.
n_vectors = 20
sample_idx = np.linspace(0, len(z) - 1, n_vectors).astype(int)
E_vectors = [ax.plot([], [], [], color="crimson", lw=1, alpha=0.6)[0] for _ in range(n_vectors)]
B_vectors = [ax.plot([], [], [], color="royalblue", lw=1, alpha=0.6)[0] for _ in range(n_vectors)]

ax.legend(loc="upper right")

def animate(frame):
    t = t_vals[frame]
    Ex = E0 * np.sin(k * z - omega * t)
    By = B0 * np.sin(k * z - omega * t)

    # Update the smooth curves
    E_line.set_data(z, Ex)
    E_line.set_3d_properties(np.zeros_like(z))

    B_line.set_data(z, np.zeros_like(z))
    B_line.set_3d_properties(By)

    # Update the little field vectors at sampled points
    for i, idx in enumerate(sample_idx):
        zi = z[idx]
        E_vectors[i].set_data([zi, zi], [0, Ex[idx]])
        E_vectors[i].set_3d_properties([0, 0])

        B_vectors[i].set_data([zi, zi], [0, 0])
        B_vectors[i].set_3d_properties([0, By[idx]])

    ax.set_title(f"Plane EM Wave Propagation (t = {t:.2f} s)")
    return [E_line, B_line] + E_vectors + B_vectors

anim = FuncAnimation(fig, animate, frames=n_frames, init_func=init,
                      interval=50, blit=False, repeat=True)

plt.tight_layout()
plt.show()
