"""
Chaotic Attractor Simulation — Lorenz System (3D, live)
-----------------------------------------------------------
Integrates the Lorenz equations, the classic model of deterministic
chaos (originally derived from a simplified atmospheric convection
model — this is literally why "chaos theory" and "the butterfly
effect" entered the popular vocabulary):

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

With the classic parameters (sigma=10, rho=28, beta=8/3) the system
never settles into a fixed point or a repeating cycle. Instead it
traces a bounded, non-repeating trajectory that folds back on itself
infinitely -- the "strange attractor" -- forming the two-lobed
butterfly shape.

To make the chaos itself visible (not just the shape), TWO
trajectories are integrated side by side from almost identical
starting points (differing by 1e-5). Because nearby trajectories in
a chaotic system diverge exponentially (sensitive dependence on
initial conditions), you'll see the two curves track each other
closely at first, then visibly peel apart and desynchronize -- the
"butterfly effect" made literal.

Integration uses RK4 (4th-order Runge-Kutta), which is accurate
enough for the trajectory shape to be trustworthy over long runs.

Runs live in a 3D matplotlib window as the script executes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------
# Lorenz system parameters (classic chaotic regime)
# ----------------------------
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

dt = 0.008                  # integration time step
steps_per_frame = 3            # sub-steps between redraws
trail_length = 3000              # how many recent points to keep drawn (older points fade off)

def lorenz_deriv(state, sigma=sigma, rho=rho, beta=beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt):
    k1 = lorenz_deriv(state)
    k2 = lorenz_deriv(state + 0.5 * dt * k1)
    k3 = lorenz_deriv(state + 0.5 * dt * k2)
    k4 = lorenz_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# ----------------------------
# Two nearly-identical initial conditions, to visualize the
# butterfly effect: sensitive dependence on initial conditions.
# ----------------------------
state_a = np.array([1.0, 1.0, 1.0])
state_b = state_a + np.array([1e-5, 0.0, 0.0])  # tiny perturbation

history_a = np.tile(state_a, (trail_length, 1))
history_b = np.tile(state_b, (trail_length, 1))

# ----------------------------
# Figure setup
# ----------------------------
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-25, 25)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz Attractor — Deterministic Chaos (two nearly identical starts)")

line_a, = ax.plot([], [], [], color="deepskyblue", lw=0.8, alpha=0.9, label="trajectory A")
line_b, = ax.plot([], [], [], color="crimson", lw=0.8, alpha=0.9, label="trajectory B (Δx₀ = 1e-5)")
point_a, = ax.plot([], [], [], color="deepskyblue", marker="o", markersize=4)
point_b, = ax.plot([], [], [], color="crimson", marker="o", markersize=4)

sim_time = 0.0
time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, family="monospace")
ax.legend(loc="upper right", fontsize=8)

def animate(frame):
    global state_a, state_b, history_a, history_b, sim_time

    for _ in range(steps_per_frame):
        state_a = rk4_step(state_a, dt)
        state_b = rk4_step(state_b, dt)
        sim_time += dt

    history_a = np.roll(history_a, -1, axis=0)
    history_a[-1] = state_a
    history_b = np.roll(history_b, -1, axis=0)
    history_b[-1] = state_b

    line_a.set_data(history_a[:, 0], history_a[:, 1])
    line_a.set_3d_properties(history_a[:, 2])
    line_b.set_data(history_b[:, 0], history_b[:, 1])
    line_b.set_3d_properties(history_b[:, 2])

    point_a.set_data([state_a[0]], [state_a[1]])
    point_a.set_3d_properties([state_a[2]])
    point_b.set_data([state_b[0]], [state_b[1]])
    point_b.set_3d_properties([state_b[2]])

    separation = np.linalg.norm(state_a - state_b)
    time_text.set_text(f"t = {sim_time:6.2f}   |A - B| = {separation:8.4f}")

    return line_a, line_b, point_a, point_b, time_text

anim = FuncAnimation(fig, animate, interval=15, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
