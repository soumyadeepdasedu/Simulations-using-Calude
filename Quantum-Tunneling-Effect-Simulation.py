"""
Quantum Wave Packet & Tunneling Simulation (1D, Crank-Nicolson)
---------------------------------------------------------------------
Solves the time-dependent Schrodinger equation

    i * hbar * d(psi)/dt = [ -hbar^2/(2m) * d^2/dx^2 + V(x) ] psi

for a Gaussian wave packet incident on a rectangular potential
barrier, using natural units (hbar = m = 1).

Numerical method: Crank-Nicolson finite differencing. This
discretizes the Hamiltonian into a tridiagonal matrix H and steps
forward with:

    (I + i*dt/2 * H) psi(t+dt) = (I - i*dt/2 * H) psi(t)

Crank-Nicolson is used (rather than simple forward-Euler time
stepping) because it is unconditionally stable and, critically,
*unitary* -- total probability is exactly conserved at every step,
which matters a lot for a simulation whose whole point is to track
how probability splits between "reflected" and "transmitted"
(tunneled) parts.

The left-hand matrix A = (I + i*dt/2*H) doesn't change over time
(the potential is static), so it's LU-factorized ONCE up front and
then reused every frame -- this is what makes the live animation
fast even though we're technically solving a linear system at every
timestep.

The key physics on display: even when the packet's mean energy is
BELOW the barrier height (classically forbidden), a portion of the
probability density leaks through to the other side -- quantum
tunneling. This underlies STM, flash memory (electrons tunneling
through the floating-gate oxide), and fusion reaction rates (nuclei
tunneling through their mutual Coulomb barrier).

Runs live in a matplotlib window as the script executes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import splu

# ----------------------------
# Grid & physical parameters (natural units: hbar = m = 1)
# ----------------------------
x_min, x_max = -40, 60
N = 2000
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]
dt = 0.05

k0 = 3.0                    # central momentum of the wave packet -> mean energy E ≈ k0^2/2
sigma0 = 3.0                  # initial width of the wave packet
x0 = -20.0                     # starting center position

E_mean = 0.5 * k0**2             # approximate mean kinetic energy of the packet

# ----------------------------
# Potential barrier: rectangular, centered at x=0
# ----------------------------
barrier_width = 1.2
barrier_height = 5.0            # set > E_mean for a genuinely classically-forbidden barrier
V = np.zeros(N)
V[(x > -barrier_width / 2) & (x < barrier_width / 2)] = barrier_height

print(f"Mean packet energy E ~ {E_mean:.2f}, barrier height V0 = {barrier_height:.2f} "
      f"({'tunneling regime, E < V0' if E_mean < barrier_height else 'over-barrier regime, E > V0'})")

# ----------------------------
# Initial Gaussian wave packet moving right (toward the barrier)
# ----------------------------
psi = np.exp(-(x - x0)**2 / (4 * sigma0**2)) * np.exp(1j * k0 * x)
psi = psi.astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)   # normalize

# ----------------------------
# Build the (static) Hamiltonian as a sparse tridiagonal matrix
#   H_ii     = 1/dx^2 + V_i
#   H_i,i+-1 = -1/(2*dx^2)
# ----------------------------
main_diag = 1.0 / dx**2 + V
off_diag = -0.5 / dx**2 * np.ones(N - 1)
H = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csc")

identity = diags([np.ones(N)], [0], format="csc")
A = (identity + 1j * dt / 2 * H).tocsc()
B = (identity - 1j * dt / 2 * H).tocsc()

# LU-factorize A once; this is the expensive part, done a single time
A_lu = splu(A)

def cn_step(psi):
    rhs = B.dot(psi)
    return A_lu.solve(rhs)

# ----------------------------
# Figure setup
# ----------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(x_min, x_max)
prob_density = np.abs(psi)**2
ax.set_ylim(0, prob_density.max() * 4)
ax.set_xlabel("x")
ax.set_ylabel("|ψ(x,t)|²")
ax.set_title("Quantum Wave Packet vs Potential Barrier — Tunneling")

line_prob, = ax.plot(x, prob_density, color="royalblue", lw=1.5, label="|ψ|² (probability density)")
fill = ax.fill_between(x, prob_density, color="royalblue", alpha=0.3)

# Show the barrier on a secondary scale so its height is visually meaningful
ax2 = ax.twinx()
ax2.plot(x, V, color="black", lw=1.2, ls="--", alpha=0.7, label="V(x) barrier")
ax2.set_ylabel("Potential V(x)")
ax2.set_ylim(0, barrier_height * 3)

time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, family="monospace")
stats_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, family="monospace", fontsize=9)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

sim_time = 0.0
steps_per_frame = 2

# Regions for computing reflected / transmitted probability, split at the barrier center
left_mask = x < -barrier_width / 2
right_mask = x > barrier_width / 2

def animate(frame):
    global psi, sim_time, fill

    for _ in range(steps_per_frame):
        psi = cn_step(psi)
        sim_time += dt

    prob = np.abs(psi)**2
    line_prob.set_data(x, prob)

    if fill is not None:
        fill.remove()
    fill = ax.fill_between(x, prob, color="royalblue", alpha=0.3)

    total_prob = np.sum(prob) * dx
    reflected = np.sum(prob[left_mask]) * dx
    transmitted = np.sum(prob[right_mask]) * dx

    time_text.set_text(f"t = {sim_time:6.2f}")
    stats_text.set_text(
        f"total P  = {total_prob:.4f}  (should stay ≈ 1.0)\n"
        f"reflected P (left)  = {reflected:.4f}\n"
        f"transmitted P (right) = {transmitted:.4f}"
    )

    return line_prob, fill, time_text, stats_text

anim = FuncAnimation(fig, animate, interval=25, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
