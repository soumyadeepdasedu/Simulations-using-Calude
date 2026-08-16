"""
Photon Wave Packet Simulation (1D, spectral/FFT propagation)
----------------------------------------------------------------
Builds a localized photon wave packet as a superposition of plane
waves (a Gaussian distribution in k-space around a central
wavenumber k0), then propagates it exactly using the photon's
dispersion relation:

    omega(k) = c * k      (linear -> non-dispersive)

Because the dispersion relation is LINEAR, every Fourier component
travels at the same speed c, so the packet's envelope moves rigidly
without spreading -- this is the key physical difference between a
photon (massless, non-dispersive) and a massive-particle wave packet
(quadratic dispersion, spreads over time). A "dispersive" toggle is
included below so you can see the contrast by adding a small
quadratic term to omega(k), similar to what a matter wave (electron)
wave packet would do.

Each frame is computed as:
    1. FFT the initial wave packet once to get its spectrum A(k)
    2. Multiply by the exact time-evolution phase exp(-i*omega(k)*t)
    3. Inverse FFT back to real space to get psi(x, t)

This is numerically exact (no time-stepping error), unlike
finite-difference wave solvers.

Runs live in a matplotlib window as the script executes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Parameters
# ----------------------------
c = 1.0                 # speed of light (natural units)
k0 = 40.0                 # central wavenumber (sets the carrier frequency)
sigma_k = 3.0               # spread in k-space -> sets packet width in x
                             # (narrow sigma_k = wide packet, and vice versa)
x_min, x_max = -20, 60
N = 4096                       # grid resolution
dispersive = False              # set True to see a matter-wave packet spread & chirp
dispersion_coeff = 0.05          # strength of the added quadratic term if dispersive

dt = 0.05                # simulated time advance per frame
frames_per_reset = 400     # loop the animation after this many frames

x = np.linspace(x_min, x_max, N, endpoint=False)
dx = x[1] - x[0]
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)   # wavenumber grid matching the FFT convention

# ----------------------------
# Initial wave packet: Gaussian envelope * carrier wave, centered at x0
# ----------------------------
x0 = -10.0                 # starting position of the packet's center
sigma_x = 1.0 / sigma_k       # real-space width consistent with sigma_k (min uncertainty)

psi0 = np.exp(-(x - x0)**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)   # normalize probability

psi0_k = np.fft.fft(psi0)   # spectrum, computed once

# ----------------------------
# Dispersion relation
# ----------------------------
def omega(k):
    if dispersive:
        # add a small quadratic term (like a massive particle) on top
        # of the linear photon term, purely to illustrate spreading
        return c * k + dispersion_coeff * k**2
    return c * k   # true photon: exactly linear, so no spreading

phase_velocity_factor = omega(k)

def psi_at(t):
    psi_k_t = psi0_k * np.exp(-1j * phase_velocity_factor * t)
    return np.fft.ifft(psi_k_t)

# ----------------------------
# Figure setup
# ----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

line_real, = ax1.plot([], [], color="crimson", lw=1, label="Re[E(x,t)]")
line_env_pos, = ax1.plot([], [], color="black", lw=1.2, ls="--", label="envelope |ψ|")
line_env_neg, = ax1.plot([], [], color="black", lw=1.2, ls="--")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(-1.2, 1.2)
ax1.set_ylabel("Field amplitude")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("Photon Wave Packet — Field Oscillation & Envelope")

line_prob, = ax2.plot([], [], color="royalblue", lw=1.5)
fill = None
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(0, None)
ax2.set_xlabel("x")
ax2.set_ylabel("|ψ(x,t)|²  (probability density)")

time_text = ax1.text(0.02, 0.92, "", transform=ax1.transAxes, family="monospace")

def init():
    line_real.set_data([], [])
    line_env_pos.set_data([], [])
    line_env_neg.set_data([], [])
    line_prob.set_data([], [])
    return line_real, line_env_pos, line_env_neg, line_prob

def animate(frame):
    global fill
    t = (frame % frames_per_reset) * dt
    psi = psi_at(t)

    amp = np.abs(psi)
    scale = 1.0 / amp.max() if amp.max() > 0 else 1.0  # keep field visually normalized

    line_real.set_data(x, psi.real * scale)
    line_env_pos.set_data(x, amp * scale)
    line_env_neg.set_data(x, -amp * scale)

    prob_density = amp**2
    line_prob.set_data(x, prob_density)
    ax2.set_ylim(0, prob_density.max() * 1.2 + 1e-9)

    if fill is not None:
        fill.remove()
    fill = ax2.fill_between(x, prob_density, color="royalblue", alpha=0.3)

    time_text.set_text(f"t = {t:5.2f}   mode: {'dispersive' if dispersive else 'photon (non-dispersive)'}")
    return line_real, line_env_pos, line_env_neg, line_prob, fill

anim = FuncAnimation(fig, animate, init_func=init, interval=30,
                      blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
