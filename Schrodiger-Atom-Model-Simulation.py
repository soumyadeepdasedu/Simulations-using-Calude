"""
3D Schrodinger Atomic Model Simulation
=========================================
Unlike the Bohr model (electrons on fixed circular orbits), the quantum
mechanical model describes an electron's location as a PROBABILITY CLOUD
governed by the Schrodinger equation. For the hydrogen atom, this equation
can be solved exactly, giving wavefunctions psi(r, theta, phi) defined by
three quantum numbers:

    n (principal)      -> energy level / shell        (1, 2, 3, ...)
    l (azimuthal)       -> orbital shape (s, p, d, f)   (0 to n-1)
    m (magnetic)        -> orbital orientation          (-l to +l)

The probability of finding the electron at a point is |psi|^2. This script
computes the exact hydrogen wavefunction, then uses rejection sampling to
scatter thousands of points with density proportional to |psi|^2 -- giving
you the real, textbook "electron cloud" shape (s, p, d orbitals).

Requires: numpy, scipy, matplotlib
Install with: pip install numpy scipy matplotlib
"""

import math
import numpy as np
from scipy.special import genlaguerre
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# scipy >= 1.15 renamed sph_harm -> sph_harm_y with a new argument order
# (theta=polar, phi=azimuthal). Older scipy used sph_harm(m, l, phi, theta).
# This wrapper keeps the script working on either version.
try:
    from scipy.special import sph_harm_y

    def spherical_harmonic(l, m, theta, phi):
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm

    def spherical_harmonic(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)


# ----------------------------
# Quantum numbers (tweak these)
# ----------------------------
N = 3        # principal quantum number  (energy level / shell)
L = 2        # azimuthal quantum number  (0=s, 1=p, 2=d, 3=f) -- must be < N
M = 0        # magnetic quantum number   (-L <= M <= L)

N_POINTS = 40000      # number of electron positions to sample
BOX_SIZE = None        # half-width of sampling box in Bohr radii (auto if None)
A0 = 1.0                # Bohr radius, set to 1 for natural (dimensionless) units


def radial_wavefunction(n, l, r):
    """Radial part R_nl(r) of the hydrogen wavefunction (in units of a0)."""
    rho = 2 * r / (n * A0)
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    norm = np.sqrt(
        (2 / (n * A0)) ** 3 * math.factorial(n - l - 1)
        / (2 * n * math.factorial(n + l))
    )
    return norm * np.exp(-rho / 2) * rho ** l * laguerre


def psi(n, l, m, r, theta, phi):
    """Full hydrogen wavefunction psi_nlm(r, theta, phi)."""
    R = radial_wavefunction(n, l, r)
    Y = spherical_harmonic(l, m, theta, phi)
    return R * Y


def sample_electron_cloud(n, l, m, n_points, box_size):
    """
    Rejection-sample 3D points (x, y, z) with density proportional to
    |psi_nlm|^2, i.e. the true quantum probability distribution.
    """
    samples = []

    # Estimate a reasonable peak probability density to normalize acceptance
    r_test = np.linspace(0.01, box_size, 400)
    theta_test = np.linspace(0, np.pi, 60)
    phi_test = np.linspace(0, 2 * np.pi, 60)
    RR, TT = np.meshgrid(r_test, theta_test, indexing="ij")
    prob_slice = np.abs(psi(n, l, m, RR, TT, 0.0)) ** 2
    max_density = prob_slice.max() * 1.5  # small safety margin

    batch = max(n_points * 4, 20000)
    while len(samples) < n_points:
        x = np.random.uniform(-box_size, box_size, batch)
        y = np.random.uniform(-box_size, box_size, batch)
        z = np.random.uniform(-box_size, box_size, batch)

        r = np.sqrt(x**2 + y**2 + z**2)
        r_safe = np.where(r == 0, 1e-9, r)
        theta = np.arccos(np.clip(z / r_safe, -1, 1))
        phi = np.arctan2(y, x)

        prob = np.abs(psi(n, l, m, r, theta, phi)) ** 2
        accept_prob = prob / max_density
        rand_vals = np.random.uniform(0, 1, batch)

        accepted = rand_vals < accept_prob
        pts = np.column_stack((x[accepted], y[accepted], z[accepted]))
        samples.extend(pts.tolist())

    samples = np.array(samples[:n_points])
    return samples[:, 0], samples[:, 1], samples[:, 2]


def main():
    if L >= N:
        raise ValueError("l must be less than n")
    if abs(M) > L:
        raise ValueError("m must satisfy -l <= m <= l")

    box_size = BOX_SIZE if BOX_SIZE else 4 * N**2 * A0  # rough extent of orbital

    print(f"Sampling {N_POINTS} electron positions for n={N}, l={L}, m={M} ...")
    x, y, z = sample_electron_cloud(N, L, M, N_POINTS, box_size)
    print("Done sampling.")

    orbital_names = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
    orbital_label = f"{N}{orbital_names.get(L, '?')}"

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # Color points by distance from nucleus for a nice glowing-cloud effect
    r = np.sqrt(x**2 + y**2 + z**2)
    sc = ax.scatter(x, y, z, c=r, cmap="plasma", s=2, alpha=0.35, linewidths=0)

    # Mark the nucleus
    ax.scatter([0], [0], [0], color="cyan", s=120, edgecolors="white",
               linewidths=1, zorder=10)

    lim = box_size * 0.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.set_title(
        f"Schrodinger Atomic Model — Orbital {orbital_label} "
        f"(n={N}, l={L}, m={M})",
        color="white", fontsize=13
    )

    plt.tight_layout()
    plt.show()

    # To save a static image instead of / in addition to an interactive window:
    # fig.savefig("orbital_cloud.png", dpi=200, facecolor="black")


if __name__ == "__main__":
    main()
