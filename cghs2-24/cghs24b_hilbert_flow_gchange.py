import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================
# Parameters
# ============================================================

Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70
Omega_inf = 50.0

S_c = 8.0
width = 2.0
epsilon_tilt = 0.02

kappa = 60.0
gamma_S = 0.08
mu_S = 0.002

EXP_CLIP = 700.0

# Geometry slices to inspect
G_values = [-12, -8, -4, -2, 0]


# ============================================================
# Helper
# ============================================================

def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


# ============================================================
# Hilbert activation structure
# ============================================================

def logistic_u(S):
    x = (S - S_c) / width
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(x))


def f_internal(S):
    return 1.0 - logistic_u(S)


# ============================================================
# Hilbert vacuum potential
# ============================================================

def Omega_S(S):

    u = logistic_u(S)

    tilt = epsilon_tilt * np.tanh((S - S_c) / width)

    return Omega_L_floor + Omega_inf * u + tilt


def dOmegaS_dS(S):

    u = logistic_u(S)
    du = -(1.0 / width) * u * (1.0 - u)

    x = (S - S_c) / width
    dtilt = (epsilon_tilt / width) * (1.0 / np.cosh(np.clip(x, -50, 50))**2)

    return Omega_inf * du + dtilt


# ============================================================
# Expansion function
# ============================================================

def E_of(G, S):

    fS = f_internal(S)

    term_r = Omega_r0 * fS * safe_exp(-4.0 * G)
    term_m = Omega_m0 * fS * safe_exp(-3.0 * G)
    term_s = Omega_S(S)

    rhs = term_r + term_m + term_s
    rhs = np.maximum(rhs, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# Hilbert dynamics
# ============================================================

def dS_source(S):
    return mu_S * logistic_u(S)


def dS_potential(G, S):

    E = E_of(G, S)
    E_safe = np.maximum(E, 1e-60)

    return -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)


def dS_damp(S):
    return -gamma_S * S * (1.0 - logistic_u(S))


def dS_dlambda(G, S):

    return (
        dS_source(S)
        + dS_potential(G, S)
        + dS_damp(S)
    )


# ============================================================
# Evaluation grid
# ============================================================

S = np.linspace(0, 20, 1200)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(11,7))

colors = cm.viridis(np.linspace(0,1,len(G_values)))

for G, c in zip(G_values, colors):

    flow = dS_dlambda(G, S)

    plt.plot(
        S,
        flow,
        color=c,
        linewidth=2.5,
        label=f"G = {G}"
    )


# reference line
plt.axhline(0, color="black", linestyle="--", linewidth=1)

# transition marker
plt.axvline(S_c, color="red", linestyle=":", linewidth=2, label=f"$S_c$ = {S_c}")

plt.xlabel("Hilbert order parameter S")
plt.ylabel(r"$dS/d\lambda$")
plt.title("Hilbert Flow Across Geometry Slices")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Hilbert Flow gchange.png", dpi=300)
plt.show()