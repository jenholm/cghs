import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters: match current model as closely as possible
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

# ------------------------------------------------------------
# Choose a fixed geometry slice for the diagnostic
# ------------------------------------------------------------
# G = ln(a).  G=0 means a=1.
# You can try G = -10, -5, 0, +2 later to compare slices.
G_fixed = 0.0


# ============================================================
# Helpers
# ============================================================
def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


# ============================================================
# Hilbert activation structure
# ============================================================
def logistic_u(S):
    """
    Plateau control:
      S << S_c -> u ~ 1
      S >> S_c -> u ~ 0
    """
    x = (S - S_c) / width
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(x))


def f_internal(S):
    """
    Released fraction.
    """
    return 1.0 - logistic_u(S)


# ============================================================
# Hilbert vacuum potential
# ============================================================
def Omega_S(S):
    """
    Hilbert vacuum sector:
      floor + inflation plateau + bounded tilt
    """
    u = logistic_u(S)
    tilt = epsilon_tilt * np.tanh((S - S_c) / width)
    return Omega_L_floor + Omega_inf * u + tilt


def dOmegaS_dS(S):
    """
    d/dS Omega_S(S)
    """
    u = logistic_u(S)
    du = -(1.0 / width) * u * (1.0 - u)

    x = (S - S_c) / width
    dtilt = (epsilon_tilt / width) * (1.0 / np.cosh(np.clip(x, -50, 50))**2)

    return Omega_inf * du + dtilt


# ============================================================
# Expansion function at fixed G
# ============================================================
def E_of(G, S):
    """
    E(G,S)^2 = Omega_r0 f(S)e^{-4G}
             + Omega_m0 f(S)e^{-3G}
             + Omega_S(S)
    """
    fS = f_internal(S)

    term_r = Omega_r0 * fS * safe_exp(-4.0 * G)
    term_m = Omega_m0 * fS * safe_exp(-3.0 * G)
    term_s = Omega_S(S)

    rhs = term_r + term_m + term_s
    rhs = np.maximum(rhs, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# dS/dlambda and its component terms
# ============================================================
def dS_source(S):
    """
    Source term driving pre-release growth.
    """
    return mu_S * logistic_u(S)


def dS_potential(G, S):
    """
    Potential-force term.
    """
    E = E_of(G, S)
    E_safe = np.maximum(E, 1e-60)
    return -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)


def dS_damp(S):
    """
    Post-release damping.
    """
    return -gamma_S * S * (1.0 - logistic_u(S))


def dS_dlambda(G, S):
    """
    Full Hilbert evolution law at fixed geometry slice G.
    """
    return dS_source(S) + dS_potential(G, S) + dS_damp(S)


# ============================================================
# Evaluate over S range
# ============================================================
S = np.linspace(0.0, 20.0, 1200)

source_vals = dS_source(S)
pot_vals = dS_potential(G_fixed, S)
damp_vals = dS_damp(S)
total_vals = dS_dlambda(G_fixed, S)


# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(11, 7))

plt.plot(S, source_vals, label="source term  $\\mu_S u(S)$", linewidth=2)
plt.plot(S, pot_vals, label="potential term  $-(\\kappa/3E)\\, d\\Omega_S/dS$", linewidth=2)
plt.plot(S, damp_vals, label="damping term  $-\\gamma_S S(1-u(S))$", linewidth=2)
plt.plot(S, total_vals, label="total  $dS/d\\lambda$", linewidth=3, color="black")

plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
plt.axvline(S_c, color="red", linestyle=":", linewidth=1.5, label=f"$S_c = {S_c}$")

plt.xlabel("Hilbert order parameter S")
plt.ylabel("Contribution to $dS/d\\lambda$")
plt.title(f"Hilbert Flow Diagnostic at Fixed Geometry Slice  G = {G_fixed}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hilbert_flow.png", dpi=300)
plt.show()