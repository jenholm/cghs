import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters copied from current model
# ============================================================
Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70
Omega_inf = 1.0

S_c = 8.0
width = 2.0
epsilon_tilt = 0.02

kappa = 60.0
gamma_S = 0.08
mu_S = 0.002

EXP_CLIP = 700


# ============================================================
# Helpers
# ============================================================
def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


def logistic_u(S):
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(x))


def f_internal(S):
    return 1.0 - logistic_u(S)


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


def expansion_terms(G, S):
    fS = f_internal(S)
    rad = Omega_r0 * fS * safe_exp(-4 * G)
    mat = Omega_m0 * fS * safe_exp(-3 * G)
    vac = Omega_S(S)
    return rad, mat, vac


def E_of(G, S):
    rad, mat, vac = expansion_terms(G, S)
    rhs = np.maximum(rad + mat + vac, 1e-12)
    return np.sqrt(rhs)


def dS_terms(G, S):
    u = logistic_u(S)
    source = mu_S * u

    E = np.maximum(E_of(G, S), 1e-60)
    potential = -(kappa / (3 * E)) * dOmegaS_dS(S)
    damping = -gamma_S * S * (1.0 - u)

    net = source + potential + damping
    return source, potential, damping, net


# ============================================================
# Grids
# ============================================================
S_grid = np.linspace(0, 20, 1200)
G_samples = [-11, -9, -7, -5, -3, 0]


# ============================================================
# Base activation / potential plots
# ============================================================
u_vals = logistic_u(S_grid)
f_vals = f_internal(S_grid)
Omega_vals = Omega_S(S_grid)
dOmega_vals = dOmegaS_dS(S_grid)

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

axs[0, 0].plot(S_grid, u_vals, label="u(S)")
axs[0, 0].plot(S_grid, f_vals, label="f_internal(S)")
axs[0, 0].axvline(S_c, color="red", linestyle=":")
axs[0, 0].set_title("Activation functions")
axs[0, 0].set_xlabel("S")
axs[0, 0].legend()
axs[0, 0].grid(alpha=0.3)

axs[0, 1].plot(S_grid, Omega_vals, label="Omega_S(S)")
axs[0, 1].axvline(S_c, color="red", linestyle=":")
axs[0, 1].set_title("Vacuum potential")
axs[0, 1].set_xlabel("S")
axs[0, 1].legend()
axs[0, 1].grid(alpha=0.3)

axs[1, 0].plot(S_grid, dOmega_vals, label="dOmegaS/dS")
axs[1, 0].axhline(0, color="black", linestyle=":")
axs[1, 0].axvline(S_c, color="red", linestyle=":")
axs[1, 0].set_title("Vacuum gradient")
axs[1, 0].set_xlabel("S")
axs[1, 0].legend()
axs[1, 0].grid(alpha=0.3)

for G in G_samples:
    E_vals = E_of(G, S_grid)
    axs[1, 1].plot(S_grid, E_vals, label=f"G={G}")

axs[1, 1].set_yscale("log")
axs[1, 1].set_title("Expansion scale E(G,S)")
axs[1, 1].set_xlabel("S")
axs[1, 1].legend()
axs[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cghs25_diag_terms_base.png", dpi=250)
plt.show()


# ============================================================
# Expansion-term audit for fixed G values
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
axs = axs.ravel()

for ax, G in zip(axs, G_samples):
    rad, mat, vac = expansion_terms(G, S_grid)
    ax.plot(S_grid, rad, label="radiation")
    ax.plot(S_grid, mat, label="matter")
    ax.plot(S_grid, vac, label="vacuum")
    ax.set_yscale("log")
    ax.set_title(f"Expansion terms at G={G}")
    ax.set_xlabel("S")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("cghs25_diag_expansion_terms.png", dpi=250)
plt.show()


# ============================================================
# dS/dlambda decomposition
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
axs = axs.ravel()

for ax, G in zip(axs, G_samples):
    source, potential, damping, net = dS_terms(G, S_grid)
    ax.plot(S_grid, source, label="source")
    ax.plot(S_grid, potential, label="potential")
    ax.plot(S_grid, damping, label="damping")
    ax.plot(S_grid, net, label="net", linestyle="--", linewidth=2)
    ax.axhline(0, color="black", linestyle=":")
    ax.set_title(f"dS/dλ terms at G={G}")
    ax.set_xlabel("S")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("cghs25_diag_dS_terms.png", dpi=250)
plt.show()


# ============================================================
# Point audit at current initial condition
# ============================================================
G0 = -11.0
S0 = 0.1

u0 = logistic_u(S0)
f0 = f_internal(S0)
rad0, mat0, vac0 = expansion_terms(G0, S0)
E0 = E_of(G0, S0)
src0, pot0, damp0, net0 = dS_terms(G0, S0)

print("\n=== Initial-condition audit ===")
print(f"G0 = {G0}")
print(f"S0 = {S0}")
print(f"u(S0) = {u0:.6e}")
print(f"f_internal(S0) = {f0:.6e}")
print(f"Omega_S(S0) = {vac0:.6e}")
print(f"radiation term = {rad0:.6e}")
print(f"matter term    = {mat0:.6e}")
print(f"E(G0,S0)       = {E0:.6e}")
print(f"source         = {src0:.6e}")
print(f"potential      = {pot0:.6e}")
print(f"damping        = {damp0:.6e}")
print(f"net dS/dλ      = {net0:.6e}")