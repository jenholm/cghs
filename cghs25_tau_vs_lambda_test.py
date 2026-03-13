import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# Cosmology parameters
# ============================================================

Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70
Omega_inf = 50.0

# Hilbert sector parameters
S_c = 8.0
width = 2.0
epsilon_tilt = 0.02

kappa = 60.0
gamma_S = 0.08
mu_S = 0.002

# emergent clock parameters
alpha_floor = 1e-4

EXP_CLIP = 700


# ============================================================
# Helpers
# ============================================================

def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


# ============================================================
# Hilbert activation
# ============================================================

def logistic_u(S):
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)
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
# Expansion law
# ============================================================

def E_of(G, S):
    fS = f_internal(S)

    rad = Omega_r0 * fS * safe_exp(-4 * G)
    mat = Omega_m0 * fS * safe_exp(-3 * G)

    # use the gated vacuum version that behaved better
    vac = Omega_S(S) * logistic_u(S)

    rhs = rad + mat + vac
    rhs = np.maximum(rhs, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# Emergent clock law
# ============================================================

def alpha_of(G, S):
    """
    Positive clock-rate functional.
    """
    fS = f_internal(S)
    E = E_of(G, S)

    alpha = alpha_floor + 0.25 * fS + 0.75 * fS / (1.0 + E)
    return np.maximum(alpha, alpha_floor)


# ============================================================
# Dynamics
# ============================================================

def dG_dlambda(G, S):
    return E_of(G, S)


def dS_dlambda(G, S):
    source = mu_S * logistic_u(S)

    E = E_of(G, S)
    E_safe = np.maximum(E, 1e-60)

    potential = -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)
    damping = -gamma_S * S * (1.0 - logistic_u(S))

    return source + potential + damping


def dtau_dlambda(G, S):
    return alpha_of(G, S)


def system(lam, y):
    G, S, tau = y
    return [
        dG_dlambda(G, S),
        dS_dlambda(G, S),
        dtau_dlambda(G, S),
    ]


# ============================================================
# Integrate model
# ============================================================

lam_span = (0.0, 80.0)
lam_eval = np.linspace(0.0, 80.0, 2000)

initial_state = [-11.0, 0.1, 0.0]

sol = solve_ivp(
    system,
    lam_span,
    initial_state,
    t_eval=lam_eval,
    method="Radau",
    rtol=1e-8,
    atol=1e-10
)

if not sol.success:
    raise RuntimeError(sol.message)

lam = sol.t
G = sol.y[0]
S = sol.y[1]
tau = sol.y[2]

a = safe_exp(G)

E = E_of(G, S)
alpha = alpha_of(G, S)

# ============================================================
# Observable comparison
# ------------------------------------------------------------
# "Computational" rate and "emergent-time" rate
# ============================================================

H_lambda = E
H_tau = E / np.maximum(alpha, 1e-12)

# numerical cross-check for H_tau = dG/dtau
dG_dlam_num = np.gradient(G, lam)
dtau_dlam_num = np.gradient(tau, lam)
dtau_dlam_num = np.where(np.abs(dtau_dlam_num) < 1e-12, 1e-12, dtau_dlam_num)
H_tau_num = dG_dlam_num / dtau_dlam_num

# ============================================================
# Component terms and fractions
# ============================================================

rad_term = Omega_r0 * f_internal(S) * safe_exp(-4 * G)
mat_term = Omega_m0 * f_internal(S) * safe_exp(-3 * G)
vac_term = Omega_S(S) * logistic_u(S)

total_term = rad_term + mat_term + vac_term
total_term = np.where(total_term < 1e-12, 1e-12, total_term)

rad_frac = rad_term / total_term
mat_frac = mat_term / total_term
vac_frac = vac_term / total_term

# ============================================================
# Effective equation of state under both parameterizations
# ------------------------------------------------------------
# lambda-based proxy:
#   w_lambda = -1 - (2/3)(1/E)(dE/dG)
#
# tau-based proxy:
#   w_tau = -1 - (2/3)(1/H_tau)(dH_tau/dG)
# ============================================================

# dE/dG via chain rule
dE_dlam = np.gradient(E, lam)
dG_dlam_safe = np.where(np.abs(dG_dlam_num) < 1e-12, 1e-12, dG_dlam_num)
dE_dG = dE_dlam / dG_dlam_safe

E_safe = np.where(np.abs(E) < 1e-12, 1e-12, E)
w_lambda = -1.0 - (2.0 / 3.0) * (dE_dG / E_safe)

# dH_tau/dG via chain rule
dHtau_dlam = np.gradient(H_tau, lam)
dHtau_dG = dHtau_dlam / dG_dlam_safe

Htau_safe = np.where(np.abs(H_tau) < 1e-12, 1e-12, H_tau)
w_tau = -1.0 - (2.0 / 3.0) * (dHtau_dG / Htau_safe)

# Trim earliest transient for cleaner comparison
valid = tau > np.percentile(tau, 2)

# ============================================================
# Diagnostics
# ============================================================

fig, axs = plt.subplots(3, 2, figsize=(13, 12))

# ------------------------------------------------------------
# 1. emergent time map
# ------------------------------------------------------------
axs[0, 0].plot(lam, tau, color="black")
axs[0, 0].set_title("Emergent time $\\tau(\\lambda)$")
axs[0, 0].set_xlabel("$\\lambda$")
axs[0, 0].set_ylabel("$\\tau$")

# ------------------------------------------------------------
# 2. clock rate
# ------------------------------------------------------------
axs[0, 1].plot(lam, alpha, color="tab:blue")
axs[0, 1].set_title("Clock rate $\\alpha(G,S)=d\\tau/d\\lambda$")
axs[0, 1].set_xlabel("$\\lambda$")
axs[0, 1].set_ylabel("$\\alpha$")

# ------------------------------------------------------------
# 3. Hubble-like comparison
# ------------------------------------------------------------
axs[1, 0].plot(G, H_lambda, label="$H_\\lambda = dG/d\\lambda = E$")
axs[1, 0].plot(G, H_tau, label="$H_\\tau = dG/d\\tau = E/\\alpha$")
axs[1, 0].set_yscale("log")
axs[1, 0].set_title("Expansion rate under two parameterizations")
axs[1, 0].set_xlabel("$G = \\ln a$")
axs[1, 0].set_ylabel("rate")
axs[1, 0].legend()

# ------------------------------------------------------------
# 4. consistency check
# ------------------------------------------------------------
axs[1, 1].plot(lam, H_tau, label="$E/\\alpha$")
axs[1, 1].plot(lam, H_tau_num, "--", label="$(dG/d\\lambda)/(d\\tau/d\\lambda)$")
axs[1, 1].set_yscale("log")
axs[1, 1].set_title("Consistency check for emergent-time rate")
axs[1, 1].set_xlabel("$\\lambda$")
axs[1, 1].set_ylabel("$H_\\tau$")
axs[1, 1].legend()

# ------------------------------------------------------------
# 5. component fractions
# ------------------------------------------------------------
axs[2, 0].plot(G, rad_frac, label="radiation fraction")
axs[2, 0].plot(G, mat_frac, label="matter fraction")
axs[2, 0].plot(G, vac_frac, label="Hilbert vacuum fraction")
axs[2, 0].set_title("Component dominance fractions")
axs[2, 0].set_xlabel("$G = \\ln a$")
axs[2, 0].set_ylabel("fraction of total")
axs[2, 0].legend()

# ------------------------------------------------------------
# 6. compare w under lambda vs tau
# ------------------------------------------------------------
axs[2, 1].plot(G[valid], w_lambda[valid], label="$w_{\\lambda}$", color="gray")
axs[2, 1].plot(G[valid], w_tau[valid], label="$w_{\\tau}$", color="black")
axs[2, 1].axhline(-1, color="red", linestyle=":", label="$w=-1$")
axs[2, 1].axhline(0, color="gray", linestyle="--", label="$w=0$")
axs[2, 1].axhline(1/3, color="gray", linestyle="-.", label="$w=1/3$")
axs[2, 1].set_title("Equation-of-state comparison")
axs[2, 1].set_xlabel("$G = \\ln a$")
axs[2, 1].set_ylabel("$w$")
axs[2, 1].legend()

plt.tight_layout()
plt.savefig("cghs_tau_vs_lambda_test.png", dpi=300)
plt.show()