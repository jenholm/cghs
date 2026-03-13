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
    vac = Omega_S(S) * logistic_u(S)

    rhs = rad + mat + vac
    rhs = np.maximum(rhs, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# Emergent clock law
# ============================================================

def alpha_of(G, S):
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
lam_eval = np.linspace(0.0, 80.0, 2500)

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
# Hubble-like rates
# ============================================================

H_lambda = E
H_tau = E / np.maximum(alpha, 1e-12)

# numerical consistency check
dG_dlam_num = np.gradient(G, lam)
dtau_dlam_num = np.gradient(tau, lam)
dtau_dlam_num = np.where(np.abs(dtau_dlam_num) < 1e-12, 1e-12, dtau_dlam_num)
H_tau_num = dG_dlam_num / dtau_dlam_num

# ============================================================
# Horizon proxies
# ============================================================

RH_lambda = 1.0 / np.maximum(H_lambda, 1e-12)
RH_tau = 1.0 / np.maximum(H_tau, 1e-12)

comov_lambda = 1.0 / np.maximum(a * H_lambda, 1e-12)
comov_tau = 1.0 / np.maximum(a * H_tau, 1e-12)

# ============================================================
# Comoving horizon integrals
# ------------------------------------------------------------
# lambda-based proxy:
#   eta_lambda = ∫ dλ / a(λ)
#
# tau-based:
#   eta_tau = ∫ dτ / a(τ) = ∫ alpha dλ / a(λ)
# ============================================================

dlam = np.gradient(lam)

eta_lambda = np.cumsum(dlam / np.maximum(a, 1e-12))
eta_tau = np.cumsum((alpha * dlam) / np.maximum(a, 1e-12))

# normalize to compare shapes
eta_lambda_norm = eta_lambda / np.max(eta_lambda)
eta_tau_norm = eta_tau / np.max(eta_tau)

# ============================================================
# Fractional difference diagnostic
# ============================================================

frac_diff_H = np.abs(H_tau - H_lambda) / np.maximum(np.abs(H_lambda), 1e-12)
frac_diff_comov = np.abs(comov_tau - comov_lambda) / np.maximum(np.abs(comov_lambda), 1e-12)

# ============================================================
# Plots
# ============================================================

fig, axs = plt.subplots(3, 2, figsize=(13, 12))

# ------------------------------------------------------------
# 1. emergent time mapping
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
# 3. Hubble comparison
# ------------------------------------------------------------
axs[1, 0].plot(G, H_lambda, label="$H_\\lambda = E$")
axs[1, 0].plot(G, H_tau, label="$H_\\tau = E/\\alpha$")
axs[1, 0].set_yscale("log")
axs[1, 0].set_title("Expansion-rate comparison")
axs[1, 0].set_xlabel("$G = \\ln a$")
axs[1, 0].set_ylabel("rate")
axs[1, 0].legend()

# ------------------------------------------------------------
# 4. comoving horizon comparison
# ------------------------------------------------------------
axs[1, 1].plot(G, comov_lambda, label="$(aH_\\lambda)^{-1}$")
axs[1, 1].plot(G, comov_tau, label="$(aH_\\tau)^{-1}$")
axs[1, 1].set_yscale("log")
axs[1, 1].set_title("Comoving horizon comparison")
axs[1, 1].set_xlabel("$G = \\ln a$")
axs[1, 1].set_ylabel("comoving horizon proxy")
axs[1, 1].legend()

# ------------------------------------------------------------
# 5. integrated horizon comparison
# ------------------------------------------------------------
axs[2, 0].plot(G, eta_lambda_norm, label="$\\eta_\\lambda$")
axs[2, 0].plot(G, eta_tau_norm, label="$\\eta_\\tau$")
axs[2, 0].set_title("Integrated horizon comparison")
axs[2, 0].set_xlabel("$G = \\ln a$")
axs[2, 0].set_ylabel("normalized conformal-like horizon")
axs[2, 0].legend()

# ------------------------------------------------------------
# 6. fractional difference
# ------------------------------------------------------------
axs[2, 1].plot(G, frac_diff_H, label="fractional diff in $H$")
axs[2, 1].plot(G, frac_diff_comov, label="fractional diff in $(aH)^{-1}$")
axs[2, 1].set_yscale("log")
axs[2, 1].set_title("How much does $\\tau$ actually change observables?")
axs[2, 1].set_xlabel("$G = \\ln a$")
axs[2, 1].set_ylabel("fractional difference")
axs[2, 1].legend()

plt.tight_layout()
plt.savefig("cghs_horizon_evolution_test.png", dpi=300)
plt.show()