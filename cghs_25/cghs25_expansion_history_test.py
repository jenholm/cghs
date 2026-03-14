import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# Parameters
# ============================================================

Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70
#Omega_inf = 50.0
Omega_inf = 1.0

S_c = 8.0
#width = 2.0
width = 1.0
epsilon_tilt = 0.02

kappa = 60.0
gamma_S = 0.08
mu_S = 0.002

# Time-current parameters
beta_t = 1.0
gamma_J = 1.2
J_inf = 1.0

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


# ============================================================
# Vacuum potential
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
    rad = Omega_r0 * safe_exp(-4 * G)
    mat = Omega_m0 * safe_exp(-3 * G)
    vac = Omega_S(S) * logistic_u(S)

    rhs = rad + mat + vac
    rhs = np.maximum(rhs, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# Geometry / Hilbert dynamics
# ============================================================

def dG_dlambda(G, S):
    return E_of(G, S)


def dS_dlambda(G, S):
    E = E_of(G, S)
    E_safe = np.maximum(E, 1e-60)

    source = mu_S * logistic_u(S)
    potential = -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)
    damping = -gamma_S * S * (1.0 - logistic_u(S))

    return source + potential + damping


# ============================================================
# Time-current dynamics
# ============================================================

def dJt_dlambda(G, S, Jt):
    u = logistic_u(S)

    # transition-localized source
    source_t = beta_t * u * (1.0 - u)

    # regulation toward classical clock
    regulator = gamma_J * (Jt - J_inf)

    return source_t - regulator


def system(lam, y):
    G, S, tau, Jt = y

    dG = dG_dlambda(G, S)
    dS = dS_dlambda(G, S)
    dtau = Jt
    dJt = dJt_dlambda(G, S, Jt)

    return [dG, dS, dtau, dJt]


# ============================================================
# Integrate model
# ============================================================

lam_span = (0.0, 80.0)
lam_eval = np.linspace(0.0, 80.0, 3000)

initial_state = [-11.0, 0.1, 0.0, 1e-3]

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
Jt = sol.y[3]

a = safe_exp(G)
E = E_of(G, S)

# computational and emergent-time expansion rates
H_lambda = E
H_eff = E / np.maximum(Jt, 1e-12)

# ============================================================
# Normalize expansion histories for shape comparison
# ------------------------------------------------------------
# use the last point as "today-like" normalization
# ============================================================

H_lambda_norm = H_lambda / H_lambda[-1]
H_eff_norm = H_eff / H_eff[-1]

# Reference LCDM-like curve (shape only, not a fit)
H_lcdm = np.sqrt(Omega_m0 * np.maximum(a, 1e-12)**(-3) + Omega_L_floor)
H_lcdm_norm = H_lcdm / H_lcdm[-1]

# Radiation+matter+vacuum reference for very early comparison
H_ref_full = np.sqrt(
    Omega_r0 * np.maximum(a, 1e-12)**(-4)
    + Omega_m0 * np.maximum(a, 1e-12)**(-3)
    + Omega_L_floor
)
H_ref_full_norm = H_ref_full / H_ref_full[-1]

# ============================================================
# Diagnostics: component fractions
# ============================================================

rad_term = Omega_r0 * safe_exp(-4 * G)
mat_term = Omega_m0 * safe_exp(-3 * G)
vac_term = Omega_S(S) * logistic_u(S)

total_term = rad_term + mat_term + vac_term
total_term = np.maximum(total_term, 1e-12)

rad_frac = rad_term / total_term
mat_frac = mat_term / total_term
vac_frac = vac_term / total_term

# ============================================================
# Diagnostics: relative deviation from reference curves
# ============================================================

reldev_lambda_lcdm = np.abs(H_lambda_norm - H_lcdm_norm) / np.maximum(H_lcdm_norm, 1e-12)
reldev_eff_lcdm = np.abs(H_eff_norm - H_lcdm_norm) / np.maximum(H_lcdm_norm, 1e-12)

reldev_lambda_full = np.abs(H_lambda_norm - H_ref_full_norm) / np.maximum(H_ref_full_norm, 1e-12)
reldev_eff_full = np.abs(H_eff_norm - H_ref_full_norm) / np.maximum(H_ref_full_norm, 1e-12)

# ============================================================
# Plot
# ============================================================

fig, axs = plt.subplots(3, 2, figsize=(13, 12))

# ------------------------------------------------------------
# 1. H(a) comparison, log-log
# ------------------------------------------------------------
axs[0, 0].plot(a, H_lambda_norm, label="$H_\\lambda$ (model)")
axs[0, 0].plot(a, H_eff_norm, label="$H_{eff}$ (model)")
axs[0, 0].plot(a, H_lcdm_norm, "--", label="reference $\\Lambda$CDM")
axs[0, 0].plot(a, H_ref_full_norm, ":", label="reference rad+mat+$\\Lambda$")
axs[0, 0].set_xscale("log")
axs[0, 0].set_yscale("log")
axs[0, 0].set_title("Normalized expansion history")
axs[0, 0].set_xlabel("$a$")
axs[0, 0].set_ylabel("$H/H_0$")
axs[0, 0].legend()

# ------------------------------------------------------------
# 2. H(G) comparison
# ------------------------------------------------------------
axs[0, 1].plot(G, H_lambda_norm, label="$H_\\lambda$ (model)")
axs[0, 1].plot(G, H_eff_norm, label="$H_{eff}$ (model)")
axs[0, 1].plot(G, H_lcdm_norm, "--", label="reference $\\Lambda$CDM")
axs[0, 1].plot(G, H_ref_full_norm, ":", label="reference rad+mat+$\\Lambda$")
axs[0, 1].set_yscale("log")
axs[0, 1].set_title("Expansion history vs $G=\\ln a$")
axs[0, 1].set_xlabel("$G$")
axs[0, 1].set_ylabel("normalized rate")
axs[0, 1].legend()

# ------------------------------------------------------------
# 3. Component fractions
# ------------------------------------------------------------
axs[1, 0].plot(G, rad_frac, label="radiation fraction")
axs[1, 0].plot(G, mat_frac, label="matter fraction")
axs[1, 0].plot(G, vac_frac, label="Hilbert vacuum fraction")
axs[1, 0].set_title("Component dominance fractions")
axs[1, 0].set_xlabel("$G$")
axs[1, 0].set_ylabel("fraction")
axs[1, 0].legend()

# ------------------------------------------------------------
# 4. Time current
# ------------------------------------------------------------
axs[1, 1].plot(G, Jt, label="$J_t$")
axs[1, 1].axhline(J_inf, color="red", linestyle=":", label="$J_\\infty$")
axs[1, 1].set_title("Time current along cosmological history")
axs[1, 1].set_xlabel("$G$")
axs[1, 1].set_ylabel("$J_t = d\\tau/d\\lambda$")
axs[1, 1].legend()

# ------------------------------------------------------------
# 5. Relative deviation from LCDM-like reference
# ------------------------------------------------------------
axs[2, 0].plot(G, reldev_lambda_lcdm, label="$H_\\lambda$ vs $\\Lambda$CDM")
axs[2, 0].plot(G, reldev_eff_lcdm, label="$H_{eff}$ vs $\\Lambda$CDM")
axs[2, 0].set_yscale("log")
axs[2, 0].set_title("Relative deviation from $\\Lambda$CDM-shaped history")
axs[2, 0].set_xlabel("$G$")
axs[2, 0].set_ylabel("relative deviation")
axs[2, 0].legend()

# ------------------------------------------------------------
# 6. Relative deviation from full rad+mat+Λ reference
# ------------------------------------------------------------
axs[2, 1].plot(G, reldev_lambda_full, label="$H_\\lambda$ vs full reference")
axs[2, 1].plot(G, reldev_eff_full, label="$H_{eff}$ vs full reference")
axs[2, 1].set_yscale("log")
axs[2, 1].set_title("Relative deviation from rad+mat+$\\Lambda$ reference")
axs[2, 1].set_xlabel("$G$")
axs[2, 1].set_ylabel("relative deviation")
axs[2, 1].legend()

plt.tight_layout()
plt.savefig("cghs25_expansion_history_test.png", dpi=300)
plt.show()