import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

# Time-current parameters
beta_t = 1.0     # source strength
gamma_J = 1.2    # clock regulation strength
J_inf = 1.0      # asymptotic classical clock current

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
# Time-current continuity model (NEW VERSION)
# ============================================================

# def dJt_dlambda(G, S, Jt):

#     fS = f_internal(S)

#     source_t = beta_t * fS
#     regulator = gamma_J * (Jt - J_inf)

#     return source_t - regulator

def dJt_dlambda(G, S, Jt):

    u = logistic_u(S)

    # # transition-localized source: only active while ordering is underway
    # source_t = beta_t * u * (1.0 - u)

    # # regulator drives clock current to the late-time classical value
    # regulator = gamma_J * (Jt - J_inf)

    u = logistic_u(S)
    source_t = beta_t * u * (1.0 - u)
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
# Integrate
# ============================================================

lam_span = (0.0, 80.0)
lam_eval = np.linspace(0.0, 80.0, 2500)

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

H_eff = E / np.maximum(Jt, 1e-12)


# ============================================================
# Component fractions
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
# Diagnostics
# ============================================================

source_t = beta_t * f_internal(S)
regulator = gamma_J * (Jt - J_inf)


# ============================================================
# Plot
# ============================================================

fig, axs = plt.subplots(3, 2, figsize=(13, 12))

axs[0,0].plot(lam, tau, color="black")
axs[0,0].set_title("Accumulated emergent time $\\tau(\\lambda)$")
axs[0,0].set_xlabel("$\\lambda$")
axs[0,0].set_ylabel("$\\tau$")

axs[0,1].plot(lam, Jt, label="$J_t$")
axs[0,1].axhline(J_inf, color="red", linestyle=":", label="$J_\\infty$")
axs[0,1].set_title("Time current")
axs[0,1].set_xlabel("$\\lambda$")
axs[0,1].set_ylabel("$d\\tau/d\\lambda$")
axs[0,1].legend()

axs[1,0].plot(lam, source_t, label="source")
axs[1,0].plot(lam, regulator, label="regulation")
axs[1,0].axhline(0,color="black")
axs[1,0].set_title("Time-current balance")
axs[1,0].legend()

axs[1,1].plot(G,E,label="$H_\\lambda$")
axs[1,1].plot(G,H_eff,label="$H_{eff}$")
axs[1,1].set_yscale("log")
axs[1,1].set_title("Expansion rate comparison")
axs[1,1].legend()

axs[2,0].plot(G,rad_frac,label="radiation")
axs[2,0].plot(G,mat_frac,label="matter")
axs[2,0].plot(G,vac_frac,label="Hilbert vacuum")
axs[2,0].set_title("Component fractions")
axs[2,0].legend()

axs[2,1].plot(G,S,color="black")
axs[2,1].axhline(S_c,color="red",linestyle=":")
axs[2,1].set_title("Phase portrait $(G,S)$")

plt.tight_layout()
plt.savefig("cghs25_time_current_test.png",dpi=300)
plt.show()