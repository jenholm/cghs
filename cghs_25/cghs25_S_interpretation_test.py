import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# PARAMETERS
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

EXP_CLIP = 700


# ============================================================
# HELPERS
# ============================================================

def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


def logistic_u(S):
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)
    return 1 / (1 + np.exp(x))


def f_internal(S):
    return 1 - logistic_u(S)


# ============================================================
# VACUUM POTENTIAL
# ============================================================

def Omega_S(S):
    u = logistic_u(S)
    tilt = epsilon_tilt * np.tanh((S - S_c) / width)
    return Omega_L_floor + Omega_inf * u + tilt


def dOmegaS_dS(S):

    u = logistic_u(S)
    du = -(1/width) * u * (1-u)

    x = (S - S_c) / width
    dtilt = (epsilon_tilt/width) * (1/np.cosh(np.clip(x,-50,50))**2)

    return Omega_inf*du + dtilt


# ============================================================
# EXPANSION LAW VARIANTS
# ============================================================

def E_variant(G,S,variant):

    if variant == "A":
        # S gates classical sectors only
        rad = Omega_r0 * f_internal(S) * safe_exp(-4*G)
        mat = Omega_m0 * f_internal(S) * safe_exp(-3*G)
        vac = Omega_S(S) * logistic_u(S)
        #rad = Omega_r0 * f_internal(S) * safe_exp(-4*G)
        #mat = Omega_m0 * f_internal(S) * safe_exp(-3*G)
        #vac = Omega_S(S)

    elif variant == "B":
        # S gates vacuum only
        rad = Omega_r0 * safe_exp(-4*G)
        mat = Omega_m0 * safe_exp(-3*G)
        vac = Omega_S(S) * logistic_u(S)

    elif variant == "C":
        # hybrid effective order parameter
        rad = Omega_r0 * f_internal(S) * safe_exp(-4*G)
        mat = Omega_m0 * f_internal(S) * safe_exp(-3*G)
        vac = Omega_S(S) * logistic_u(S)

    rhs = rad + mat + vac
    rhs = np.maximum(rhs,1e-12)

    return np.sqrt(rhs)


# ============================================================
# DYNAMICS
# ============================================================

def system(lam,y,variant):

    G,S = y

    E = E_variant(G,S,variant)

    dG = E

    source = mu_S * logistic_u(S)

    potential = -(kappa/(3*E)) * dOmegaS_dS(S)

    damping = -gamma_S * S * (1 - logistic_u(S))

    dS = source + potential + damping

    return [dG,dS]


# ============================================================
# RUN MODEL
# ============================================================

def run_model(variant):

    lam_span = (0,80)
    lam_eval = np.linspace(0,80,2000)

    init = [-11,0.1]

    sol = solve_ivp(
        lambda lam,y: system(lam,y,variant),
        lam_span,
        init,
        t_eval=lam_eval,
        method="Radau"
    )

    lam = sol.t
    G = sol.y[0]
    S = sol.y[1]

    a = safe_exp(G)

    rad = Omega_r0 * safe_exp(-4*G)
    mat = Omega_m0 * safe_exp(-3*G)
    vac = Omega_S(S)

    total = rad+mat+vac

    return lam,G,S,a,rad/total,mat/total,vac/total


# ============================================================
# RUN ALL THREE VARIANTS
# ============================================================

results = {}

for v in ["A","B","C"]:
    results[v] = run_model(v)


# ============================================================
# PLOTS
# ============================================================

fig,axs = plt.subplots(2,2,figsize=(12,10))

colors = {"A":"tab:blue","B":"tab:orange","C":"tab:green"}

# S evolution
for v in results:
    lam,G,S,a,rf,mf,vf = results[v]
    axs[0,0].plot(lam,S,color=colors[v],label=f"Variant {v}")

axs[0,0].axhline(S_c,color="red",linestyle=":")
axs[0,0].set_title("Hilbert order parameter S")
axs[0,0].set_xlabel("λ")
axs[0,0].legend()


# Geometry evolution
for v in results:
    lam,G,S,a,rf,mf,vf = results[v]
    axs[0,1].plot(lam,G,color=colors[v],label=f"Variant {v}")

axs[0,1].set_title("Geometry variable G = ln(a)")
axs[0,1].set_xlabel("λ")
axs[0,1].legend()


# Phase portrait
for v in results:
    lam,G,S,a,rf,mf,vf = results[v]
    axs[1,0].plot(G,S,color=colors[v],label=f"Variant {v}")

axs[1,0].axhline(S_c,color="red",linestyle=":")
axs[1,0].set_title("Phase portrait (G,S)")
axs[1,0].set_xlabel("G")
axs[1,0].set_ylabel("S")
axs[1,0].legend()


# Component fractions (variant C)
lam,G,S,a,rf,mf,vf = results["C"]

axs[1,1].plot(G,rf,label="radiation")
axs[1,1].plot(G,mf,label="matter")
axs[1,1].plot(G,vf,label="Hilbert vacuum")

axs[1,1].set_title("Component fractions (variant C)")
axs[1,1].set_xlabel("G")
axs[1,1].legend()


plt.tight_layout()
plt.savefig("cghs25_S_interpretation_test.png",dpi=300)
plt.show()