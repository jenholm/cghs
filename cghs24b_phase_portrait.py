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

EXP_CLIP = 700.0


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
# Hilbert vacuum
# ============================================================

def Omega_S(S):

    u = logistic_u(S)
    tilt = epsilon_tilt * np.tanh((S - S_c)/width)

    return Omega_L_floor + Omega_inf*u + tilt


def dOmegaS_dS(S):

    u = logistic_u(S)
    du = -(1.0/width)*u*(1.0-u)

    x = (S-S_c)/width
    dtilt = (epsilon_tilt/width)*(1.0/np.cosh(np.clip(x,-50,50))**2)

    return Omega_inf*du + dtilt


# ============================================================
# Expansion law
# ============================================================

def E_of(G,S):

    fS = f_internal(S)

    rad = Omega_r0 * fS * safe_exp(-4*G)
    mat = Omega_m0 * fS * safe_exp(-3*G)
    vac = Omega_S(S)

    rhs = rad + mat + vac
    rhs = np.maximum(rhs,1e-12)

    return np.sqrt(rhs)


# ============================================================
# Hilbert evolution
# ============================================================

def dS_dlambda(G,S):

    source = mu_S * logistic_u(S)

    E = E_of(G,S)
    E_safe = np.maximum(E,1e-60)

    potential = -(kappa/(3*E_safe))*dOmegaS_dS(S)

    damping = -gamma_S * S * (1-logistic_u(S))

    return source + potential + damping


# ============================================================
# Geometry evolution
# ============================================================

def dG_dlambda(G,S):

    return E_of(G,S)


# ============================================================
# Combined system
# ============================================================

def system(lam, y):

    G,S = y

    return [
        dG_dlambda(G,S),
        dS_dlambda(G,S)
    ]


# ============================================================
# Phase grid
# ============================================================

G_vals = np.linspace(-12,2,30)
S_vals = np.linspace(0,20,30)

G_grid, S_grid = np.meshgrid(G_vals,S_vals)

U = np.zeros_like(G_grid)
V = np.zeros_like(S_grid)

for i in range(G_grid.shape[0]):
    for j in range(G_grid.shape[1]):

        G = G_grid[i,j]
        S = S_grid[i,j]

        U[i,j] = dG_dlambda(G,S)
        V[i,j] = dS_dlambda(G,S)


# normalize arrows for readability
mag = np.sqrt(U**2 + V**2)
U /= (mag + 1e-8)
V /= (mag + 1e-8)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10,7))

plt.quiver(
    G_grid,
    S_grid,
    U,
    V,
    mag,
    cmap="viridis",
    scale=40
)

plt.axhline(S_c,color="red",linestyle=":",label=f"S_c = {S_c}")

plt.xlabel("Geometry  G = ln(a)")
plt.ylabel("Hilbert order parameter  S")

plt.title("Cosmology Phase Portrait (G,S)")

plt.colorbar(label="Flow speed")

plt.grid(alpha=0.3)

# ============================================================
# Example trajectories
# ============================================================

initial_conditions = [
    (-11,0.1),
    (-10,0.5),
    (-9,0.2),
]

for ic in initial_conditions:

    sol = solve_ivp(
        system,
        [0,80],
        ic,
        max_step=0.1
    )

    plt.plot(sol.y[0],sol.y[1],linewidth=2)

plt.legend()
plt.savefig("phase portrait.png", dpi=300)
plt.show()