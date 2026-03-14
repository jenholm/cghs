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

    tilt = epsilon_tilt * np.tanh((S - S_c)/width)

    return Omega_L_floor + Omega_inf*u + tilt


def dOmegaS_dS(S):

    u = logistic_u(S)

    du = -(1.0/width)*u*(1.0-u)

    x = (S - S_c)/width
    dtilt = (epsilon_tilt/width)*(1.0/np.cosh(np.clip(x,-50,50))**2)

    return Omega_inf*du + dtilt


# ============================================================
# Expansion law
# ============================================================

def E_of(G,S):

    fS = f_internal(S)

    rad = Omega_r0 * fS * safe_exp(-4*G)
    mat = Omega_m0 * fS * safe_exp(-3*G)
    #vac = Omega_S(S)
    vac = Omega_S(S) * logistic_u(S)

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
# Coupled system
# ============================================================

def system(lam,y):

    G,S = y

    return [
        dG_dlambda(G,S),
        dS_dlambda(G,S)
    ]


# ============================================================
# Integrate model
# ============================================================

lam_span = (0,80)

lam_eval = np.linspace(0,80,1500)

initial_state = [-11.0,0.1]

sol = solve_ivp(
    system,
    lam_span,
    initial_state,
    t_eval = lam_eval,
    method = "Radau"
)

G = sol.y[0]
S = sol.y[1]
lam = sol.t

a = safe_exp(G)


# ============================================================
# Diagnostics
# ============================================================

fig,axs = plt.subplots(3,2,figsize=(12,12))

# ------------------------------------------------
# Hilbert evolution
# ------------------------------------------------

axs[0,0].plot(lam,S)
axs[0,0].axhline(S_c,color="red",linestyle=":")
axs[0,0].set_title("Hilbert order parameter")

# ------------------------------------------------
# Geometry evolution
# ------------------------------------------------

axs[0,1].plot(lam,G)
axs[0,1].set_title("Geometry variable G = ln(a)")

# ------------------------------------------------
# Scale factor
# ------------------------------------------------

axs[1,0].plot(lam,a)
axs[1,0].set_yscale("log")
axs[1,0].set_title("Scale factor a")

# ------------------------------------------------
# Phase portrait
# ------------------------------------------------

axs[1,1].plot(G,S)
axs[1,1].axhline(S_c,color="red",linestyle=":")
axs[1,1].set_title("Phase portrait (G,S)")

# ============================================================
# Expansion components
# ============================================================

rad_term = Omega_r0 * f_internal(S) * safe_exp(-4*G)
mat_term = Omega_m0 * f_internal(S) * safe_exp(-3*G)
vac_term = Omega_S(S)

axs[2,0].plot(G, rad_term, label="radiation")
axs[2,0].plot(G, mat_term, label="matter")
axs[2,0].plot(G, vac_term, label="Hilbert vacuum")

axs[2,0].set_yscale("log")
axs[2,0].set_title("Expansion components")
axs[2,0].legend()

# ------------------------------------------------
# Leave last panel empty for now
# ------------------------------------------------

#axs[2,1].axis("off")
# ============================================================
# Cosmology phase portrait flow field
# ============================================================

# G_grid = np.linspace(np.min(G)-2, np.max(G)+2, 30)
# S_grid = np.linspace(0, 20, 30)

# GG, SS = np.meshgrid(G_grid, S_grid)

# dG = np.zeros_like(GG)
# dS = np.zeros_like(SS)

# for i in range(GG.shape[0]):
#     for j in range(GG.shape[1]):

#         g = GG[i,j]
#         s = SS[i,j]

#         dG[i,j] = dG_dlambda(g,s)
#         dS[i,j] = dS_dlambda(g,s)

# axs[2,1].streamplot(
#     GG, SS,
#     dG, dS,
#     density=1.2,
#     color="gray"
# )

# axs[2,1].plot(G, S, color="black", linewidth=2)

# axs[2,1].axhline(S_c, color="red", linestyle=":")

# axs[2,1].set_title("Cosmology phase portrait")
# axs[2,1].set_xlabel("G")
# axs[2,1].set_ylabel("S")


# plt.tight_layout()
# plt.savefig("cghs25.png", dpi=300)
# plt.show()

# ============================================================
# Component fractions
# ============================================================

E2 = E_of(G, S)**2
E2_safe = np.where(E2 < 1e-12, 1e-12, E2)

rad_frac = rad_term / E2_safe
mat_frac = mat_term / E2_safe
vac_frac = vac_term / E2_safe

axs[2,1].plot(G, rad_frac, label="radiation fraction")
axs[2,1].plot(G, mat_frac, label="matter fraction")
axs[2,1].plot(G, vac_frac, label="Hilbert vacuum fraction")

axs[2,1].set_title("Component dominance fractions")
axs[2,1].set_xlabel("G = ln(a)")
axs[2,1].set_ylabel("fraction of E²")
axs[2,1].legend()

# ============================================================
# 3D Spacetime tube
# ============================================================

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

fig3 = plt.figure(figsize=(10,8))
ax3 = fig3.add_subplot(111, projection="3d")

theta = np.linspace(0,2*np.pi,80)

LAM,TH = np.meshgrid(lam,theta)

# tube radius proportional to scale factor
a_norm = a / np.max(a)
radius = a_norm**0.18 + 0.02

R = np.tile(radius,(len(theta),1))

X = LAM
Y = R*np.cos(TH)
Z = R*np.sin(TH)

norm = Normalize(vmin=np.min(S), vmax=np.max(S))

colors = cm.viridis_r(norm(S))
facecolors = np.tile(colors,(len(theta),1,1))

ax3.plot_surface(
    X,Y,Z,
    facecolors=facecolors,
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.35
)

ax3.plot_wireframe(
    X, Y, Z,
    rstride=6,
    cstride=6,
    color="black",
    linewidth=0.3,
    alpha=0.2
)

# center spine
ax3.plot(lam,np.zeros_like(lam),np.zeros_like(lam),color="black")

# mark Hilbert transition
cross_idx = np.where(S >= S_c)[0]

if len(cross_idx)>0:
    i = cross_idx[0]

    ax3.scatter(
        lam[i],0,0,
        color="red",
        s=80,
        label=f"S_c = {S_c}"
    )

ax3.set_title("Spacetime tube from cosmology trajectory")

ax3.set_xlabel("λ")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")

ax3.view_init(elev=18, azim=-120)
ax3.set_box_aspect((3,1,1))
# colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)
mappable.set_array([])

cbar = plt.colorbar(mappable, ax=ax3, shrink=0.7)
cbar.set_label("Hilbert order parameter S")
plt.savefig("cghs25_tube.png", dpi=300)
plt.show()