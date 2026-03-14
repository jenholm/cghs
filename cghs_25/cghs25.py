import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# Cosmology parameters
# ============================================================

Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70
#Omega_inf = 50.0
Omega_inf = 1.0

# Hilbert sector parameters
S_c = 8.0
width = 2.0
epsilon_tilt = 0.02

kappa = 60.0
gamma_S = 0.08
mu_S = 0.002

# ============================================================
# Time current parameters
# ============================================================

beta_t = 0.8
gamma_t = 1.2
J_inf = 1.0

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


#def f_internal(S):
#    return 1.0 - logistic_u(S)
def f_internal(S):
    return (1.0 - logistic_u(S))**8

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

#def dG_dlambda(G,S):

#    return E_of(G,S)

#def dG_dlambda(G,S,J):
#    return E_of(G,S) / np.maximum(J,1e-12)

def dG_dlambda(G, S):
    return E_of(G, S)

# def dG_dlambda(G, S, J):

#     E = E_of(G, S)

#     return E / np.maximum(J, 1e-12)

# ============================================================
# Time current evolution
# ============================================================

def dJ_dlambda(S,J):

     u = logistic_u(S)

     production = beta_t * u * (1-u)

     relaxation = -gamma_t * (J - J_inf)

     return production + relaxation





# ============================================================
# Coupled system
# ============================================================

# def system(lam,y):

#     G,S,J = y

#     dG = dG_dlambda(G,S)
#     dS = dS_dlambda(G,S)
#     dJ = dJ_dlambda(S,J)

#     return [dG,dS,dJ]

# def system(lam,y):

#     G,S,J = y

#     dG = dG_dlambda(G,S,J)
#     dS = dS_dlambda(G,S)
#     dJ = dJ_dlambda(S,J)

def system(lam, y):
    G, S, J = y

    dG = dG_dlambda(G, S)
    dS = dS_dlambda(G, S)
    dJ = dJ_dlambda(S, J)

    return np.array([dG, dS, dJ])



# def system(lam, y):

#     G, S, J = y

#     dG = dG_dlambda(G, S, J)
#     dS = dS_dlambda(G, S)
#     dJ = dJ_dlambda(S, J)

#     return np.array([dG, dS, dJ])

# ============================================================
# Integrate model
# ============================================================

lam_span = (0,80)

lam_eval = np.linspace(0,80,1500)

initial_state = [-11.0,0.1,1.0]

sol = solve_ivp(
    system,
    lam_span,
    initial_state,
    t_eval = lam_eval,
    method = "Radau"
)

G = sol.y[0]
S = sol.y[1]
J = sol.y[2]
lam = sol.t

a = safe_exp(G)

# ============================================================
# Reconstruct physical time
# ============================================================

tau = np.zeros_like(lam)

for i in range(1,len(lam)):

    dt = lam[i]-lam[i-1]

    tau[i] = tau[i-1] + 0.5*(J[i]+J[i-1])*dt


# ============================================================
# Observable expansion rate
# ============================================================

E = E_of(G,S)

#H_eff = E / np.maximum(J,1e-12)
H_eff = E_of(G, S) / J

# ============================================================
# Diagnostics
# ============================================================

fig,axs = plt.subplots(3,2,figsize=(12,12))

# Hilbert evolution

axs[0,0].plot(tau,S)
axs[0,0].axhline(S_c,color="red",linestyle=":")
axs[0,0].set_title("Hilbert order parameter S(τ)")

# Geometry

axs[0,1].plot(tau,G)
axs[0,1].set_title("Geometry G = ln(a)")

# Scale factor

axs[1,0].plot(tau,a)
axs[1,0].set_yscale("log")
axs[1,0].set_title("Scale factor a(τ)")

# Phase portrait

axs[1,1].plot(G,S)
axs[1,1].axhline(S_c,color="red",linestyle=":")
axs[1,1].set_title("Phase portrait (G,S)")

# Time current

axs[2,0].plot(tau,J)
axs[2,0].set_title("Emergent time current J_t")

# Hubble rate

axs[2,1].plot(tau,H_eff)
axs[2,1].set_title("Observable expansion H_eff")

plt.tight_layout()
plt.savefig("cghs25_time_current.png", dpi=300)
plt.show()


# ============================================================
# 3D spacetime tube
# ============================================================

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

fig3 = plt.figure(figsize=(10,8))
ax3 = fig3.add_subplot(111, projection="3d")

theta = np.linspace(0,2*np.pi,80)

TAU,TH = np.meshgrid(tau,theta)

a_norm = a/np.max(a)

#radius = a_norm**0.18 + 0.02
radius = np.log10(a + 1)
radius = radius / np.max(radius)
radius = 0.1 + 0.9*radius

R = np.tile(radius,(len(theta),1))

X = TAU
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
    X,Y,Z,
    rstride=6,
    cstride=6,
    color="black",
    linewidth=0.3,
    alpha=0.2
)

ax3.plot(tau,np.zeros_like(tau),np.zeros_like(tau),color="black")

ax3.set_title("Spacetime tube with emergent time")

ax3.set_xlabel("τ")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")

plt.savefig("cghs25_tube_time.png", dpi=300)
plt.show()