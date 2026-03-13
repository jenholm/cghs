import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib.colors import Normalize

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
    vac = Omega_S(S)

    rhs = rad + mat + vac
    rhs = np.maximum(rhs, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# Coupled dynamics
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


def system(lam, y):
    G, S = y
    return [dG_dlambda(G, S), dS_dlambda(G, S)]


# ============================================================
# Integrate a representative trajectory
# ============================================================

lam_span = (0.0, 80.0)
lam_eval = np.linspace(lam_span[0], lam_span[1], 1200)

# You can vary these if you want to test nearby trajectories
G0 = -11.0
S0 = 0.1

sol = solve_ivp(
    system,
    lam_span,
    [G0, S0],
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

# ============================================================
# Build 3D spacetime tube
# ------------------------------------------------------------
# Axis direction  : lambda
# Tube radius     : normalized exp(G)
# Tube color      : S
# ============================================================

# Raw scale factor along the trajectory
a_raw = safe_exp(G - np.max(G))   # normalize so the largest radius is ~1
a_raw = np.maximum(a_raw, 1e-12)

# Optional contrast stretch so early tube is still visible
#radius = a_raw**0.25
# geometry-driven radius (better scaling)
G_shift = G - np.min(G)
radius = (G_shift / np.max(G_shift)) * 1.2 + 0.02

theta = np.linspace(0, 2 * np.pi, 80)
LAM, TH = np.meshgrid(lam, theta)

R = np.tile(radius, (len(theta), 1))

X = LAM
Y = R * np.cos(TH)
Z = R * np.sin(TH)

# Color by S along the trajectory
norm = Normalize(vmin=np.min(S), vmax=np.max(S))
#colors_along_traj = cm.plasma(norm(S))            # shape (N, 4)
colors_along_traj = cm.viridis_r(norm(S))
facecolors = np.tile(colors_along_traj, (len(theta), 1, 1))

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    facecolors=facecolors,
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.95
)

# Draw center spine
ax.plot(lam, np.zeros_like(lam), np.zeros_like(lam), color="black", linewidth=1.2, alpha=0.7)

# Mark the transition crossing near S = S_c if it exists
cross_idx = np.where(S >= S_c)[0]
if len(cross_idx) > 0:
    i = cross_idx[0]
    ax.scatter(
        [lam[i]], [0], [0],
        color="red", s=80, marker="o", label=f"Transition near S_c = {S_c}"
    )

# Labels
ax.set_title("3D Spacetime Tube from Phase-Portrait Trajectory")
ax.set_xlabel("Trajectory parameter λ")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Better view
ax.view_init(elev=24, azim=-63)

# Colorbar for S
#mappable = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08)
cbar.set_label("Hilbert order parameter S")

if len(cross_idx) > 0:
    ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("phase portrait_trajectory.png", dpi=300)
plt.show()