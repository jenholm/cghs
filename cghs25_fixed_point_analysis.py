import numpy as np
import matplotlib.pyplot as plt

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
# Flow equations
# ============================================================

def dS_dlambda(G, S):
    E = E_of(G, S)
    E_safe = np.maximum(E, 1e-60)

    source = mu_S * logistic_u(S)
    potential = -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)
    damping = -gamma_S * S * (1.0 - logistic_u(S))

    return source + potential + damping


def dJt_dlambda(S, Jt):
    u = logistic_u(S)

    # transition-localized source
    source_t = beta_t * u * (1.0 - u)

    # regulation toward classical clock
    regulator = gamma_J * (Jt - J_inf)

    return source_t - regulator


# ============================================================
# Root finding helpers
# ============================================================

def find_roots_on_grid(xgrid, yvals):
    """
    Find approximate roots of y(x)=0 by sign changes and linear interpolation.
    """
    roots = []

    for i in range(len(xgrid) - 1):
        y1 = yvals[i]
        y2 = yvals[i + 1]

        if np.isnan(y1) or np.isnan(y2):
            continue

        # exact grid hit
        if y1 == 0:
            roots.append(xgrid[i])
            continue

        # sign change
        if y1 * y2 < 0:
            x1 = xgrid[i]
            x2 = xgrid[i + 1]

            # linear interpolation
            xr = x1 - y1 * (x2 - x1) / (y2 - y1)
            roots.append(xr)

    return roots


def local_slope(xgrid, yvals, x0):
    """
    Numerical slope dy/dx near x0 using nearest grid point.
    """
    idx = np.argmin(np.abs(xgrid - x0))

    if idx == 0:
        idx = 1
    if idx == len(xgrid) - 1:
        idx = len(xgrid) - 2

    x1, x2 = xgrid[idx - 1], xgrid[idx + 1]
    y1, y2 = yvals[idx - 1], yvals[idx + 1]

    return (y2 - y1) / (x2 - x1)


# ============================================================
# Scan fixed points of S for many G values
# ============================================================

G_scan = np.linspace(-11.0, 9.0, 140)
S_grid = np.linspace(0.0, 35.0, 3000)

stable_G = []
stable_S = []

unstable_G = []
unstable_S = []

for G in G_scan:
    flow_vals = dS_dlambda(G, S_grid)
    roots = find_roots_on_grid(S_grid, flow_vals)

    for r in roots:
        slope = local_slope(S_grid, flow_vals, r)

        if slope < 0:
            stable_G.append(G)
            stable_S.append(r)
        else:
            unstable_G.append(G)
            unstable_S.append(r)


# ============================================================
# Time-current fixed point branch
# ------------------------------------------------------------
# dJt/dlambda = 0  =>  Jt* = J_inf + source_t/gamma_J
# where source_t = beta_t u(1-u)
# ============================================================

S_scan_for_J = np.linspace(0.0, 35.0, 1200)
u_scan = logistic_u(S_scan_for_J)
Jt_star = J_inf + (beta_t / gamma_J) * u_scan * (1.0 - u_scan)

# stability check:
# d/dJ [dJ/dlambda] = -gamma_J < 0 always stable
Jt_is_stable = True


# ============================================================
# Optional: show the flow field in S at selected G slices
# ============================================================

G_slices = [-10.0, -6.0, -3.0, 0.0, 4.0, 8.0]

# ============================================================
# Plot
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

# ------------------------------------------------------------
# 1. Fixed-point branches for S*(G)
# ------------------------------------------------------------
axs[0, 0].scatter(stable_G, stable_S, s=10, label="stable fixed points")
axs[0, 0].scatter(unstable_G, unstable_S, s=12, marker="x", label="unstable fixed points")
axs[0, 0].axhline(S_c, color="red", linestyle=":", label="$S_c$")
axs[0, 0].set_title("Fixed-point branches for $dS/d\\lambda = 0$")
axs[0, 0].set_xlabel("$G = \\ln a$")
axs[0, 0].set_ylabel("$S_*(G)$")
axs[0, 0].legend()

# ------------------------------------------------------------
# 2. Flow slices dS/dlambda vs S
# ------------------------------------------------------------
for G in G_slices:
    axs[0, 1].plot(S_grid, dS_dlambda(G, S_grid), label=f"G={G:.0f}")
axs[0, 1].axhline(0, color="black", linewidth=1)
axs[0, 1].axvline(S_c, color="red", linestyle=":")
axs[0, 1].set_title("Hilbert-sector flow slices")
axs[0, 1].set_xlabel("$S$")
axs[0, 1].set_ylabel("$dS/d\\lambda$")
axs[0, 1].legend()

# ------------------------------------------------------------
# 3. Time-current fixed point Jt*(S)
# ------------------------------------------------------------
axs[1, 0].plot(S_scan_for_J, Jt_star, color="tab:blue", label="$J_t^*(S)$")
axs[1, 0].axhline(J_inf, color="red", linestyle=":", label="$J_\\infty$")
axs[1, 0].axvline(S_c, color="gray", linestyle="--", label="$S_c$")
axs[1, 0].set_title("Time-current fixed point from $dJ_t/d\\lambda = 0$")
axs[1, 0].set_xlabel("$S$")
axs[1, 0].set_ylabel("$J_t^*$")
axs[1, 0].legend()

# ------------------------------------------------------------
# 4. Stability derivative summaries
# ------------------------------------------------------------
# For S sector, plot slope at stable/unstable roots
stable_slopes = []
for G, Sstar in zip(stable_G, stable_S):
    flow_vals = dS_dlambda(G, S_grid)
    stable_slopes.append(local_slope(S_grid, flow_vals, Sstar))

unstable_slopes = []
for G, Sstar in zip(unstable_G, unstable_S):
    flow_vals = dS_dlambda(G, S_grid)
    unstable_slopes.append(local_slope(S_grid, flow_vals, Sstar))

axs[1, 1].scatter(stable_G, stable_slopes, s=10, label="stable branch slopes")
axs[1, 1].scatter(unstable_G, unstable_slopes, s=12, marker="x", label="unstable branch slopes")
axs[1, 1].axhline(0, color="black", linewidth=1)
axs[1, 1].set_title("Local stability slopes for $S$ fixed points")
axs[1, 1].set_xlabel("$G = \\ln a$")
axs[1, 1].set_ylabel("$\\partial(dS/d\\lambda)/\\partial S$")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("cghs25_fixed_point_analysis.png", dpi=300)
plt.show()