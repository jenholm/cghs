import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.gridspec as gridspec

# ============================================================
# CLOSED HILBERT + CONFORMAL GEOMETRY MODEL (cosmic time)
#
# State variables:
#   G(t) = ln a(t)          (geometry operator)
#   S(t) = Hilbert variable (order/state operator)
#
# Derived via constraint manifold (Friedmann-like):
#   H_geom(G,S)^2 = H0^2 * [ Ωr0 e^{-4G} + Ωm0 e^{-3G} + ΩS(S) ]
#
# ODEs (2 equations, closed):
#   dG/dt = +H_geom(G,S)
#   dS/dt = -(1/(3 H_geom)) * dV/dS   (slow-roll, Lyapunov guaranteed)
#
# Lyapunov:
#   V(S) decreases monotonically since dV/dt = V'(S)*Sdot = -(V'(S))^2/(3H) <= 0
# ============================================================

# ----------------------------
# Units
# ----------------------------
year = 3.154e7
Gyr = 1e9 * year

# ----------------------------
# "Present-day" H0 and matter/radiation parameters
# (These set the late-time scale of expansion.)
# ----------------------------
H0 = 70.0 * 1e3 / (3.086e22)  # 70 km/s/Mpc in 1/s
Omega_r0 = 9e-5
Omega_m0 = 0.30

# ----------------------------
# Hilbert-sector potential V(S) and its contribution ΩS(S)
# We use a "plateau -> floor" potential:
#   V(S) = ΩΛ + Ωinf / (1 + exp((S-Sc)/Δ))
# So as S increases, V(S) relaxes from ΩΛ+Ωinf down to ΩΛ.
# ----------------------------
Omega_L_floor = 0.70          # late-time residual (Λ-like)
Omega_inf = 1e12              # early-time vacuum-like contribution (drives inflation-like phase)
S_c = 8.0                     # center of the transition
Delta = 0.8                   # width of the transition

def V_potential(S):
    """Dimensionless 'energy' in critical-density units (acts like ΩS)."""
    x = (S - S_c) / Delta
    return Omega_L_floor + Omega_inf / (1.0 + np.exp(x))

def dV_dS(S):
    """Derivative of V(S) w.r.t S."""
    x = (S - S_c) / Delta
    ex = np.exp(x)
    # d/dS [ Ωinf / (1+e^x) ] = Ωinf * d/dS[(1+e^x)^-1]
    # = Ωinf * (-(1+e^x)^-2) * e^x * (1/Δ)
    return -(Omega_inf / Delta) * (ex / (1.0 + ex)**2)

def Omega_S(S):
    """Hilbert-sector density parameter contribution."""
    return V_potential(S)

# ----------------------------
# Constraint manifold: Hubble derived from (G,S)
# ----------------------------
def H_geom(G, S):
    """
    Geometry Hubble rate derived from the constraint:
      H^2 = H0^2 [ Ωr0 a^-4 + Ωm0 a^-3 + ΩS(S) ]
    where a = exp(G).
    """
    # Use exp(-nG) directly to avoid overflow from exp(G) at huge negative G.
    term_r = Omega_r0 * np.exp(-4.0 * G)
    term_m = Omega_m0 * np.exp(-3.0 * G)
    term_s = Omega_S(S)
    rhs = term_r + term_m + term_s

    # Safety: avoid tiny negative from rounding
    rhs = np.maximum(rhs, 0.0)
    return H0 * np.sqrt(rhs)

# ----------------------------
# 2D ODE system in cosmic time
# ----------------------------
def rhs(t, y):
    G, S = y
    H = H_geom(G, S)

    # Geometry equation
    dGdt = H

    # Hilbert equation (slow-roll; Lyapunov monotone)
    # Guard against division by ~0
    H_safe = max(H, 1e-60)
    dSdt = -(1.0 / (3.0 * H_safe)) * dV_dS(S)

    return [dGdt, dSdt]

# ============================================================
# Integrate from very early a to today
# ============================================================
t_today = 13.8 * Gyr

# Initial conditions
a_init = 1e-60
G0 = np.log(a_init)
S0 = 0.0  # start on the high plateau (inflation-like), then roll toward floor

# Log-spaced evaluation times (resolve early dynamics)
t_eval = np.geomspace(1e-40, t_today, 9000)
t_eval = np.unique(np.concatenate([[0.0], t_eval]))

sol = solve_ivp(
    rhs,
    (0.0, t_today),
    [G0, S0],
    t_eval=t_eval,
    method="Radau",     # stiff-friendly
    rtol=1e-8,
    atol=1e-12
)

t = sol.t
G = sol.y[0]
S = sol.y[1]

# ============================================================
# Derived series (UN-NORMALIZED physical integration)
# ============================================================
H = np.array([H_geom(G[i], S[i]) for i in range(len(t))])
a = np.exp(G)
V_vol = a**3                       # volume metric (proportional)
comoving_hubble = 1.0 / (a * H)    # (aH)^(-1)

# Constraint residual (should be ~0 because H is derived from it)
# C = H^2/H0^2 - (Ωr a^-4 + Ωm a^-3 + ΩS)
Omega_sum = Omega_r0 * a**(-4) + Omega_m0 * a**(-3) + Omega_S(S)
C_resid = (H/H0)**2 - Omega_sum

# Lyapunov / budget (should be monotone decreasing)
L = V_potential(S)

# Deceleration parameter q(t) = -1 - (dH/dt)/H^2
# Use numerical derivative (for diagnostics only)
dHdt = np.gradient(H, t, edge_order=2)
q = -1.0 - (dHdt / np.maximum(H**2, 1e-80))

# ============================================================
# Normalize a(t) for plotting convenience ONLY
# (Does not change the dynamics; just makes plots readable)
# ============================================================
a_plot = a / a[-1]
V_plot = a_plot**3

# For plotting comoving Hubble radius, use normalized a as well (shape remains meaningful)
comoving_hubble_plot = 1.0 / (a_plot * H)

t_gyr = t / Gyr

# ============================================================
# Build 3D tube (radius proportional to a_plot)
# ============================================================
theta = np.linspace(0, 2*np.pi, 80)
T, TH = np.meshgrid(t_gyr, theta)

R = a_plot
R_mesh = np.tile(R, (len(theta), 1))

X = T
Y = R_mesh * np.cos(TH)
Z = R_mesh * np.sin(TH)

# ============================================================
# ONE FIGURE: tube + metrics + "performance" diagnostics
# ============================================================
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# --- (1) 3D tube ---
ax0 = fig.add_subplot(gs[0, 0], projection="3d")
ax0.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0)
ax0.set_title("Co-primary Tube (Radius ∝ a)")
ax0.set_xlabel("Time (Gyr)")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")

# --- (2) Scale factor a(t) ---
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(t_gyr, a_plot)
ax1.set_title("Scale factor a(t) (normalized to 1 today)")
ax1.set_xlabel("Time (Gyr)")
ax1.set_ylabel("a(t)")

# --- (3) Volume metric V ∝ a^3 ---
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(t_gyr, V_plot)
ax2.set_title("Volume metric: V(t) ∝ a(t)^3 (normalized)")
ax2.set_xlabel("Time (Gyr)")
ax2.set_ylabel("V(t)")

# --- (4) Inflation diagnostic: (aH)^(-1) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t + 1e-40, comoving_hubble_plot)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Inflation diagnostic: comoving Hubble radius (aH)^(-1)")
ax3.set_xlabel("t (seconds, log)")
ax3.set_ylabel("(aH)^(-1) (arb, log)")

# --- (5) Constraint residual (manifold health) ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_gyr, C_resid)
ax4.set_title("Constraint residual C(t) = H^2/H0^2 - Ω_sum (should ~0)")
ax4.set_xlabel("Time (Gyr)")
ax4.set_ylabel("C(t)")

# --- (6) Lyapunov + q(t) (performance / regime markers) ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(t_gyr, L, label="Lyapunov L=V(S)")
ax5_twin = ax5.twinx()
ax5_twin.plot(t_gyr, q, linestyle="--", label="q(t)")
ax5.set_title("Budget (Lyapunov) + Deceleration q(t)")
ax5.set_xlabel("Time (Gyr)")
ax5.set_ylabel("L")
ax5_twin.set_ylabel("q")

# Legends (combined for twin axes)
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5_twin.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.savefig("coprimary_hilbert_conformal_closed.png", dpi=300)
plt.show()

# ============================================================
# Print summary diagnostics
# ============================================================
print("=== Co-primary closed system diagnostics ===")
print(f"Final a_plot(t_today) = {a_plot[-1]:.6f} (should be 1)")
print(f"Constraint residual: max|C| = {np.max(np.abs(C_resid)):.3e}")
print(f"Lyapunov: L_start={L[0]:.3e}, L_end={L[-1]:.3e} (should not increase)")
print(f"q(t) sample: q_start={q[0]:.3f}, q_mid={q[len(q)//2]:.3f}, q_end={q[-1]:.3f}")