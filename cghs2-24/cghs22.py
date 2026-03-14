import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp

# ============================================================
# Co-primary Conformal–Hilbert Cosmology (Closed GS, Option A)
# - State: (G, S)
# - Radiation/matter amplitudes are INTERNAL functions of S:
#     Ω_r(G,S) = Ω_r0 * f(S) * exp(-4G)
#     Ω_m(G,S) = Ω_m0 * f(S) * exp(-3G)
# - No pre-existing a^-4 term leaking through a smooth gate tail.
# - One combined 2x3 figure (like your original script).
# ============================================================

# ----------------------------
# Units / time normalizations
# ----------------------------
Gyr = 365.25 * 24 * 3600 * 1e9  # seconds in a gigayear
H0  = 2.2683e-18                # ~70 km/s/Mpc in 1/s (rough)
t_today = 13.8 * Gyr
tau_today = H0 * t_today        # dimensionless τ = H0 t

# ----------------------------
# "Today" density parameters (for late-time normalization)
# ----------------------------
Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70            # residual late vacuum floor

# ----------------------------
# Hilbert vacuum structure Ω_S(S)
# ----------------------------
S_c   = 8.0
Delta = 0.9

Omega_inf = 1.0e4               # inflationary plateau height (dimensionless)

# ----------------------------
# Numeric safety helpers (avoid exp overflow / 0*inf NaNs)
# ----------------------------
EXP_CLIP = 700.0  # exp(700) ~ 1e304 (near float max), prevents overflow

def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))

def logistic_u(S):
    """
    u(S) = 1/(1+exp((S-Sc)/Δ)), but implemented to avoid overflow.
    For x >> 0, u -> 0; for x << 0, u -> 1.
    """
    x = (S - S_c) / Delta

    # Scalar-fast path (solve_ivp passes scalars)
    if np.isscalar(x):
        if x > 50.0:
            return 0.0
        if x < -50.0:
            return 1.0
        ex = np.exp(x)
        return 1.0 / (1.0 + ex)

    # Vector path (for post-processing arrays)
    x = np.asarray(x, dtype=float)
    u = np.empty_like(x)

    u[x > 50.0] = 0.0
    u[x < -50.0] = 1.0
    mid = (x >= -50.0) & (x <= 50.0)
    u[mid] = 1.0 / (1.0 + np.exp(x[mid]))
    return u

def Omega_S(S):
    return Omega_L_floor + Omega_inf * logistic_u(S)

def dOmegaS_dS(S):
    # derivative of Omega_inf * logistic_u(S)
    u = logistic_u(S)
    du = -(1.0 / Delta) * u * (1.0 - u)
    return Omega_inf * du

# ----------------------------
# Internal activation f(S) for emergent radiation/matter amplitudes
# ----------------------------
def f_raw(S):
    # smooth 0->1 sigmoid near Sc
    return 0.5 * (1.0 + np.tanh((S - S_c) / Delta))

def f_internal(S, hard_width=8.0):
    """
    INTERNAL "handoff" function that is EXACTLY ZERO deep in inflation,
    then transitions smoothly near Sc.

    hard_width * Delta sets how far below Sc we force exact zero.
    This eliminates 'gate leakage' multiplying exp(-4G) at tiny a.
    """
    S = np.asarray(S)
    cutoff = S_c - hard_width * Delta
    fr = f_raw(S)
    return np.where(S < cutoff, 0.0, fr)

# ----------------------------
# Constraint manifold: E(G,S)
# ----------------------------
def E_of(G, S):
    """
    E^2 = Ω_r0 f(S) e^{-4G} + Ω_m0 f(S) e^{-3G} + Ω_S(S)
    (numeric-safe implementation)
    """
    fS = float(f_internal(S))

    # safe exponentials prevent overflow if solver probes extreme G
    e4 = safe_exp(-4.0 * G)
    e3 = safe_exp(-3.0 * G)

    term_r = Omega_r0 * fS * e4
    term_m = Omega_m0 * fS * e3
    term_s = Omega_S(S)

    rhs = term_r + term_m + term_s

    # Guard against 0*inf -> NaN or any non-finite RHS
    if (not np.isfinite(rhs)) or (rhs < 0.0):
        rhs = term_s  # fall back to vacuum-only (keeps closure, prevents crash)

    return np.sqrt(rhs)
# ----------------------------
# Hilbert dynamics (same spirit as your asymmetric form)
# ----------------------------
k_roll = 1.0
gamma  = 0.15
beta   = 0.35
mu_exit = 8.0

def rhs_tau(tau, y):
    G, S = y
    E = E_of(G, S)
    E_safe = max(E, 1e-60)

    dG = E

    # inflation-only drift (helps guarantee exit without external field)
    drift = mu_exit * E * logistic_u(S)

    # geometry->Hilbert damping activates with f(S) (post-inflation)
    fS = float(f_internal(S))

    dS = (
        -(k_roll / (3.0 * E_safe)) * dOmegaS_dS(S)     # roll / vacuum gradient
        - gamma * E * S * fS                           # geometric damping (OFF in inflation)
        + beta  * E * np.tanh((S - S_c) / Delta)        # sign-flip feedback near transition
        + drift
    )

    return [dG, dS]

# ============================================================
# Integrate
# ============================================================
a_init = 1e-30
G0 = np.log(a_init)
S0 = 0.0

tau_eval = np.geomspace(1e-18, tau_today, 9000)
tau_eval = np.unique(np.concatenate([[0.0], tau_eval]))

sol = solve_ivp(
    rhs_tau,
    (0.0, tau_today),
    [G0, S0],
    t_eval=tau_eval,
    method="Radau",
    rtol=1e-8,
    atol=1e-10,
)

tau = sol.t
G = sol.y[0]
S = sol.y[1]

# ============================================================
# Derived observables
# ============================================================
E = np.array([E_of(G[i], S[i]) for i in range(len(tau))])

# time in seconds and Gyr
t = tau / H0
t_gyr = t / Gyr

# scale factor
a = np.exp(G)
a_plot = a / a[-1]  # normalize to 1 today
V_plot = (a_plot ** 3) / (a_plot[-1] ** 3)

# Inflation diagnostic: comoving Hubble radius (aH)^-1 (arb units)
# H = H0 E, so (aH)^-1 ∝ 1/(a E)
comoving_hubble = 1.0 / np.maximum(a * E, 1e-300)

# Density terms + fractions on the manifold
fS_vec = f_internal(S)
term_r = Omega_r0 * fS_vec * safe_exp(-4.0 * G)
term_m = Omega_m0 * fS_vec * safe_exp(-3.0 * G)
term_s = Omega_S(S)
rhs_E2 = term_r + term_m + term_s

C_resid = E**2 - rhs_E2

Or_eff = term_r / np.maximum(E**2, 1e-300)
Om_eff = term_m / np.maximum(E**2, 1e-300)
Os_eff = term_s / np.maximum(E**2, 1e-300)

# Deceleration parameter: q = -1 - (dE/dτ)/E^2  (since H ∝ E)
dE_dtau = np.gradient(E, tau, edge_order=2)
q = -1.0 - dE_dtau / np.maximum(E**2, 1e-300)

# ============================================================
# 3D tube geometry (radius proportional to a_plot)
# ============================================================
theta = np.linspace(0, 2*np.pi, 80)
T, TH = np.meshgrid(t_gyr, theta)
R_mesh = np.tile(a_plot, (len(theta), 1))

X = T
Y = R_mesh * np.cos(TH)
Z = R_mesh * np.sin(TH)

# ============================================================
# One combined figure (2x3)
# ============================================================
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# (1) 3D tube
ax0 = fig.add_subplot(gs[0, 0], projection="3d")
ax0.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0)
ax0.set_title("Co-primary tube (radius ∝ a)")
ax0.set_xlabel("Time (Gyr)")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")

# (2) a(t)
ax1 = fig.add_subplot(gs[0, 1])
ax1.semilogy(t_gyr, a_plot)
ax1.set_title("Scale factor a(t) (normalized to 1 today)")
ax1.set_xlabel("Time (Gyr)")
ax1.set_ylabel("a(t) (log)")

# (3) Volume metric V ∝ a^3
ax2 = fig.add_subplot(gs[0, 2])
ax2.semilogy(t_gyr, V_plot)
ax2.set_title("Volume metric: V(t) ∝ a(t)^3 (normalized)")
ax2.set_xlabel("Time (Gyr)")
ax2.set_ylabel("V(t) (log)")

# (4) Inflation diagnostic: (aH)^(-1)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t + 1e-40, comoving_hubble)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Inflation diagnostic: comoving Hubble radius (aH)^(-1)")
ax3.set_xlabel("t (seconds, log)")
ax3.set_ylabel("(aH)^(-1) (arb, log)")

# (5) Effective density fractions on the manifold
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_gyr, Or_eff, label="Ω_r,eff")
ax4.plot(t_gyr, Om_eff, label="Ω_m,eff")
ax4.plot(t_gyr, Os_eff, label="Ω_S,eff")
ax4.set_title("Manifold fractions (who dominates)")
ax4.set_xlabel("Time (Gyr)")
ax4.set_ylabel("Fraction of E^2")
ax4.legend(loc="best")

# (6) Constraint residual + q(t) + S(t)
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(t_gyr, C_resid, label="Constraint residual (E^2 - RHS)", alpha=0.9)
ax5.set_title("Constraint + dynamics markers")
ax5.set_xlabel("Time (Gyr)")
ax5.set_ylabel("Constraint residual")

ax5b = ax5.twinx()
ax5b.plot(t_gyr, q, linestyle="--", label="q(t)")
ax5b.plot(t_gyr, S, linestyle=":", label="S(t)")
ax5b.set_ylabel("q(t), S(t)")

lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5b.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.savefig("coprimary_closed_internal_amplitude.png", dpi=300)
plt.show()