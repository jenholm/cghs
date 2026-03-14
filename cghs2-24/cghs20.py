import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.gridspec as gridspec

# ============================================================
# CO-PRIMARY CLOSED SYSTEM (Cosmic time, but integrated in dimensionless τ=H0 t)
#
# State:
#   G(τ) = ln a
#   S(τ) = Hilbert operator (order/state)
#
# Measurement map:
#   a = exp(G)
#
# Constraint manifold (Friedmann-like governor):
#   E(G,S)^2 = gate(S) * [Ωr0 e^{-4G} + Ωm0 e^{-3G}] + ΩS(S)
#   H = H0 * E
#
# ODEs (2 equations, closed):
#   dG/dτ = +E
#   dS/dτ = -(k_roll/(3E)) * dΩS/dS  - γ E S  + β E * tanh((S-Sc)/Δ)
#
# Key asymmetry:
#   - During inflation (S<Sc): gate~0, so geometry is driven by ΩS and
#     radiation/matter are effectively "not yet on".
#   - Post inflation (S>Sc): gate->1, radiation/matter turn on; the
#     tanh term flips sign, changing geometry's net influence on S.
#
# Diagnostics:
#   - comoving Hubble radius (aH)^(-1) should DECREASE during inflation
#   - q(t) should be negative during inflation
#   - constraint residual should stay ~0
# ============================================================

# ----------------------------
# Units
# ----------------------------
year = 3.154e7
Gyr = 1e9 * year

# ----------------------------
# Late-time parameters (defined at a=1, today)
# ----------------------------
H0 = 70.0 * 1e3 / (3.086e22)  # 70 km/s/Mpc in 1/s
Omega_r0 = 9e-5
Omega_m0 = 0.30

# ----------------------------
# Hilbert-sector "vacuum" contribution ΩS(S): plateau -> floor
# ΩS acts like an effective vacuum energy density parameter on the manifold.
# ----------------------------
Omega_L_floor = 0.70          # late-time residual
Omega_inf = 1e2              # inflationary plateau height (dimensionless in E^2 units)
S_c = 8.0                     # transition center
Delta = 0.9                   # transition width


def logistic_u(S):
    """u ~ 1 early (S << Sc), u ~ 0 late (S >> Sc)."""
    x = (S - S_c) / Delta
    return 1.0 / (1.0 + np.exp(x))

def Omega_S(S):
    """Hilbert-sector contribution to E^2."""
    return Omega_L_floor + Omega_inf * logistic_u(S)

def dOmegaS_dS(S):
    """Derivative of Omega_S with respect to S."""
    x = (S - S_c) / Delta
    ex = np.exp(x)
    # d/dS [1/(1+e^x)] = -(e^x)/(1+e^x)^2 * (1/Δ)
    du_dS = -(ex / (1.0 + ex)**2) * (1.0 / Delta)
    return Omega_inf * du_dS  # floor term derivative = 0

# ----------------------------
# Reheating / handoff gate: turns on radiation+matter smoothly after Sc
# This is the "different law before system took over", but still closed:
# gate depends on S(t).
# ----------------------------
def gate(S):
    """0 (inflation) -> 1 (hot big bang) smoothly."""
    return 0.5 * (1.0 + np.tanh((S - S_c) / Delta))

# ----------------------------
# Constraint manifold: E(G,S)
# ----------------------------
def E_of(G, S):
    """
    Dimensionless Hubble E = H/H0 derived from the manifold:
      E^2 = gate(S)*(Ωr0 a^-4 + Ωm0 a^-3) + ΩS(S)
    with a = exp(G).
    """
    g = gate(S)
    term_r = Omega_r0 * np.exp(-4.0 * G)
    term_m = Omega_m0 * np.exp(-3.0 * G)
    term_s = Omega_S(S)
    rhs = g * (term_r + term_m) + term_s
    rhs = np.maximum(rhs, 0.0)
    return np.sqrt(rhs)

# ----------------------------
# Asymmetric Hilbert dynamics parameters
# ----------------------------
k_roll = 1.0
gamma = 0.15
beta  = 0.35
#beta = 1.4
#gamma = 0.15

#mu_exit = 0.02     # NEW: slow drift toward Sc during inflation (in τ units)
                   # tune in [1e-3, 1e-1] depending on behavior

mu_exit = 8.0   # order unity, because τ_today ~ 1


def rhs_tau(tau, y):
    G, S = y
    E = E_of(G, S)
    E_safe = max(E, 1e-60)

    dG = E

    # flip is +1 in inflation (S<Sc), -1 post-inflation if you keep this definition
    flip = np.tanh((S_c - S) / Delta)

    # NEW: inflation-only drift that guarantees exit
    #drift = mu_exit * logistic_u(S)   # ~mu_exit during inflation, ~0 after
    drift = mu_exit * E * logistic_u(S)

    dS = (
        -(k_roll / (3.0 * E_safe)) * dOmegaS_dS(S)
        - gamma * E * S * gate(S)
        #+ beta * E * flip
        #+ beta * E * gate(S) * np.tanh((S - S_c)/Delta)
        + beta * E * np.tanh((S - S_c)/Delta)
        + drift
    )

    #dS += mu_exit * logistic_u(S)

    return [dG, dS]



# ============================================================
# Integrate from "very early" to today in τ units
# ============================================================
t_today = 13.8 * Gyr
tau_today = H0 * t_today

# Initial conditions
a_init = 1e-30          # do NOT start at 1e-60; we gate radiation anyway, but this keeps numbers sane
G0 = np.log(a_init)
S0 = 0.0                # starts deep in inflation regime (S << Sc)

# Log-ish sampling in τ to resolve early behavior
tau_eval = np.geomspace(1e-18, tau_today, 9000)
tau_eval = np.unique(np.concatenate([[0.0], tau_eval]))

sol = solve_ivp(
    rhs_tau,
    (0.0, tau_today),
    [G0, S0],
    t_eval=tau_eval,
    method="Radau",
    rtol=1e-8,
    atol=1e-10
)


tau = sol.t
G = sol.y[0]
S = sol.y[1]

# Convert back to physical time for labels
t = tau / H0
t_gyr = t / Gyr


# ============================================================
# Diagnostics: find exit / reheating markers
# ============================================================
idx_cross = np.where(S >= S_c)[0]
if len(idx_cross) > 0:
    i0 = idx_cross[0]
    print(f"EXIT: S crossed Sc at t = {t[i0]:.3e} s  (={t_gyr[i0]:.3e} Gyr)")
    print(f"      gate(S) there = {gate(S[i0]):.3f}")
    # "e-folds" proxy in this model is ΔG while in inflation regime
    N_proxy = (G[i0] - G[0])
    print(f"      N_proxy = ΔG(inflation) = {N_proxy:.3f}")
else:
    print("EXIT: S never crossed Sc (no reheating). Increase mu_exit or adjust beta/gamma.")

# ============================================================
# Derived series (LOG-CONSISTENT VERSION)
# ============================================================

E = np.array([E_of(G[i], S[i]) for i in range(len(tau))])
H = H0 * E

# ---- Normalize in log space ----
G_shifted = G - G[-1]        # ensures a(today)=1
a_plot = np.exp(G_shifted)   # SAFE: max exponent is 0

V_plot = a_plot**3

# ---- Inflation diagnostic ----
comoving_hubble = np.where(
    a_plot * H > 0,
    1.0 / (a_plot * H),
    np.nan
)

# ---- Constraint residual (use exp(-nG), NOT a**(-n)) ----
gS = gate(S)

term_r = Omega_r0 * np.exp(-4.0 * G)
term_m = Omega_m0 * np.exp(-3.0 * G)
rhs_E2 = gS * (term_r + term_m) + Omega_S(S)

C_resid = E**2 - rhs_E2

rel_resid = C_resid / np.maximum(E**2, 1.0)
print("max|relative constraint residual| =", np.max(np.abs(rel_resid)))

# ---- Density fractions ----
Or_eff = (gS * term_r) / np.maximum(E**2, 1e-300)
Om_eff = (gS * term_m) / np.maximum(E**2, 1e-300)
Os_eff = Omega_S(S) / np.maximum(E**2, 1e-300)

# Deceleration parameter:
# q = -1 - (dH/dt)/H^2
# with H = H0 E, t = τ/H0, so dH/dt = H0^2 dE/dτ
# => q = -1 - (dE/dτ)/E^2
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


# --- Term audit at final time ---
i = -1
Gf, Sf = G[i], S[i]
Ef = E_of(Gf, Sf)
gf = gate(Sf)
flipf = np.tanh((S_c - Sf)/Delta)  # or whatever you currently use

term_roll = -(k_roll/(3.0*max(Ef,1e-60))) * dOmegaS_dS(Sf)
term_fric = -gamma * Ef * Sf              # CHANGE to *gf if using gating
term_beta =  beta * Ef * flipf            # or *gf if you gate it
term_exit =  mu_exit * logistic_u(Sf)

print("\n=== S-term audit (final step) ===")
print(f"S_end={Sf:.6f}, gate_end={gf:.6f}, E_end={Ef:.6f}, logistic_u={logistic_u(Sf):.6f}, flip={flipf:.6f}")
print(f"roll  = {term_roll:+.6e}")
print(f"fric  = {term_fric:+.6e}")
print(f"beta  = {term_beta:+.6e}")
print(f"exit  = {term_exit:+.6e}")
print(f"dSnet = {(term_roll+term_fric+term_beta+term_exit):+.6e}\n")

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
ax1.plot(t_gyr, a_plot)
ax1.set_title("Scale factor a(t) (normalized to 1 today)")
ax1.set_xlabel("Time (Gyr)")
ax1.set_ylabel("a(t)")
# a(t) on log scale (otherwise it's a flatline)
ax1.semilogy(t_gyr, a_plot)
ax1.set_ylabel("a(t) (log scale)")



# (3) Volume metric V ∝ a^3
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(t_gyr, V_plot)
ax2.set_title("Volume metric: V(t) ∝ a(t)^3 (normalized)")
ax2.set_xlabel("Time (Gyr)")
ax2.set_ylabel("V(t)")
# V(t) also on log scale
ax2.semilogy(t_gyr, V_plot)
ax2.set_ylabel("V(t) (log scale)")

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

# Combine legends
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5b.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.savefig("coprimary_asymmetric_closed.png", dpi=300)
plt.show()

# ============================================================
# Print quick sanity readout
# ============================================================
print("=== Quick diagnostics ===")
print(f"a_plot(today) = {a_plot[-1]:.6f} (should be 1)")
print(f"max|constraint residual| = {np.max(np.abs(C_resid)):.3e}")
print(f"q at start/mid/end = {q[0]:.3f}, {q[len(q)//2]:.3f}, {q[-1]:.3f}")
print(f"S at start/end = {S[0]:.3f}, {S[-1]:.3f}")
print(f"gate(S) at start/end = {gate(S[0]):.3f}, {gate(S[-1]):.3f}")