import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# CGHS-25 vs LCDM Comparison Script
# ------------------------------------------------------------
# Computes the stable attractor branch S_*(G), reconstructs
# emergent physical time tau(G), and compares:
#
#   1. a(tau)
#   2. H_eff(tau)
#   3. q(tau)
#
# against a reference flat LCDM model.
#
# Output:
#   cghs25_lcdm_comparison.png
# ============================================================

# ============================================================
# Parameters
# ============================================================
Omega_r0 = 9.0e-5
Omega_m0 = 0.30
Omega_L  = 0.70

# Hilbert-ordering dynamics
sigma   = 0.15
p_src   = 0.80
mu      = 3.00
nu      = 0.05
S_inf   = 8.00
S_c     = 1.50
width   = 0.35

# Semi-analytic transition proxy in G
G_c     = -5.0
dG_gate = 1.00

# Time-current parameters
beta_t  = 0.80
gamma_t = 1.20
J_inf   = 1.00

# Domain
G_min, G_max = -12.0, 2.0
nG = 1400

OUTFILE = "cghs25_lcdm_comparison.png"


# ============================================================
# Core functions
# ============================================================
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def u_of_S(S):
    return logistic((S - S_c) / width)

def u_of_G(G):
    return logistic((G - G_c) / dG_gate)

def source_term(G):
    return sigma * np.exp(-p_src * G)

def E2_from_u(G, u):
    return (
        Omega_r0 * np.exp(-4.0 * G)
        + Omega_m0 * np.exp(-3.0 * G)
        + Omega_L * u
    )

def E2(G, S):
    return E2_from_u(G, u_of_S(S))

def E(G, S):
    return np.sqrt(np.maximum(E2(G, S), 1.0e-14))

def vacuum_pull(G, S):
    return mu * (S_inf * u_of_S(S) - S)

def damping(G, S):
    return -nu * E(G, S) * S

def dS_dlambda(G, S):
    return source_term(G) + vacuum_pull(G, S) + damping(G, S)

def rhs_partial_S(G, S, eps=1.0e-5):
    return (dS_dlambda(G, S + eps) - dS_dlambda(G, S - eps)) / (2.0 * eps)

def attractor_analytic(G):
    uG = u_of_G(G)
    E_bg = np.sqrt(np.maximum(E2_from_u(G, uG), 1.0e-14))
    return (source_term(G) + mu * S_inf * uG) / (mu + nu * E_bg)

def J_star_from_S(S):
    u = u_of_S(S)
    return J_inf + (beta_t / gamma_t) * u * (1.0 - u)

def H_eff(G, S):
    return E(G, S) / J_star_from_S(S)

def fractions(G, S):
    u = u_of_S(S)
    Er = Omega_r0 * np.exp(-4.0 * G)
    Em = Omega_m0 * np.exp(-3.0 * G)
    EL = Omega_L * u
    Etot = Er + Em + EL
    return Er / Etot, Em / Etot, EL / Etot


# ============================================================
# Numerical fixed-point branch
# ============================================================
def find_fixed_point_for_G(G, S_hint=None):
    S_lo, S_hi = 0.0, max(1.5 * S_inf, 20.0)
    grid = np.linspace(S_lo, S_hi, 1800)
    vals = np.array([dS_dlambda(G, s) for s in grid])

    brackets = []
    for i in range(len(grid) - 1):
        f1, f2 = vals[i], vals[i + 1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 == 0.0:
            brackets.append((grid[i], grid[i]))
        elif f1 * f2 < 0.0:
            brackets.append((grid[i], grid[i + 1]))

    roots = []
    for a, b in brackets:
        if a == b:
            roots.append(a)
        else:
            try:
                roots.append(brentq(lambda s: dS_dlambda(G, s), a, b, maxiter=200))
            except ValueError:
                pass

    if not roots:
        s_est = attractor_analytic(G)
        return float(np.clip(s_est, S_lo, S_hi))

    #roots = sorted(set(np.round(r, 10) for r in roots)))
    roots = sorted(set(np.round(r, 10) for r in roots))
    stable_roots = [r for r in roots if rhs_partial_S(G, r) < 0.0]
    candidates = stable_roots if stable_roots else roots

    if S_hint is None:
        # choose the lowest stable root by default
        return float(candidates[0])

    return float(min(candidates, key=lambda r: abs(r - S_hint)))


def compute_branch(G_vals):
    S_num = np.zeros_like(G_vals)
    slope = np.zeros_like(G_vals)

    prev = None
    for i, G in enumerate(G_vals):
        root = find_fixed_point_for_G(G, S_hint=prev)
        S_num[i] = root
        slope[i] = rhs_partial_S(G, root)
        prev = root

    return S_num, slope


# ============================================================
# Build CGHS attractor history
# ============================================================
G = np.linspace(G_min, G_max, nG)
a = np.exp(G)

S_num, slope_num = compute_branch(G)
u_num = u_of_S(S_num)
J_num = J_star_from_S(S_num)
E_num = E(G, S_num)
H_num = H_eff(G, S_num)

# Emergent time:
#   dG/dlambda = E
#   dtau/dlambda = J
# so
#   dtau/dG = J/E
dG_step = G[1] - G[0]
dtau_dG = J_num / np.maximum(E_num, 1.0e-14)

tau = np.zeros_like(G)
tau[1:] = np.cumsum(0.5 * (dtau_dG[1:] + dtau_dG[:-1]) * dG_step)

# Shift so that today (a=1, G=0) is tau = 0
idx_today = np.argmin(np.abs(G - 0.0))
tau = tau - tau[idx_today]

# Numerical derivatives with respect to tau
def safe_gradient(y, x):
    return np.gradient(y, x, edge_order=2)

a_tau = a
H_tau = H_num

da_dtau = safe_gradient(a_tau, tau)
dH_dtau = safe_gradient(H_tau, tau)

# q = - a a'' / a'^2 = -1 - (1/H^2) dH/dtau
q_cghs = -1.0 - dH_dtau / np.maximum(H_tau**2, 1.0e-14)

# Alternative consistency check from a'' if wanted
d2a_dtau2 = safe_gradient(da_dtau, tau)
q_from_a = -a_tau * d2a_dtau2 / np.maximum(da_dtau**2, 1.0e-14)

fr = np.zeros_like(G)
fm = np.zeros_like(G)
fL = np.zeros_like(G)
for i, (g, s) in enumerate(zip(G, S_num)):
    fr[i], fm[i], fL[i] = fractions(g, s)


# ============================================================
# Reference flat LCDM model
# ------------------------------------------------------------
# Use same present-day densities:
#   E_LCDM(a)^2 = Om_r a^-4 + Om_m a^-3 + Om_L
#
# with physical time tau_LCDM from
#   dtau/dG = 1 / E_LCDM
# because dG/dtau = H/H0 = E_LCDM
# ============================================================
def E_lcdm_of_G(G):
    return np.sqrt(np.maximum(
        Omega_r0 * np.exp(-4.0 * G)
        + Omega_m0 * np.exp(-3.0 * G)
        + Omega_L,
        1.0e-14
    ))

E_lcdm = E_lcdm_of_G(G)
dtau_dG_lcdm = 1.0 / np.maximum(E_lcdm, 1.0e-14)

tau_lcdm = np.zeros_like(G)
tau_lcdm[1:] = np.cumsum(0.5 * (dtau_dG_lcdm[1:] + dtau_dG_lcdm[:-1]) * dG_step)
tau_lcdm = tau_lcdm - tau_lcdm[idx_today]

H_lcdm = E_lcdm

dH_lcdm_dtau = safe_gradient(H_lcdm, tau_lcdm)
q_lcdm = -1.0 - dH_lcdm_dtau / np.maximum(H_lcdm**2, 1.0e-14)


# ============================================================
# Interpolate both models onto a common tau grid
# ============================================================
tau_min = max(np.min(tau), np.min(tau_lcdm))
tau_max = min(np.max(tau), np.max(tau_lcdm))
tau_common = np.linspace(tau_min, tau_max, 1200)

# Need monotone tau arrays for interpolation
sort_cghs = np.argsort(tau)
sort_lcdm = np.argsort(tau_lcdm)

tau_cghs_sorted = tau[sort_cghs]
a_cghs_sorted = a_tau[sort_cghs]
H_cghs_sorted = H_tau[sort_cghs]
q_cghs_sorted = q_cghs[sort_cghs]
q2_cghs_sorted = q_from_a[sort_cghs]
u_cghs_sorted = u_num[sort_cghs]
J_cghs_sorted = J_num[sort_cghs]

tau_lcdm_sorted = tau_lcdm[sort_lcdm]
a_lcdm_sorted = a[sort_lcdm]
H_lcdm_sorted = H_lcdm[sort_lcdm]
q_lcdm_sorted = q_lcdm[sort_lcdm]

a_cghs_i = np.interp(tau_common, tau_cghs_sorted, a_cghs_sorted)
H_cghs_i = np.interp(tau_common, tau_cghs_sorted, H_cghs_sorted)
q_cghs_i = np.interp(tau_common, tau_cghs_sorted, q_cghs_sorted)
q2_cghs_i = np.interp(tau_common, tau_cghs_sorted, q2_cghs_sorted)
u_cghs_i = np.interp(tau_common, tau_cghs_sorted, u_cghs_sorted)
J_cghs_i = np.interp(tau_common, tau_cghs_sorted, J_cghs_sorted)

a_lcdm_i = np.interp(tau_common, tau_lcdm_sorted, a_lcdm_sorted)
H_lcdm_i = np.interp(tau_common, tau_lcdm_sorted, H_lcdm_sorted)
q_lcdm_i = np.interp(tau_common, tau_lcdm_sorted, q_lcdm_sorted)

# Normalize H for shape comparison
H_cghs_norm = H_cghs_i / H_cghs_i[np.argmin(np.abs(tau_common))]
H_lcdm_norm = H_lcdm_i / H_lcdm_i[np.argmin(np.abs(tau_common))]

# Relative differences
eps = 1.0e-12
rel_H_diff = (H_cghs_i - H_lcdm_i) / np.maximum(H_lcdm_i, eps)
rel_a_diff = (a_cghs_i - a_lcdm_i) / np.maximum(a_lcdm_i, eps)


# ============================================================
# Plot
# ============================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

# Panel 1: a(tau)
ax = axs[0, 0]
ax.plot(tau_common, a_cghs_i, label="CGHS-25", linewidth=2)
ax.plot(tau_common, a_lcdm_i, "--", label=r"$\Lambda$CDM", linewidth=2)
ax.axvline(0.0, linestyle=":", linewidth=1)
ax.set_title(r"Scale Factor $a(\tau)$")
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$a$")
ax.grid(alpha=0.3)
ax.legend()

# Panel 2: H(tau)
ax = axs[0, 1]
ax.plot(tau_common, H_cghs_norm, label="CGHS-25", linewidth=2)
ax.plot(tau_common, H_lcdm_norm, "--", label=r"$\Lambda$CDM", linewidth=2)
ax.axvline(0.0, linestyle=":", linewidth=1)
ax.set_title(r"Normalized Expansion Rate $H(\tau)/H(\tau_0)$")
ax.set_xlabel(r"$\tau$")
ax.set_ylabel("normalized H")
ax.grid(alpha=0.3)
ax.legend()

# Panel 3: q(tau)
ax = axs[1, 0]
ax.plot(tau_common, q_cghs_i, label="CGHS-25 from $H$", linewidth=2)
ax.plot(tau_common, q2_cghs_i, ":", label="CGHS-25 from $a$", linewidth=2)
ax.plot(tau_common, q_lcdm_i, "--", label=r"$\Lambda$CDM", linewidth=2)
ax.axhline(0.0, linestyle=":", linewidth=1)
ax.axvline(0.0, linestyle=":", linewidth=1)
ax.set_title(r"Deceleration Parameter $q(\tau)$")
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$q$")
ax.grid(alpha=0.3)
ax.legend()

# Panel 4: residual diagnostics
ax = axs[1, 1]
ax.plot(tau_common, rel_H_diff, label=r"$(H_{\rm CGHS}-H_{\Lambda CDM})/H_{\Lambda CDM}$", linewidth=2)
ax.plot(tau_common, rel_a_diff, label=r"$(a_{\rm CGHS}-a_{\Lambda CDM})/a_{\Lambda CDM}$", linewidth=2)
ax.plot(tau_common, u_cghs_i, label=r"$u(S_*)$", linewidth=2, alpha=0.8)
ax.plot(tau_common, J_cghs_i, label=r"$J_t^*$", linewidth=2, alpha=0.8)
ax.axhline(0.0, linestyle=":", linewidth=1)
ax.axvline(0.0, linestyle=":", linewidth=1)
ax.set_title("Comparison Residuals and CGHS Internal Diagnostics")
ax.set_xlabel(r"$\tau$")
ax.set_ylabel("diagnostic value")
ax.grid(alpha=0.3)
ax.legend()

fig.suptitle("CGHS-25 vs ΛCDM Comparison", fontsize=16)
fig.savefig(OUTFILE, dpi=220, bbox_inches="tight")
print(f"Saved: {OUTFILE}")

# ============================================================
# Text summary
# ============================================================
today_idx_common = np.argmin(np.abs(tau_common))
print("\n--- CGHS-25 vs LCDM summary ---")
print(f"tau range overlap          = [{tau_min:.6f}, {tau_max:.6f}]")
print(f"a_CGHS(today)              = {a_cghs_i[today_idx_common]:.6f}")
print(f"a_LCDM(today)              = {a_lcdm_i[today_idx_common]:.6f}")
print(f"H_CGHS(today)              = {H_cghs_i[today_idx_common]:.6f}")
print(f"H_LCDM(today)              = {H_lcdm_i[today_idx_common]:.6f}")
print(f"q_CGHS(today)              = {q_cghs_i[today_idx_common]:.6f}")
print(f"q_LCDM(today)              = {q_lcdm_i[today_idx_common]:.6f}")
print(f"max |relative H diff|      = {np.max(np.abs(rel_H_diff)):.6e}")
print(f"max |relative a diff|      = {np.max(np.abs(rel_a_diff)):.6e}")
print(f"J_t early / late           = {J_num[0]:.6f} / {J_num[-1]:.6f}")
print(f"u(S_*) early / late        = {u_num[0]:.6f} / {u_num[-1]:.6f}")