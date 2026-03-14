import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# CGHS-25 Attractor Branch Diagnostics
# ------------------------------------------------------------
# Semi-analytic derivation + numerical fixed-point branch
# for the Hilbert ordering attractor S_*(G).
#
# Output:
#   cghs25_attractor_diagnostics.png
# ============================================================

# ============================================================
# Parameters
# ============================================================
Omega_r0 = 9.0e-5
Omega_m0 = 0.30
Omega_L  = 0.70

# Hilbert-ordering dynamics
sigma   = 0.15     # source amplitude
p_src   = 0.80     # source decay with G
mu      = 3.00     # vacuum restoring strength
nu      = 0.05     # damping coefficient
S_inf   = 8.00     # late-time ordered vacuum scale
S_c     = 1.50     # logistic midpoint in S
width   = 0.35     # logistic sharpness in S

# Semi-analytic transition proxy in G
G_c     = -5.0
dG      = 1.00

# Time-current parameters
beta_t  = 0.80
gamma_t = 1.20
J_inf   = 1.00

# Solve / plot domain
G_min, G_max = -12.0, 2.0
nG = 900

# Output file
OUTFILE = "cghs25_attractor_diagnostics.png"


# ============================================================
# Core functions
# ============================================================
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def u_of_S(S):
    return logistic((S - S_c) / width)

def u_of_G(G):
    return logistic((G - G_c) / dG)

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
    # pulls S toward the ordered vacuum level S_inf * u(S)
    return mu * (S_inf * u_of_S(S) - S)

def damping(G, S):
    return -nu * E(G, S) * S

def dS_dlambda(G, S):
    return source_term(G) + vacuum_pull(G, S) + damping(G, S)

def dG_dlambda(G, S):
    # geometry flow in computational parameter lambda
    return E(G, S)

def rhs_partial_S(G, S, eps=1.0e-5):
    return (dS_dlambda(G, S + eps) - dS_dlambda(G, S - eps)) / (2.0 * eps)


# ============================================================
# Semi-analytic attractor estimate
# ------------------------------------------------------------
# From dS/dlambda ≈ 0, replace u(S) by a transition proxy u_G(G):
#
#   0 ≈ source(G) + mu*(S_inf*u_G - S) - nu*E_bg(G)*S
#
# giving
#
#   S_approx(G) = [source(G) + mu*S_inf*u_G] / [mu + nu*E_bg(G)]
# ============================================================
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
    """
    Solve dS/dlambda(G, S) = 0 for the stable fixed-point branch.
    We scan for sign changes, root-find, then choose the stable root
    nearest the previous solution to continue the branch smoothly.
    """
    S_lo, S_hi = 0.0, max(1.5 * S_inf, 20.0)
    grid = np.linspace(S_lo, S_hi, 1400)
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

    roots = sorted(set(np.round(r, 10) for r in roots))

    stable_roots = [r for r in roots if rhs_partial_S(G, r) < 0.0]
    candidates = stable_roots if stable_roots else roots

    if S_hint is None:
        return float(candidates[0])

    return float(min(candidates, key=lambda r: abs(r - S_hint)))

def compute_branch(G_vals):
    S_num = np.zeros_like(G_vals)
    S_an  = attractor_analytic(G_vals)
    slope = np.zeros_like(G_vals)

    prev = None
    for i, G in enumerate(G_vals):
        root = find_fixed_point_for_G(G, S_hint=prev)
        S_num[i] = root
        slope[i] = rhs_partial_S(G, root)
        prev = root

    return S_an, S_num, slope


# ============================================================
# Main computation
# ============================================================
G = np.linspace(G_min, G_max, nG)
a = np.exp(G)

S_an, S_num, slope_num = compute_branch(G)

u_num = u_of_S(S_num)
J_num = J_star_from_S(S_num)
E_num = E(G, S_num)
Heff_num = H_eff(G, S_num)

fr = np.zeros_like(G)
fm = np.zeros_like(G)
fL = np.zeros_like(G)

src = source_term(G)
vac = np.array([vacuum_pull(g, s) for g, s in zip(G, S_num)])
dmp = np.array([damping(g, s) for g, s in zip(G, S_num)])
net = src + vac + dmp

for i, (g, s) in enumerate(zip(G, S_num)):
    fr[i], fm[i], fL[i] = fractions(g, s)

# approximate emergent time on attractor
dlam = G[1] - G[0]
tau = np.cumsum(J_num) * dlam


# ============================================================
# Vector field for phase portrait
# ============================================================
Gv = np.linspace(G_min, G_max, 34)
Sv = np.linspace(0.0, max(1.2 * S_inf, 10.0), 34)
GG, SS = np.meshgrid(Gv, Sv)

UG = dG_dlambda(GG, SS)
US = dS_dlambda(GG, SS)

speed = np.sqrt(UG**2 + US**2)
UGn = UG / np.maximum(speed, 1e-12)
USn = US / np.maximum(speed, 1e-12)


# ============================================================
# Plot
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

# Panel 1: attractor branch
ax = axs[0, 0]
ax.plot(G, S_an, label="semi-analytic attractor", linewidth=2)
ax.plot(G, S_num, "--", label="numerical fixed-point branch", linewidth=2)
ax.set_title("Hilbert Attractor Branch $S_*(G)$")
ax.set_xlabel("$G = \\ln a$")
ax.set_ylabel("$S_*(G)$")
ax.grid(alpha=0.3)
ax.legend()

# Panel 2: ordering gate + stability
ax = axs[0, 1]
ax.plot(G, u_num, label="$u(S_*)$", linewidth=2)
ax.plot(G, slope_num, label="$\\partial (dS/d\\lambda)/\\partial S|_{S_*}$", linewidth=2)
ax.axhline(0.0, linestyle=":", linewidth=1)
ax.set_title("Ordering Gate and Local Stability")
ax.set_xlabel("$G = \\ln a$")
ax.set_ylabel("diagnostic value")
ax.grid(alpha=0.3)
ax.legend()

# Panel 3: energy fractions
ax = axs[0, 2]
ax.plot(G, fr, label="radiation fraction", linewidth=2)
ax.plot(G, fm, label="matter fraction", linewidth=2)
ax.plot(G, fL, label="Hilbert-vacuum fraction", linewidth=2)
ax.set_title("Energy Budget Along Attractor")
ax.set_xlabel("$G = \\ln a$")
ax.set_ylabel("fraction of $E^2$")
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.3)
ax.legend()

# Panel 4: time-current and observable Hubble rate
ax = axs[1, 0]
ax.plot(G, J_num, label="$J_t^*(G)$", linewidth=2)
ax.plot(G, Heff_num / np.max(Heff_num), label="$H_{eff}/\\max(H_{eff})$", linewidth=2)
ax.set_title("Emergent Time Current and Observable Expansion")
ax.set_xlabel("$G = \\ln a$")
ax.set_ylabel("normalized / absolute")
ax.grid(alpha=0.3)
ax.legend()

# Panel 5: phase portrait
ax = axs[1, 1]
ax.quiver(GG, SS, UGn, USn, speed, angles="xy", pivot="mid", alpha=0.7)
ax.plot(G, S_num, color="black", linewidth=2.5, label="stable attractor")
ax.set_title("Phase Flow in $(G,S)$")
ax.set_xlabel("$G$")
ax.set_ylabel("$S$")
ax.grid(alpha=0.3)
ax.legend()

# Panel 6: RHS decomposition
ax = axs[1, 2]
ax.plot(G, src, label="source", linewidth=2)
ax.plot(G, vac, label="vacuum pull", linewidth=2)
ax.plot(G, dmp, label="damping", linewidth=2)
ax.plot(G, net, "--", label="net residual", linewidth=2)
ax.axhline(0.0, linestyle=":", linewidth=1)
ax.set_title("Attractor RHS Decomposition")
ax.set_xlabel("$G$")
ax.set_ylabel("term value")
ax.grid(alpha=0.3)
ax.legend()

fig.suptitle("CGHS-25 Attractor Diagnostics", fontsize=16)
fig.savefig(OUTFILE, dpi=220, bbox_inches="tight")
print(f"Saved: {OUTFILE}")

# ============================================================
# Compact text summary
# ============================================================
print("\n--- Summary diagnostics ---")
print(f"S_*(early)            = {S_num[0]:.6f}")
print(f"S_*(late)             = {S_num[-1]:.6f}")
print(f"u_early               = {u_num[0]:.6f}")
print(f"u_late                = {u_num[-1]:.6f}")
print(f"J_early               = {J_num[0]:.6f}")
print(f"J_late                = {J_num[-1]:.6f}")
print(f"min stability slope   = {np.min(slope_num):.6e}")
print(f"max |net residual|    = {np.max(np.abs(net)):.6e}")