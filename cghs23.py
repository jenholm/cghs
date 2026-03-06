import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp

# ============================================================
# Co-primary Conformal–Hilbert Cosmology (Closed GS, Option A)
# Focus shift:
#   Geometry and Hilbert structure are dual aspects of ONE constraint manifold.
#   We keep state (G,S) but:
#     STEP 2: tie f(S) to vacuum release: f(S) = 1 - logistic_u(S)
#     STEP 1: replace ad hoc dS/dτ with a principled manifold flow rule:
#             damped gradient flow of a single scalar "potential" U(S)
# ============================================================

# ----------------------------
# Units / time normalizations
# ----------------------------
Gyr = 365.25 * 24 * 3600 * 1e9  # seconds in a gigayear
H0  = 2.2683e-18                # ~70 km/s/Mpc in 1/s (rough)
t_today = 13.8 * Gyr
tau_today = H0 * t_today        # dimensionless τ = H0 t

# ----------------------------
# "Today" density parameters (late-time normalization)
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

# A tiny "tilt" breaks the perfect plateau-flatness so S can drift without
# adding a separate "exit motor" term. Set to 0.0 if you want perfect flat.
#epsilon_tilt = 1e-6             # << keep tiny (dimensionless)
epsilon_tilt = 0.02

# ----------------------------
# Numeric safety helpers
# ----------------------------
EXP_CLIP = 700.0  # exp(700) ~ 1e304, prevents overflow

def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))

def logistic_u(S):
    """
    u(S) = 1/(1+exp((S-Sc)/Δ)), overflow-safe.
    u ~ 1 for S << Sc (inflation), u ~ 0 for S >> Sc (post-exit).
    """
    x = (S - S_c) / Delta

    if np.isscalar(x):
        if x > 50.0:
            return 0.0
        if x < -50.0:
            return 1.0
        ex = np.exp(x)
        return 1.0 / (1.0 + ex)

    x = np.asarray(x, dtype=float)
    u = np.empty_like(x)
    u[x > 50.0] = 0.0
    u[x < -50.0] = 1.0
    mid = (x >= -50.0) & (x <= 50.0)
    u[mid] = 1.0 / (1.0 + np.exp(x[mid]))
    return u

def Omega_S(S):
    """
    Vacuum sector: floor + plateau + bounded tilt.
    Using tanh keeps the "slope to move" but prevents Ω_S → -∞ for S → -∞.
    """
    u = logistic_u(S)
    tilt = epsilon_tilt * np.tanh((S - S_c) / Delta)
    return Omega_L_floor + Omega_inf * u + tilt

def dOmegaS_dS(S):
    """
    d/dS [Omega_inf * logistic_u(S) + epsilon_tilt*tanh((S-Sc)/Δ)]
    """
    u = logistic_u(S)
    du = -(1.0 / Delta) * u * (1.0 - u)

    x = (S - S_c) / Delta
    dtilt = (epsilon_tilt / Delta) * (1.0 / np.cosh(x)**2)  # sech^2(x)

    return Omega_inf * du + dtilt

# ============================================================
# ### STEP 2: Tie reheating emergence to vacuum release
# We define "released fraction" R(S) = 1 - logistic_u(S).
# Then matter/radiation amplitudes are proportional to R(S).
# We preserve "exact zero deep in inflation" using a hard cutoff.
# ============================================================
def f_internal(S, hard_width=8.0):
    """
    Emergent amplitude tied to vacuum release:
        f(S) = 1 - logistic_u(S)
    with exact zero for S far below Sc (prevents leakage at tiny a).
    """
    cutoff = S_c - hard_width * Delta

    # scalar path for solver
    if np.isscalar(S):
        if S < cutoff:
            return 0.0
        return float(1.0 - logistic_u(S))

    # vector path for post-processing
    S = np.asarray(S)
    f = 1.0 - logistic_u(S)
    f[S < cutoff] = 0.0
    return f

# ----------------------------
# Constraint manifold: E(G,S)
# ----------------------------
def E_of(G, S):
    """
    E^2 = Ω_r0 f(S) e^{-4G} + Ω_m0 f(S) e^{-3G} + Ω_S(S)
    """
    fS = float(f_internal(S))
    e4 = safe_exp(-4.0 * G)
    e3 = safe_exp(-3.0 * G)

    term_r = Omega_r0 * fS * e4
    term_m = Omega_m0 * fS * e3
    term_s = Omega_S(S)

    rhs = term_r + term_m + term_s
    if (not np.isfinite(rhs)) or (rhs < 0.0):
        rhs = term_s

    return np.sqrt(rhs)

# ============================================================
# ### STEP 1: Replace ad hoc Hilbert RHS with manifold + flow rule
#
# Interpretation:
#   - The constraint defines the manifold through E(G,S).
#   - The system moves on that manifold.
#   - S evolves by damped gradient flow of a single scalar "potential" U(S).
#
# Choose U(S) = Ω_S(S) (vacuum compatibility potential).
# Then S "wants" to move downhill in Ω_S, i.e., reduce inflationary vacuum.
#
# Geometry→Hilbert influence is modeled only as post-exit damping proportional
# to expansion rate and turned on by f(S). (You can set gamma=0 to remove it.)
# ============================================================
#kappa = 1.0      # strength of "flow down U"
kappa = 500.0
gamma = 0.15     # post-exit geometric damping strength

def rhs_tau(tau, y):
    G, S = y
    # Numeric guardrail: keep S in a reasonable toy-model domain
    # (prevents runaway to huge negative values that can destabilize the manifold)
    S = float(np.clip(S, -50.0, 50.0))
    E = E_of(G, S)
    E_safe = max(E, 1e-60)

    # Geometry flow: still dG/dτ = E
    dG = E

    # Manifold-driven Hilbert flow:
    # "descend" U(S) = Ω_S(S) with a gentle scaling by 1/(3E)
    # (keeps the flow from becoming stiff when E is large).
    dS_potential = -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)
    #dS_potential = -kappa * dOmegaS_dS(S)
    # Geometry-to-Hilbert damping that activates only when vacuum has released
    fS = float(f_internal(S))
    dS_damp = - gamma * E * S * fS

    dS = dS_potential + dS_damp
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

t = tau / H0
t_gyr = t / Gyr

#a = np.exp(G)
#a_plot = a / a[-1]
#V_plot = (a_plot ** 3) / (a_plot[-1] ** 3)

# --- Numerically safe normalization: avoid exp overflow ---
G_shift = G - G[-1]                 # so that a_plot(t_today)=1 exactly
a_plot = np.exp(np.clip(G_shift, -EXP_CLIP, EXP_CLIP))
V_plot = a_plot**3

#comoving_hubble = 1.0 / np.maximum(a * E, 1e-300)
# proportional to 1/(aE); using normalized a_plot is fine up to a constant factor
comoving_hubble = 1.0 / np.maximum(a_plot * E, 1e-300)

fS_vec = f_internal(S)
term_r = Omega_r0 * fS_vec * safe_exp(-4.0 * G)
term_m = Omega_m0 * fS_vec * safe_exp(-3.0 * G)
term_s = Omega_S(S)
rhs_E2 = term_r + term_m + term_s

C_resid = E**2 - rhs_E2

Or_eff = term_r / np.maximum(E**2, 1e-300)
Om_eff = term_m / np.maximum(E**2, 1e-300)
Os_eff = term_s / np.maximum(E**2, 1e-300)

dE_dtau = np.gradient(E, tau, edge_order=2)
q = -1.0 - dE_dtau / np.maximum(E**2, 1e-300)

# ============================================================
# 3D tube geometry
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

ax0 = fig.add_subplot(gs[0, 0], projection="3d")
ax0.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0)
ax0.set_title("Co-primary tube (radius ∝ a)")
ax0.set_xlabel("Time (Gyr)")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")

ax1 = fig.add_subplot(gs[0, 1])
ax1.semilogy(t_gyr, a_plot)
ax1.set_title("Scale factor a(t) (normalized to 1 today)")
ax1.set_xlabel("Time (Gyr)")
ax1.set_ylabel("a(t) (log)")

ax2 = fig.add_subplot(gs[0, 2])
ax2.semilogy(t_gyr, V_plot)
ax2.set_title("Volume metric: V(t) ∝ a(t)^3 (normalized)")
ax2.set_xlabel("Time (Gyr)")
ax2.set_ylabel("V(t) (log)")

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t + 1e-40, comoving_hubble)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Inflation diagnostic: comoving Hubble radius (aH)^(-1)")
ax3.set_xlabel("t (seconds, log)")
ax3.set_ylabel("(aH)^(-1) (arb, log)")

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_gyr, Or_eff, label="Ω_r,eff")
ax4.plot(t_gyr, Om_eff, label="Ω_m,eff")
ax4.plot(t_gyr, Os_eff, label="Ω_S,eff")
ax4.set_title("Manifold fractions (who dominates)")
ax4.set_xlabel("Time (Gyr)")
ax4.set_ylabel("Fraction of E^2")
ax4.legend(loc="best")

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
plt.savefig("coprimary_closed_dual_manifold_step1_step2.png", dpi=300)
plt.show()


# ============================================================
# SWEEP FRAMEWORK (epsilon_tilt x kappa) for boundary behavior
# Paste BELOW the model code (functions + rhs_tau) and ABOVE any plt.show()
# ============================================================

import itertools
import csv

def integrate_model(
    epsilon_tilt_val: float,
    kappa_val: float,
    gamma_val: float = None,
    Delta_val: float = None,
    t_end_gyr: float = 13.8,
    n_eval: int = 6000,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """
    Runs ONE simulation with specified params and returns dict of diagnostics.
    Minimal changes: uses your existing global functions but temporarily
    overwrites globals epsilon_tilt/kappa/gamma/Delta.
    """

    # --- capture globals we will override ---
    global epsilon_tilt, kappa, gamma, Delta, tau_today, t_today

    old_epsilon = epsilon_tilt
    old_kappa   = kappa
    old_gamma   = gamma
    old_Delta   = Delta
    old_t_today = t_today
    old_tau_today = tau_today

    try:
        # --- apply overrides ---
        epsilon_tilt = float(epsilon_tilt_val)
        kappa = float(kappa_val)
        if gamma_val is not None:
            gamma = float(gamma_val)
        if Delta_val is not None:
            Delta = float(Delta_val)

        # --- time horizon override (keep H0 constant) ---
        t_today = float(t_end_gyr) * Gyr
        tau_today = H0 * t_today

        # --- integrate ---
        a_init = 1e-30
        G0 = np.log(a_init)
        S0 = 0.0

        # eval grid (geom helps early time resolution)
        tau_eval = np.geomspace(1e-18, tau_today, n_eval)
        tau_eval = np.unique(np.concatenate([[0.0], tau_eval]))

        def event_cross_Sc(tau, y):
            # event when S - S_c crosses 0
            return y[1] - S_c
        event_cross_Sc.terminal = False
        event_cross_Sc.direction = +1

        # sol = solve_ivp(
        #     rhs_tau,
        #     (0.0, tau_today),
        #     [G0, S0],
        #     t_eval=tau_eval,
        #     method=method,
        #     rtol=rtol,
        #     atol=atol,
        # )

        sol = solve_ivp(
            rhs_tau,
            (0.0, tau_today),
            [G0, S0],
            t_eval=tau_eval,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
            events=[event_cross_Sc],
        )

        if len(sol.t_events[0]) > 0:
            print("First crossing tau:", sol.t_events[0][0], "t(s):", sol.t_events[0][0]/H0)

        # --- basic sanity flags ---
        success = bool(sol.success)
        tau = sol.t
        G = sol.y[0]
        S = sol.y[1]

        finite_ok = (
            np.all(np.isfinite(tau)) and
            np.all(np.isfinite(G)) and
            np.all(np.isfinite(S))
        )

        # If solver failed or produced non-finite values, stop early with diagnostics.
        if (not success) or (not finite_ok) or (len(tau) < 5):
            return {
                "success": int(success),
                "finite_ok": int(finite_ok),
                "epsilon_tilt": epsilon_tilt,
                "kappa": kappa,
                "gamma": gamma,
                "Delta": Delta,
                "t_end_gyr": t_end_gyr,
                "crossed_Sc": 0,
                "t_cross_s": np.nan,
                "t_cross_gyr": np.nan,
                "N_proxy": np.nan,
                "S_end": float(S[-1]) if len(S) else np.nan,
                "G_end": float(G[-1]) if len(G) else np.nan,
                "max_abs_constraint": np.nan,
                "min_rhs_E2": np.nan,
                "note": "solver_failed_or_nan",
            }

        # --- compute E and constraint residual ---
        E = np.array([E_of(G[i], S[i]) for i in range(len(tau))])

        fS_vec = f_internal(S)
        term_r = Omega_r0 * fS_vec * safe_exp(-4.0 * G)
        term_m = Omega_m0 * fS_vec * safe_exp(-3.0 * G)
        term_s = Omega_S(S)
        rhs_E2 = term_r + term_m + term_s

        C_resid = E**2 - rhs_E2
        max_abs_constraint = float(np.nanmax(np.abs(C_resid)))
        min_rhs_E2 = float(np.nanmin(rhs_E2))

        # --- crossing metrics ---
        crossed = np.any(S >= S_c)
        if crossed:
            idx = int(np.argmax(S >= S_c))  # first True
            tau_cross = float(tau[idx])
            t_cross_s = tau_cross / H0
            t_cross_gyr = t_cross_s / Gyr
            N_proxy = float(G[idx] - G[0])  # ΔG until crossing
        else:
            t_cross_s = np.nan
            t_cross_gyr = np.nan
            N_proxy = np.nan

        # --- return compact record ---
        return {
            "success": int(success),
            "finite_ok": int(finite_ok),
            "epsilon_tilt": float(epsilon_tilt),
            "kappa": float(kappa),
            "gamma": float(gamma),
            "Delta": float(Delta),
            "t_end_gyr": float(t_end_gyr),
            "crossed_Sc": int(bool(crossed)),
            "t_cross_s": float(t_cross_s) if np.isfinite(t_cross_s) else np.nan,
            "t_cross_gyr": float(t_cross_gyr) if np.isfinite(t_cross_gyr) else np.nan,
            "N_proxy": float(N_proxy) if np.isfinite(N_proxy) else np.nan,
            "S_end": float(S[-1]),
            "G_end": float(G[-1]),
            "max_abs_constraint": max_abs_constraint,
            "min_rhs_E2": min_rhs_E2,
            "note": "ok" if crossed else "no_cross",
        }

    except Exception as e:
        return {
            "success": 0,
            "finite_ok": 0,
            "epsilon_tilt": float(epsilon_tilt_val),
            "kappa": float(kappa_val),
            "gamma": float(gamma_val) if gamma_val is not None else float(gamma),
            "Delta": float(Delta_val) if Delta_val is not None else float(Delta),
            "t_end_gyr": float(t_end_gyr),
            "crossed_Sc": 0,
            "t_cross_s": np.nan,
            "t_cross_gyr": np.nan,
            "N_proxy": np.nan,
            "S_end": np.nan,
            "G_end": np.nan,
            "max_abs_constraint": np.nan,
            "min_rhs_E2": np.nan,
            "note": f"exception:{type(e).__name__}",
        }

    finally:
        # restore globals
        epsilon_tilt = old_epsilon
        kappa = old_kappa
        gamma = old_gamma
        Delta = old_Delta
        t_today = old_t_today
        tau_today = old_tau_today


def run_sweep(
    eps_list,
    kappa_list,
    gamma_val=None,
    Delta_val=None,
    t_end_gyr=13.8,
    n_eval=4000,
    out_csv="sweep_results.csv",
):
    """
    Sweeps epsilon_tilt x kappa and writes a CSV.
    Returns list of dict rows.
    """
    rows = []
    total = len(eps_list) * len(kappa_list)
    done = 0

    for eps, kap in itertools.product(eps_list, kappa_list):
        done += 1
        row = integrate_model(
            epsilon_tilt_val=eps,
            kappa_val=kap,
            gamma_val=gamma_val,
            Delta_val=Delta_val,
            t_end_gyr=t_end_gyr,
            n_eval=n_eval,
        )
        rows.append(row)
        print(f"[{done:4d}/{total}] eps={eps:g}, kappa={kap:g} -> crossed={row['crossed_Sc']} note={row['note']}")

    # write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote: {out_csv}")
    return rows


def plot_heatmaps(rows, eps_list, kappa_list, out_png="sweep_heatmaps.png"):
    """
    Produces two heatmaps:
      - crossed_Sc (0/1)
      - N_proxy (ΔG at crossing) for successful crossings
    """
    # Build grids
    eps_list = list(eps_list)
    kappa_list = list(kappa_list)

    crossed_grid = np.full((len(eps_list), len(kappa_list)), np.nan)
    N_grid = np.full((len(eps_list), len(kappa_list)), np.nan)

    # index maps
    eps_idx = {v: i for i, v in enumerate(eps_list)}
    kap_idx = {v: j for j, v in enumerate(kappa_list)}

    for r in rows:
        i = eps_idx.get(r["epsilon_tilt"])
        j = kap_idx.get(r["kappa"])
        if i is None or j is None:
            continue
        crossed_grid[i, j] = r["crossed_Sc"]
        if r["crossed_Sc"] == 1 and np.isfinite(r["N_proxy"]):
            N_grid[i, j] = r["N_proxy"]

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(
        crossed_grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    ax1.set_title("Boundary crossing: crossed_Sc")
    ax1.set_xlabel("kappa index")
    ax1.set_ylabel("epsilon_tilt index")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(
        N_grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    ax2.set_title("N_proxy = ΔG at crossing (only where crossed)")
    ax2.set_xlabel("kappa index")
    ax2.set_ylabel("epsilon_tilt index")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # annotate axis ticks with actual values (sparse)
    def sparse_ticks(vals, max_ticks=10):
        if len(vals) <= max_ticks:
            return list(range(len(vals))), [f"{v:g}" for v in vals]
        idx = np.linspace(0, len(vals) - 1, max_ticks).astype(int)
        return idx.tolist(), [f"{vals[i]:g}" for i in idx]

    xt, xl = sparse_ticks(kappa_list, 9)
    yt, yl = sparse_ticks(eps_list, 9)

    for ax in [ax1, ax2]:
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=45, ha="right")
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()
    print(f"Wrote: {out_png}")


# ============================================================
# SWEEP ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    # --- Choose sweep ranges ---
    # epsilon_tilt is sensitive; include very small and modest values
    eps_list = [0.0, 1e-6, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]

    # kappa spans stiffness; use log-ish spacing
    kappa_list = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    # Fix gamma and Delta for now (you can sweep later)
    gamma_val = 0.15
    Delta_val = 0.9

    rows = run_sweep(
        eps_list=eps_list,
        kappa_list=kappa_list,
        gamma_val=gamma_val,
        Delta_val=Delta_val,
        t_end_gyr=13.8,
        n_eval=4000,
        out_csv="sweep_results.csv",
    )

    plot_heatmaps(
        rows,
        eps_list=eps_list,
        kappa_list=kappa_list,
        out_png="sweep_heatmaps.png",
    )