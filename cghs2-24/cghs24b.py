import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp

# ============================================================
# Co-Primary Geometry–Hilbert Cosmology
# ============================================================
#
# MATHEMATICAL REFERENCE SHEET
#
# ONTOLOGY
#   G = ln(a)             : logarithmic geometry variable
#   a = exp(G)            : scale factor
#   S                     : Hilbert order parameter
#
# CLOCK
#   tau                   : emergent cosmic time
#   lambda                : integration parameter (compatibility flow)
#
# DYNAMICS
#   u(S) = 1 / (1 + exp((S - Sc)/Delta))
#   f(S) = 1 - u(S)
#
#   Omega_S(S) = Omega_L_floor
#                + Omega_inf * u(S)
#                + epsilon_tilt * tanh((S - Sc)/Delta)
#
#   E(G,S)^2 = Omega_r0 * f(S) * exp(-4G)
#            + Omega_m0 * f(S) * exp(-3G)
#            + Omega_S(S)
#
#   alpha(G,S) = alpha_floor
#              + 0.2 * f(S)
#              + 0.8 * f(S) / E(G,S)
#
#   dG/dlambda   = E(G,S)
#
#   dS/dlambda   = mu_S * u(S)
#                - (kappa / (3E)) * dOmega_S/dS
#                - gamma_S * S * (1 - u(S))
#
#   dtau/dlambda = alpha(G,S)
#
# OBSERVABLES
#   t = tau / H0
#   a = exp(G)
#   V ~ a^3
#   (aH)^(-1) proxy ~ 1 / (a * E)
#
# OUTPUTS
#   1) 3D cosmological tube
#   2) scale factor a(t)
#   3) volume metric V(t)
#   4) comoving Hubble radius proxy
#   5) Hilbert order parameter S(t) + normalized alpha(t)
#   6) cosmology diagram panel
#   7) phase-space portrait (G,S)
# ============================================================


# ============================================================
# Units / normalization
# ============================================================
Gyr = 365.25 * 24 * 3600 * 1e9
H0 = 2.2683e-18  # ~70 km/s/Mpc in 1/s

# ============================================================
# Model parameters
# ============================================================
Omega_r0 = 9e-5
Omega_m0 = 0.30
Omega_L_floor = 0.70
#Omega_inf = 1.0e4
Omega_inf = 50

S_c = 8.0
Delta = 0.9
#epsilon_tilt = 0.02
epsilon_tilt = 0.002

kappa = 60.0
gamma_S = 0.08
#mu_S = 0.02
#mu_S = .002
mu_S = .001
alpha_floor = 1e-8

a_init = 1e-30
G0 = np.log(a_init)
S0 = 0.0
tau0 = 0.0

lambda_max = 80.0
n_eval = 8000

EXP_CLIP = 700.0


# ============================================================
# Numeric helpers
# ============================================================
def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


def normalize01(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmax - xmin < eps):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


# ============================================================
# ONTOLOGY / DYNAMICS: Hilbert activation structure
# ============================================================
def logistic_u(S):
    """
    u(S) = 1 / (1 + exp((S - S_c)/Delta))

    Plateau control:
      S << S_c  -> u ~ 1
      S >> S_c  -> u ~ 0
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
    out = np.empty_like(x)
    out[x > 50.0] = 0.0
    out[x < -50.0] = 1.0
    mid = (x >= -50.0) & (x <= 50.0)
    out[mid] = 1.0 / (1.0 + np.exp(x[mid]))
    return out


def f_internal(S):
    """
    f(S) = 1 - u(S)

    Released fraction of the Hilbert sector.
    """
    return 1.0 - logistic_u(S)


def Omega_S(S):
    """
    Omega_S(S) = Omega_L_floor
               + Omega_inf * u(S)
               + epsilon_tilt * tanh((S - S_c)/Delta)

    Hilbert vacuum sector:
      - late-time vacuum floor
      - inflation plateau
      - bounded smooth tilt
    """
    u = logistic_u(S)
    tilt = epsilon_tilt * np.tanh((S - S_c) / Delta)
    return Omega_L_floor + Omega_inf * u + tilt


def dOmegaS_dS(S):
    """
    d/dS Omega_S(S)
    """
    u = logistic_u(S)
    du = -(1.0 / Delta) * u * (1.0 - u)

    x = (S - S_c) / Delta
    dtilt = (epsilon_tilt / Delta) * (1.0 / np.cosh(np.clip(x, -50, 50)) ** 2)

    return Omega_inf * du + dtilt


# ============================================================
# DYNAMICS: Constraint manifold
# ============================================================
def E_of(G, S):
    """
    E(G,S)^2 = Omega_r0 * f(S) * exp(-4G)
             + Omega_m0 * f(S) * exp(-3G)
             + Omega_S(S)
    """
    fS = float(f_internal(S))
    term_r = Omega_r0 * fS * safe_exp(-4.0 * G)
    term_m = Omega_m0 * fS * safe_exp(-3.0 * G)
    term_s = Omega_S(S)

    rhs = term_r + term_m + term_s
    if (not np.isfinite(rhs)) or (rhs < 0.0):
        rhs = max(term_s, 1e-12)

    return np.sqrt(rhs)


# ============================================================
# CLOCK: emergent time flow
# ============================================================
def alpha_of(G, S):
    """
    alpha(G,S) = alpha_floor + 0.2 f(S) + 0.8 f(S)/E(G,S)

    Interpretation:
      - inflation / locked sector => alpha ~ alpha_floor
      - released sector           => stronger clock flow
    """
    E = E_of(G, S)
    fS = float(f_internal(S))
    alpha = alpha_floor + 0.2 * fS + 0.8 * fS / (E + 1e-12)
    return float(max(alpha, alpha_floor))


# ============================================================
# DYNAMICS: ODE system in compatibility parameter lambda
# ============================================================
def rhs_lambda(lam, y):
    """
    dG/dlambda   = E(G,S)

    dS/dlambda   = mu_S * u(S)
                 - (kappa / (3E)) * dOmega_S/dS
                 - gamma_S * S * (1 - u(S))

    dtau/dlambda = alpha(G,S)
    """
    G, S, tau = y
    S = float(np.clip(S, -50.0, 50.0))

    E = E_of(G, S)
    E_safe = max(E, 1e-60)
    alpha = alpha_of(G, S)

    # Geometry
    dG = E

    # Hilbert order parameter
    u = logistic_u(S)
    #dS_source = mu_S * u
    dS_source = mu_S * logistic_u(S) * (1 - np.exp(-S))
    dS_potential = -(kappa / (3.0 * E_safe)) * dOmegaS_dS(S)
    dS_damp = -gamma_S * S * (1.0 - u)
    dS = dS_source + dS_potential + dS_damp

    # Emergent clock
    dtau = alpha

    return [dG, dS, dtau]


# ============================================================
# Regime classifier
# ============================================================
def classify_regimes(a, S, fS, hubble_proxy):
    """
    Simple qualitative classifier by release fraction.
    """
    regimes = np.zeros(len(a), dtype=int)

    for i in range(len(a)):
        if fS[i] < 0.05:
            regimes[i] = 0
        elif fS[i] < 0.5:
            regimes[i] = 1
        elif fS[i] < 0.9:
            regimes[i] = 2
        else:
            regimes[i] = 3

    return regimes


regime_names = {
    0: "Inflation",
    1: "Transition / Reheating",
    2: "Matter-dominated expansion",
    3: "Dark-energy dominated",
}


# ============================================================
# Phase-space vector field (consistent with integrated ODE)
# ============================================================
def compute_phase_field(G_range, S_range, resolution=60):
    G_vals = np.linspace(*G_range, resolution)
    S_vals = np.linspace(*S_range, resolution)

    dG = np.zeros((resolution, resolution))
    dS = np.zeros((resolution, resolution))

    for i, Gv in enumerate(G_vals):
        for j, Sv in enumerate(S_vals):
            E = E_of(Gv, Sv)
            E_safe = max(E, 1e-60)

            u = logistic_u(Sv)
            dG_val = E
            dS_source = mu_S * u
            dS_potential = -(kappa / (3.0 * E_safe)) * dOmegaS_dS(Sv)
            dS_damp = -gamma_S * Sv * (1.0 - u)
            dS_val = dS_source + dS_potential + dS_damp

            dG[i, j] = dG_val
            dS[i, j] = dS_val

    return G_vals, S_vals, dG, dS


def plot_phase_portrait(ax, G_vals, S_vals, dG, dS, G_traj, S_traj):
    speed = np.sqrt(dG**2 + dS**2)
    speed = speed / (np.max(speed) + 1e-12)

    dG_plot = dG / (np.max(np.abs(dG)) + 1e-12)
    dS_plot = dS / (np.max(np.abs(dS)) + 1e-12)

    ax.streamplot(
        G_vals,
        S_vals,
        dG_plot.T,
        dS_plot.T,
        density=1.2,
        color=speed.T,
        cmap="viridis",
        linewidth=1,
    )

    ax.plot(G_traj, S_traj, color="red", linewidth=2, label="Universe trajectory")
    ax.set_title("Phase-space portrait (G,S)")
    ax.set_xlabel("G = ln(a)")
    ax.set_ylabel("Hilbert order parameter S")
    ax.legend()


# ============================================================
# Diagram helpers
# ============================================================
def build_vertices(tvals, fvals, Svals, S_cross, threshold=0.18):
    """
    Interaction vertices appear when release activates
    and S is near the transition region.
    """
    near_cross = np.abs(Svals - S_cross) < 2.0
    active = (fvals > threshold) & near_cross

    idx = np.where(active)[0]
    if idx.size > 24:
        idx = idx[np.linspace(0, idx.size - 1, 24).astype(int)]
    return idx



def plot_cosmology_diagram(ax, tvals, a_vals, Svals, fvals, hubble_proxy, S_cross, regimes):
    """
    Feynman-style cosmology glyph panel:
      - tube walls = geometry propagation
      - Hilbert spine = internal quantum structure
      - stars = interaction/coupling vertices
      - X glyphs = causal/horizon sketch
    """

    # Tube walls
    ax.fill_between(tvals, -a_vals, a_vals, alpha=0.08)
    ax.plot(tvals, +a_vals, linewidth=2.0)
    ax.plot(tvals, -a_vals, linewidth=2.0)

    # Hilbert spine
    S_strength = normalize01(np.abs(Svals))
    f_strength = normalize01(fvals)

    for i in range(len(tvals) - 1):
        lw = 0.6 + 5.5 * S_strength[i]
        al = 0.10 + 0.85 * f_strength[i]
        ax.plot([tvals[i], tvals[i + 1]], [0.0, 0.0], linewidth=lw, alpha=al)

    # Interaction vertices
    v_idx = build_vertices(tvals, fvals, Svals, S_cross, threshold=0.18)
    if len(v_idx) > 0:
        ax.scatter(tvals[v_idx], np.zeros_like(v_idx), marker="*", s=130, zorder=5)

    # Causal / horizon glyphs
    ch_n = normalize01(np.log10(np.maximum(hubble_proxy, 1e-300)))
    slope = 0.15 + 1.25 * (1.0 - ch_n)

    anchors = np.linspace(0, len(tvals) - 1, 7).astype(int)
    dt = max((tvals[-1] - tvals[0]) * 0.045, 1e-6)

    for j in anchors:
        tj = tvals[j]
        m = slope[j]
        ax.plot([tj - dt, tj + dt], [-m * dt, +m * dt], alpha=0.35, linewidth=1.0)
        ax.plot([tj - dt, tj + dt], [+m * dt, -m * dt], alpha=0.35, linewidth=1.0)

    ax.set_title("Cosmology diagram: geometry walls, Hilbert spine, vertices, causal glyphs")
    ax.set_xlabel("Emergent cosmic time (Gyr)")
    ax.set_ylabel("Tube radius ±a(t), spine at 0")

    ymax = max(np.max(a_vals), 1.0)
    ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax.grid(True, alpha=0.25)

    # -------------------------------------------------
    # Improved regime labels (no stacking)
    # -------------------------------------------------

    label_style = dict(
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
    )

    label_levels = [0.85, 0.70, 0.55, 0.40, 0.25]  # vertical stagger
    level_i = 0

    current = regimes[0]
    start = 0

    for i in range(1, len(regimes)):
        if regimes[i] != current:
            mid = (tvals[start] + tvals[i]) / 2

            ax.text(
                mid,
                label_levels[level_i % len(label_levels)] * ymax,
                regime_names[current],
                **label_style,
            )

            level_i += 1
            start = i
            current = regimes[i]

    mid = (tvals[start] + tvals[-1]) / 2

    ax.text(
        mid,
        label_levels[level_i % len(label_levels)] * ymax,
        regime_names[current],
        **label_style,
    )

# ============================================================
# Integrate
# ============================================================
lam_eval = np.linspace(0.0, lambda_max, n_eval)

sol = solve_ivp(
    rhs_lambda,
    (0.0, lambda_max),
    [G0, S0, tau0],
    t_eval=lam_eval,
    method="Radau",
    rtol=1e-8,
    atol=1e-10,
)

if not sol.success:
    raise RuntimeError(f"Integration failed: {sol.message}")

lam = sol.t
G = sol.y[0]
S = sol.y[1]
tau = sol.y[2]


# ============================================================
# OBSERVABLES
# ============================================================
E = np.array([E_of(G[i], S[i]) for i in range(len(G))])
alpha = np.array([alpha_of(G[i], S[i]) for i in range(len(G))])
fS_vec = f_internal(S)

# Emergent cosmic time
t = tau / H0
t_gyr = t / Gyr

# Scale factor normalized to 1 at final time
G_shift = G - G[-1]
a_plot = np.exp(np.clip(G_shift, -EXP_CLIP, EXP_CLIP))

# Volume metric
V_plot = a_plot**3

#"Emergent-time comoving Hubble radius (a H_eff)^(-1)"
#comoving_hubble = 1.0 / np.maximum(a_plot * E, 1e-300)
H_eff = H0 * E / np.maximum(alpha, 1e-300)
comoving_hubble = 1.0 / np.maximum(a_plot * H_eff, 1e-300)

# Regimes
regimes = classify_regimes(a_plot, S, fS_vec, comoving_hubble)

# Effective manifold fractions
term_r = Omega_r0 * fS_vec * safe_exp(-4.0 * G)
term_m = Omega_m0 * fS_vec * safe_exp(-3.0 * G)
term_s = Omega_S(S)

denom = np.maximum(E**2, 1e-300)
Or_eff = term_r / denom
Om_eff = term_m / denom
Os_eff = term_s / denom


# ============================================================
# Phase-space field
# ============================================================
G_center = np.mean(G)
G_range = (G_center - 5, G_center + 5)
S_range = (-1, S_c + 3)

G_vals, S_vals, dG_field, dS_field = compute_phase_field(
    G_range,
    S_range,
    resolution=60,
)


# ============================================================
# 3D tube geometry
# ============================================================
theta = np.linspace(0, 2 * np.pi, 80)
T, TH = np.meshgrid(t_gyr, theta)
R_mesh = np.tile(a_plot, (len(theta), 1))

X = T
Y = R_mesh * np.cos(TH)
Z = R_mesh * np.sin(TH)


# ============================================================
# Figure layout: 7 panels
# ============================================================
fig = plt.figure(figsize=(18, 13))
gs = gridspec.GridSpec(4, 2, figure=fig)

# ------------------------------------------------------------
# 1) 3D tube
# ------------------------------------------------------------
ax0 = fig.add_subplot(gs[0, 0], projection="3d")
ax0.plot_surface(X, Y, Z, alpha=0.85, linewidth=0)
ax0.set_title("Cosmological tube (radius ∝ a)")
ax0.set_xlabel("Emergent time (Gyr)")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")

# ------------------------------------------------------------
# 2) Scale factor
# ------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])
ax1.semilogy(t_gyr, a_plot)
ax1.set_title("Scale factor a(t) normalized to 1 today")
ax1.set_xlabel("Emergent time (Gyr)")
ax1.set_ylabel("a(t)")

# ------------------------------------------------------------
# 3) Volume
# ------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])
ax2.semilogy(t_gyr, V_plot)
ax2.set_title("Volume metric V(t) ∝ a(t)^3")
ax2.set_xlabel("Emergent time (Gyr)")
ax2.set_ylabel("V(t)")

# ------------------------------------------------------------
# 4) Comoving Hubble radius proxy
# ------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(np.maximum(t + 1e-40, 1e-40), comoving_hubble)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Comoving Hubble radius proxy (aH)^(-1)")
ax3.set_xlabel("Emergent time (seconds)")
ax3.set_ylabel("(aH)^(-1)")

# ------------------------------------------------------------
# 5) Hilbert order parameter and clock
# ------------------------------------------------------------
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t_gyr, S, label="S(t)")
ax4.plot(t_gyr, alpha / np.max(alpha), linestyle="--", label="normalized clock rate α(t)")
ax4.set_title("Hilbert order parameter and emergent clock")
ax4.set_xlabel("Emergent time (Gyr)")
ax4.set_ylabel("S(t), α(t)/max")
ax4.legend(loc="best")

# ------------------------------------------------------------
# 6) Cosmology diagram
# ------------------------------------------------------------
ax5 = fig.add_subplot(gs[2, 1])
plot_cosmology_diagram(
    ax5,
    t_gyr,
    a_plot,
    S,
    fS_vec,
    comoving_hubble,
    S_c,
    regimes,
)

# ------------------------------------------------------------
# 7) Phase portrait
# ------------------------------------------------------------
ax6 = fig.add_subplot(gs[3, :])
plot_phase_portrait(ax6, G_vals, S_vals, dG_field, dS_field, G, S)

plt.tight_layout()
plt.savefig("coprimary_geometry_hilbert_emergent_time.png", dpi=300)
plt.show()