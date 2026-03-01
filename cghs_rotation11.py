import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import matplotlib.cm as cm

# ============================================================
# PASS SWITCHES
# ============================================================
USE_MULTIFLUID_FRW = True          # multi-fluid FRW backbone (radiation + matter + dark)
USE_CURVATURE = True              # include Ω_k term (a^{-2})
USE_RK4 = True                    # RK4 integrator (recommended for small a)
STOP_AT_A_EQ_1 = True             # stop integration when a reaches 1 ("today")

USE_TWIST_DYNAMICS = True
USE_DYNAMIC_ORDER = True
USE_FEEDBACK_MEMORY = True
USE_EVENTS = False

# ============================================================
# Helpers
# ============================================================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def save_plot(x, ys, labels, title, ylabel, filename):
    plt.figure()
    if not isinstance(ys, (list, tuple)):
        ys = [ys]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel("t (dimensionless)")
    plt.ylabel(ylabel)
    if any(lab for lab in labels):
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# Cosmology + dark-sector controller (θ–ω–O–fb)
# ============================================================
def run_model(
    use_efold=True,
    dN=1e-4,
    # --- time control ---
    T=200000,
    dt=2e-5,  # dimensionless dt in units of 1/H0 (smaller is safer for tiny a)

    # --- FRW normalization + present-day fractions ---
    H0=1.0,              # by definition in these units
    Omega_r0=9.0e-5,     # (photons + neutrinos approx) tune as desired
    Omega_m0=0.30,       # matter (baryons+CDM)
    Omega_k0=0.00,       # curvature (positive=open, negative=closed)
    Omega_sigma0=0.0,   # Bianchi I shear term today (σ^2 ∝ a^{-6})
    Omega_X0=None,       # dark sector (if None -> closure)

    # --- initial conditions ---
    a0=1e-8,
    #a0=1e-2,
    theta0=0.01,
    omega0=0.0,
    order0=0.50,
    fb0=0.0,

    # ---- dark-sector equation of state mapping ----
    # Interpret w_eff as w_X(t) ONLY (radiation/matter are fixed).
    w0=-1.0,
    alpha_O=-0.8,
    O_star=0.65,
    alpha_fb=-0.8,
    alpha_omega=0.15,
    omega_star=0.15,

    # ---- late-time attractor forcing: w_X -> -1 ----
    FORCE_W_TO_MINUS_ONE=True,
    a_late=0.30,         # when to start strongly blending toward -1 (in units where a=1 today)
    a_width=0.08,        # smoothness of the transition

    # ---- numerical safety ----
    H_min=1e-12,
    H_max=1e8,

    # ---- twist dynamics ----
    gamma_drag=0.8,
    beta_theta=0.002,
    k0=0.003,
    k1_twist=0.010,

    # ---- order dynamics ----
    lam_prod=0.12,
    eta_omega=0.7,
    S_star=0.25,
    mu_relax=0.10,
    O_min=0.50,

    # Make order production scale with the *dark importance* to protect early universe
    ORDER_SCALES_WITH_DARK_FRACTION=True,

    # ---- feedback memory ----
    tau_fb=18.0,

    # ---- optional events ----
    event_times=(25, 80),
    event_domega=(+0.25, -0.25),
    event_dfb=(0.0, 0.0),
):
    """
    Multi-fluid flat/open/closed FRW with RK4 integration.

    In dimensionless units (t -> H0 t_phys), set H0=1 and interpret dt accordingly.
    Closure (with curvature):
      H^2 = H0^2 [ Ω_r0 a^{-4} + Ω_m0 a^{-3} + Ω_k0 a^{-2} + Ω_X(a) ]
    Dark continuity:
      dΩ_X/dt = -3 H (1+w_X) Ω_X
    a evolution:
      da/dt = a H

    θ–ω–O–fb is treated as internal dark-sector dynamics that only influences w_X(t),
    not radiation or matter.
    """
    if Omega_X0 is None:
        Omega_X0 = 1.0 - Omega_r0 - Omega_m0 - (Omega_k0 if USE_CURVATURE else 0.0)
        Omega_X0 = max(Omega_X0, 1e-12)

    # event lookup
    event_map = {}
    if USE_EVENTS and len(event_times) > 0:
        for tt, d_om, d_fb in zip(event_times, event_domega, event_dfb):
            event_map[float(tt)] = (float(d_om), float(d_fb))

    def H_from(a, rhoX_bar):
        # Guard tiny a
        a = max(a, 1e-40)

        rho_r_bar = Omega_r0 / (a**4)
        rho_m_bar = Omega_m0 / (a**3)
        rho_sigma_bar = Omega_sigma0 / (a**6)

        term = rho_r_bar + rho_m_bar + rhoX_bar + rho_sigma_bar

        if USE_CURVATURE:
            rho_k_bar = Omega_k0 / (a**2)
            term += rho_k_bar

        H = H0 * np.sqrt(max(term, 0.0))
        return clamp(H, H_min, H_max)
    
    def OmegaX_fraction(a, rhoX_bar):
        a = max(a, 1e-40)

        rho_r_bar = Omega_r0 / (a**4)
        rho_m_bar = Omega_m0 / (a**3)
        rho_sigma_bar = Omega_sigma0 / (a**6)

        total = rho_r_bar + rho_m_bar + rhoX_bar + rho_sigma_bar

        if USE_CURVATURE:
            total += Omega_k0 / (a**2)

        return rhoX_bar / total

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def wX_from_state(O, fb, om, a):
        # base mapping (your current framework)
        w_raw = (
            w0
            + alpha_O * (O - O_star)
            + alpha_fb * fb
            + alpha_omega * np.tanh(om / max(omega_star, 1e-12))
        )
        w_raw = clamp(w_raw, -1.2, +1.0)

        # enforce late-time attractor toward -1 (phenomenological but useful)
        if FORCE_W_TO_MINUS_ONE:
            # blend weight rises as a approaches 1
            wgt = sigmoid((a - a_late) / max(a_width, 1e-6))
            w = (1.0 - wgt) * w_raw + wgt * (-1.0)
        else:
            w = w_raw

        return clamp(w, -1.2, +1.0)

    
    # ------------------------------------------------------------
    # Time variable choice:
    #   - use_efold=True integrates in N = ln a (adaptive cosmic time via dt = dN/H)
    #   - use_efold=False integrates directly in dimensionless cosmic time t
    # ------------------------------------------------------------
    def rhs_N(N, y):
        """
        y = [a, OmegaX, theta, omega, O, fb]
        Returns dy/dN for integration in N = ln a.
        """

        a, OmegaX, theta, omega, O, fb = y

        # H from FRW closure
        H = H_from(a, OmegaX)
        H_safe = max(H, 1e-30)

        # --------------------------------------------------
        # Feedback memory
        # Original: dfb/dt = ((O - O_star) - fb)/tau_fb
        # Convert: dfb/dN = (1/H) dfb/dt
        # --------------------------------------------------
        if USE_FEEDBACK_MEMORY:
            dfb = (((O - O_star) - fb) / max(tau_fb, 1e-12)) / H_safe
        else:
            dfb = 0.0

        # --------------------------------------------------
        # Dark equation of state
        # --------------------------------------------------
        wX = wX_from_state(O, fb, omega, a)

        # --------------------------------------------------
        # Dark continuity
        # Original: dOmegaX/dt = -3 H (1+w) OmegaX
        # Convert: divide by H → no H remains
        # --------------------------------------------------
        dOmegaX = -3.0 * (1.0 + wX) * OmegaX

        # --------------------------------------------------
        # Scale factor
        # da/dt = a H
        # da/dN = a
        # --------------------------------------------------
        da = a

        # --------------------------------------------------
        # Twist sector
        # --------------------------------------------------
        if USE_TWIST_DYNAMICS:
            kO = k0 + k1_twist * (O - O_star)
            kO = max(kO, 0.0)

            # dtheta/dN = (dtheta/dt)/H = omega/H
            dtheta = omega / H_safe

            # domega/dN = (-gamma_drag H omega - (kO + beta_theta) theta)/H
            domega = -gamma_drag * omega - ((kO + beta_theta) * theta) / H_safe
        else:
            dtheta = 0.0
            domega = 0.0

        # --------------------------------------------------
        # Order dynamics
        # --------------------------------------------------
        if USE_DYNAMIC_ORDER:
            activity = (theta**2 + eta_omega * (omega**2)) / max(S_star, 1e-12)
            sigma = np.tanh(activity)

            prod = lam_prod * sigma

            if ORDER_SCALES_WITH_DARK_FRACTION:
                prod *= clamp(OmegaX, 0.0, 1.0)

            # Original: dO/dt = prod - mu_relax * H * (O - O_min)
            # Convert: divide by H
            dO = (prod / H_safe) - mu_relax * (O - O_min)
        else:
            dO = 0.0

        return np.array([da, dOmegaX, dtheta, domega, dO, dfb], dtype=float), H, wX

    # Allocate (oversized) arrays, then truncate
    t = np.zeros(T, dtype=float)
    a_arr = np.zeros(T, dtype=float)
    H_arr = np.zeros(T, dtype=float)
    theta_arr = np.zeros(T, dtype=float)
    omega_arr = np.zeros(T, dtype=float)
    O_arr = np.zeros(T, dtype=float)
    fb_arr = np.zeros(T, dtype=float)
    wX_arr = np.zeros(T, dtype=float)
    OmegaX_arr = np.zeros(T, dtype=float)

# Independent variable: N = ln a
    N = np.zeros(T, dtype=float)
    N_now = np.log(a0)

    # init state (keep your variable names)
    y = np.array([a0, Omega_X0, theta0, omega0, clamp(order0, 0.0, 1.0), fb0], dtype=float)

    # record initial
    deriv0, H0_now, w0_now = rhs_N(N_now, y)
    a_arr[0], OmegaX_arr[0], theta_arr[0], omega_arr[0], O_arr[0], fb_arr[0] = y
    H_arr[0] = H0_now
    wX_arr[0] = w0_now
    N[0] = N_now

    n = 1

    for i in range(1, T):

        if USE_RK4:
            k1v, _, _ = rhs_N(N_now, y)
            k2v, _, _ = rhs_N(N_now + 0.5*dN, y + 0.5*dN*k1v)
            k3v, _, _ = rhs_N(N_now + 0.5*dN, y + 0.5*dN*k2v)
            k4v, _, _ = rhs_N(N_now + dN, y + dN*k3v)
            y_new = y + (dN/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        else:
            dy, _, _ = rhs_N(N_now, y)
            y_new = y + dN*dy

        # positivity / bounds
        y_new[0] = max(y_new[0], 1e-40)          # a
        y_new[1] = max(y_new[1], 0.0)            # OmegaX
        y_new[4] = clamp(y_new[4], 0.0, 1.0)     # O

        # optional discrete events (if you truly want them in N-space, keep this;
        # otherwise turn USE_EVENTS off for now)
        if USE_EVENTS:
            if float(N_now) in event_map:
                d_om, d_fb = event_map[float(N_now)]
                y_new[3] += d_om
                y_new[5] += d_fb

        # commit
        y = y_new
        N_now += dN

        deriv, H_now, w_now = rhs_N(N_now, y)

        N[i] = N_now
        a_arr[i], OmegaX_arr[i], theta_arr[i], omega_arr[i], O_arr[i], fb_arr[i] = y
        H_arr[i] = H_now
        wX_arr[i] = w_now
        n = i + 1

        if STOP_AT_A_EQ_1 and a_arr[i] >= 1.0:
            break

    # truncate
    N = N[:n]
    a_arr = a_arr[:n]
    H_arr = H_arr[:n]
    theta_arr = theta_arr[:n]
    omega_arr = omega_arr[:n]
    O_arr = O_arr[:n]
    fb_arr = fb_arr[:n]
    wX_arr = wX_arr[:n]
    OmegaX_arr = OmegaX_arr[:n]

    # --- Compute actual dark fraction Ω_X(a) ---
    OmegaX_frac = np.zeros_like(OmegaX_arr)

    for i in range(len(OmegaX_arr)):
        OmegaX_frac[i] = OmegaX_fraction(a_arr[i], OmegaX_arr[i])

    V = a_arr**3
    t = N  # if the rest of your code expects "t", reuse it as N for now
    return t, a_arr, H_arr, V, theta_arr, omega_arr, wX_arr, O_arr, fb_arr, OmegaX_arr, OmegaX_frac
# ============================================================
# 3D Metric Tube
# ============================================================
def plot_metric_tube(t, a, theta, w_eff, max_points=2500, phi_res=64):
    # Downsample aggressively to avoid gigantic surfaces (can freeze the renderer)
    t = np.asarray(t); a = np.asarray(a); theta = np.asarray(theta); w_eff = np.asarray(w_eff)
    if t.size > max_points:
        idx = np.linspace(0, t.size-1, max_points).astype(int)
        t = t[idx]; a = a[idx]; theta = theta[idx]; w_eff = w_eff[idx]

    a_norm = a / np.max(a)
    R_profile = np.power(a_norm, 0.05)

    phi = np.linspace(0, 2*np.pi, phi_res)
    T_grid, Phi = np.meshgrid(t, phi)

    R = np.outer(R_profile, np.ones_like(phi)).T
    twist = np.outer(theta, np.ones_like(phi)).T

    X = R * np.cos(Phi + twist)
    Y = R * np.sin(Phi + twist)
    Z = T_grid

    # Color normalize w
    wmin = np.percentile(w_eff, 5)
    wmax = np.percentile(w_eff, 95)
    w_clip = np.clip(w_eff, wmin, wmax)
    norm = (w_clip - wmin) / (wmax - wmin + 1e-12)
    colors = cm.plasma(norm)
    colors = np.tile(colors, (len(phi), 1, 1))

    # Normalize t to [0, 1] to keep alpha within [0, 1] even when t spans negatives.
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    denom = t_max - t_min
    if denom <= 0.0 or not np.isfinite(denom):
        t_norm = np.zeros_like(t, dtype=float)
    else:
        t_norm = (t - t_min) / denom
    alpha_profile = 0.35 + 0.45 * t_norm
    for i in range(len(phi)):
        colors[i, :, 3] = alpha_profile

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    ax.plot([0]*len(t), [0]*len(t), t, linewidth=1.0)
    ax.plot_wireframe(X, Y, Z, color='k', alpha=0.05)

    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_zlabel("Cosmic time (dimensionless)")
    ax.set_title("Metric Tube\nTwist = θ(t), Radius = a(t), Color = w_X(t)")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("metric_tube_3D.png", dpi=300)
    plt.close()

# ============================================================
# Run
# ============================================================

# ============================================================
# OBSERVABLES (supernova / BAO helpers)
# ============================================================

def E_of_a(a, H, H0):
    """Dimensionless expansion rate E(a)=H(a)/H0."""
    return np.asarray(H, dtype=float) / float(H0)

def z_of_a(a):
    a = np.asarray(a, dtype=float)
    return 1.0/np.maximum(a, 1e-30) - 1.0

def comoving_distance_z(z, z_grid, E_grid, H0=1.0, c_over_H0=1.0):
    """Comoving distance (in units of c/H0 if c_over_H0=1) via interpolation on precomputed grid."""
    # Integral: chi(z)=c/H0 ∫0^z dz'/E(z')
    z = np.asarray(z, dtype=float)
    # cumulative trapezoid on grid
    dz = np.diff(z_grid)
    integrand = 1.0/np.maximum(E_grid, 1e-30)
    cum = np.concatenate([[0.0], np.cumsum(0.5*dz*(integrand[1:]+integrand[:-1]))])
    chi_grid = c_over_H0 * cum
    return np.interp(z, z_grid, chi_grid)

def transverse_comoving_distance(chi, Omega_k0):
    """D_M from chi for curvature Omega_k0 (dimensionless), in units of c/H0."""
    ok = float(Omega_k0)
    chi = np.asarray(chi, dtype=float)
    if abs(ok) < 1e-12:
        return chi
    # FRW: D_M = S_k( sqrt(|Ok|) * chi ) / sqrt(|Ok|)
    x = np.sqrt(abs(ok)) * chi
    if ok > 0:
        return np.sinh(x) / np.sqrt(ok)
    else:
        return np.sin(x) / np.sqrt(abs(ok))

def luminosity_distance_z(z, z_grid, E_grid, Omega_k0=0.0, H0=1.0, c_over_H0=1.0):
    chi = comoving_distance_z(z, z_grid, E_grid, H0=H0, c_over_H0=c_over_H0)
    DM = transverse_comoving_distance(chi, Omega_k0)
    return (1.0 + np.asarray(z, dtype=float)) * DM  # in units of c/H0

def distance_modulus_from_dL(dL_Mpc):
    """mu = 5 log10(dL/10pc). Input dL in Mpc."""
    dL_Mpc = np.asarray(dL_Mpc, dtype=float)
    return 5.0*np.log10(np.maximum(dL_Mpc, 1e-30)) + 25.0

def make_z_grid_from_solution(a_hist, H_hist, H0):
    """Build monotone z-grid and E(z) from the model history."""
    a_hist = np.asarray(a_hist, dtype=float)
    H_hist = np.asarray(H_hist, dtype=float)
    z_hist = z_of_a(a_hist)
    # Ensure increasing z grid (early times -> large z). Sort by z.
    order = np.argsort(z_hist)
    z_grid = z_hist[order]
    E_grid = E_of_a(a_hist[order], H_hist[order], H0)
    # Remove duplicates (flat segments can happen early)
    z_grid_u, idx = np.unique(z_grid, return_index=True)
    return z_grid_u, E_grid[idx]

def sn_chi2(z_sn, mu_obs, mu_err, a_hist, H_hist, H0, Omega_k0=0.0, c_over_H0_Mpc=None, M_offset=0.0, H0_km_s_Mpc=70.0, c_km_s=299792.458):
    """
    Supernova chi^2 with a free nuisance offset M_offset (or fix it).
    If c_over_H0_Mpc is None, uses c/H0=1 in dimensionless units and compares only shapes (not absolute scale).
    """
    z_grid, E_grid = make_z_grid_from_solution(a_hist, H_hist, H0)
    c_over_H0 = 1.0 if (c_over_H0_Mpc is None) else float(c_over_H0_Mpc)
    dL = luminosity_distance_z(z_sn, z_grid, E_grid, Omega_k0=Omega_k0, H0=H0, c_over_H0=c_over_H0)
    mu_model = distance_modulus_from_dL(dL) + float(M_offset)
    r = (mu_model - np.asarray(mu_obs, float)) / np.maximum(np.asarray(mu_err, float), 1e-30)
    return float(np.sum(r*r))

def bao_DV_over_rs(z_bao, a_hist, H_hist, H0, Omega_k0=0.0, rs=1.0, c_over_H0=1.0):
    """Compute D_V(z)/r_s in units where distances are (c/H0). Provide rs in same units."""
    z_grid, E_grid = make_z_grid_from_solution(a_hist, H_hist, H0)
    chi = comoving_distance_z(z_bao, z_grid, E_grid, H0=H0, c_over_H0=c_over_H0)
    DM = transverse_comoving_distance(chi, Omega_k0)
    Hz = np.interp(z_bao, z_grid, H0*E_grid)
    z = np.asarray(z_bao, float)
    DV = (( (1.0+z)**2 * DM**2 * (c_over_H0/np.maximum(Hz, 1e-30)) * z ) )**(1.0/3.0)
    return DV/float(rs)


if __name__ == "__main__":
    t, a, H, V, theta, omega, wX, order_score, fb_hist, rhoX_bar, OmegaX = run_model()

    save_plot(t, H, ["H(t)"], "Hubble rate H(t) (multi-fluid + curvature)", "H(t)", "H_evolution.png")
    save_plot(t, a, ["a(t)"], "Scale factor a(t)", "a(t)", "a_evolution.png")
    save_plot(t, theta, ["θ(t)"], "Rotation angle θ(t)", "θ(t)", "theta_evolution.png")
    save_plot(t, omega, ["ω(t)"], "Angular rate ω(t)", "ω(t)", "omega_evolution.png")
    save_plot(t, wX, ["w_X(t)"], "Dark-sector equation of state w_X(t)", "w_X", "w_eff_evolution.png")
    save_plot(t, order_score, ["order(t)"], "Persistent order", "order", "order_recovery.png")
    save_plot(t, fb_hist, ["fb(t)"], "Feedback state", "fb", "feedback_state.png")
    #save_plot(t, OmegaX, ["Ω_X(t)"], "Dark density fraction proxy Ω_X(t)", "Ω_X", "OmegaX_evolution.png")
    save_plot(t, OmegaX, ["Ω_X(t)"], 
          "Dark density fraction Ω_X(t)", 
          "Ω_X", 
          "OmegaX_evolution.png")
    
    plt.figure()
    plt.plot(t, theta, label="θ(t)")
    plt.plot(t, wX, label="w_X(t)")
    plt.plot(t, order_score, label="order(t)")
    plt.legend()
    plt.title("Coupled observables")
    plt.xlabel("t (dimensionless)")
    plt.tight_layout()
    plt.savefig("combined_theta_w_order.png", dpi=300)
    plt.close()

    plot_metric_tube(t, a, theta, wX)
    print("All plots saved successfully.")
