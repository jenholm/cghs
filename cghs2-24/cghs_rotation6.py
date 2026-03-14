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
# DEBUG CONTROLS
# ============================================================
DEBUG = True              # master 
DEBUG_EVERY = 5000        # print lightweight status every N steps (set big to avoid spam)
DEBUG_RHS_SHAPES = True   # print only when RHS detects non-scalars
DEBUG_BREAK_ON_BAD = True # raise immediately with diagnostics

# ============================================================
# Helpers
# ============================================================

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

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

def _shape(x):
    try:
        return np.shape(x)
    except Exception:
        return None

def _dtype(x):
    try:
        return getattr(x, "dtype", type(x))
    except Exception:
        return type(x)

def _is_scalar_like(x):
    # True for python scalars, numpy scalars
    return np.isscalar(x) or isinstance(x, (np.generic,))

def _dbg_var(name, x):
    return f"{name}: type={type(x).__name__}, dtype={_dtype(x)}, shape={_shape(x)}, val={x}"

# ============================================================
# Cosmology + dark-sector controller (θ–ω–O–fb)
# ============================================================
def run_model(
    # --- time control ---
    T=200000,
    dt=2e-5,  # dimensionless dt in units of 1/H0 (smaller is safer for tiny a)

    # --- FRW normalization + present-day fractions ---
    H0=1.0,              # by definition in these units
    Omega_r0=9.0e-5,     # (photons + neutrinos approx) tune as desired
    Omega_m0=0.30,       # matter (baryons+CDM)
    Omega_k0=0.00,       # curvature (positive=open, negative=closed)
    Omega_X0=None,       # dark sector (if None -> closure)

    # --- initial conditions ---
    a0=1e-8,
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
    k1=0.010,

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

    def H_from(a, OmegaX):
        # Guard tiny a
        a = max(a, 1e-40)
        term = (Omega_r0 / (a**4)) + (Omega_m0 / (a**3)) + OmegaX
        if USE_CURVATURE:
            term += Omega_k0 / (a**2)
        H = H0 * np.sqrt(max(term, 0.0))
        return clamp(H, H_min, H_max)

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

    def rhs(t, y):
        """
        y = [a, OmegaX, theta, omega, O, fb]
        Returns dy/dt for RK4.
        """

                # ---- DEBUG: verify y is the shape we expect ----
        if DEBUG and DEBUG_RHS_SHAPES:
            if not isinstance(y, np.ndarray):
                print("[DBG rhs] y is not ndarray:", type(y), y)
            else:
                if y.shape != (6,):
                    print("[DBG rhs] y.shape != (6,):", y.shape, y)

        a, OmegaX, theta, omega, O, fb = y
        if OmegaX < 0.0:
            OmegaX = 0.0
        if a < 1e-40:
            a = 1e-40

        # ---- DEBUG: ensure these are scalar-like ----
        if DEBUG and DEBUG_RHS_SHAPES:
            bad = []
            for nm, vv in [("a", a), ("OmegaX", OmegaX), ("theta", theta), ("omega", omega), ("O", O), ("fb", fb)]:
                if not _is_scalar_like(vv):
                    bad.append(_dbg_var(nm, vv))
            if bad:
                print("\n[DBG rhs] NON-SCALAR STATE COMPONENT(S) DETECTED")
                for line in bad:
                    print("  ", line)
                if DEBUG_BREAK_ON_BAD:
                    raise RuntimeError("[DBG rhs] y contains non-scalar component(s); see diagnostics above.")
        a, OmegaX, theta, omega, O, fb = y

        # H from FRW closure
        H = H_from(a, OmegaX)

        # feedback memory
        if USE_FEEDBACK_MEMORY:
            dfb = ((O - O_star) - fb) / max(tau_fb, 1e-12)
        else:
            dfb = 0.0

        # dark equation of state
        wX = wX_from_state(O, fb, omega, a)

        # dark continuity
        dOmegaX = -3.0 * H * (1.0 + wX) * OmegaX

        # scale factor
        da = a * H

        # twist
        if USE_TWIST_DYNAMICS:
            # ---- twist ----
            kO = k0 + k1 * (O - O_star)

            # DEBUG: check coefficients are scalar
            if DEBUG and DEBUG_RHS_SHAPES:
                for nm, vv in [("k0", k0), ("k1", k1), ("kO", kO), ("beta_theta", beta_theta), ("gamma_drag", gamma_drag)]:
                    if not _is_scalar_like(vv):
                        print("\n[DBG rhs] NON-SCALAR COEFFICIENT IN TWIST EQUATION")
                        print("  t =", t)
                        print("  ", _dbg_var(nm, vv))
                        print("  ", _dbg_var("k0", k0))
                        print("  ", _dbg_var("k1", k1))
                        print("  ", _dbg_var("beta_theta", beta_theta))
                        print("  ", _dbg_var("gamma_drag", gamma_drag))
                        if DEBUG_BREAK_ON_BAD:
                            raise RuntimeError("[DBG rhs] Twist coefficient went non-scalar.")
                                    
            # force scalar positivity without numpy broadcasting
            if kO < 0.0:
                kO = 0.0

            # force domega to scalar and crash *right here* if it isn't
            domega_expr = -gamma_drag * H * omega - (kO + beta_theta) * theta
            try:
                domega = float(domega_expr)
            except Exception as e:
                print("\n[DBG rhs] domega_expr could not be cast to float")
                print("  t =", t)
                print("  ", _dbg_var("domega_expr", domega_expr))
                print("  ", _dbg_var("gamma_drag", gamma_drag))
                print("  ", _dbg_var("H", H))
                print("  ", _dbg_var("omega", omega))
                print("  ", _dbg_var("kO", kO))
                print("  ", _dbg_var("beta_theta", beta_theta))
                print("  ", _dbg_var("theta", theta))
                raise

            dtheta = float(omega)
            kO = np.maximum(kO, 0.0)
            domega = -gamma_drag * H * omega - (kO + beta_theta) * theta
            dtheta = omega
        else:
            domega = 0.0
            dtheta = 0.0

        # order
        if USE_DYNAMIC_ORDER:
            activity = (theta**2 + eta_omega * (omega**2)) / max(S_star, 1e-12)
            sigma = np.tanh(activity)

            prod = lam_prod * sigma
            if ORDER_SCALES_WITH_DARK_FRACTION:
                # "turn on" order production only as dark sector matters
                prod *= clamp(OmegaX, 0.0, 1.0)

            dO = prod - mu_relax * H * (O - O_min)
        else:
            dO = 0.0

                # ---- DEBUG: detect which derivative is a sequence/array ----
        if DEBUG and DEBUG_RHS_SHAPES:
            derivs = [("da", da), ("dOmegaX", dOmegaX), ("dtheta", dtheta),
                      ("domega", domega), ("dO", dO), ("dfb", dfb)]
            bad = []
            for nm, vv in derivs:
                if not _is_scalar_like(vv):
                    bad.append(_dbg_var(nm, vv))
            if bad:
                print("\n[DBG rhs] NON-SCALAR DERIVATIVE(S) DETECTED")
                print("  t =", t)
                print("  ", _dbg_var("a", a))
                print("  ", _dbg_var("OmegaX", OmegaX))
                print("  ", _dbg_var("theta", theta))
                print("  ", _dbg_var("omega", omega))
                print("  ", _dbg_var("O", O))
                print("  ", _dbg_var("fb", fb))
                print("  ", _dbg_var("H", H))
                print("  ", _dbg_var("wX", wX))
                for line in bad:
                    print("  ", line)
                if DEBUG_BREAK_ON_BAD:
                    raise RuntimeError("[DBG rhs] RHS produced non-scalar derivative(s); see diagnostics above.")

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

    # init
    y = np.array([a0, Omega_X0, theta0, omega0, clamp(order0,0.0,1.0), fb0], dtype=float)

    # record initial
    deriv0, H0_now, w0_now = rhs(0.0, y)
    a_arr[0], OmegaX_arr[0], theta_arr[0], omega_arr[0], O_arr[0], fb_arr[0] = y
    H_arr[0] = H0_now
    wX_arr[0] = w0_now

    n = 1
    for i in range(1, T):
        ti = (i-1) * dt
        if DEBUG and (i % DEBUG_EVERY == 0):
            print(f"[DBG step] i={i} t={ti:.6g} a={a_arr[i-1]:.6g} H={H_arr[i-1]:.6g} OmegaX={OmegaX_arr[i-1]:.6g}")

        if USE_RK4:
            rk1, _, _ = rhs(ti, y)
            rk2, _, _ = rhs(ti + 0.5*dt, y + 0.5*dt*rk1)
            rk3, _, _ = rhs(ti + 0.5*dt, y + 0.5*dt*rk2)
            rk4, _, _ = rhs(ti + dt, y + dt*rk3)
            y_new = y + (dt/6.0)*(rk1 + 2*rk2 + 2*rk3 + rk4)
        else:
            dy, _, _ = rhs(ti, y)
            y_new = y + dt*dy

        # positivity / bounds
        #y_new[0] = max(y_new[0], 1e-40)          # a
        #y_new[1] = max(y_new[1], 0.0)            # OmegaX
        y_new[0] = np.maximum(y_new[0], 1e-40)
        y_new[1] = np.maximum(y_new[1], 0.0)
        y_new[4] = clamp(y_new[4], 0.0, 1.0)     # O

        # optional discrete events (kick omega/fb only)
        if USE_EVENTS:
            if float(ti) in event_map:
                d_om, d_fb = event_map[float(ti)]
                y_new[3] += d_om
                y_new[5] += d_fb

        # commit
        y = y_new
        deriv, H_now, w_now = rhs(ti + dt, y)

        t[i] = ti + dt
        a_arr[i], OmegaX_arr[i], theta_arr[i], omega_arr[i], O_arr[i], fb_arr[i] = y
        H_arr[i] = H_now
        wX_arr[i] = w_now
        n = i + 1

        if STOP_AT_A_EQ_1 and a_arr[i] >= 1.0:
            break

    # truncate
    t = t[:n]
    a_arr = a_arr[:n]
    H_arr = H_arr[:n]
    theta_arr = theta_arr[:n]
    omega_arr = omega_arr[:n]
    O_arr = O_arr[:n]
    fb_arr = fb_arr[:n]
    wX_arr = wX_arr[:n]
    OmegaX_arr = OmegaX_arr[:n]

    V = a_arr**3
    return t, a_arr, H_arr, V, theta_arr, omega_arr, wX_arr, O_arr, fb_arr, OmegaX_arr

# ============================================================
# 3D Metric Tube
# ============================================================
def plot_metric_tube(t, a, theta, w_eff):
    a_norm = a / np.max(a)
    R_profile = np.power(a_norm, 0.05)

    phi = np.linspace(0, 2*np.pi, 100)
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

    alpha_profile = 0.35 + 0.45 * (t / np.max(t))
    for i in range(len(phi)):
        colors[i, :, 3] = alpha_profile

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=True)
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
if __name__ == "__main__":
    t, a, H, V, theta, omega, wX, order_score, fb_hist, OmegaX = run_model()

    save_plot(t, H, ["H(t)"], "Hubble rate H(t) (multi-fluid + curvature)", "H(t)", "H_evolution.png")
    save_plot(t, a, ["a(t)"], "Scale factor a(t)", "a(t)", "a_evolution.png")
    save_plot(t, theta, ["θ(t)"], "Rotation angle θ(t)", "θ(t)", "theta_evolution.png")
    save_plot(t, omega, ["ω(t)"], "Angular rate ω(t)", "ω(t)", "omega_evolution.png")
    save_plot(t, wX, ["w_X(t)"], "Dark-sector equation of state w_X(t)", "w_X", "w_eff_evolution.png")
    save_plot(t, order_score, ["order(t)"], "Persistent order", "order", "order_recovery.png")
    save_plot(t, fb_hist, ["fb(t)"], "Feedback state", "fb", "feedback_state.png")
    save_plot(t, OmegaX, ["Ω_X(t)"], "Dark density fraction proxy Ω_X(t)", "Ω_X", "OmegaX_evolution.png")

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
