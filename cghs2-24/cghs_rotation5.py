import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import matplotlib.cm as cm

# ============================================================
# PASS SWITCHES
# ============================================================
# If True, evolve H dynamically from w_eff (recommended)
USE_DYNAMIC_H = True

# If True, evolve theta with inertia + Hubble drag + potential torque (recommended)
USE_TWIST_DYNAMICS = True

# If True, evolve order dynamically (recommended)
USE_DYNAMIC_ORDER = True

# If True, include feedback memory state fb(t) and let it influence w_eff
USE_FEEDBACK_MEMORY = True

# Optional: discrete "events" that kick omega/fb (kept off by default)
USE_EVENTS = False

# ============================================================
# Helpers
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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
    plt.xlabel("t")
    plt.ylabel(ylabel)
    if any(lab for lab in labels):
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# Core coupled model (recommended backbone)
# ============================================================
def run_model(
    T=120,
    dt=1.0,
    H_inf=1.2,
    # Initial conditions
    a0=1e-3,
    H0=1.2,
    theta0=0.01,
    omega0=0.0,
    order0=0.50,
    fb0=0.0,

    # ---- w_eff coupling ----
    w0=-1.0,            # baseline
    alpha_O=-0.8,       # order -> w_eff (negative makes higher order more DE-like)
    O_star=0.65,        # reference order level
    alpha_fb=-0.8,      # feedback memory -> w_eff
    alpha_omega=0.15,   # omega -> w_eff via tanh
    omega_star=0.15,    # tanh scale for omega

    # ---- H dynamics ----
    H_min=1e-6,
    H_max=1.6,

    # ---- Twist dynamics ----
    gamma_drag=0.8,     # Hubble drag coefficient in domega/dt = -gamma*H*omega - dU/dtheta
    beta_theta=0.002,   # linear restoring torque strength via dU/dtheta = k(O)*theta
    k0=0.003,           # base stiffness
    k1=0.010,           # order-dependent stiffness shift

    # ---- Order dynamics ----
    lam_prod=0.12,      # production strength
    eta_omega=0.7,      # weight of omega^2 in activity
    S_star=0.25,        # activity scale
    mu_relax=0.10,      # relaxation toward O_min with expansion
    O_min=0.50,         # baseline order floor

    # ---- Feedback memory ----
    tau_fb=18.0,        # fb relaxation timescale (in t units)

    # ---- Optional events ----
    event_times=(25, 80),
    event_domega=(+0.25, -0.25),
    event_dfb=(0.0, 0.0),
):
    t = np.arange(T, dtype=float) * dt

    a = np.zeros(T)
    H = np.zeros(T)
    theta = np.zeros(T)
    omega = np.zeros(T)
    w_eff = np.zeros(T)
    order_score = np.zeros(T)
    fb_hist = np.zeros(T)

    # init
    a[0] = a0
    H[0] = H0
    theta[0] = theta0
    omega[0] = omega0
    order_score[0] = clamp(order0, 0.0, 1.0)
    fb = fb0
    fb_hist[0] = fb

    # event lookup
    event_map = {}
    if USE_EVENTS and len(event_times) > 0:
        for tt, d_om, d_fb in zip(event_times, event_domega, event_dfb):
            event_map[float(tt)] = (float(d_om), float(d_fb))

    for i in range(1, T):
        # ----------------------------
        # 1) derive w_eff from current state (previous step)
        # ----------------------------
        O_prev = order_score[i - 1]
        om_prev = omega[i - 1]
        th_prev = theta[i - 1]
        H_prev = H[i - 1]

        # feedback memory update (continuous-time leaky integrator)
        if USE_FEEDBACK_MEMORY:
            fb += dt * ((O_prev - O_star) - fb) / max(tau_fb, 1e-9)
        else:
            fb = 0.0

        # w_eff is derived (do NOT kick it directly)
        w = (
            w0
            + alpha_O * (O_prev - O_star)
            + alpha_fb * fb
            + alpha_omega * np.tanh(om_prev / max(omega_star, 1e-12))
        )

        # keep w in a sane band for numerical stability / plotting
        w = clamp(w, -1.2, +1.0)
        w_eff[i] = w
        fb_hist[i] = fb

        # ----------------------------
        # 2) evolve H (dynamic Friedmann-like)
        # ----------------------------
        if USE_DYNAMIC_H:
            dH = -(3.0 / 2.0) * (1.0 + w) * (H_prev ** 2)
            H_now = H_prev + dt * dH
        else:
            # fallback: a gentle monotone decay if you want to compare
            H_now = H_prev * (1.0 - 0.01 * dt)

        H_now = clamp(H_now, H_min, H_max)
        if t[i] < 6.0:
            H_now = H_inf
        H[i] = H_now

        # ----------------------------
        # 3) evolve a (scale factor)
        # ----------------------------
        a[i] = a[i - 1] * np.exp(H_now * dt)

        # ----------------------------
        # 4) twist dynamics (theta, omega)
        # ----------------------------
        if USE_TWIST_DYNAMICS:
            # order-dependent stiffness: k(O) = k0 + k1*(O - O_star)
            kO = k0 + k1 * (O_prev - O_star)
            kO = max(kO, 0.0)

            # potential torque: dU/dtheta = (kO + beta_theta)*theta
            # domega/dt = -gamma*H*omega - dU/dtheta
            domega = -gamma_drag * H_now * om_prev - (kO + beta_theta) * th_prev
            omega[i] = om_prev + dt * domega
            theta[i] = th_prev + dt * omega[i]
        else:
            # fallback: integrate theta from H difference (older behavior)
            theta[i] = theta[i - 1] + (H0 - H_now) * dt
            omega[i] = (theta[i] - theta[i - 1]) / dt

        # ----------------------------
        # 5) order dynamics (production - dilution)
        # ----------------------------
        if USE_DYNAMIC_ORDER:
            # activity sigma = tanh((theta^2 + eta*omega^2)/S_star)
            activity = (theta[i] ** 2 + eta_omega * (omega[i] ** 2)) / max(S_star, 1e-12)
            sigma = np.tanh(activity)

            # dO/dt = lam*sigma - mu*H*(O - O_min)
            dO = lam_prod * sigma - mu_relax * H_now * (O_prev - O_min)
            O_now = O_prev + dt * dO
        else:
            # fallback: smooth toward an instantaneous stimulus from omega
            inst = 0.5 + 0.5 * np.tanh(6.0 * abs(omega[i]))
            O_now = O_prev + 0.06 * (inst - O_prev)

        order_score[i] = clamp(O_now, 0.0, 1.0)

        # ----------------------------
        # 6) optional discrete events (kick omega/fb only)
        # ----------------------------
        if USE_EVENTS:
            ti = float(t[i])
            if ti in event_map:
                d_om, d_fb = event_map[ti]
                omega[i] += d_om
                fb += d_fb
                fb_hist[i] = fb  # record after event

    # A "volume proxy" if you still like V(t)
    V = a ** 3

    return t, H, V, theta, omega, w_eff, order_score, fb_hist

# ============================================================
# 3D Metric Tube
# ============================================================
def plot_metric_tube(t, H, theta, order_score, w_eff):
    dt = t[1] - t[0]

    # --- Scale factor ---
    a = np.exp(np.cumsum(H) * dt)
    a = a / np.max(a)

    # softer geometry profile
    #a_star = 0.25
    a = np.exp(np.cumsum(H) * dt)
    a = a / np.max(a)
    R_profile = np.power(a, 0.05)
    nu = 0.85
    #R_profile = np.power(a / (a + a_star), nu)

    phi = np.linspace(0, 2*np.pi, 100)
    T_grid, Phi = np.meshgrid(t, phi)

    R = np.outer(R_profile, np.ones_like(phi)).T
    twist = np.outer(theta, np.ones_like(phi)).T

    X = R * np.cos(Phi + twist)
    Y = R * np.sin(Phi + twist)
    Z = T_grid

    # -------------------------------------------------
    # Better color normalization
    # -------------------------------------------------
    wmin = np.percentile(w_eff, 5)
    wmax = np.percentile(w_eff, 95)

    w_clip = np.clip(w_eff, wmin, wmax)
    norm = (w_clip - wmin) / (wmax - wmin)

    colors = cm.plasma(norm)  # better gradient than coolwarm
    colors = np.tile(colors, (len(phi), 1, 1))

    # Add transparency gradient over time
    alpha_profile = 0.35 + 0.45 * (t / np.max(t))
    for i in range(len(phi)):
        colors[i, :, 3] = alpha_profile

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True
    )
    ax.plot([0]*len(t), [0]*len(t), t, linewidth=1.0)
    ax.plot_wireframe(X, Y, Z, color='k', alpha=0.05)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_zlabel("Cosmic time")

    ax.set_title("Metric Tube\nTwist = θ(t), Radius = a(t), Color = w_eff")

    # remove grid clutter
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("metric_tube_3D.png", dpi=300)
    plt.close()

    print("Updated metric tube saved.")

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    t, H, V, theta, omega, w_eff, order_score, fb_hist = run_model()

    save_plot(t, H, ["H(t)"], "Hubble-like rate H(t)", "H(t)", "H_evolution.png")
    save_plot(t, theta, ["θ(t)"], "Rotation angle θ(t)", "θ(t)", "theta_evolution.png")
    save_plot(t, omega, ["ω(t)"], "Angular rate ω(t)", "ω(t)", "omega_evolution.png")
    save_plot(t, w_eff, ["w_eff(t)"], "Effective equation of state", "w_eff", "w_eff_evolution.png")
    save_plot(t, order_score, ["order(t)"], "Persistent order", "order", "order_recovery.png")
    save_plot(t, fb_hist, ["fb(t)"], "Feedback state", "fb", "feedback_state.png")

    plt.figure()
    plt.plot(t, theta, label="θ(t)")
    plt.plot(t, w_eff, label="w_eff(t)")
    plt.plot(t, order_score, label="order(t)")
    plt.legend()
    plt.title("Coupled observables")
    plt.xlabel("t")
    plt.tight_layout()
    plt.savefig("combined_theta_w_order.png", dpi=300)
    plt.close()

    plot_metric_tube(t, H, theta, order_score, w_eff)

    print("All plots saved successfully.")