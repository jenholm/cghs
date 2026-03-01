import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ============================================================
# PASS SWITCHES
# ============================================================
USE_OPTION_A_RATE_ORDER = True
USE_OPTION_B_INERTIA = True
USE_OPTION_C_FEEDBACK = True

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
# Core model
# ============================================================
def run_model(
    T=120,
    H_inf=1.2,
    theta0=0.01,
    omega0=0.0,
    alpha=2.0,
    Gamma=0.25,
    beta=0.002,
    lam=6.0,
    rate_scale=1.0,
    order_smooth=0.06,
    eps_feedback=0.35,
    order_target=0.75,
    feedback_smooth=0.15
):
    t = np.arange(T, dtype=float)

    V = np.zeros(T)
    H = np.zeros(T)
    theta = np.zeros(T)
    omega = np.zeros(T)
    w_eff = np.zeros(T)
    order_score = np.zeros(T)
    fb_hist = np.zeros(T)

    V[0] = 10.0
    theta[0] = theta0
    omega[0] = omega0
    order_score[0] = 0.5

    # --- Cosmology-inspired H profile ---
    t0 = 5.0
    t1 = 25.0
    t2 = 80.0

    H_base = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < t0:
            H_base[i] = H_inf
        elif ti < t1:
            H_base[i] = 1.0 / (2.0 * (ti - t0 + 1.0))
        elif ti < t2:
            H_base[i] = 2.0 / (3.0 * (ti - t1 + 5.0))
        else:
            H_base[i] = 0.02 + 0.005 * np.exp((ti - t2)/40.0)

    H[0] = H_base[0]
    fb = 0.0

    for i in range(1, T):

        H_now = H_base[i]

        if USE_OPTION_C_FEEDBACK:
            drive = (order_score[i - 1] - order_target)
            fb = (1.0 - feedback_smooth) * fb + feedback_smooth * drive
            H_now = H_now * (1.0 - eps_feedback * fb)

        H_now = clamp(H_now, 1e-6, H_inf * 1.25)

        H[i] = H_now
        fb_hist[i] = fb

        V[i] = V[i - 1] * np.exp(3.0 * H[i])

        # --- Theta dynamics ---
        if USE_OPTION_B_INERTIA:
            dH = H[i] - H[i - 1]
            accel = (-alpha * dH) - (Gamma * omega[i - 1]) - (beta * theta[i - 1])
            omega[i] = omega[i - 1] + accel
            theta[i] = theta[i - 1] + omega[i]
        else:
            theta[i] = theta[i - 1] + (H_inf - H[i])
            omega[i] = theta[i] - theta[i - 1]

        # --- Effective equation of state ---
        denom = H[i] + 1e-12
        w_eff[i] = -1.0 - (2.0 / 3.0) * (H[i] - H[i - 1]) / denom

        # --- Order stimulus ---
        if USE_OPTION_A_RATE_ORDER:
            x = rate_scale * np.abs(omega[i])
            order_inst = 0.5 + 0.5 * np.tanh(lam * x)
        else:
            order_inst = 0.5 + 0.5 * np.tanh(lam * np.sin(theta[i]))

        order_inst = clamp(order_inst, 0.0, 1.0)
        order_score[i] = order_score[i - 1] + order_smooth * (order_inst - order_score[i - 1])
        order_score[i] = clamp(order_score[i], 0.0, 1.0)

    return t, H, V, theta, omega, w_eff, order_score, fb_hist

# ============================================================
# 3D Metric Tube
# ============================================================
def plot_metric_tube(t, H, theta, order_score, w_eff):

    dt = t[1] - t[0]

    # --- Scale factor ---
    a = np.exp(np.cumsum(H) * dt)

    # compress dynamic range for visualization only
    a = a / np.max(a)
    a = np.power(a, 0.6)

    phi = np.linspace(0, 2*np.pi, 80)
    T_grid, Phi = np.meshgrid(t, phi)

    R = np.outer(a, np.ones_like(phi)).T

    twist = np.outer(theta, np.ones_like(phi)).T

    X = R * np.cos(Phi + twist)
    Y = R * np.sin(Phi + twist)
    Z = T_grid

    # --- Robust color mapping ---
    w_centered = w_eff + 1.0

    # clip extremes to avoid spike domination
    clip = 0.5
    w_centered = np.clip(w_centered, -clip, clip)

    norm = (w_centered - (-clip)) / (2*clip)

    colors = cm.coolwarm(norm)
    colors = np.tile(colors, (len(phi), 1, 1))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_zlabel("Cosmic time")

    ax.set_title("Metric Tube\nTwist = θ(t), Radius = a(t), Color = w_eff")

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