import numpy as np
import matplotlib.pyplot as plt

#pass 1
#USE_OPTION_A_RATE_ORDER = True
#USE_OPTION_B_INERTIA = False
#USE_OPTION_C_FEEDBACK = False
#pass 2
USE_OPTION_A_RATE_ORDER = True
USE_OPTION_B_INERTIA = True
USE_OPTION_C_FEEDBACK = False

# pass 3
#USE_OPTION_A_RATE_ORDER = True
#USE_OPTION_B_INERTIA = True
#USE_OPTION_C_FEEDBACK = True

# ============================================================
# SWITCHES (run in sequence by flipping these)
# ============================================================
#USE_OPTION_A_RATE_ORDER = True     # A: order depends on |dtheta/dt|
#USE_OPTION_B_INERTIA = False       # B: theta second-order dynamics
#USE_OPTION_C_FEEDBACK = False      # C: order feeds back into H

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
    # Era controls
    H_inf=0.35,
    t_rad=25,
    t_mat=60,
    width_rad=3.0,
    width_mat=6.0,
    # Theta dynamics controls
    kappa=2.0,
    gamma=0.05,
    # Option B (inertia) controls
    alpha=2.0,
    Gamma=0.25,
    beta=0.002,          # <-- NEW restoring term
    theta0=0.01,
    omega0=0.0,
    # Order controls
    lam=6.0,
    rate_scale=1.0,
    # Persistent order controls (NEW)
    order_tau=18.0,        # memory timescale (bigger = more persistence)
    order_floor=0.50,      # baseline level
    order_decay=0.003,     # slow leak back toward floor (set 0.0 for none)
    # Feedback controls (Option C)
    eps_feedback=0.35,
    order_target=0.75,
    feedback_smooth=0.15
    
):
    """
    Returns arrays: H, V, theta, w_eff, order_score
    """

    t = np.arange(T, dtype=float)

    V = np.zeros(T, dtype=float)
    H = np.zeros(T, dtype=float)
    theta = np.zeros(T, dtype=float)
    w_eff = np.zeros(T, dtype=float)
    order_score = np.zeros(T, dtype=float)

    # initial conditions
    V[0] = 10.0
    theta[0] = theta0
    order_score[0] = order_floor

    # For Option B
    omega = np.zeros(T, dtype=float)  # omega = dtheta/dt
    omega[0] = omega0

    # For Option C: smooth feedback state
    fb = 0.0

    # Precompute smooth mixing functions for eras
    rad_mix = sigmoid((t - t_rad) / width_rad)     # inflation -> radiation
    mat_mix = sigmoid((t - t_mat) / width_mat)     # radiation -> matter

    # baseline era H targets
    H_rad = 1.0 / (2.0 * (t + 1.0))
    H_mat = 2.0 / (3.0 * (t + 1.0))

    # Construct a smooth H(t) curve (before feedback)
    H_base = (
        (1.0 - rad_mix) * H_inf
        + rad_mix * (1.0 - mat_mix) * H_rad
        + rad_mix * mat_mix * H_mat
    )

    # Main loop
    for i in range(1, T):

        # ====================================================
        # H(t): smooth eras + optional feedback
        # ====================================================
        H_now = H_base[i]

        if USE_OPTION_C_FEEDBACK:
            # Compute last order, drive fb smoothly (low-pass)
            # fb grows when order > target, shrinks otherwise
            drive = (order_score[i - 1] - order_target)
            fb = (1.0 - feedback_smooth) * fb + feedback_smooth * drive

            # Apply feedback as a multiplicative damping or boost of H
            # Positive fb reduces H (more "structure" slows expansion)
            H_now = H_now * (1.0 - eps_feedback * fb)

            # Keep sane bounds
            H_now = clamp(H_now, 1e-6, H_inf * 1.25)

        H[i] = H_now

        # Update volume proxy
        V[i] = V[i - 1] * np.exp(3.0 * H[i])

        # ====================================================
        # Theta dynamics
        # ====================================================
        if USE_OPTION_B_INERTIA:
            # Second-order with restoring term:
            # theta'' + Gamma theta' + beta theta = alpha (H_inf - H)
            dH = H[i] - H[i - 1]

            accel = (
            -alpha * dH         # drive only during era change
            - Gamma * omega[i - 1]
            - beta * theta[i - 1]
            )

            omega[i] = omega[i - 1] + accel
            theta[i] = theta[i - 1] + omega[i]
        else:
            # First-order:
            theta[i] = theta[i - 1] + kappa * (H_inf - H[i]) - gamma * theta[i - 1]

        # ====================================================
        # Effective equation of state proxy
        # (kept consistent with your current definition)
        # ====================================================
        denom = (H[i] + 1e-12)
        if i == 1:
            w_eff[i] = -1.0
        else:
            denom = (H[i] + 1e-12)
            w_eff[i] = -1.0 - (2.0 / 3.0) * (H[i] - H[i - 1]) / denom

        # ====================================================
        # Order recovery (PERSISTENT MEMORY)
        # ====================================================
        if USE_OPTION_A_RATE_ORDER:
            dtheta = theta[i] - theta[i - 1]
            x = rate_scale * np.abs(dtheta)
            o_star = 0.5 + 0.5 * np.tanh(lam * x)          # instantaneous "drive"
        else:
            o_star = 0.5 + 0.5 * np.tanh(lam * np.sin(theta[i]))

        prev = order_score[i - 1]

        # Relax toward o_star over timescale order_tau
        relax = (o_star - prev) / max(order_tau, 1e-12)

        # Optional slow leak back toward order_floor
        leak = -order_decay * (prev - order_floor)

        order_score[i] = prev + relax + leak
        order_score[i] = clamp(order_score[i], 0.0, 1.0)
    return H, V, theta, w_eff, order_score

# ============================================================
# Run + plots
# ============================================================
if __name__ == "__main__":
    H, V, theta, w_eff, order_score = run_model()

    t = np.arange(len(theta))

    # Individual PNGs
    save_plot(t, H, ["H(t)"], "Hubble-like rate H(t)", "H(t)", "H_evolution.png")
    save_plot(t, theta, ["theta(t)"], "Rotation angle θ(t)", "θ(t)", "theta_evolution.png")
    save_plot(t, w_eff, ["w_eff(t)"], "Effective equation of state w_eff(t)", "w_eff(t)", "w_eff_evolution.png")
    save_plot(t, order_score, ["order(t)"], "Causal-order recovery vs time", "order accuracy", "order_recovery.png")

    # Combined “watch the gears mesh” plot
    plt.figure()
    plt.plot(t, theta, label="θ(t)")
    plt.plot(t, w_eff, label="w_eff(t)")
    plt.plot(t, order_score, label="order(t)")
    plt.title("Coupled observables: θ(t), w_eff(t), order(t)")
    plt.xlabel("t")
    plt.tight_layout()
    plt.legend()
    plt.savefig("combined_theta_w_order.png", dpi=300)
    plt.close()

    print("Saved PNGs: H_evolution.png, theta_evolution.png, w_eff_evolution.png, order_recovery.png, combined_theta_w_order.png")
    print(f"Options: A(rate-order)={USE_OPTION_A_RATE_ORDER}, B(inertia)={USE_OPTION_B_INERTIA}, C(feedback)={USE_OPTION_C_FEEDBACK}")