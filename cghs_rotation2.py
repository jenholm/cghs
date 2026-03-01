import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Three-era Hubble schedule
# ----------------------------
def hubble_three_era(t, t_rad=25, t_mat=60, H_inf=0.35):
    if t < t_rad:
        return H_inf
    elif t < t_mat:
        return 1.0 / (2.0 * (t + 1.0))
    else:
        return 2.0 / (3.0 * (t + 1.0))


# ----------------------------
# Stable θ evolution
# ----------------------------
def update_theta(theta_prev, H_inf, H_t, kappa=0.8, gamma=0.03):
    theta_new = theta_prev + kappa * (H_inf - H_t) - gamma * theta_prev
    return theta_new % (2.0 * np.pi)


# ----------------------------
# Order strength (magnitude only)
# ----------------------------
def order_from_theta(theta_t, lam=4.0):
    return 0.5 + 0.5 * np.tanh(lam * np.abs(np.sin(theta_t)))


# ----------------------------
# Main model
# ----------------------------
def run_model(T=120):

    V = np.zeros(T)
    H = np.zeros(T)
    w_eff = np.zeros(T)
    theta = np.zeros(T)
    order_score = np.zeros(T)

    V[0] = 10.0
    theta[0] = 0.01

    H_inf = 0.35
    t_rad = 25
    t_mat = 60

    kappa = 0.8
    gamma = 0.03
    lam = 4.0

    for t in range(1, T):

        # --- ERA STRUCTURE (hard transitions) ---
        H[t] = hubble_three_era(t, t_rad, t_mat, H_inf)

        # Volume evolution
        V[t] = V[t-1] * np.exp(3 * H[t])

        # --- θ dynamics (stabilized + wrapped) ---
        theta[t] = update_theta(theta[t-1], H_inf, H[t], kappa, gamma)

        # --- w_eff ---
        if t > 1:
            w_eff[t] = -1 - (2/3)*(H[t] - H[t-1])/(H[t] + 1e-8)

        # --- order recovery (magnitude only) ---
        order_score[t] = order_from_theta(theta[t], lam)

    return theta, w_eff, order_score


# ----------------------------
# Run model
# ----------------------------
theta, w_eff, order_score = run_model()


# ----------------------------
# PNG saving utility
# ----------------------------
def save_plot(x, y, title, ylabel, filename):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


t = np.arange(len(theta))

save_plot(t, theta,
          "Rotation angle θ(t)",
          "θ(t)",
          "theta_evolution.png")

save_plot(t, w_eff,
          "Effective equation of state w_eff(t)",
          "w_eff(t)",
          "w_eff_evolution.png")

save_plot(t, order_score,
          "Causal-order recovery vs time",
          "order accuracy",
          "order_recovery.png")

print("Saved upgraded PNGs.")