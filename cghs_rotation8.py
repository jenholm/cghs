# ============================================================
# CGHS ROTATION MODEL — BIANCHI I + MULTIFLUID + OBSERVABLES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

H0 = 1.0
Omega_r0 = 8.4e-5
Omega_m0 = 0.3
Omega_k0 = 0.0

# Dark sector parameters
alpha_w = 4.0
beta_w = 3.0
gamma_w = 2.5
delta_w = 1.2

# Shear coupling
alpha_sigma = 0.05

# Integration settings
dt = 5e-4
T = 40000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def H_from(a, OmegaX, sigma):
    rho_std = (
        Omega_r0 * a**(-4)
        + Omega_m0 * a**(-3)
        + Omega_k0 * a**(-2)
        + OmegaX
    )
    return np.sqrt(H0**2 * rho_std + sigma**2 / 6.0)


def rhs(t, y):
    a, OmegaX, theta, omega, order, fb, sigma = y

    H = H_from(a, OmegaX, sigma)

    # -----------------------------
    # Scale factor
    # -----------------------------
    da = H * a

    # -----------------------------
    # Dark energy evolution
    # -----------------------------
    w_eff = -1 + np.tanh(alpha_w * theta - beta_w * order)

    dOmegaX = -3 * H * (1 + w_eff) * OmegaX

    # -----------------------------
    # Rotation sector
    # -----------------------------
    dtheta = omega
    domega = -gamma_w * H * omega + delta_w * OmegaX * theta

    # -----------------------------
    # Order parameter
    # -----------------------------
    dorder = H * (1 - order)

    # -----------------------------
    # Feedback
    # -----------------------------
    dfb = -0.2 * H * fb

    # -----------------------------
    # Shear evolution (Bianchi I)
    # -----------------------------
    dsigma = -3 * H * sigma + alpha_sigma * (omega**2) * order

    return np.array([da, dOmegaX, dtheta, domega, dorder, dfb, dsigma])


# ============================================================
# RUN MODEL
# ============================================================

def run_model():

    y = np.array([
        1e-3,   # a
        0.7,    # OmegaX
        0.01,   # theta
        0.0,    # omega
        0.5,    # order
        0.0,    # feedback
        1e-5    # sigma (small initial shear)
    ])

    history = []
    t_vals = []

    t = 0.0

    for _ in range(T):
        history.append(y.copy())
        t_vals.append(t)

        k1 = rhs(t, y)
        k2 = rhs(t + dt/2, y + dt*k1/2)
        k3 = rhs(t + dt/2, y + dt*k2/2)
        k4 = rhs(t + dt, y + dt*k3)

        y = y + dt * (k1 + 2*k2 + 2*k3 + k4)/6
        t += dt

        # Stop when a ~ 1
        if y[0] >= 1.0:
            break

    history = np.array(history)
    t_vals = np.array(t_vals)

    a = history[:, 0]
    OmegaX = history[:, 1]
    theta = history[:, 2]
    omega = history[:, 3]
    order = history[:, 4]
    fb = history[:, 5]
    sigma = history[:, 6]

    H_vals = np.array([H_from(a[i], OmegaX[i], sigma[i]) for i in range(len(a))])

    # Derived quantities
    Omega_sigma = sigma**2 / (6 * H_vals**2)
    Omega_r = Omega_r0 * a**(-4) / (H_vals**2 / H0**2)
    Omega_m = Omega_m0 * a**(-3) / (H_vals**2 / H0**2)
    Omega_k = Omega_k0 * a**(-2) / (H_vals**2 / H0**2)

    constraint = Omega_r + Omega_m + Omega_k + OmegaX + Omega_sigma - 1

    return (
        t_vals,
        a,
        H_vals,
        theta,
        omega,
        order,
        fb,
        OmegaX,
        sigma,
        Omega_sigma,
        constraint
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    (
        t,
        a,
        H,
        theta,
        omega,
        order,
        fb,
        OmegaX,
        sigma,
        Omega_sigma,
        constraint
    ) = run_model()

    # -----------------------------
    # PLOTS
    # -----------------------------

    plt.figure()
    plt.plot(t, a)
    plt.title("Scale factor a(t)")
    plt.show()

    plt.figure()
    plt.plot(t, H)
    plt.title("Hubble rate H(t)")
    plt.show()

    plt.figure()
    plt.plot(t, theta, label="theta")
    plt.plot(t, omega, label="omega")
    plt.legend()
    plt.title("Rotation sector")
    plt.show()

    plt.figure()
    plt.plot(t, sigma)
    plt.title("Shear σ(t) (Bianchi I)")
    plt.show()

    plt.figure()
    plt.plot(t, Omega_sigma)
    plt.title("Shear density parameter Ω_sigma")
    plt.show()

    plt.figure()
    plt.plot(t, constraint)
    plt.title("Friedmann constraint residual")
    plt.show()

    print("Final Ω_sigma:", Omega_sigma[-1])
    print("Final constraint residual:", constraint[-1])