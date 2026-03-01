import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Simulation core
# ============================================================

def clip(x, lo, hi):
    return max(lo, min(x, hi))


def simulate(
    T=120,
    gA=1.0,     # coherence decay coupling
    gR=0.45,    # geometric response strength
    sR=0.0015,  # density backreaction
    A0=0.98
):
    """
    Self-regulating layered growth model.
    Returns arrays: V, a, A, H, Z
    """

    V = np.zeros(T+1)
    a = np.zeros(T+1)
    A = np.zeros(T+1)
    H = np.zeros(T+1)
    Z = np.zeros(T+1)

    V[0] = 10.0
    A[0] = A0
    a[0] = V[0]**(1/3)

    for t in range(T):

        # density proxy
        rho = 1.0 / max(V[t], 1e-12)

        # coherence evolution
        A[t+1] = A[t] * (1.0 - gA * rho)
        A[t+1] = clip(A[t+1], 1e-6, 1.0)

        # geometric growth feedback
        growth = gR * (1.0 - A[t]) - sR * rho
        V[t+1] = V[t] * (1.0 + growth)

        if V[t+1] < 1e-6:
            V[t+1] = 1e-6

        a[t+1] = V[t+1]**(1/3)

        H[t] = np.log(V[t+1] / V[t])
        Z[t] = -np.log(A[t])

    H[T] = H[T-1]
    Z[T] = Z[T-1]

    return V, a, A, H, Z


# ============================================================
# 2. Effective equation of state
# ============================================================

def compute_w_eff(a, H):
    eps = 1e-12
    lnH = np.log(np.maximum(H, eps))
    lna = np.log(np.maximum(a, eps))

    dlnH = lnH[1:] - lnH[:-1]
    dlna = lna[1:] - lna[:-1]

    dlnH_dlna = dlnH / np.maximum(dlna, eps)

    w_eff = -1.0 - (2.0/3.0) * dlnH_dlna
    return w_eff


# ============================================================
# 3. Local power-law exponent
# ============================================================

def sliding_powerlaw_exponent(a, window=15):
    T = len(a) - 1
    p_hat = np.full(T+1, np.nan)

    for t0 in range(2, T-window):
        tt = np.arange(t0, t0+window)
        x = np.log(tt)
        y = np.log(a[tt])
        p = np.polyfit(x, y, 1)[0]
        p_hat[t0 + window//2] = p

    return p_hat


# ============================================================
# 4. Late-time exponential fit
# ============================================================

def fit_exponential_rate(a, t_min):
    tt = np.arange(t_min, len(a))
    y = np.log(a[tt])
    h, b = np.polyfit(tt, y, 1)
    return h


# ============================================================
# 5. Perturbation growth proxy
# ============================================================

def perturbation_growth(A, H, g0=0.08, kappa=4.0):
    T = len(A) - 1
    delta = np.zeros(T+1)
    delta[0] = 1e-3

    for t in range(T):
        g = g0 * (1.0 - A[t]) / (1.0 + kappa * H[t])
        delta[t+1] = delta[t] * (1.0 + g)

    return delta


# ============================================================
# 6. Run model
# ============================================================

V, a, A, H, Z = simulate()

w_eff = compute_w_eff(a, H)
p_hat = sliding_powerlaw_exponent(a)
h_late = fit_exponential_rate(a, t_min=len(a)//2)
delta = perturbation_growth(A, H)

t = np.arange(len(a))


# ============================================================
# 7. Plots (one per figure)
# ============================================================

plt.figure()
plt.plot(t, V)
plt.xlabel("layer t")
plt.ylabel("V(t)")
plt.title("Volume proxy V(t)")
plt.show()

plt.figure()
plt.plot(t, a)
plt.xlabel("layer t")
plt.ylabel("a(t)")
plt.title("Scale factor proxy a(t)")
plt.show()

plt.figure()
plt.plot(t, H)
plt.xlabel("layer t")
plt.ylabel("H(t)")
plt.title("Hubble-like rate H(t)")
plt.show()

plt.figure()
plt.plot(t[:-1], w_eff)
plt.xlabel("layer t")
plt.ylabel("w_eff(t)")
plt.title("Effective equation of state w_eff(t)")
plt.show()

plt.figure()
plt.plot(t, p_hat)
plt.xlabel("layer t")
plt.ylabel("p(t)")
plt.title("Local power-law exponent p(t)")
plt.show()

plt.figure()
plt.plot(t, A)
plt.xlabel("layer t")
plt.ylabel("A(t)")
plt.title("Coherence proxy A(t)")
plt.show()

plt.figure()
plt.plot(t, Z)
plt.xlabel("layer t")
plt.ylabel("Z(t)")
plt.title("Redshift-like attenuation Z(t)")
plt.show()

plt.figure()
plt.plot(t, np.log(delta))
plt.xlabel("layer t")
plt.ylabel("log delta(t)")
plt.title("Toy structure growth")
plt.show()


print("Late-time exponential rate h ≈", h_late)
