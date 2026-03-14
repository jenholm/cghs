#!/usr/bin/env python3
"""
Null-boundary toy cosmology (Option B): natural inflation exit via coherence-gated braking.

Mechanism:
- Early coherence A ~ 1 -> braking gate ~ 0 -> rapid growth (inflation-like).
- As coherence decays -> braking turns on smoothly -> graceful inflation exit.

Toy dynamics:
  rho_t = V_t / (1+t)^3                     (dilution proxy)
  gate  = (1 - A_t)^q                        (coherence gate)

  V_{t+1} = V_t * base^(1 + kappa*A_t)
            * exp(-B * gate * rho_t^r)
            + geometric_floor

  A_{t+1} = A_t * exp(-beta*rho_t - beta2*V_t)

Derived proxies:
  a(t) = (V/V0)^(1/3)
  H(t) = Δ ln a
  Z(t) = -ln A
  p(t) = local slope in log a vs log t  (a ~ t^p locally)
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Utility
# ---------------------------

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def local_power_exponent(t, a, win=11):
    """
    Estimate local p(t) where a ~ t^p via sliding regression.
    """
    n = len(t)
    p = np.full(n, np.nan, dtype=float)
    half = win // 2
    eps = 1e-12

    for i in range(n):
        j0 = max(1, i - half)        # exclude t=0
        j1 = min(n, i + half + 1)

        tt = t[j0:j1]
        aa = a[j0:j1]

        if np.any(tt <= 0) or np.any(aa <= 0):
            continue

        x = np.log(tt + eps)
        y = np.log(aa + eps)

        xm, ym = x.mean(), y.mean()
        denom = np.sum((x - xm)**2)
        if denom < 1e-14:
            continue

        p[i] = np.sum((x - xm)*(y - ym)) / denom

    return p


# ---------------------------
# Core Simulation (Option B)
# ---------------------------

def run_sim(
    T=140,
    V0=10.0,

    # growth
    base=1.13,
    kappa=1.25,

    # coherence
    A0=0.999,
    beta=0.012,
    beta2=0.00003,
    A_floor=1e-9,

    # braking
    brake_strength=2.2,
    brake_power=2.4,
    rho_power=1.0,

    geometric_floor=2.0
):

    t = np.arange(T + 1, dtype=float)
    V = np.zeros(T + 1)
    A = np.zeros(T + 1)

    V[0] = V0
    A[0] = A0

    for i in range(T):

        # dilution density proxy
        rho = V[i] / ((1.0 + t[i])**3)

        # coherence gate
        gate = (max(0.0, 1.0 - A[i]))**brake_power

        growth_factor = base**(1.0 + kappa*A[i])
        brake_factor  = np.exp(-brake_strength * gate * (rho**rho_power))

        V_next = V[i] * growth_factor * brake_factor + geometric_floor
        V[i+1] = max(geometric_floor, V_next)

        # coherence decay
        A_next = A[i] * np.exp(-beta*rho - beta2*V[i])
        A[i+1] = clip(A_next, A_floor, 1.0)

    # ---------------------------
    # Derived quantities
    # ---------------------------

    eps = 1e-12
    a = (V / V[0])**(1.0/3.0)

    ln_a = np.log(a + eps)
    H = np.zeros_like(a)
    H[1:] = ln_a[1:] - ln_a[:-1]

    Z = -np.log(A + eps)

    p = local_power_exponent(t, a)

    return t, V, a, H, A, Z, p


# ---------------------------
# Run + Plot
# ---------------------------

if __name__ == "__main__":

    t, V, a, H, A, Z, p = run_sim()

    plt.figure()
    plt.plot(t, V)
    plt.title("Volume proxy V(t) (Option B)")
    plt.xlabel("layer t")
    plt.ylabel("V(t)")
    plt.show()

    plt.figure()
    plt.plot(t, a)
    plt.title("Scale factor proxy a(t)")
    plt.xlabel("layer t")
    plt.ylabel("a(t)")
    plt.show()

    plt.figure()
    plt.plot(t, H)
    plt.title("Hubble-like rate H(t) = Δ ln a")
    plt.xlabel("layer t")
    plt.ylabel("H(t)")
    plt.show()

    plt.figure()
    plt.plot(t, A)
    plt.title("Coherence proxy A(t)")
    plt.xlabel("layer t")
    plt.ylabel("A(t)")
    plt.show()

    plt.figure()
    plt.plot(t, Z)
    plt.title("Redshift-like attenuation Z(t)")
    plt.xlabel("layer t")
    plt.ylabel("Z(t)")
    plt.show()

    plt.figure()
    plt.plot(t, p)
    plt.title("Local power-law exponent p(t)")
    plt.xlabel("layer t")
    plt.ylabel("p(t)")
    plt.show()