#!/usr/bin/env python3
"""
Self-regulating layered toy cosmology with a geometric expansion floor (Option A).
# python selfreg_bigbang_floor.py --plots
State:
  V(t): volume proxy
  A(t): coherence proxy in (0,1]
Derived:
  a(t) = (V(t)/V0)^(1/d_eff)
  H(t) = log(V(t+1)/V(t))
  Z(t) = -log(A(t))

Dynamics:
  rho(t) = V(t)/V0

  gamma_raw(t) = exp( g0 + gA*(A(t)-A_star) - gR*log(1+rho(t)) )

  Option A floor:
    gamma(t) = gamma_min + gamma_raw(t)    (default gamma_min = 1.0)

  V(t+1) = ceil( gamma(t) * V(t) )

  A(t+1) = clip( A(t) + dt*( sH*H(t)*(1-A(t)) - sR*rho(t)*A(t) ), eps, 1 )
"""

import argparse
import math
import numpy as np


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def simulate(
    T=120,
    V0=10,
    d_eff=3.0,
    # growth parameters
    g0=0.02,
    gA=1.2,
    A_star=0.35,
    gR=0.45,
    gamma_min=1.0,  # <-- geometric floor (Option A)
    # coherence parameters
    dt=1.0,
    sH=0.35,
    sR=0.0015,
    eps=1e-9,
    A0=0.98,
):
    V = np.zeros(T + 1, dtype=float)
    A = np.zeros(T + 1, dtype=float)
    H = np.zeros(T + 1, dtype=float)
    Z = np.zeros(T + 1, dtype=float)

    V[0] = V0
    A[0] = A0
    Z[0] = -math.log(max(A[0], eps))

    for t in range(T):
        rho = V[t] / V0

        # Growth with geometric floor
        gamma_raw = math.exp(g0 + gA * (A[t] - A_star) - gR * math.log(1.0 + rho))
        gamma = gamma_min + gamma_raw
        V[t + 1] = max(1.0, math.ceil(gamma * V[t]))

        # Hubble-like rate
        H[t] = math.log(V[t + 1] / max(V[t], 1e-12))

        # Coherence update (kernel responds to expansion + density)
        dA = dt * (sH * H[t] * (1.0 - A[t]) - sR * rho * A[t])
        A[t + 1] = clip(A[t] + dA, eps, 1.0)

        Z[t + 1] = -math.log(max(A[t + 1], eps))

    H[T] = H[T - 1]
    a = (V / V0) ** (1.0 / d_eff)
    return V, a, A, H, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--V0", type=float, default=10)
    ap.add_argument("--d_eff", type=float, default=3.0)

    ap.add_argument("--g0", type=float, default=0.02)
    ap.add_argument("--gA", type=float, default=1.2)
    ap.add_argument("--A_star", type=float, default=0.35)
    ap.add_argument("--gR", type=float, default=0.45)
    ap.add_argument("--gamma_min", type=float, default=1.0)

    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sH", type=float, default=0.35)
    ap.add_argument("--sR", type=float, default=0.0015)
    ap.add_argument("--A0", type=float, default=0.98)

    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    V, a, A, H, Z = simulate(
        T=args.T, V0=args.V0, d_eff=args.d_eff,
        g0=args.g0, gA=args.gA, A_star=args.A_star, gR=args.gR,
        gamma_min=args.gamma_min,
        dt=args.dt, sH=args.sH, sR=args.sR,
        A0=args.A0,
    )

    t = np.arange(args.T + 1)

    print("Final values:")
    print("  V(T) =", V[-1])
    print("  a(T) =", a[-1])
    print("  A(T) =", A[-1])
    print("  Z(T) =", Z[-1])
    print("  mean H(last 20) =", float(np.mean(H[-20:])))

    if args.plots:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t, V)
        plt.xlabel("layer t")
        plt.ylabel("V(t)")
        plt.title("Volume proxy V(t) (self-regulating, geometric floor)")
        plt.show()

        plt.figure()
        plt.plot(t, a)
        plt.xlabel("layer t")
        plt.ylabel("a_proxy(t)")
        plt.title("Scale factor proxy a(t)")
        plt.show()

        plt.figure()
        plt.plot(t, H)
        plt.xlabel("layer t")
        plt.ylabel("H(t)=log(V(t+1)/V(t))")
        plt.title("Hubble-like rate H(t)")
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
        plt.ylabel("Z_like(t)=-log(A(t))")
        plt.title("Redshift-like attenuation Z(t)")
        plt.show()

        plt.figure()
        plt.plot(np.log(t[1:]), np.log(a[1:]))
        plt.xlabel("log t")
        plt.ylabel("log a(t)")
        plt.title("log-log check: power-law if ~linear")
        plt.show()


if __name__ == "__main__":
    main()
