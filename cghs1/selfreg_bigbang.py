#!/usr/bin/env python3
"""
Self-regulating layered "Big Bang" toy model from a coupled (geometry, coherence) system.
python3 selfreg_bigbang.py --plots

State variables per layer t:
  V(t): event volume proxy (events in layer)
  A(t): coherence proxy in [0,1] (summarizes kernel structure)
Derived:
  a(t) = (V(t)/V(0))^(1/d_eff)
  H(t) = log(V(t+1)/V(t))  (discrete Hubble-like rate)
  Z(t) = -log(A(t))        (redshift-like attenuation proxy)

Dynamics:
  Growth factor:
    gamma(t) = exp( g0 + gA*(A(t)-A*) - gR*log(1+rho(t)) )
    V(t+1) = ceil(gamma(t)*V(t))

  Coherence update:
    A(t+1) = clip( A(t) + dt*( sH*H(t)*(1-A(t)) - sR*rho(t)*A(t) ), eps, 1)

Interpretation:
  - A high coherence can drive rapid expansion early.
  - Density growth brakes expansion and damps coherence.
  - Expansion can (partially) rebuild coherence.
  - Together these can generate self-regulating behavior (attractor).

This is a sandbox, not a cosmology solver.
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
    g0=0.06,
    gA=1.8,
    A_star=0.35,
    gR=0.55,
    # coherence parameters
    dt=1.0,
    sH=0.85,
    sR=0.0035,
    eps=1e-9
):
    V = np.zeros(T+1, dtype=float)
    A = np.zeros(T+1, dtype=float)
    H = np.zeros(T+1, dtype=float)
    Z = np.zeros(T+1, dtype=float)

    V[0] = V0
    A[0] = 0.98   # start highly coherent "near-bang"
    Z[0] = -math.log(max(A[0], eps))

    for t in range(T):
        rho = V[t] / V0

        # growth factor (self-regulating)
        gamma = math.exp(g0 + gA*(A[t] - A_star) - gR*math.log(1.0 + rho))
        V[t+1] = max(1.0, math.ceil(gamma * V[t]))

        # discrete hubble-like rate
        H[t] = math.log(V[t+1] / max(V[t], 1e-12))

        # coherence update (kernel responds to expansion + density)
        dA = dt*( sH*H[t]*(1.0 - A[t]) - sR*rho*A[t] )
        A[t+1] = clip(A[t] + dA, eps, 1.0)

        Z[t+1] = -math.log(max(A[t+1], eps))

    # define H[T] (last point) for plotting continuity
    H[T] = H[T-1]
    a = (V / V0) ** (1.0 / d_eff)
    return V, a, A, H, Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--V0", type=float, default=10)
    ap.add_argument("--d_eff", type=float, default=3.0)

    ap.add_argument("--g0", type=float, default=0.06)
    ap.add_argument("--gA", type=float, default=1.8)
    ap.add_argument("--A_star", type=float, default=0.35)
    ap.add_argument("--gR", type=float, default=0.55)

    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sH", type=float, default=0.85)
    ap.add_argument("--sR", type=float, default=0.0035)

    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    V, a, A, H, Z = simulate(
        T=args.T, V0=args.V0, d_eff=args.d_eff,
        g0=args.g0, gA=args.gA, A_star=args.A_star, gR=args.gR,
        dt=args.dt, sH=args.sH, sR=args.sR
    )

    t = np.arange(args.T+1)

    print("Final values:")
    print("  V(T) =", V[-1])
    print("  a(T) =", a[-1])
    print("  A(T) =", A[-1])
    print("  Z(T) =", Z[-1])
    print("  mean H(last 20) =", float(np.mean(H[-20:])))

    if args.plots:
        import matplotlib.pyplot as plt

        # Plot V(t)
        plt.figure()
        plt.plot(t, V)
        plt.xlabel("layer t")
        plt.ylabel("V(t)")
        plt.title("Volume proxy V(t) (self-regulating)")
        plt.show()

        # Plot a(t)
        plt.figure()
        plt.plot(t, a)
        plt.xlabel("layer t")
        plt.ylabel("a_proxy(t)")
        plt.title("Scale factor proxy a(t) (self-regulating)")
        plt.show()

        # Plot H(t)
        plt.figure()
        plt.plot(t, H)
        plt.xlabel("layer t")
        plt.ylabel("H(t)=log(V(t+1)/V(t))")
        plt.title("Hubble-like rate H(t) (self-regulating)")
        plt.show()

        # Plot A(t)
        plt.figure()
        plt.plot(t, A)
        plt.xlabel("layer t")
        plt.ylabel("A(t)")
        plt.title("Coherence proxy A(t) (kernel summary)")
        plt.show()

        # Plot Z(t)
        plt.figure()
        plt.plot(t, Z)
        plt.xlabel("layer t")
        plt.ylabel("Z_like(t)=-log(A(t))")
        plt.title("Redshift-like attenuation Z(t)")
        plt.show()

        # Optional: log-log plot to check power-law-ish behavior
        # (skip t=0)
        plt.figure()
        plt.plot(np.log(t[1:]), np.log(a[1:]))
        plt.xlabel("log t")
        plt.ylabel("log a(t)")
        plt.title("log-log check: power-law if ~linear")
        plt.show()

if __name__ == "__main__":
    main()
