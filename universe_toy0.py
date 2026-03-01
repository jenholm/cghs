
#!/usr/bin/env python3
"""
Universe toy model: self-regulating layered growth with a geometric floor (Option A),
plus diagnostics:

A) w_eff(t) from discrete FRW identity
B) local power-law exponent p(t) (sliding log-log fit)
C) late-time exponential rate h_late (log a vs t fit)
D) toy perturbation growth delta(t)
E) optional parameter sweep (quick phase scan)

Key update vs prior version:
- density braking in the growth exponent can be power-law: -(1+rho)^alpha_rho
  (instead of weak log braking), which helps produce "era"-like behavior.

Run:
  python3 universe_selfreg_floor_eras.py --plots

Try tuning:
  --alpha_rho 1.0 --gR 0.35
  --alpha_rho 1.3 --gR 0.40   (stronger early braking)
  --alpha_rho 0.7 --gR 0.30   (weaker early braking)

Notes:
- This is a discrete toy. Interpret diagnostics as qualitative proxies.
"""

import argparse
import math
import numpy as np


# ============================================================
# Utilities
# ============================================================

def clip(x, lo, hi):
    return max(lo, min(x, hi))


# ============================================================
# 1) Core simulation (Option A geometric floor)
# ============================================================

def simulate(
    T=120,
    V0=10.0,
    d_eff=3.0,

    # growth kernel parameters
    g0=0.02,
    gA=1.2,
    A_star=0.35,
    gR=0.35,
    alpha_rho=1.0,      # <-- NEW: power-law density braking exponent
    use_log_brake=False, # if True, use -gR*log(1+rho) instead of power braking
    gamma_min=1.0,      # <-- Option A geometric floor

    # coherence parameters
    dt=1.0,
    sH=0.35,
    sR=0.0015,
    eps=1e-12,
    A0=0.98,
):
    """
    State:
      V(t): volume proxy
      A(t): coherence proxy in (0,1]
    Derived:
      a(t) = (V/V0)^(1/d_eff)
      H(t) = log(V(t+1)/V(t))
      Z(t) = -log(A(t))

    Density proxy:
      rho(t) = V(t)/V0   (in this toy, "density-like" or "load" measure)

    Growth:
      gamma_raw(t) = exp( g0 + gA*(A-A_star) - braking(rho) )
      gamma(t) = gamma_min + gamma_raw(t)  (Option A floor)
      V(t+1) = ceil( gamma(t) * V(t) )

    Coherence update:
      A(t+1) = clip( A(t) + dt*( sH*H(t)*(1-A(t)) - sR*rho(t)*A(t) ), eps, 1 )
    """
    V = np.zeros(T + 1, dtype=float)
    A = np.zeros(T + 1, dtype=float)
    H = np.zeros(T + 1, dtype=float)
    Z = np.zeros(T + 1, dtype=float)

    V[0] = V0
    A[0] = clip(A0, eps, 1.0)
    Z[0] = -math.log(max(A[0], eps))

    for t in range(T):
        rho = V[t] / V0

        if use_log_brake:
            brake = gR * math.log(1.0 + rho)
        else:
            # power-law density braking: strong early, fades later
            brake = gR * ((1.0 + rho) ** alpha_rho)

        gamma_raw = math.exp(g0 + gA * (A[t] - A_star) - brake)
        gamma = gamma_min + gamma_raw

        V[t + 1] = max(1.0, math.ceil(gamma * V[t]))

        H[t] = math.log(V[t + 1] / max(V[t], eps))

        dA = dt * (sH * H[t] * (1.0 - A[t]) - sR * rho * A[t])
        A[t + 1] = clip(A[t] + dA, eps, 1.0)

        Z[t + 1] = -math.log(max(A[t + 1], eps))

    H[T] = H[T - 1]
    return V, A, H, Z


def derived_scale_factor(V, V0=10.0, d_eff=3.0):
    return (V / V0) ** (1.0 / d_eff)


# ============================================================
# 2) Diagnostics: w_eff(t)
# ============================================================

def compute_w_eff(a, H, burn_in=5, eps=1e-12):
    """
    Effective equation of state proxy:
      w_eff = -1 - (2/3) d ln H / d ln a

    Uses centered differences where possible, and ignores early burn-in layers.
    Returns:
      t_idx (indices where defined), w_eff
    """
    a = np.asarray(a, dtype=float)
    H = np.asarray(H, dtype=float)

    lnH = np.log(np.maximum(H, eps))
    lna = np.log(np.maximum(a, eps))

    # centered differences (avoid endpoints)
    dlnH = lnH[2:] - lnH[:-2]
    dlna = lna[2:] - lna[:-2]
    slope = dlnH / np.maximum(dlna, eps)

    w = -1.0 - (2.0 / 3.0) * slope

    t_idx = np.arange(1, len(a) - 1)  # corresponds to centered index

    mask = t_idx >= burn_in
    return t_idx[mask], w[mask]


# ============================================================
# 3) Diagnostics: local power-law exponent p(t)
# ============================================================

def sliding_powerlaw_exponent(a, window=15, eps=1e-12):
    """
    Fit log a ~ p log t + b over sliding windows.
    Returns p_hat array (NaN where not defined).
    """
    a = np.asarray(a, dtype=float)
    T = len(a) - 1
    p_hat = np.full(T + 1, np.nan)

    for t0 in range(2, T - window):
        tt = np.arange(t0, t0 + window)
        x = np.log(tt)
        y = np.log(np.maximum(a[tt], eps))
        p = np.polyfit(x, y, 1)[0]
        p_hat[t0 + window // 2] = p

    return p_hat


# ============================================================
# 4) Diagnostics: late-time exponential rate
# ============================================================

def fit_exponential_rate(a, t_min=60, eps=1e-12):
    """
    Fit log a ~ h t + b over t >= t_min.
    Returns (h, b).
    """
    a = np.asarray(a, dtype=float)
    tt = np.arange(t_min, len(a))
    y = np.log(np.maximum(a[tt], eps))
    h, b = np.polyfit(tt, y, 1)
    return float(h), float(b)


# ============================================================
# 5) Diagnostics: toy perturbation growth
# ============================================================

def perturbation_growth(A, H, g0=0.08, kappa=4.0):
    """
    Toy structure growth:
      delta_{t+1} = delta_t * (1 + g(t))
      g(t) = g0*(1-A)/(1+kappa*H)
    """
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    T = len(A) - 1
    delta = np.zeros(T + 1, dtype=float)
    delta[0] = 1e-3

    for t in range(T):
        g = g0 * (1.0 - A[t]) / (1.0 + kappa * H[t])
        delta[t + 1] = delta[t] * (1.0 + g)

    return delta


# ============================================================
# 6) Optional: quick sweep
# ============================================================

def sweep(params_base, grid, T=120):
    """
    params_base: dict of simulate() kwargs (excluding T)
    grid: dict param_name -> list of values
    Returns list of dict rows with summaries.
    """
    rows = []
    keys = list(grid.keys())

    def run_one(overrides):
        kw = dict(params_base)
        kw.update(overrides)
        V, A, H, Z = simulate(T=T, **kw)
        a = derived_scale_factor(V, V0=kw["V0"], d_eff=kw["d_eff"])
        t_w, w = compute_w_eff(a, H, burn_in=5)
        w_late = float(np.mean(w[-20:])) if len(w) >= 20 else float(np.mean(w))
        H_late = float(np.mean(H[-20:]))
        A_late = float(np.mean(A[-20:]))
        oscZ = float(np.std(Z[-40:])) if len(Z) >= 40 else float(np.std(Z))
        return {
            **overrides,
            "V_final": float(V[-1]),
            "H_late": H_late,
            "w_late": w_late,
            "A_late": A_late,
            "oscZ": oscZ,
        }

    # simple cartesian product
    def rec(i, current):
        if i == len(keys):
            rows.append(run_one(current))
            return
        k = keys[i]
        for v in grid[k]:
            cur2 = dict(current)
            cur2[k] = v
            rec(i + 1, cur2)

    rec(0, {})
    return rows


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--V0", type=float, default=10.0)
    ap.add_argument("--d_eff", type=float, default=3.0)

    # growth
    ap.add_argument("--g0", type=float, default=0.02)
    ap.add_argument("--gA", type=float, default=1.2)
    ap.add_argument("--A_star", type=float, default=0.35)
    ap.add_argument("--gR", type=float, default=0.35)
    ap.add_argument("--alpha_rho", type=float, default=1.0)
    ap.add_argument("--use_log_brake", action="store_true")
    ap.add_argument("--gamma_min", type=float, default=1.0)

    # coherence
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sH", type=float, default=0.35)
    ap.add_argument("--sR", type=float, default=0.0015)
    ap.add_argument("--A0", type=float, default=0.98)

    # diagnostics
    ap.add_argument("--burn_in", type=int, default=5)
    ap.add_argument("--p_window", type=int, default=15)
    ap.add_argument("--late_fit_start", type=int, default=60)

    # sweep
    ap.add_argument("--do_sweep", action="store_true")
    ap.add_argument("--plots", action="store_true")

    args = ap.parse_args()

    params = dict(
        V0=args.V0,
        d_eff=args.d_eff,
        g0=args.g0,
        gA=args.gA,
        A_star=args.A_star,
        gR=args.gR,
        alpha_rho=args.alpha_rho,
        use_log_brake=args.use_log_brake,
        gamma_min=args.gamma_min,
        dt=args.dt,
        sH=args.sH,
        sR=args.sR,
        A0=args.A0,
    )

    V, A, H, Z = simulate(T=args.T, **params)
    a = derived_scale_factor(V, V0=args.V0, d_eff=args.d_eff)

    t = np.arange(args.T + 1)

    # Diagnostics
    t_w, w_eff = compute_w_eff(a, H, burn_in=args.burn_in)
    p_hat = sliding_powerlaw_exponent(a, window=args.p_window)

    h_late, _ = fit_exponential_rate(a, t_min=args.late_fit_start)
    delta = perturbation_growth(A, H)

    print("Summary:")
    print("  V(T)            =", float(V[-1]))
    print("  a(T)            =", float(a[-1]))
    print("  A(T)            =", float(A[-1]))
    print("  H_late (mean20) =", float(np.mean(H[-20:])))
    print("  w_late (mean20) =", float(np.mean(w_eff[-20:])) if len(w_eff) >= 20 else float(np.mean(w_eff)))
    print("  h_late          =", h_late)
    print("  use_log_brake   =", bool(args.use_log_brake))
    print("  alpha_rho       =", args.alpha_rho)

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
        plt.ylabel("a(t)")
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
        plt.ylabel("Z(t)=-log(A(t))")
        plt.title("Redshift-like attenuation Z(t)")
        plt.show()

        plt.figure()
        plt.plot(t_w, w_eff)
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
        plt.plot(t, np.log(np.maximum(delta, 1e-30)))
        plt.xlabel("layer t")
        plt.ylabel("log delta(t)")
        plt.title("Toy structure growth log(delta(t))")
        plt.show()

        plt.figure()
        plt.plot(np.log(t[1:]), np.log(np.maximum(a[1:], 1e-12)))
        plt.xlabel("log t")
        plt.ylabel("log a(t)")
        plt.title("log-log check: power-law if ~linear")
        plt.show()

    if args.do_sweep:
        # quick grid (small by default)
        base = dict(params)
        grid = {
            "gR": [0.30, 0.35, 0.40],
            "alpha_rho": [0.7, 1.0, 1.3],
            "gA": [1.0, 1.2, 1.4],
        }
        rows = sweep(base, grid, T=args.T)

        # Print top candidates closest to w_late ~ -1 but not crazy oscillatory
        rows_sorted = sorted(rows, key=lambda r: (abs(r["w_late"] + 1.0), r["oscZ"]))
        print("\nTop sweep results (closest to w_late ~ -1, low oscZ):")
        for r in rows_sorted[:10]:
            print(r)


if __name__ == "__main__":
    main()
