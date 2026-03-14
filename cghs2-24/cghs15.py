#!/usr/bin/env python3
"""
python cghs15.py --mode frw --prefix run1
python cghs15.py --mode sandbox --prefix run1

photonuniverse_next_direction.py

A two-track "Photonland" scaffold built from the Feb 21–27 logs:

MODE 1 (default): FRW-reduced toy dark sector in e-fold time N=ln a
- Integrates a small coupled system with:
    OmegaX(N): dark-energy density parameter (effective)
    theta(N): internal twist coordinate (do NOT over-interpret as global vorticity)
    omega(N): dtheta/dt proxy
    O(N): coherence/order proxy in [0,1]
- Stable RK4 integration in N (addresses tiny-a stability issues).
- Produces diagnostics:
    E(z)=H(z)/H0, w_X(z), closure drift proxy, and plots.

MODE 2 (--sandbox): Layered causal growth + PSD kernel K = Phi Phi^T
- Generates a layered DAG (cosmic slices), propagates features Phi forward,
  and computes per-layer "cosmology proxies" and inter-layer correlation summaries.

This is a *directional* harness. It is not "the final theory".
It exists so you can iterate cleanly and decide what to promote into an APS-legible action.

Author: Jake + Novella
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

# -----------------------------
# Utilities
# -----------------------------

# ============================================================
# 3D Metric Tube (Works with photonuniverse_next_direction.py)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def _compress01(x, p_hi=99.0, power=0.08, floor=0.02):
    """
    Robust normalization + dynamic range compression to [floor, 1].
    """
    x = np.asarray(x, dtype=float)
    hi = np.percentile(x[np.isfinite(x)], p_hi) if np.any(np.isfinite(x)) else 1.0
    x = x / (hi + 1e-30)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, power)
    x = np.maximum(x, floor)
    return x

def plot_metric_tube_from_series(
    N,
    a,
    Omega=None,
    psi=None,
    color_by=None,
    *,
    max_points=2500,
    phi_res=72,
    radius_power=0.08,
    radius_floor=0.02,
    fade_lo=0.30,
    fade_hi=0.85,
    cmap=cm.plasma,
    elev=20,
    azim=35,
    outpath="metric_tube_3D.png",
    title="Metric Tube"
):
    """
    Create a 3D "metric tube" surface.

    Parameters
    ----------
    N : array-like
        Longitudinal coordinate (e.g., e-fold time N = ln a or layer index t).
    a : array-like
        Scale proxy.
    Omega : array-like or None
        Conformal factor proxy. If None, uses 1.
    psi : array-like or None
        Hilbert state proxy for twist. Accepts:
          - shape (T,2+) for Arg(psi[:,1], psi[:,0])
          - shape (T,) for direct phase (already an angle)
          - None -> no twist
    color_by : array-like or None
        Scalar to color along time (e.g., wX, H, Z). If None, uses zeros.

    Notes
    -----
    - Radius uses R_profile = a * Omega, then robust-compresses.
    - Twist uses Arg(psi) if possible, else zeros.
    - Colors are per-time, broadcast around the circumference.
    """
    N = np.asarray(N, dtype=float)
    a = np.asarray(a, dtype=float)

    if Omega is None:
        Omega = np.ones_like(a)
    else:
        Omega = np.asarray(Omega, dtype=float)

    if color_by is None:
        color_by = np.zeros_like(a)
    else:
        color_by = np.asarray(color_by, dtype=float)

    # Downsample if large
    T = N.size
    if T > max_points:
        idx = np.linspace(0, T - 1, max_points).astype(int)
        N = N[idx]
        a = a[idx]
        Omega = Omega[idx]
        color_by = color_by[idx]
        if psi is not None:
            psi = np.asarray(psi)
            psi = psi[idx]

    # ----------------------------------------
    # Radius = a · Ω  (compressed)
    # ----------------------------------------
    R_profile = a * Omega
    R_profile = _compress01(R_profile, p_hi=99.0, power=radius_power, floor=radius_floor)

    # ----------------------------------------
    # Twist from psi
    # ----------------------------------------
    theta = np.zeros_like(R_profile)
    if psi is not None:
        psi = np.asarray(psi)
        if psi.ndim == 1:
            # interpret as phase directly
            theta = psi.astype(float)
        elif psi.ndim == 2 and psi.shape[1] >= 2:
            theta = np.arctan2(psi[:, 1], psi[:, 0])
        else:
            theta = np.zeros_like(R_profile)

    # unwrap to avoid sudden jumps
    theta = np.unwrap(theta)

    # ----------------------------------------
    # Build surface
    # ----------------------------------------
    phi = np.linspace(0, 2 * np.pi, phi_res)
    T_grid, Phi = np.meshgrid(N, phi, indexing="xy")

    R = np.outer(R_profile, np.ones_like(phi)).T
    twist = np.outer(theta, np.ones_like(phi)).T

    X = R * np.cos(Phi + twist)
    Y = R * np.sin(Phi + twist)
    Z = T_grid

    # ----------------------------------------
    # Color
    # ----------------------------------------
    c = np.asarray(color_by, dtype=float)
    c_finite = c[np.isfinite(c)]
    if c_finite.size == 0:
        cmin, cmax = 0.0, 1.0
    else:
        cmin = np.percentile(c_finite, 5)
        cmax = np.percentile(c_finite, 95)
        if abs(cmax - cmin) < 1e-12:
            cmax = cmin + 1.0

    c_clip = np.clip(c, cmin, cmax)
    norm = (c_clip - cmin) / (cmax - cmin + 1e-12)

    colors = cmap(norm)                     # (T,4)
    colors = np.tile(colors, (len(phi), 1, 1))  # (phi,T,4)

    # Fade with time
    t_min = float(np.min(N))
    t_max = float(np.max(N))
    t_norm = (N - t_min) / (t_max - t_min + 1e-12)
    alpha_profile = fade_lo + (fade_hi - fade_lo) * t_norm
    for i in range(len(phi)):
        colors[i, :, 3] = alpha_profile

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    # Centerline
    ax.plot([0]*len(N), [0]*len(N), N, linewidth=1.0)

    # Light wireframe
    ax.plot_wireframe(X, Y, Z, alpha=0.05, linewidth=0.3)

    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_zlabel("N (or layer index)")
    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath


# ------------------------------------------------------------
# Convenience wrappers for your two modes in the scaffold script
# ------------------------------------------------------------

def plot_metric_tube_frw(out: dict, *, outpath="metric_tube_frw.png"):
    """
    out is the dict returned by run_frw() in photonuniverse_next_direction.py
    Radius: a · Ω, with Ω proxied by OmegaX_norm (safe, already normalized-ish).
    Twist: Arg(Ψ) from (theta, omega) treated as 2D state vector.
    Color: wX
    """
    N = out["N"]
    a = out["a"]

    # Ω proxy: use OmegaX_norm (clamped) as a conformal-ish modulation
    Omega = np.clip(out.get("OmegaX_norm", np.ones_like(a)), 0.0, 2.0)

    # ψ proxy from internal phase-plane (theta, omega)
    psi = np.column_stack([out["theta"], out["omega"]])

    return plot_metric_tube_from_series(
        N=N,
        a=a,
        Omega=Omega,
        psi=psi,
        color_by=out["wX"],
        title="FRW Metric Tube\nRadius=a·Ω(proxy), Twist=Arg(theta+i·omega), Color=wX",
        outpath=outpath
    )

def plot_metric_tube_sandbox(stats: dict, *, outpath="metric_tube_sandbox.png"):
    """
    stats is the dict returned by sandbox_stats()
    Radius: a(t)
    Twist: optional phase from A(t) (monotone decay -> smooth twist)
    Color: H(t) by default (you can swap to Z easily)
    """
    t = np.arange(len(stats["a"]), dtype=float)
    a = stats["a"]

    # A(t) is positive; use its log as a smooth phase source
    A = np.clip(stats.get("A", np.ones_like(a)), 1e-15, 1.0)
    phase = np.unwrap(np.log(A))  # gentle twist

    return plot_metric_tube_from_series(
        N=t,
        a=a,
        Omega=None,
        psi=phase,
        color_by=stats.get("H", np.zeros_like(a)),
        title="Sandbox Metric Tube\nRadius=a(t), Twist=log A(t), Color=H(t)",
        outpath=outpath
    )

def sigmoid(x: float) -> float:
    # stable-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def safe_log(x: float, eps: float = 1e-15) -> float:
    return math.log(max(eps, x))

# ============================================================
# MODE 1: FRW-REDUCED TOY ODE IN N = ln a
# ============================================================

@dataclass
class FRWParams:
    # Background
    H0: float = 1.0
    Omega_r0: float = 9e-5
    Omega_m0: float = 0.30
    Omega_k0: float = 0.0

    # Dark sector knobs
    gamma_drag: float = 2.2      # damping on omega
    beta_theta: float = 0.8      # stiffness coupling theta
    alpha_drive: float = 0.15    # drive from dH/dN into omega (toy "binding to time metric")
    tau_O: float = 0.8           # relaxation time for coherence O

    # w_X shape controls (toy)
    w0: float = -1.0
    w1: float = 0.35
    w_theta: float = 0.7
    w_O: float = 1.2

    # Coherence target function (Phi)
    O_floor: float = 1e-6
    O_ceiling: float = 1.0
    Phi_bias: float = 0.15
    Phi_theta_scale: float = 1.2

    # Numerical
    N_min: float = math.log(1e-6)   # start scale factor
    N_max: float = 0.0             # today (a=1)
    dN: float = 2e-4

def E2_of_N(N: float, OmegaX: float, p: FRWParams) -> float:
    a = math.exp(N)
    # standard components
    rho_std = (
        p.Omega_r0 * a**(-4)
        + p.Omega_m0 * a**(-3)
        + p.Omega_k0 * a**(-2)
    )
    # effective closure: we let OmegaX fill the rest (toy)
    return max(1e-30, rho_std + OmegaX)

def H_of_N(N: float, OmegaX: float, p: FRWParams) -> float:
    return p.H0 * math.sqrt(E2_of_N(N, OmegaX, p))

def wX_of_state(theta: float, O: float, p: FRWParams) -> float:
    # A bounded, interpretable w_X that can approach -1 at late times.
    # (This is a toy; later you can replace this with action-derived pressure.)
    # Make wX tend to w0 when O is high and theta is small.
    x = p.w_theta * theta - p.w_O * (O - 0.5)
    return max(-1.5, min(0.5, p.w0 + p.w1 * math.tanh(x)))

def Phi_of_state(theta: float, p: FRWParams) -> float:
    # "desired" coherence given twist amplitude:
    # larger |theta| reduces Phi (more disorder), with a bias floor.
    return max(p.O_floor, min(p.O_ceiling, 1.0 - sigmoid(p.Phi_theta_scale * abs(theta)) + p.Phi_bias))

def dH_dN_numeric(N: float, OmegaX: float, p: FRWParams, h: float = 1e-4) -> float:
    # small finite difference for dH/dN
    Hp = H_of_N(N + h, OmegaX, p)
    Hm = H_of_N(N - h, OmegaX, p)
    return (Hp - Hm) / (2.0 * h)

def rhs_N(N: float, y: np.ndarray, p: FRWParams) -> Tuple[np.ndarray, float, float]:
    """
    y = [OmegaX, theta, omega, O]
    Returns dy/dN, H, wX
    """
    OmegaX, theta, omega, O = float(y[0]), float(y[1]), float(y[2]), float(y[3])

    H = H_of_N(N, OmegaX, p)
    wX = wX_of_state(theta, O, p)

    # Dark energy evolution in N:
    # dOmegaX/dN = -3(1+wX)OmegaX  (toy: in a flat-ish closure interpretation)
    dOmegaX = -3.0 * (1.0 + wX) * OmegaX

    # Twist phase dynamics:
    # In cosmic time: dtheta/dt = omega
    # Convert to N: dtheta/dN = (1/H) dtheta/dt = omega / H
    dtheta = omega / max(1e-30, H)

    # omega equation in N with:
    # - drag term ~ -gamma*omega
    # - stiffness term ~ -(beta/H)*theta
    # - drive term from the "time metric" dH/dN (your binding intuition)
    dH_dN = dH_dN_numeric(N, OmegaX, p)
    domega = -p.gamma_drag * omega - (p.beta_theta / max(1e-30, H)) * theta - p.alpha_drive * dH_dN

    # Coherence relaxes toward Phi(theta)
    Phi = Phi_of_state(theta, p)
    dO = (Phi - O) / max(1e-30, p.tau_O)

    return np.array([dOmegaX, dtheta, domega, dO], dtype=float), H, wX

def rk4_step(N: float, y: np.ndarray, dN: float, p: FRWParams) -> np.ndarray:
    k1, _, _ = rhs_N(N, y, p)
    k2, _, _ = rhs_N(N + 0.5*dN, y + 0.5*dN*k1, p)
    k3, _, _ = rhs_N(N + 0.5*dN, y + 0.5*dN*k2, p)
    k4, _, _ = rhs_N(N + dN, y + dN*k3, p)
    return y + (dN/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_frw(p: FRWParams, OmegaX0: float = 1e-6, theta0: float = 0.0, omega0: float = 0.0, O0: float = 0.95):
    # grid
    n_steps = int(math.ceil((p.N_max - p.N_min) / p.dN)) + 1
    Ns = np.linspace(p.N_min, p.N_max, n_steps)

    y = np.array([OmegaX0, theta0, omega0, O0], dtype=float)

    OmegaX = np.zeros_like(Ns)
    theta  = np.zeros_like(Ns)
    omega  = np.zeros_like(Ns)
    O      = np.zeros_like(Ns)
    H      = np.zeros_like(Ns)
    wX     = np.zeros_like(Ns)

    for i, N in enumerate(Ns):
        OmegaX[i], theta[i], omega[i], O[i] = y
        dy, Hi, wi = rhs_N(N, y, p)
        H[i], wX[i] = Hi, wi

        if i < len(Ns) - 1:
            y = rk4_step(N, y, p.dN, p)
            # keep O bounded
            y[3] = min(p.O_ceiling, max(p.O_floor, y[3]))
            # keep OmegaX non-negative
            y[0] = max(0.0, y[0])

    # Convert to z, E(z)
    a = np.exp(Ns)
    z = 1.0 / a - 1.0
    E = H / p.H0

    # Closure drift proxy: Omega_std + OmegaX - 1 (toy, since OmegaX is treated as a parameter)
    Omega_std = p.Omega_r0 * a**(-4) / (E**2) + p.Omega_m0 * a**(-3) / (E**2) + p.Omega_k0 * a**(-2) / (E**2)
    OmegaX_norm = OmegaX / (E**2)
    closure = (Omega_std + OmegaX_norm) - 1.0

    return {
        "N": Ns, "a": a, "z": z,
        "OmegaX": OmegaX, "OmegaX_norm": OmegaX_norm,
        "theta": theta, "omega": omega, "O": O,
        "H": H, "E": E, "wX": wX,
        "closure": closure,
    }

def save_frw_outputs(out: Dict[str, np.ndarray], prefix: str):
    import csv
    # Save a compact CSV
    path = f"{prefix}_frw_series.csv"
    keys = ["N","a","z","E","OmegaX","OmegaX_norm","theta","omega","O","wX","closure"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(out["N"])):
            w.writerow([float(out[k][i]) for k in keys])
    return path

def plot_frw(out: Dict[str, np.ndarray], prefix: str):
    import matplotlib.pyplot as plt

    z = out["z"]
    # sort by increasing z for nicer plots (integration went from high z -> 0)
    idx = np.argsort(z)
    z = z[idx]

    def save(x, y, title, ylabel, filename):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("z")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filename, dpi=250)
        plt.close()

    # Use log10(1+z) for stability and interpretability at high z
    x = np.log10(1.0 + z)

    save(x, out["E"][idx], "E(z)=H(z)/H0 (toy)", "E(z)", f"{prefix}_E_of_z.png")
    save(x, out["wX"][idx], "w_X(z) (toy)", "w_X", f"{prefix}_wX_of_z.png")
    save(x, out["OmegaX_norm"][idx], "Omega_X(z) normalized (toy)", "Omega_X", f"{prefix}_OmegaX_of_z.png")
    save(x, out["O"][idx], "Coherence O(z) (toy)", "O", f"{prefix}_O_of_z.png")
    save(x, out["closure"][idx], "Closure drift proxy (Omega_sum-1)", "Δ", f"{prefix}_closure_drift.png")

    return [
        f"{prefix}_E_of_z.png",
        f"{prefix}_wX_of_z.png",
        f"{prefix}_OmegaX_of_z.png",
        f"{prefix}_O_of_z.png",
        f"{prefix}_closure_drift.png",
    ]

# ============================================================
# MODE 2: LAYERED CAUSAL GROWTH + PSD KERNEL SANDBOX
# ============================================================

@dataclass
class SandboxParams:
    T: int = 80                       # number of layers
    V0: int = 10                      # events in layer 0
    mean_new: float = 12.0            # mean new events per layer (Poisson)
    parents_k: int = 3                # parents per new event
    parent_window: int = 3            # how many prior layers allowed as parents
    mix_parents: bool = False         # if False prefer immediate previous layer
    alpha: float = 0.65               # ancestor influence decay
    d_eff: float = 3.0                # effective dimension for a(t) proxy
    seed: int = 7

def build_layers(params: SandboxParams) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    rng = random.Random(params.seed)

    layer_of: List[int] = []
    parents: List[List[int]] = []
    children: List[List[int]] = []

    layer_nodes: List[List[int]] = []

    # initialize layer 0
    for _ in range(params.V0):
        v = len(layer_of)
        layer_of.append(0)
        parents.append([])
        children.append([])
        if not layer_nodes:
            layer_nodes.append([])
        layer_nodes[0].append(v)

    # add layers 1..T
    for t in range(1, params.T + 1):
        # Poisson number of new events
        # (use max(1, ...) to avoid accidental empty layers)
        lam = max(1.0, params.mean_new)
        # naive Poisson via exponential inter-arrivals for smallish sizes is fine; use numpy for simplicity
        n_new = int(np.random.default_rng(params.seed + t).poisson(lam=lam))
        n_new = max(1, n_new)

        layer_nodes.append([])
        new_nodes = []
        for _ in range(n_new):
            v = len(layer_of)
            layer_of.append(t)
            parents.append([])
            children.append([])
            layer_nodes[t].append(v)
            new_nodes.append(v)

        # connect respecting partial order
        past_layers_start = max(0, t - params.parent_window)
        candidate_pool = [v for L in range(past_layers_start, t) for v in layer_nodes[L]]
        if not candidate_pool:
            continue

        for v in new_nodes:
            if params.mix_parents:
                pool = candidate_pool
            else:
                pool = layer_nodes[t - 1] if layer_nodes[t - 1] else candidate_pool
            k = min(params.parents_k, len(pool))
            chosen = rng.sample(pool, k=k)
            parents[v] = chosen
            for pnode in chosen:
                children[pnode].append(v)

    return layer_of, parents, children

def compute_Phi(layer_of: List[int], parents: List[List[int]], params: SandboxParams) -> np.ndarray:
    """
    Build Φ(event) as a vector over layers capturing ancestor influence with decay alpha.
    Construction (from log):
      w_v = e_{t(v)} + alpha * average_{p in Parents(v)} w_p
    Then K = Φ Φ^T is PSD by construction (Gram matrix).
    """
    max_layer = max(layer_of)
    dim = max_layer + 1
    n = len(layer_of)

    Phi = np.zeros((n, dim), dtype=float)

    for v in range(n):
        t = layer_of[v]
        Phi[v, t] += 1.0
        ps = parents[v]
        if ps:
            avg_parent = np.mean(Phi[ps, :], axis=0)
            Phi[v, :] += params.alpha * avg_parent

    return Phi

def sandbox_stats(layer_of: List[int], parents: List[List[int]], children: List[List[int]], Phi: np.ndarray, params: SandboxParams):
    n = len(layer_of)
    L = max(layer_of)

    V = np.bincount(np.array(layer_of), minlength=L+1).astype(float)
    a = (V / max(1.0, V[0])) ** (1.0 / max(1e-9, params.d_eff))
    H = np.zeros_like(a)
    H[:-1] = np.log((a[1:] + 1e-15) / (a[:-1] + 1e-15))

    # inter-layer mean correlations:
    # mean K(Li, Lj) = average dot products between Phi vectors in layers i and j
    layer_indices = [np.where(np.array(layer_of) == t)[0] for t in range(L+1)]
    corr = np.zeros((L+1, L+1), dtype=float)

    for i in range(L+1):
        Ii = layer_indices[i]
        if len(Ii) == 0:
            continue
        Pi = Phi[Ii, :]
        for j in range(L+1):
            Ij = layer_indices[j]
            if len(Ij) == 0:
                continue
            Pj = Phi[Ij, :]
            # mean dot product
            corr[i, j] = float(np.mean(Pi @ Pj.T))

    # define a simple coherence proxy A(t): correlation of layer 0 with layer t, normalized
    A = np.zeros(L+1, dtype=float)
    base = corr[0, 0] if corr[0, 0] > 0 else 1.0
    for t in range(L+1):
        A[t] = corr[0, t] / base

    Z = -np.log(np.maximum(1e-15, A))

    return {
        "V": V, "a": a, "H": H,
        "A": A, "Z": Z,
        "corr": corr,
    }

def save_sandbox_outputs(layer_of, parents, children, stats, prefix: str):
    import csv

    # event table
    ev_path = f"{prefix}_sandbox_events.csv"
    with open(ev_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "layer", "indeg", "outdeg"])
        for v in range(len(layer_of)):
            w.writerow([v, layer_of[v], len(parents[v]), len(children[v])])

    # layer stats
    ly_path = f"{prefix}_sandbox_layers.csv"
    with open(ly_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "V", "a", "H", "A", "Z"])
        for t in range(len(stats["V"])):
            w.writerow([t, float(stats["V"][t]), float(stats["a"][t]), float(stats["H"][t]), float(stats["A"][t]), float(stats["Z"][t])])

    return ev_path, ly_path

def plot_sandbox(stats: Dict[str, np.ndarray], prefix: str):
    import matplotlib.pyplot as plt

    t = np.arange(len(stats["V"]))

    def save(x, y, title, ylabel, filename):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("t (layer)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filename, dpi=250)
        plt.close()

    save(t, stats["V"], "Layer volume V(t)", "V", f"{prefix}_V.png")
    save(t, stats["a"], "Scale proxy a(t)", "a", f"{prefix}_a.png")
    save(t, stats["H"], "Discrete H proxy H(t)=Δ ln a", "H", f"{prefix}_H.png")
    save(t, stats["A"], "Coherence proxy A(t)=corr(layer0, layert)/corr(0,0)", "A", f"{prefix}_A.png")
    save(t, stats["Z"], "Z(t)=-ln A(t)", "Z", f"{prefix}_Z.png")

    return [f"{prefix}_V.png", f"{prefix}_a.png", f"{prefix}_H.png", f"{prefix}_A.png", f"{prefix}_Z.png"]

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frw", "sandbox"], default="frw", help="which scaffold to run")
    ap.add_argument("--prefix", default="photonuniverse", help="output file prefix")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib outputs")
    args = ap.parse_args()

    if args.mode == "frw":
        p = FRWParams()
        out = run_frw(p)
        csv_path = save_frw_outputs(out, args.prefix)
        plots = []
        if not args.no_plots:
            plots = plot_frw(out, args.prefix)
        print("FRW mode complete.")
        print(f"  wrote {csv_path}")
        for pp in plots:
            print(f"  wrote {pp}")

        if not args.no_plots:
            plots = plot_frw(out, args.prefix)
            tube = plot_metric_tube_frw(out, outpath=f"{args.prefix}_metric_tube_frw.png")
            print(f"  wrote {tube}")

    else:
        sp = SandboxParams()
        layer_of, parents, children = build_layers(sp)
        Phi = compute_Phi(layer_of, parents, sp)
        stats = sandbox_stats(layer_of, parents, children, Phi, sp)
        ev_path, ly_path = save_sandbox_outputs(layer_of, parents, children, stats, args.prefix)
        plots = []
        if not args.no_plots:
            plots = plot_sandbox(stats, args.prefix)
        print("Sandbox mode complete.")
        print(f"  wrote {ev_path}")
        print(f"  wrote {ly_path}")
        for pp in plots:
            print(f"  wrote {pp}")

        if not args.no_plots:
            plots = plot_sandbox(stats, args.prefix)
            tube = plot_metric_tube_sandbox(stats, outpath=f"{args.prefix}_metric_tube_sandbox.png")
            print(f"  wrote {tube}")

if __name__ == "__main__":
    main()
