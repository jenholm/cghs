#!/usr/bin/env python3
"""
python k-space.py --layers 50 --n0 10 --infl_layers 7 --infl_factor 2.0 --post_factor 1.10 \
  --parents_k 3 --parent_window 2 --alpha 0.65 --d_eff 3 --seed 42 --plots

Layered causal growth + positive kernel toy model ("Big Bang" sandbox)

Minimal seed:
  - Events E arrive in layers (cosmic time) t=0..T
  - Causal admissibility: edges only from earlier -> later layers => a partial order ≺
  - Kernel K(x,y) is built as a Gram matrix K = Φ Φ^T, guaranteeing PSD (Hilbert metric)

Interpretation:
  - Layer sizes V(t) act as a toy "spatial volume" proxy
  - a(t) ~ V(t)^(1/d_eff) is a toy scale-factor proxy
  - Kernel correlations between early and late layers act as a toy "redshift-like" attenuation statistic

This is NOT a physical cosmology simulation.
It is a relational toy that lets you test whether the framework can
generate qualitatively FRW-ish signatures under plausible growth rules.

Author: (you + Novella)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class Params:
    layers: int
    n0: int
    infl_layers: int
    infl_factor: float
    post_factor: float
    parents_k: int
    parent_window: int
    alpha: float
    mix_parents: bool
    d_eff: float
    seed: int
    plots: bool


def layer_sizes(params: Params) -> List[int]:
    """
    Piecewise growth:
      - layers 0..infl_layers grow by infl_factor
      - remaining grow by post_factor
    """
    sizes = [params.n0]
    for t in range(1, params.layers + 1):
        if t <= params.infl_layers:
            nxt = max(1, int(round(sizes[-1] * params.infl_factor)))
        else:
            nxt = max(1, int(round(sizes[-1] * params.post_factor)))
        sizes.append(nxt)
    return sizes


def build_layered_dag(params: Params) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    """
    Build events in layers and connect each new event to k parents in recent past layers.
    Returns:
      - layer_of[event_id]
      - parents[event_id] list of parent ids
      - children[event_id] list of child ids
    """
    rng = random.Random(params.seed)

    sizes = layer_sizes(params)
    layer_nodes: List[List[int]] = []
    layer_of: List[int] = []
    parents: List[List[int]] = []
    children: List[List[int]] = []

    # Create nodes layer by layer
    eid = 0
    for t, n in enumerate(sizes):
        nodes = list(range(eid, eid + n))
        layer_nodes.append(nodes)
        for v in nodes:
            layer_of.append(t)
            parents.append([])
            children.append([])
        eid += n

    # Connect edges respecting the partial order (earlier -> later)
    for t in range(1, len(layer_nodes)):
        past_layers_start = max(0, t - params.parent_window)
        candidate_pool = [v for L in range(past_layers_start, t) for v in layer_nodes[L]]
        if not candidate_pool:
            continue

        for v in layer_nodes[t]:
            # choose parents; optionally prefer immediate previous layer to mimic local causality
            if params.mix_parents:
                pool = candidate_pool
            else:
                pool = layer_nodes[t - 1] if layer_nodes[t - 1] else candidate_pool

            k = min(params.parents_k, len(pool))
            chosen = rng.sample(pool, k=k)
            parents[v] = chosen
            for p in chosen:
                children[p].append(v)

    return layer_of, parents, children


def compute_features(layer_of: List[int], parents: List[List[int]], params: Params) -> np.ndarray:
    """
    Build Φ(event) as a vector over layers capturing "ancestor influence" with decay alpha.

    Let w_v be a (T+1)-dim vector where index ℓ summarizes influence of ancestors in layer ℓ.
    We define:
      w_v = e_{t(v)} + alpha * average_{p in Parents(v)} w_p

    This is a simple causal propagation model:
      - Influence flows forward along ≺
      - Decays with alpha per layer-step (in expectation, since parents are in recent layers)

    Then K = Φ Φ^T is PSD by construction.

    Returns Φ as an (N x (L+1)) array.
    """
    max_layer = max(layer_of)
    dim = max_layer + 1
    n = len(layer_of)

    Phi = np.zeros((n, dim), dtype=float)

    # process in increasing event id works because construction was layered
    for v in range(n):
        t = layer_of[v]
        Phi[v, t] += 1.0

        ps = parents[v]
        if ps:
            avg_parent = np.mean(Phi[ps, :], axis=0)
            Phi[v, :] += params.alpha * avg_parent

    # Optional normalization to keep scales sane
    # (comment out if you want raw growth to dominate)
    # norms = np.linalg.norm(Phi, axis=1, keepdims=True) + 1e-12
    # Phi = Phi / norms

    return Phi


def layer_statistics(layer_of: List[int], parents: List[List[int]], children: List[List[int]], Phi: np.ndarray, params: Params):
    """
    Compute per-layer stats and kernel correlation summaries.
    """
    n = len(layer_of)
    L = max(layer_of)

    # basic layer volumes
    V = np.bincount(np.array(layer_of), minlength=L + 1)

    # toy scale factor proxy
    # You can compare shapes to FRW power laws in post-analysis (very approximate).
    a = (V / max(1, V[0])) ** (1.0 / max(1e-9, params.d_eff))

    # degrees
    indeg = np.array([len(parents[v]) for v in range(n)], dtype=int)
    outdeg = np.array([len(children[v]) for v in range(n)], dtype=int)

    # Kernel is K = Phi Phi^T, but we do NOT materialize NxN for large N.
    # Instead compute inter-layer mean correlations efficiently:
    # mean K between layer i and j = mean over x in Li, y in Lj of <Phi_x, Phi_y>
    # = average dot products between their feature vectors.

    # precompute layer feature sums
    layer_ids: List[np.ndarray] = [np.where(np.array(layer_of) == t)[0] for t in range(L + 1)]
    layer_sum = np.zeros((L + 1, Phi.shape[1]), dtype=float)
    for t in range(L + 1):
        if len(layer_ids[t]) > 0:
            layer_sum[t, :] = Phi[layer_ids[t], :].sum(axis=0)

    # mean dot products between layers:
    # (sum_{x in Li} sum_{y in Lj} Phi_x·Phi_y) / (|Li||Lj|)
    # = ( (sumPhi_i) · (sumPhi_j) ) / (|Li||Lj|)
    K_layer_mean = np.zeros((L + 1, L + 1), dtype=float)
    for i in range(L + 1):
        for j in range(L + 1):
            denom = max(1, V[i]) * max(1, V[j])
            K_layer_mean[i, j] = float(np.dot(layer_sum[i, :], layer_sum[j, :]) / denom)

    # A simple "redshift-like" attenuation score from early layer 0 to later layers:
    # Z(t) = log( meanK(0,0) / meanK(0,t) )
    # If meanK(0,t) decreases with t, Z(t) grows.
    base = max(1e-12, K_layer_mean[0, 0])
    Z = np.array([math.log(base / max(1e-12, K_layer_mean[0, t])) for t in range(L + 1)], dtype=float)

    # Package layer stats
    layer_rows = []
    for t in range(L + 1):
        ids = layer_ids[t]
        layer_rows.append({
            "layer": t,
            "V": int(V[t]),
            "a_proxy": float(a[t]),
            "mean_indeg": float(indeg[ids].mean()) if len(ids) else 0.0,
            "mean_outdeg": float(outdeg[ids].mean()) if len(ids) else 0.0,
            "mean_normPhi": float(np.linalg.norm(Phi[ids, :], axis=1).mean()) if len(ids) else 0.0,
            "meanK_0_t": float(K_layer_mean[0, t]),
            "Z_like": float(Z[t]),
        })

    # Flatten kernel matrix as rows for CSV
    kernel_rows = []
    for i in range(L + 1):
        for j in range(L + 1):
            kernel_rows.append({
                "layer_i": i,
                "layer_j": j,
                "meanK": float(K_layer_mean[i, j]),
            })

    return layer_rows, kernel_rows, V, a, Z


def maybe_plot(layer_rows, V, a, Z, params: Params, out_prefix: str):
    if not params.plots:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    t = np.array([r["layer"] for r in layer_rows], dtype=int)

    plt.figure()
    plt.plot(t, V)
    plt.xlabel("layer t")
    plt.ylabel("V(t) = events in layer")
    plt.title("Layer volume proxy V(t)")
    plt.savefig(f"{out_prefix}_V.png", dpi=160)

    plt.figure()
    plt.plot(t, a)
    plt.xlabel("layer t")
    plt.ylabel("a_proxy(t) ~ V(t)^(1/d_eff)")
    plt.title("Scale factor proxy a(t)")
    plt.savefig(f"{out_prefix}_a.png", dpi=160)

    plt.figure()
    plt.plot(t, Z)
    plt.xlabel("layer t")
    plt.ylabel("Z_like(t) = log(meanK(0,0)/meanK(0,t))")
    plt.title("Redshift-like attenuation from kernel")
    plt.savefig(f"{out_prefix}_Z.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40, help="number of layers after t=0")
    ap.add_argument("--n0", type=int, default=8, help="size of initial layer")
    ap.add_argument("--infl_layers", type=int, default=6, help="how many layers of 'inflation-like' fast growth")
    ap.add_argument("--infl_factor", type=float, default=1.8, help="growth factor per layer during inflation")
    ap.add_argument("--post_factor", type=float, default=1.12, help="growth factor per layer after inflation")
    ap.add_argument("--parents_k", type=int, default=3, help="number of parents for each event")
    ap.add_argument("--parent_window", type=int, default=2, help="how many previous layers can be chosen as parents")
    ap.add_argument("--alpha", type=float, default=0.65, help="decay factor in kernel feature propagation")
    ap.add_argument("--mix_parents", action="store_true", help="if set, choose parents from the full window, not just previous layer")
    ap.add_argument("--d_eff", type=float, default=3.0, help="assumed effective spatial dimension for a_proxy")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--plots", action="store_true", help="write PNG plots (requires matplotlib)")
    ap.add_argument("--out_prefix", type=str, default="universe", help="output filename prefix")
    args = ap.parse_args()

    params = Params(
        layers=args.layers,
        n0=args.n0,
        infl_layers=args.infl_layers,
        infl_factor=args.infl_factor,
        post_factor=args.post_factor,
        parents_k=args.parents_k,
        parent_window=args.parent_window,
        alpha=args.alpha,
        mix_parents=args.mix_parents,
        d_eff=args.d_eff,
        seed=args.seed,
        plots=args.plots,
    )

    layer_of, parents, children = build_layered_dag(params)
    Phi = compute_features(layer_of, parents, params)

    layer_rows, kernel_rows, V, a, Z = layer_statistics(layer_of, parents, children, Phi, params)

    # Write outputs
    out_prefix = args.out_prefix

    # Event-level table
    event_rows = []
    for v in range(len(layer_of)):
        event_rows.append({
            "event_id": v,
            "layer": layer_of[v],
            "indeg": len(parents[v]),
            "outdeg": len(children[v]),
            "normPhi": float(np.linalg.norm(Phi[v, :])),
        })

    if pd is None:
        # fallback to simple CSV writing
        def write_csv(path: str, rows: List[Dict]):
            if not rows:
                return
            cols = list(rows[0].keys())
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")

        write_csv(f"{out_prefix}_events.csv", event_rows)
        write_csv(f"{out_prefix}_layer_stats.csv", layer_rows)
        write_csv(f"{out_prefix}_kernel_stats.csv", kernel_rows)
    else:
        pd.DataFrame(event_rows).to_csv(f"{out_prefix}_events.csv", index=False)
        pd.DataFrame(layer_rows).to_csv(f"{out_prefix}_layer_stats.csv", index=False)
        pd.DataFrame(kernel_rows).to_csv(f"{out_prefix}_kernel_stats.csv", index=False)

    maybe_plot(layer_rows, V, a, Z, params, out_prefix)

    # Console summary (quick glance)
    print(f"Built layered universe with N={len(layer_of)} events, layers=0..{max(layer_of)}")
    print("First few layer stats:")
    for r in layer_rows[:min(8, len(layer_rows))]:
        print(
            f"t={r['layer']:>2d}  V={r['V']:>5d}  a={r['a_proxy']:.3f}  "
            f"meanK(0,t)={r['meanK_0_t']:.4e}  Z_like={r['Z_like']:.3f}"
        )
    print(f"Wrote: {out_prefix}_events.csv, {out_prefix}_layer_stats.csv, {out_prefix}_kernel_stats.csv")
    if params.plots:
        print(f"Wrote plots: {out_prefix}_V.png, {out_prefix}_a.png, {out_prefix}_Z.png")


if __name__ == "__main__":
    main()
