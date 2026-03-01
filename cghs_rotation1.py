import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def make_layers(T=40, base=6, growth=1.05, jitter=0.3, seed=0):
    """
    Make layered events: layer t has n_t events.
    Assign each event:
      - time coordinate tau = t
      - spatial coord r in R^d (here d=2) with slowly increasing spread
    """
    rng = np.random.default_rng(seed)
    layers = []
    coords = []
    layer_id = []
    idx = 0
    for t in range(T):
        n_t = int(base * (growth ** t) + 1)
        # spatial spread increases with t a bit
        spread = 1.0 + 0.03 * t
        r = rng.normal(0, spread, size=(n_t, 2))
        tau = np.full((n_t, 1), float(t))
        x = np.hstack([tau, r])  # (tau, x, y)
        coords.append(x)
        layer_id += [t] * n_t
        layers.append(list(range(idx, idx + n_t)))
        idx += n_t
    X = np.vstack(coords)
    layer_id = np.array(layer_id, dtype=int)
    return X, layers, layer_id

def causal_admissible(i, j, layer_id):
    """Layered partial order: i precedes j iff layer(i) < layer(j)."""
    return layer_id[i] < layer_id[j]

def build_causal_matrix(layer_id):
    """
    C_ij = +1 if i≺j, -1 if j≺i, 0 otherwise (same layer).
    This is antisymmetric by construction.
    """
    N = len(layer_id)
    C = np.zeros((N, N), dtype=float)
    # Vectorized-ish using broadcasting
    Li = layer_id[:, None]
    Lj = layer_id[None, :]
    C[Li < Lj] = +1.0
    C[Li > Lj] = -1.0
    # diagonal and same-layer remain 0
    return C

def rbf_kernel(X, ell=2.5):
    """
    Symmetric positive-definite kernel g(x,y) = exp(-||x-y||^2 / (2 ell^2)).
    Works as a Hilbert metric seed.
    """
    # squared Euclidean distances
    XX = np.sum(X**2, axis=1)
    D2 = XX[:, None] + XX[None, :] - 2.0 * (X @ X.T)
    g = np.exp(-D2 / (2.0 * ell**2))
    # small diagonal jitter to help numerical stability
    g += 1e-10 * np.eye(len(X))
    return g

def causal_phase_kernel(C, alpha=0.8, decay=0.15, layer_id=None):
    """
    Antisymmetric 'causal phase' omega:
      omega_ij = C_ij * exp(-decay * |Δlayer|)
    """
    if layer_id is None:
        raise ValueError("layer_id required for decay")
    Li = layer_id[:, None]
    Lj = layer_id[None, :]
    dL = np.abs(Li - Lj)
    omega = alpha * C * np.exp(-decay * dL)
    # Ensure strict antisymmetry numerically
    omega = 0.5 * (omega - omega.T)
    return omega

def order_recovery_score(omega, layer_id, sample_pairs=20000, seed=0):
    """
    Try to recover the causal direction from sign(omega_ij):
      predict i≺j if omega_ij > 0, predict j≺i if omega_ij < 0.
    Score = accuracy on sampled pairs where layers differ.
    """
    rng = np.random.default_rng(seed)
    N = len(layer_id)
    # sample pairs
    ii = rng.integers(0, N, size=sample_pairs)
    jj = rng.integers(0, N, size=sample_pairs)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]

    Li, Lj = layer_id[ii], layer_id[jj]
    mask = Li != Lj
    ii, jj, Li, Lj = ii[mask], jj[mask], Li[mask], Lj[mask]

    true = (Li < Lj)  # True means i≺j
    pred = (omega[ii, jj] > 0)  # omega sign predicts direction
    return np.mean(pred == true)

def min_eig(A):
    # symmetric eigenvalues
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    return float(np.min(w))

# ----------------------------
# Main experiment: "rotation"
# ----------------------------
def run_rotation_experiment(
    T=35,
    base=6,
    growth=1.05,
    ell=2.5,
    alpha=0.8,
    decay=0.15,
    thetas=60,
    seed=0
):
    X, layers, layer_id = make_layers(T=T, base=base, growth=growth, seed=seed)
    N = len(X)

    # Build the two "orthogonal components"
    g = rbf_kernel(X, ell=ell)                      # symmetric + positive
    C = build_causal_matrix(layer_id)               # antisymmetric sign order
    omega = causal_phase_kernel(C, alpha=alpha, decay=decay, layer_id=layer_id)  # antisymmetric

    theta_grid = np.linspace(0, 0.5*np.pi, thetas)

    # Track: positivity of Re(M), order recoverability of Im(M)
    pos_min_eigs = []
    order_scores = []
    norms_re = []
    norms_im = []

    for th in theta_grid:
        # unified object (like a "field tensor" split):
        # M_th = cos(th) g + i sin(th) omega
        Re = np.cos(th) * g
        Im = np.sin(th) * omega

        pos_min_eigs.append(min_eig(Re))
        order_scores.append(order_recovery_score(Im, layer_id, sample_pairs=15000, seed=seed+1))

        norms_re.append(np.linalg.norm(Re, ord="fro") / np.sqrt(N))
        norms_im.append(np.linalg.norm(Im, ord="fro") / np.sqrt(N))

    return theta_grid, np.array(pos_min_eigs), np.array(order_scores), np.array(norms_re), np.array(norms_im)

def plot_results(theta, pos_min, order_score, norms_re, norms_im):
    # Plot 1: "Hilbert positivity" via min eigenvalue of Re(M_theta)
    plt.figure()
    plt.plot(theta, pos_min)
    plt.title("Hilbert-positivity proxy: min eigenvalue of Re(Mθ)")
    plt.xlabel("θ")
    plt.ylabel("min eig(Re(Mθ))")
    plt.axhline(0.0)
    plt.tight_layout()

    # Plot 2: Causal order recoverability from sign(Im(M_theta))
    plt.figure()
    plt.plot(theta, order_score)
    plt.title("Causal-order proxy: accuracy from sign(Im(Mθ))")
    plt.xlabel("θ")
    plt.ylabel("order recovery accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()

    # Plot 3: component norms (how much you're leaning into each side)
    plt.figure()
    plt.plot(theta, norms_re, label="||Re(Mθ)||")
    plt.plot(theta, norms_im, label="||Im(Mθ)||")
    plt.title("Component strength vs θ (E/B-style rotation)")
    plt.xlabel("θ")
    plt.ylabel("Frobenius norm / √N")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Tunable knobs (crayon-friendly):
    # - ell: geometric locality scale for g
    # - alpha, decay: strength and range of causal phase kernel omega
    theta, pos_min, order_score, norms_re, norms_im = run_rotation_experiment(
        T=35,
        base=6,
        growth=1.05,
        ell=2.8,
        alpha=1.0,
        decay=0.12,
        thetas=60,
        seed=7
    )
    plot_results(theta, pos_min, order_score, norms_re, norms_im)