import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ============================================================
# 3D Metric Tube (Conformal-Hilbert Version)
# ============================================================
def plot_metric_tube(N, a, Omega, psi, w_eff,
                     max_points=2500, phi_res=64):

    N = np.asarray(N)
    a = np.asarray(a)
    Omega = np.asarray(Omega)
    psi = np.asarray(psi)      # shape (T, n)
    w_eff = np.asarray(w_eff)

    # Downsample if large
    if N.size > max_points:
        idx = np.linspace(0, N.size-1, max_points).astype(int)
        N = N[idx]
        a = a[idx]
        Omega = Omega[idx]
        psi = psi[idx]
        w_eff = w_eff[idx]

    # ----------------------------------------
    # Radius = physical scale factor
    # ----------------------------------------
    R_profile = a * Omega

    # robust normalization and dynamic range compression
    R_profile = R_profile / (np.percentile(R_profile, 99) + 1e-30)
    R_profile = np.clip(R_profile, 0.0, 1.0)
    R_profile = np.power(R_profile, 0.08)     # 0.05–0.15 are useful
    R_profile = np.maximum(R_profile, 0.02)   # avoid “needle” collapse

    # ----------------------------------------
    # Twist from Hilbert phase
    # ----------------------------------------
    if psi.shape[1] >= 2:
        theta = np.arctan2(psi[:,1], psi[:,0])
    else:
        theta = np.zeros_like(R_profile)

    # ----------------------------------------
    # Build surface
    # ----------------------------------------
    phi = np.linspace(0, 2*np.pi, phi_res)
    T_grid, Phi = np.meshgrid(N, phi)

    R = np.outer(R_profile, np.ones_like(phi)).T
    twist = np.outer(theta, np.ones_like(phi)).T

    X = R * np.cos(Phi + twist)
    Y = R * np.sin(Phi + twist)
    Z = T_grid

    # ----------------------------------------
    # Color by w_eff
    # ----------------------------------------
    wmin = np.percentile(w_eff, 5)
    wmax = np.percentile(w_eff, 95)
    w_clip = np.clip(w_eff, wmin, wmax)
    norm = (w_clip - wmin) / (wmax - wmin + 1e-12)

    colors = cm.plasma(norm)
    colors = np.tile(colors, (len(phi), 1, 1))

    # Fade with time
    t_min = float(np.min(N))
    t_max = float(np.max(N))
    denom = t_max - t_min
    t_norm = (N - t_min) / (denom + 1e-12)
    alpha_profile = 0.35 + 0.45 * t_norm
    for i in range(len(phi)):
        colors[i, :, 3] = alpha_profile

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, facecolors=colors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    ax.plot([0]*len(N), [0]*len(N), N, linewidth=1.0)
    ax.plot_wireframe(X, Y, Z, color='k', alpha=0.05)

    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_zlabel("N = ln a")
    ax.set_title("Conformal Metric Tube\nRadius = a·Ω, Twist = Arg(Ψ), Color = w_eff")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("metric_tube_3D.png", dpi=300)
    plt.close()

# ----------------------------
# Utilities
# ----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def V_Omega(Omega, V0=1e-8, p=0.0):
    """
    Minimal potential.
    p=0 -> V=const behaves like Lambda when phi kinetic is small.
    p>0 -> mild Omega-dependence.
    """
    if p == 0.0:
        return V0
    return V0 * (Omega**p)

def dVdOmega(Omega, V0=1e-8, p=0.0):
    if p == 0.0:
        return 0.0
    return V0 * p * (Omega**(p-1.0))

def build_A(n=2, m2=(1e-4, 2e-4)):
    """
    Example self-adjoint Hilbert operator A (symmetric matrix).
    """
    A = np.zeros((n, n), float)
    for i in range(n):
        A[i, i] = float(m2[i])
    return A

# ----------------------------
# Model RHS in N = ln a
# ----------------------------
def rhs_N(N, y, params):
    """
    y = [a, phi, v, psi(0..n-1), u(0..n-1)]
    where:
      phi = ln Omega
      v   = dphi/dN
      psi = Psi components
      u   = dPsi/dN

    Uses penalty constraint with lambda_eff = kappa * <Psi, C Psi>.
    """
    Mp = params["Mp"]
    A = params["A"]
    mu = params["mu"]
    kappa = params["kappa"]

    Omega_r0 = params["Omega_r0"]
    Omega_m0 = params["Omega_m0"]
    H0 = params["H0"]

    V0 = params["V0"]
    Vp = params["Vp"]

    n = params["n"]

    a = y[0]
    phi = y[1]
    v = y[2]
    psi = y[3:3+n]
    u = y[3+n:3+2*n]

    # Conformal factor
    Omega = np.exp(phi)

    # Operators
    C = A - (mu**2)*(Omega**2)*np.eye(n)

    # Constraint value S = <Psi, C Psi>
    S = float(psi @ (C @ psi))

    # Effective Lagrange multiplier (penalty approximation)
    lambda_eff = kappa * S

    # Densities (dimensionless, scaled by 3 Mp^2 H0^2 if you want; here keep simple)
    # We'll work in units where H0 is a scale; treat Omega_r0, Omega_m0 as present fractions.
    rho_r = Omega_r0 / (a**4)
    rho_m = Omega_m0 / (a**3)

    # Energies of fields in these same "H0^2 Mp^2" units:
    # Convert N-derivatives to time derivatives: dot = H * d/dN
    # We'll compute H from Friedmann self-consistently.
    # First, compute "kinetic pieces" in terms of H^2:
    # rho_phi = 0.5 Mp^2 (phi_dot)^2 + V(Omega)
    #         = 0.5 Mp^2 (H^2 v^2) + V
    # rho_psi = 0.5 |Psi_dot|^2 + 0.5 <Psi, A Psi>
    #         = 0.5 H^2 |u|^2 + 0.5 <Psi, A Psi>

    V = V_Omega(Omega, V0=V0, p=Vp)
    dV = dVdOmega(Omega, V0=V0, p=Vp)

    # Potential term from A
    psi_A_psi = float(psi @ (A @ psi))

    # Solve for H^2 from Friedmann:
    # H^2 = H0^2 * [ rho_r + rho_m + 0.5 Mp^2 H^2 v^2 + V + 0.5 H^2 |u|^2 + 0.5 psi_A_psi ] / (3 Mp^2)
    # Rearrange: H^2 * [1 - H0^2*(0.5 Mp^2 v^2 + 0.5 |u|^2)/(3 Mp^2)] = H0^2*(rho_r+rho_m+V+0.5 psi_A_psi)/(3 Mp^2)
    # We'll absorb H0^2 into H by setting H0=1 in typical runs; still keep it explicit.
    kin_coeff = (0.5*Mp**2 * v*v + 0.5*np.dot(u, u)) / (3.0*Mp**2)
    rhs_const = (rho_r + rho_m + V + 0.5*psi_A_psi) / (3.0*Mp**2)

    denom = max(1.0 - kin_coeff, 1e-12)
    H2 = (H0**2) * rhs_const / denom
    H = np.sqrt(max(H2, 0.0))

    # Pressure pieces (for diagnostics / H' calculation)
    # p_phi = 0.5 Mp^2 H^2 v^2 - V
    p_phi = 0.5*Mp**2 * H2 * v*v - V
    # p_psi = 0.5 H^2 |u|^2 - 0.5 <Psi,A Psi>
    p_psi = 0.5*H2*np.dot(u,u) - 0.5*psi_A_psi

    p_r = rho_r/3.0
    p_m = 0.0

    rho_phi = 0.5*Mp**2 * H2 * v*v + V
    rho_psi = 0.5*H2*np.dot(u,u) + 0.5*psi_A_psi

    rho_tot = rho_r + rho_m + rho_phi + rho_psi
    p_tot = p_r + p_m + p_phi + p_psi

    # d ln H / dN from Friedmann: H' / H = (1/2) d ln(H^2)/dN
    # Use GR identity: dot H = -(rho + p)/(2 Mp^2)
    # Convert: dH/dN = dotH / H = -(rho+p)/(2 Mp^2 * H)
    # => (H'/H) = -(rho+p)/(2 Mp^2 * H^2)
    Hp_over_H = -(rho_tot + p_tot) / max(2.0*Mp**2 * H2, 1e-30)

    # ----------------------------
    # Equations of motion in N
    # ----------------------------
    # a' = a
    da = a

    # phi'' + (3 + H'/H) phi' + (Omega dV/dOmega)/ (Mp^2 H^2) - (2 lambda mu^2 Omega^2 <Psi,Psi>)/(Mp^2 H^2) = 0
    psi_norm2 = float(np.dot(psi, psi))
    source_phi = (Omega * dV) / max(Mp**2 * H2, 1e-30)
    # penalty coupling term (acts like lambda term)
    # exact theory: - (2 lambda mu^2 Omega^2 <Psi,Psi>)/(Mp^2 H^2)
    # here lambda_eff is our effective multiplier
    coupling_phi = (2.0 * lambda_eff * (mu**2) * (Omega**2) * psi_norm2) / max(Mp**2 * H2, 1e-30)

    dv = -(3.0 + Hp_over_H)*v - source_phi + coupling_phi
    dphi = v

    # Psi'' + (3 + H'/H) Psi' + (A Psi)/(H^2) - (2 lambda C Psi)/(H^2) = 0
    # (again, penalty lambda_eff used)
    Au = A @ psi
    Cu = C @ psi
    du = -(3.0 + Hp_over_H)*u - (Au / max(H2, 1e-30)) + (2.0*lambda_eff)*(Cu / max(H2, 1e-30))
    dpsi = u

    # pack
    dydN = np.zeros_like(y)
    dydN[0] = da
    dydN[1] = dphi
    dydN[2] = dv
    dydN[3:3+n] = dpsi
    dydN[3+n:3+2*n] = du

    # diagnostics we want
    diag = {
        "H": H,
        "H2": H2,
        "Omega": Omega,
        "rho_r": rho_r,
        "rho_m": rho_m,
        "rho_phi": rho_phi,
        "rho_psi": rho_psi,
        "p_phi": p_phi,
        "p_psi": p_psi,
        "rho_tot": rho_tot,
        "p_tot": p_tot,
        "OmegaX": (rho_phi + rho_psi) / max(rho_tot, 1e-30),
        "w_eff": p_tot / max(rho_tot, 1e-30),
        "lambda_eff": lambda_eff,
        "constraint_S": S
    }
    return dydN, diag

# ----------------------------
# Integrate
# ----------------------------
def integrate_model(
    N0=-18.4, N1=0.0, dN=1e-3,
    a0=np.exp(-18.4),
    Omega0=1.0,          # initial conformal factor
    v0=0.0,              # initial dphi/dN
    psi0=(1.0, 0.0),
    u0=(0.0, 0.0),
    params=None
):
    if params is None:
        params = {}

    n = params["n"]
    # state: [a, phi, v, psi..., u...]
    y = np.zeros(3 + 2*n, float)
    y[0] = a0
    y[1] = np.log(max(Omega0, 1e-30))
    y[2] = v0
    y[3:3+n] = np.array(psi0, float)
    y[3+n:3+2*n] = np.array(u0, float)

    steps = int(np.ceil((N1 - N0)/dN)) + 1
    N_arr = np.zeros(steps, float)

    out = {
        "a": np.zeros(steps),
        "Omega": np.zeros(steps),
        "H": np.zeros(steps),
        "OmegaX": np.zeros(steps),
        "w_eff": np.zeros(steps),
        "lambda_eff": np.zeros(steps),
        "constraint_S": np.zeros(steps),
        "rho_tot": np.zeros(steps),
        "psi": np.zeros((steps, n)),
        "rho_r": np.zeros(steps),
        "rho_m": np.zeros(steps),
        "rho_phi": np.zeros(steps),
        "rho_psi": np.zeros(steps),
        "H2": np.zeros(steps)
    }
    N = N0
    for i in range(steps):
        N_arr[i] = N

        y[0] = max(y[0], 1e-40)
        dydN, diag = rhs_N(N, y, params)
        out["a"][i] = y[0]
        out["Omega"][i] = diag["Omega"]
        out["H"][i] = diag["H"]
        out["OmegaX"][i] = diag["OmegaX"]
        out["w_eff"][i] = diag["w_eff"]
        out["lambda_eff"][i] = diag["lambda_eff"]
        out["constraint_S"][i] = diag["constraint_S"]
        out["rho_tot"][i] = diag["rho_tot"]
        out["psi"][i, :] = y[3:3+n]
        out["rho_r"][i]  = diag["rho_r"]
        out["rho_m"][i]  = diag["rho_m"]
        out["rho_phi"][i]= diag["rho_phi"]
        out["rho_psi"][i]= diag["rho_psi"]
        out["H2"][i]     = diag["H2"]

        # RK4 step
        k1, _ = rhs_N(N, y, params)
        k2, _ = rhs_N(N + 0.5*dN, y + 0.5*dN*k1, params)
        k3, _ = rhs_N(N + 0.5*dN, y + 0.5*dN*k2, params)
        k4, _ = rhs_N(N + dN,     y + dN*k3, params)
        y = y + (dN/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # guard
        y[0] = max(y[0], 1e-40)

        N += dN
        if N > N1 + 1e-12:
            break

    # ----------------------------
    # Trim arrays to actual length
    # ----------------------------
    n_used = i + 1  # last filled index + 1

    N_arr = N_arr[:n_used]
    for k in out:
        out[k] = out[k][:n_used]   # works for 1D and for psi's first axis

    # convert to z (now no trailing zeros)
    a = out["a"]
    z = 1.0/np.maximum(a, 1e-30) - 1.0
    return N_arr, z, out

# ----------------------------
# Example run + plots
# ----------------------------
if __name__ == "__main__":
    # Parameters (dimensionless demo units)
    n = 2
    params = {
        "n": n,
        "Mp": 1.0,
        "H0": 1.0,
        "Omega_r0": 9e-5,
        "Omega_m0": 0.30,
        "A": build_A(n=n, m2=(1e-4, 2e-4)),
        "mu": 1e-2,
        "kappa": 1e6,  # penalty strength; increase to enforce constraint more tightly
        "V0": 1e-8,    # sets late-time DE scale
        #"V0": 0.7,
        "Vp": 0.0      # constant potential (Lambda-like)
    }

    N_arr, z, out = integrate_model(
        N0=np.log(1e-8),
        N1=0.0,
        dN=2e-3,
        a0=1e-8,
        Omega0=1.0,
        v0=0.0,
        psi0=(1.0, 1e-3),
        u0=(0.0, 0.0),
        params=params
    )

    # Sort by increasing z for plotting H(z)
    order = np.argsort(z)
    z_sorted = z[order]

    plt.figure()
    plt.plot(z_sorted, out["H"][order])
    plt.gca().invert_xaxis()
    plt.xlabel("z")
    plt.ylabel("H(z) (dimensionless)")
    plt.title("H(z)")
    plt.tight_layout()
    plt.savefig("H_z.png", dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(z_sorted, out["OmegaX"][order])
    plt.gca().invert_xaxis()
    plt.xlabel("z")
    plt.ylabel(r"$\Omega_X(z)$")
    plt.title(r"Dark fraction $\Omega_X(z) = (\rho_\Omega+\rho_\Psi)/\rho_{\rm tot}$")
    plt.tight_layout()
    plt.savefig("dark_fraction.png", dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(z_sorted, out["w_eff"][order])
    plt.gca().invert_xaxis()
    plt.xlabel("z")
    plt.ylabel(r"$w_{\rm eff}(z)$")
    plt.title(r"Effective EOS $w_{\rm eff}(z)=p_{\rm tot}/\rho_{\rm tot}$")
    plt.tight_layout()
    plt.savefig("Effective_EOS.png", dpi=300)
    #plt.show()

    plt.figure()
    plt.semilogy(N_arr, np.abs(out["constraint_S"]) + 1e-30)
    plt.xlabel("N = ln a")
    plt.ylabel(r"$\langle\Psi, C(\Omega)\Psi\rangle$")
    plt.title("Constraint residual (penalty-enforced)")
    plt.tight_layout()
    plt.savefig("constraints_residual.png", dpi=300)
    #plt.show()
    # psi time series needs shape (T, n)
    psi_series = out["psi"]

    plot_metric_tube(
        N_arr,
        out["a"],
        out["Omega"],
        psi_series,
        out["w_eff"]
    )

    zmax = 1e4   # try 1e4, 1e3, 50, 10 depending what you want to see
mask = z_sorted <= zmax

plt.figure()
plt.plot(z_sorted[mask], out["OmegaX"][order][mask])
plt.gca().invert_xaxis()
plt.xlabel("z")
plt.ylabel(r"$\Omega_X(z)$")
plt.title(r"Dark fraction (zoomed to $z \leq %.0g$)" % zmax)
plt.tight_layout()
plt.savefig("dark_fraction_zoom.png", dpi=300)
plt.close()

plt.figure()
plt.plot(z_sorted[mask], out["w_eff"][order][mask])
plt.gca().invert_xaxis()
plt.xlabel("z")
plt.ylabel(r"$w_{\rm eff}(z)$")
plt.title(r"Effective EOS (zoomed to $z \leq %.0g$)" % zmax)
plt.tight_layout()
plt.savefig("Effective_EOS_zoom.png", dpi=300)
plt.close()