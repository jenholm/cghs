import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Units
# ============================================================
year = 3.154e7  # seconds

# ============================================================
# Late-time cosmology (flat ΛCDM)
# ============================================================
H0 = 70.0 * 1e3 / (3.086e22)  # 70 km/s/Mpc in 1/s
Omega_r = 9e-5
Omega_m = 0.30
Omega_L = 0.70

# ============================================================
# Inflation model (slow-roll with constant epsilon)
#   dH/dt = -epsilon H^2
#   H(t) = H_i / (1 + epsilon H_i t)
#   a(t) = a_i * (1 + epsilon H_i t)^(1/epsilon)
# E-folds: N = ln(a_end/a_i) = (1/epsilon) ln(1 + epsilon H_i t_end)
# ============================================================
epsilon = 0.01     # slow-roll parameter (<< 1 is inflationary)
N_target = 60.0    # target e-folds
H_i = 1e37         # 1/s (typical inflationary scale order; adjustable)

# Compute inflation duration to achieve N_target
t_inf_end = (np.exp(epsilon * N_target) - 1.0) / (epsilon * H_i)  # seconds

# Pick a tiny initial scale factor (arbitrary normalization; we'll normalize to a=1 today)
a_i = 1e-60
a_end_inf = a_i * np.exp(N_target)  # consistent by construction

def H_inflation(t):
    return H_i / (1.0 + epsilon * H_i * t)

def a_inflation(t):
    return a_i * (1.0 + epsilon * H_i * t) ** (1.0 / epsilon)

# Deceleration parameter during inflation:
# q = -1 - (dH/dt)/H^2 = -1 + epsilon
q_infl = -1.0 + epsilon

# ============================================================
# Post-inflation ΛCDM ODE: da/dt = a * H_LCDM(a)
# ============================================================
def H_lcdm(a):
    return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_L)

def dadt_lcdm(t, a):
    return a * H_lcdm(a)

# ============================================================
# Time span: from inflation start (t=0) to today (~13.8 Gyr)
# ============================================================
t_today = 13.8e9 * year

# We'll sample time in two parts:
#   - dense/log sampling around inflation and early universe
#   - then out to today
t_min_post = max(t_inf_end, 1e-40)  # avoid exact 0 for geomspace
t_eval_post = np.geomspace(t_min_post, t_today, 6000)

# Solve ΛCDM starting right after inflation ends
sol_post = solve_ivp(
    dadt_lcdm,
    (t_inf_end, t_today),
    [a_end_inf],
    t_eval=t_eval_post,
    rtol=1e-8,
    atol=1e-14
)

t_post = sol_post.t
a_post = sol_post.y[0]

# Build inflation arrays (for plotting + concatenation)
# Use a log-ish sampling inside inflation so we can see it on log plots
t_eval_inf = np.geomspace(1e-44, t_inf_end, 1500) if t_inf_end > 0 else np.array([0.0])
t_eval_inf = np.unique(np.concatenate([np.array([0.0]), t_eval_inf]))
a_inf = a_inflation(t_eval_inf)

# Concatenate full history
t_full = np.concatenate([t_eval_inf, t_post])
a_full = np.concatenate([a_inf, a_post])

# Normalize so that a(t_today)=1
a_full = a_full / a_full[-1]

# Build H(t) for comoving Hubble radius diagnostic
# During inflation, use inflation H(t); after, use H_lcdm(a)
H_full = np.empty_like(a_full)
H_full[:len(t_eval_inf)] = H_inflation(t_eval_inf)
H_full[len(t_eval_inf):] = H_lcdm(a_full[len(t_eval_inf):])  # uses normalized a (fine for diagnostic shape)

# Comoving Hubble radius: (aH)^(-1)
comoving_hubble = 1.0 / (a_full * H_full)

# Volume proxy (for a comoving cube): V ∝ a^3
V_full = a_full**3

# ============================================================
# 3D Tube Geometry (same orientation as before)
# Use time in billions of years for x-axis
# Use radius proportional to scale factor (normalized)
# ============================================================
t_gyr = t_full / year / 1e9
R = a_full

theta = np.linspace(0, 2*np.pi, 90)
T, TH = np.meshgrid(t_gyr, theta)
R_mesh = np.tile(R, (len(theta), 1))
X = T
Y = R_mesh * np.cos(TH)
Z = R_mesh * np.sin(TH)

# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(18, 9))

# 3D tube
ax1 = fig.add_subplot(221, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.82, linewidth=0)
ax1.set_title("Inflation + ΛCDM Expansion Tube (R ∝ a)")
ax1.set_xlabel("Time (Gyr)")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# a(t)
ax2 = fig.add_subplot(222)
ax2.plot(t_gyr, a_full)
ax2.set_title("Scale Factor a(t) (normalized to 1 today)")
ax2.set_xlabel("Time (Gyr)")
ax2.set_ylabel("a(t)")

# Volume
ax3 = fig.add_subplot(223)
ax3.plot(t_gyr, V_full)
ax3.set_title("Volume Proxy: V(t) ∝ a(t)^3")
ax3.set_xlabel("Time (Gyr)")
ax3.set_ylabel("V(t)")

# Inflation diagnostic: comoving Hubble radius
ax4 = fig.add_subplot(224)
# Use log-x (seconds) for this diagnostic because inflation happens insanely early
ax4.plot(t_full + 1e-44, comoving_hubble)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_title("Inflation Diagnostic: Comoving Hubble Radius (aH)^(-1)")
ax4.set_xlabel("t (seconds, log scale)")
ax4.set_ylabel("(aH)^(-1) (arb units, log scale)")

plt.tight_layout()
plt.savefig("universe_exp_3D.png", dpi=300)
plt.show()

print("Inflation settings:")
print(f"  epsilon = {epsilon}")
print(f"  q_infl ≈ -1 + epsilon = {q_infl:.5f}  (accelerating if q<0)")
print(f"  Target e-folds N = {N_target}")
print(f"  Inflation end time t_inf_end = {t_inf_end:.3e} s")
print(f"  a_end/a_start = exp(N) = {np.exp(N_target):.3e}")
