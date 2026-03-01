import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Cosmological Parameters (ΛCDM)
# ============================================================
H0 = 70.0 * 1e3 / (3.086e22)      # H0 in 1/s  (70 km/s/Mpc)
Omega_r = 9e-5
Omega_m = 0.3
Omega_L = 0.7

# Convert time to seconds
year = 3.154e7
t_end = 13.8e9 * year

# ============================================================
# Friedmann Equation
# ============================================================
def H(a):
    return H0 * np.sqrt(
        Omega_r / a**4 +
        Omega_m / a**3 +
        Omega_L
    )

def dadt(t, a):
    return a * H(a)

# ============================================================
# Integrate from early universe to present
# ============================================================
a_initial = 1e-8
t_span = (0.0, t_end)

# Log-spaced time sampling to resolve early expansion
t_eval = np.geomspace(1e3, t_end, 4000)

sol = solve_ivp(
    dadt,
    t_span,
    [a_initial],
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-12
)

t = sol.t
a = sol.y[0]

# Normalize scale factor to 1 at present
a = a / a[-1]

# Radius proportional to scale factor
R = a

# Volume proportional to a^3
V = a**3

# ============================================================
# Build 3D Tube (same orientation as before)
# ============================================================
theta = np.linspace(0, 2*np.pi, 90)
T, TH = np.meshgrid(t/year/1e9, theta)  # time in billions of years
R_mesh = np.tile(R, (len(theta), 1))

X = T
Y = R_mesh * np.cos(TH)
Z = R_mesh * np.sin(TH)

# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.82, linewidth=0)
ax1.set_title("ΛCDM Expansion Tube")
ax1.set_xlabel("Time (billion years)")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(222)
ax2.plot(t/year/1e9, R)
ax2.set_title("Scale Factor a(t)")
ax2.set_xlabel("Time (billion years)")
ax2.set_ylabel("a(t)")

ax3 = fig.add_subplot(224)
ax3.plot(t/year/1e9, V)
ax3.set_title("Volume ∝ a(t)^3")
ax3.set_xlabel("Time (billion years)")
ax3.set_ylabel("V(t)")

plt.tight_layout()
plt.savefig("universe_exp_3D.png", dpi=300)
plt.show()