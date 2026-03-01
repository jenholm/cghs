import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# TIME (years)
# ============================================================
t_end = 1.3e9
n = 3000
t = np.linspace(0.0, t_end, n)

# ============================================================
# VISUALIZATION AXIS (this is the secret sauce)
# This makes the early universe visible without breaking physics.
# Try: x = log10(t+1), or asinh(t/t_scale), etc.
# ============================================================
t_scale = 5e6
x = np.arcsinh(t / t_scale)   # smooth log-like warp
# x = np.log10(t + 1.0)       # alternative warp

# ============================================================
# RADIUS MODEL = inflation jump + slow drift + late accel
# This is intentionally a "shape law" (what the picture depicts).
# ============================================================

# Baseline radius scale (pure geometry units)
R0 = 0.02

# --- Inflation: a sharp step up early ---
t_inf_center = 2e6     # where the flare happens
t_inf_width  = 4e5     # sharpness of the flare edge
A_inf = 0.55           # how much inflation increases radius

S_inf = 0.5 * (1 + np.tanh((t - t_inf_center) / t_inf_width))  # 0 -> 1
R_inf = A_inf * S_inf

# --- Post-inflation slow widening: concave growth ---
# Using a sublinear power of time gives the “gentle trumpet” body.
A_body = 0.55
p_body = 0.55
R_body = A_body * (t / t_end)**p_body

# --- Late acceleration: small extra flare near the end ---
A_late = 0.25
t_late_center = 0.82 * t_end
t_late_width  = 0.10 * t_end
S_late = 0.5 * (1 + np.tanh((t - t_late_center) / t_late_width))
R_late = A_late * S_late * ((t - t_late_center).clip(min=0) / (t_end - t_late_center))**1.2

# Total radius
R = R0 + R_inf + R_body + R_late

# Optional: normalize so the plot always fits nicely
# R = R / R.max()

V = np.pi * R**2  # cross-sectional volume per unit length

# ============================================================
# BUILD 3D TUBE
# ============================================================
theta = np.linspace(0, 2*np.pi, 90)
X, TH = np.meshgrid(x, theta)
R_mesh = np.tile(R, (len(theta), 1))

Y = R_mesh * np.cos(TH)
Z = R_mesh * np.sin(TH)

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.82, linewidth=0)
ax1.set_title("Poster-faithful Universe Tube (warped time axis)")
ax1.set_xlabel("x = asinh(t/t_scale)")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(222)
ax2.plot(t, R)
ax2.set_title("Radius R(t) on real time axis")
ax2.set_xlabel("t (years)")
ax2.set_ylabel("R")

ax3 = fig.add_subplot(224)
ax3.plot(t, V)
ax3.set_title("Volume V(t) = πR(t)^2")
ax3.set_xlabel("t (years)")
ax3.set_ylabel("V")

plt.tight_layout()
plt.savefig("metric_tube_3D.png", dpi=300)
plt.show()