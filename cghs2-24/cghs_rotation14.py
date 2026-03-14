import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
alpha = 3.0
beta  = 1.0
gamma = 1.2
kappa = 0.8

C_inf = 0.4        # equilibrium radius
C_initial = 1.2    # initially "compressed"
H_initial = 0.0

# -------------------------------------------------------
# System
# -------------------------------------------------------
def rhs(t, y):
    H, C = y
    
    dHdt = alpha * (C - C_inf) - beta * H
    dCdt = -gamma * H - kappa * (C - C_inf)
    
    return [dHdt, dCdt]

# -------------------------------------------------------
# Solve
# -------------------------------------------------------
t_span = (0, 6)
t_eval = np.linspace(*t_span, 600)

sol = solve_ivp(rhs, t_span, [H_initial, C_initial], t_eval=t_eval)

t = sol.t
H = sol.y[0]
C = sol.y[1]

dt = t[1] - t[0]
Z = np.cumsum(H) * dt

# -------------------------------------------------------
# Tube
# -------------------------------------------------------
theta = np.linspace(0, 2*np.pi, 60)
T, Theta = np.meshgrid(t, theta)

R = np.tile(C, (len(theta), 1))
Z_mesh = np.tile(Z, (len(theta), 1))

X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z_mesh)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (axial inflation)")
plt.savefig("cghs.png", dpi=300)
#plt.show()
plt.close()