import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Hilbert potential parameters (match your simulation)
# ============================================================

Omega_inf = 50.0
S_c = 8.0
#width = 0.6
width = 2.0

# ============================================================
# Hilbert vacuum potential Ω_S(S)
# ============================================================

def omega_S(S):
    """
    Hilbert vacuum energy term Ω_S(S)

    Large for small S (inflation plateau)
    Drops near S_c
    """
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)
    return Omega_inf / (1.0 + np.exp(x))


# ============================================================
# Analytical derivative dΩ_S/dS
# ============================================================

def dOmega_dS(S):
    """
    Derivative of Ω_S with respect to S
    """
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)

    expx = np.exp(x)

    return -(Omega_inf / width) * expx / (1 + expx)**2


# ============================================================
# S range
# ============================================================

S = np.linspace(0, 20, 800)


# ============================================================
# Evaluate functions
# ============================================================

Omega_vals = omega_S(S)
dOmega_vals = dOmega_dS(S)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(S, Omega_vals, linewidth=3, label=r"$\Omega_S(S)$")

plt.plot(
    S,
    np.abs(dOmega_vals),
    linewidth=2,
    linestyle="--",
    label=r"$|d\Omega_S/dS|$"
)

plt.axvline(S_c, color="black", linestyle=":", label="Transition $S_c$")

plt.xlabel("Hilbert order parameter S")
plt.ylabel("Potential / slope magnitude")
plt.title("Hilbert Vacuum Potential and Its Slope")

plt.legend()
plt.grid(True)
plt.savefig("Hilbert Vacuum Potential and Its Slope.png", dpi=300)
plt.show()