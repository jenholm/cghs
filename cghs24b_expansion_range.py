import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Cosmological parameters (approximate ΛCDM reference values)
# ============================================================

Omega_r0 = 9e-5
Omega_m0 = 0.30

# Hilbert-sector parameters (test values)
Omega_inf = 50
S_c = 8.0
width = 0.6


# ============================================================
# Geometry mapping
# ============================================================

def scale_factor(G):
    """a = e^G"""
    return np.exp(G)


# ============================================================
# Hilbert activation function f(S)
# ============================================================

def f_S(S):
    """
    Hilbert activation of classical sectors.
    Logistic switch from 0 → 1 around S_c.
    """
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# Radiation contribution
# ============================================================

def radiation_term(G, S):
    """
    Ω_r0 f(S) e^{-4G}
    """
    return Omega_r0 * f_S(S) * np.exp(-4 * G)


# ============================================================
# Matter contribution
# ============================================================

def matter_term(G, S):
    """
    Ω_m0 f(S) e^{-3G}
    """
    return Omega_m0 * f_S(S) * np.exp(-3 * G)


# ============================================================
# Hilbert vacuum energy
# ============================================================

def omega_S(S):
    """
    Hilbert vacuum energy term Ω_S(S)

    Large when S < S_c (inflation)
    Drops after transition.
    """
    x = (S - S_c) / width
    x = np.clip(x, -60, 60)
    return Omega_inf / (1.0 + np.exp(x))


# ============================================================
# Total expansion function
# ============================================================

def E2(G, S):
    """
    E(G,S)^2
    """
    return (
        radiation_term(G, S)
        + matter_term(G, S)
        + omega_S(S)
    )


# ============================================================
# Test inputs
# ============================================================

# Geometry range (log scale factor)
G = np.linspace(-15, 5, 1000)

# Fix Hilbert sector for testing
#S_test = 10.0
S_values = [0, S_c, 2*S_c]

for S_test in S_values:

    rad = radiation_term(G, S_test)
    mat = matter_term(G, S_test)
    vac = omega_S(S_test) * np.ones_like(G)
    E_total = rad + mat + vac

    plt.plot(G, E_total, label=f"S = {S_test}")

plt.savefig("stest_range.png", dpi=300)

# ============================================================
# Evaluate terms
# ============================================================

rad = radiation_term(G, S_test)
mat = matter_term(G, S_test)
vac = omega_S(S_test) * np.ones_like(G)
E_total = rad + mat + vac


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(G, rad, label="Radiation term")
plt.plot(G, mat, label="Matter term")
plt.plot(G, vac, label="Hilbert vacuum Ω_S")
plt.plot(G, E_total, linewidth=3, label="Total E(G,S)^2")

plt.yscale("log")

plt.xlabel("G = ln(a)")
plt.ylabel("Contribution to E(G,S)^2")
plt.title("Dissection of Expansion Equation Terms")

plt.legend()
plt.grid(True)
plt.savefig("radiation_matter_egs.png", dpi=300)
plt.show()