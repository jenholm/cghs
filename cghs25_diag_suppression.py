import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Cosmology parameters
# ============================================================

Omega_r0 = 9e-5
Omega_m0 = 0.30

# ============================================================
# Hilbert parameters
# ============================================================

S_c = 8.0
width = 2.0

# ============================================================
# Initial condition to test
# ============================================================

G0 = -11.0
S0 = 0.1

EXP_CLIP = 700


# ============================================================
# Helpers
# ============================================================

def safe_exp(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


def logistic_u(S):

    x = (S - S_c) / width
    x = np.clip(x, -60, 60)

    return 1.0 / (1.0 + np.exp(x))


# ============================================================
# Classical sector terms
# ============================================================

def classical_terms(G, S, n):

    u = logistic_u(S)

    f = (1.0 - u)**n

    rad = Omega_r0 * f * safe_exp(-4*G)
    mat = Omega_m0 * f * safe_exp(-3*G)

    return rad, mat, f


# ============================================================
# Scan suppression powers
# ============================================================

powers = [1,2,4,6,8,10]

results = []

for n in powers:

    rad, mat, f = classical_terms(G0, S0, n)

    results.append((n, f, rad, mat))


# ============================================================
# Print audit
# ============================================================

print("\n=== Suppression power scan ===\n")

print(f"G0 = {G0}")
print(f"S0 = {S0}")
print(f"u(S0) = {logistic_u(S0):.6e}")
print()

for n, f, rad, mat in results:

    print(f"n = {n}")
    print(f"f_internal = {f:.6e}")
    print(f"radiation term = {rad:.6e}")
    print(f"matter term    = {mat:.6e}")
    print()


# ============================================================
# Continuous S scan
# ============================================================

S_grid = np.linspace(0,20,1000)

fig,ax = plt.subplots(figsize=(10,6))

for n in powers:

    u = logistic_u(S_grid)
    f = (1-u)**n

    rad = Omega_r0 * f * safe_exp(-4*G0)

    ax.plot(S_grid, rad, label=f"n={n}")

ax.set_yscale("log")
ax.set_title("Radiation term vs S for different suppression powers")
ax.set_xlabel("S")
ax.set_ylabel("radiation term")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cghs25_diag_suppression_scan.png", dpi=300)
plt.show()