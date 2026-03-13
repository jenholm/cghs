import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# import model pieces
from cghs25 import *

lam_span = (0,80)
lam_eval = np.linspace(0,80,1500)

sol = solve_ivp(
    system,
    lam_span,
    [-11.0,0.1],
    t_eval=lam_eval,
    method="Radau"
)

G = sol.y[0]
S = sol.y[1]

E = E_of(G,S)

# derivative of E with respect to G
dE_dG = np.gradient(E, G)

w_eff = -1 - (2/3)*(dE_dG/E)

plt.figure(figsize=(8,5))
plt.plot(G, w_eff)

plt.axhline(-1,color="black",linestyle=":")
plt.axhline(0,color="gray",linestyle=":")
plt.axhline(1/3,color="gray",linestyle=":")

plt.xlabel("G = ln(a)")
plt.ylabel("w_eff")
plt.title("Effective equation of state")
plt.savefig("Effective equation of state.png")
plt.show()