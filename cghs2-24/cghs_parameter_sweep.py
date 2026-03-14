import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cghs25 import *

lam_span = (0,80)
lam_eval = np.linspace(0,80,1500)

mu_values = [0.0005,0.001,0.002,0.005]

plt.figure(figsize=(8,6))

for mu in mu_values:

    mu_S = mu

    sol = solve_ivp(
        system,
        lam_span,
        [-11.0,0.1],
        t_eval=lam_eval,
        method="Radau"
    )

    G = sol.y[0]
    S = sol.y[1]

    plt.plot(G,S,label=f"μ={mu}")

plt.axhline(S_c,color="red",linestyle=":")

plt.xlabel("G")
plt.ylabel("S")

plt.title("Parameter robustness")

plt.legend()
plt.savefig("parameter_robustness")
plt.show()