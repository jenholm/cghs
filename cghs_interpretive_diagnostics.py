import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

source = mu_S * logistic_u(S)

E = E_of(G,S)

potential = -(kappa/(3*E))*dOmegaS_dS(S)

damping = -gamma_S*S*(1-logistic_u(S))

plt.figure(figsize=(8,6))

plt.plot(G,source,label="source")
plt.plot(G,potential,label="potential")
plt.plot(G,damping,label="damping")

plt.axhline(0,color="black")

plt.xlabel("G")
plt.ylabel("Contribution to dS/dλ")

plt.title("Hilbert sector forces")

plt.legend()
plt.savefig('interpretive_diagnostics.png')
plt.show()