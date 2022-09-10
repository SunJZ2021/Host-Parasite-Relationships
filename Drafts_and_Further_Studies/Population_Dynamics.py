import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %matplotlib inline

# Values of parameters for the dynamics of the resident strain
rho = 0.00144
mu = 0.014
alpha_1 = 0.01
alpha_11 = 0.02
sigma_S = 1
sigma_I = 2
beta_1 = 0.4
beta_11 = 0.8


# Dynamics of the resident strain
def dy_resident(x, t):
    lambda_1 = beta_1 * x[1] + beta_11 * x[2]
    dSdt = rho - mu * x[0] - sigma_S * lambda_1 * x[0]
    dIdt = sigma_S * lambda_1 * x[0] - (mu + alpha_1 + sigma_I * lambda_1) * x[1]
    dDdt = sigma_I * lambda_1 * x[1] - (mu + alpha_11) * x[2]
    return dSdt, dIdt, dDdt


# Time points
tmax = 2000
ts = np.linspace(0, 200, tmax)

# Initial conditions x0 = [S0, I0, D0]
x0 = [0.4, 0.1, 0.05]

xs = odeint(dy_resident, x0, ts)
Ss = xs[:, 0]
Is = xs[:, 1]
Ds = xs[:, 2]
# Find the equilibrium
S = Ss[tmax - 1]
I = Is[tmax - 1]
D = Ds[tmax - 1]
print("These are the equilibrium points:", "S =", S, "I =", I, "D =", D)

# Values of parameters for the dynamics of the mutant strain
S_e = S  # Equilibrium points
I_e = I  # Equilibrium points
alpha_m = 0.02
alpha_1m = 0.03
beta_m = 0.5
beta_m1 = 0.9


# Dynamics of the mutant strain
def dy_mutant(x, t):
    dI_mdt = (beta_m * x[0] + beta_m1 * x[1]) * S_e - (beta_1 * I_e + beta_m1 * x[1]) * x[0] - (mu + alpha_m) * x[0]
    dD_1mdt = (beta_1 * I_e + beta_m1 * x[1]) * x[0] + (beta_m * x[0] + beta_m1 * x[1]) * I_e - (mu + alpha_1m) * x[1]
    return dI_mdt, dD_1mdt


# Initial conditions m0 = [I_m0, D_1m0]
m0 = [0.02, 0.01]

ms = odeint(dy_mutant, m0, ts)
I_ms = ms[:, 0]
D_1ms = ms[:, 1]

# Plots of the population dynamics
f = plt.figure()
ax = f.add_subplot(1, 2, 1)
plt.plot(ts, Ss)
plt.plot(ts, Is)
plt.plot(ts, Ds)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Densities', fontsize=14)
plt.xlim((0, 200))
plt.ylim((0))
plt.legend(['$S$', '$I_1$', '$D_{11}$'], loc='center right')

ax = f.add_subplot(1, 2, 2)
plt.plot(ts, I_ms)
plt.plot(ts, D_1ms)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Densities', fontsize=14)
plt.xlim((0, 200))
plt.ylim((0))
plt.legend(['$I_m$', '$D_{1m}$'], loc='center right')
f.tight_layout()
# plt.show()
