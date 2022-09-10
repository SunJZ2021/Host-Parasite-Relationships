import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

be = 0.4  # Singular strategy
al_be = 0.02  # alpha(beta*)
dal_be = 0.025  # alpha'(beta*)
ddal_be = 0.03  # alpha''(beta*)


# The trade-off function
def alpha(x):
    al = al_be - (((dal_be) ** 2) / ddal_be) * (1 - np.exp((ddal_be * (x - be)) / dal_be))
    return al


# Values of parameters for the dynamics of the resident strain
beta_1 = 0.3
rho = 0.012
mu = rho
sigma_S = 1
sigma_I = 2
beta_11beta_1 = 1  # beta_11 = beta_1*2 Or beta_11=beta_1


# Dynamics of the resident strain
def dy_resident(x, t):
    beta_11 = beta_11beta_1 * beta_1
    alpha_1 = alpha(beta_1)
    alpha_11 = alpha(beta_11)
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
beta_m = 0.2
beta_m1 = beta_m
alpha_m = alpha(beta_m)
alpha_1m = alpha(beta_m1)


# Dynamics of the mutant strain
def dy_mutant(x, t):
    dI_mdt = (beta_m * x[0] + beta_m1 * x[1]) * S_e - (beta_1 * I_e + beta_m1 * x[1]) * x[0] - (mu + alpha_m) * x[0]
    dD_1mdt = (beta_1 * I_e + beta_m1 * x[1]) * x[0] + (beta_m * x[0] + beta_m1 * x[1]) * I_e - (mu + alpha_1m) * x[1]
    return dI_mdt, dD_1mdt


# Initial conditions m0 = [I_m0, D_1m0]
m0 = [0.1, 0.05]

ms = odeint(dy_mutant, m0, ts)
I_ms = ms[:, 0]
D_1ms = ms[:, 1]


# The fitness of the mutant strain (the reproduction ratio), currently not used.
#def R_m(beta1, betam, betam1):
#    beta11 = beta_11beta_1 * beta1
#    xs2 = odeint(dy_resident, x0, ts, args=(beta1,))
#    Se = xs2[:, 0][tmax - 1]
#    Ie = xs2[:, 1][tmax - 1]
#    De = xs2[:, 2][tmax - 1]
#    lambda1 = beta1 * Ie + beta11 * De
#    Rm = sigma_S * ((betam + (betam1 / (mu + alpha(betam1))) * sigma_I * lambda1) / (
#            mu + alpha(betam) + sigma_I * lambda1)) * Se + sigma_I * (betam1 / (mu + alpha(betam1))) * Ie
#    return Rm


f = plt.figure()
# Plot of the trade-off function
ax = f.add_subplot(2, 2, 1)
betas = np.linspace(0, 1, 500)
alphas = alpha(betas)
plt.plot(betas, alphas)
plt.xlabel('$\\beta$', fontsize=14)
plt.ylabel('$\\alpha$', fontsize=14)
plt.xlim((0, 1))

# Plots of the population dynamics
ax = f.add_subplot(2, 2, 2)
plt.plot(ts, Ss)
plt.plot(ts, Is)
plt.plot(ts, Ds)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Densities', fontsize=14)
plt.xlim((0, 200))
plt.ylim((0))
plt.legend(['$S$', '$I_1$', '$D_{11}$'], loc='center right')

ax = f.add_subplot(2, 2, 4)
plt.plot(ts, I_ms)
plt.plot(ts, D_1ms)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Densities', fontsize=14)
plt.xlim((0, 200))
plt.ylim((0))
plt.legend(['$I_m$', '$D_{1m}$'], loc='center right')
f.tight_layout()
# plt.show()
