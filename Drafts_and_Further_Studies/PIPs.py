import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

be = 0.4  # Singular strategy
al_be = 0.02  # alpha(beta*)
dal_be = 0.025  # alpha'(beta*)
ddal_be = 0.03  # alpha''(beta*)

# Values of parameters for the dynamics of the resident strain
rho = 0.012
mu = 0.014
sigma_S = 1
sigma_I = 2

k = 0  # 0 or 1: beta_11 = beta_1*2 Or beta_11=beta_1


# The trade-off function
def alpha(x):
    al = al_be - ((dal_be ** 2) / ddal_be) * (1 - np.exp((ddal_be * (x - be)) / dal_be))
    return al


# Dynamics of the resident strain
def dy_resident(x, t, beta_1=be):
    beta_11 = (1 + k) * beta_1
    lambda_1 = beta_1 * x[1] + beta_11 * x[2]
    dSdt = rho - mu * x[0] - sigma_S * lambda_1 * x[0]
    dIdt = sigma_S * lambda_1 * x[0] - (mu + alpha(beta_1) + sigma_I * lambda_1) * x[1]
    dDdt = sigma_I * lambda_1 * x[1] - (mu + alpha(beta_11)) * x[2]
    return dSdt, dIdt, dDdt


# The fitness of the mutant strain (the reproduction ratio), currently not used.
def R_m(beta1, betam):
    beta11 = (1 + k) * beta1
    betam1 = betam + k * beta1

    # Time points
    tmax = 2000
    ts = np.linspace(0, 200, tmax)

    # Initial conditions x0 = [S0, I0, D0]
    x0 = [0.4, 0.1, 0.05]

    xs2 = odeint(dy_resident, x0, ts, args=(beta1,))
    Se = xs2[:, 0][tmax - 1]
    Ie = xs2[:, 1][tmax - 1]
    De = xs2[:, 2][tmax - 1]

    lambda1 = beta1 * Ie + beta11 * De

    Rm = sigma_S * ((betam + (betam1 / (mu + alpha(betam1))) * sigma_I * lambda1) / (
            mu + alpha(betam) + sigma_I * lambda1)) * Se + sigma_I * (betam1 / (mu + alpha(betam1))) * Ie - 1
    return Rm


steps = 100
betas = np.linspace(0, 1, steps)
beta1s, betams = np.meshgrid(np.linspace(0, 1, steps), np.linspace(0, 1, steps))
Rms = np.frompyfunc(R_m, 2, 1)
PIP = Rms(beta1s, betams)
plt.contourf(beta1s, betams, PIP, levels=[PIP.min(), 0, PIP.max()], colors=['w', 'k'])
plt.plot(betas, betas)
plt.xlabel(r'Resident Strategy, $\beta_1$', fontsize=16)
plt.ylabel(r'Mutant Strategy, $\beta_m$', fontsize=16)
