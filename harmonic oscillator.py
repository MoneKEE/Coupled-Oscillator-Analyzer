import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import ipywidgets as ipw

#Step response
def ode(X, t, zeta, omega0):
    """
    Free Harmonic Oscillator ODE
    """
    x, dotx = X
    ddotx = -2*zeta*omega0*dotx - omega0**2*x
    return [dotx, ddotx]

def update(zeta = 0.05, omega0 = 2.*np.pi):
    """
    Update function.
    """
    X0 = [1., 0.]
    sol = integrate.odeint(ode, X0, t, args = (zeta, omega0))
    line0.set_ydata(sol[:, 0])

# Nt = 1000
# t = np.linspace(0., 10., Nt)
# dummy = np.zeros_like(t)
# fig = plt.figure()
# line0, = plt.plot(t, dummy, label = "position")
# plt.grid()
# plt.ylim(-1., 1.)
# plt.xlabel("Time, $t$")
# plt.ylabel("Amplitude, $a$")
# plt.legend()

# ipw.interact(update, zeta = (0., 1., 0.01),
#            omega0 = (2.*np.pi*0.05, 2.*np.pi*5, 2.*np.pi*0.01));