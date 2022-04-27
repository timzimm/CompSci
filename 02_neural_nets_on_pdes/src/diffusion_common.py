import numpy as np

def ftcs_cfl_condition(dx):
    return 0.5 * dx**2

def u0(x):
    return np.sin(np.pi * x)

def u_analytical(x, t):
    return np.sin(np.pi * x) *np.exp(-(np.pi)**2 * t)