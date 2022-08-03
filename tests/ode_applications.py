import matplotlib.pyplot as plt
from src.ode.ode import *

def application_ode_FE():
    """ Population growth demo"""
    def f(u, t):
        return 0.1*u

    u, t = ode_FE(f, 100, 0.5, 20)
    plt.plot(t, u, label='Numerical')
    plt.plot(t, 100*exp(0.1*t), label='Exact')
    plt.legend()
    plt.grid()
    plt.show()
