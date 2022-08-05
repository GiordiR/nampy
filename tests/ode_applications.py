import matplotlib.pyplot as plt
from src.ode.ode import *
from numpy import exp, pi, cos


def application_ode_FE():
    """ Population growth demo"""

    def f(u, t):
        return 0.1 * u

    u, t = ode_FE(f, 100, 0.5, 20)
    plt.plot(t, u, label='Numerical')
    plt.plot(t, 100 * exp(0.1 * t), label='Exact')
    plt.legend()
    plt.grid()
    plt.show()


def application_ode_system_FE():
    """ Spreading diseases demo """

    def f(u, t):
        S, I, R = u
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

    beta = 10. / (40 * 8 * 24)
    gamma = 3. / (15 * 24)
    dt = 0.1
    D = 30
    Nt = int(D * 24 / dt)
    T = dt * Nt
    U0 = [50, 1, 0]

    u, t = ode_system_FE(f, U0, dt, T)

    S = u[:, 0]
    I = u[:, 1]
    R = u[:, 2]

    # Solution plot
    fig = plt.figure()
    l1, l2, l3 = plt.plot(t, S, t, I, t, R)
    fig.legend((l1, l2, l3), ('S', 'I', 'R'), 'lower right')
    plt.xlabel('hours')
    plt.show()

    # Consistency check
    N = S[0] + I[0] + R[0]
    eps = 1E-12
    for n in range(len(S)):
        SIR_sum = S[n] + I[n] + R[n]
        if abs(SIR_sum - N) > eps:
            print('*** consistency check failed: S+I+R=%g != %g' % (SIR_sum, N))


def application_ode_EC_linear_damping():
    """ Linear damping demo"""
    b = 0.3
    f = lambda v: b * v
    s = lambda u: k * u
    F = lambda t: 0

    m = 1
    k = 1
    U_0 = 1
    V_0 = 0

    T = 12 * pi
    dt = T / 5000.

    u, v, t = ode_EulerCromer(f=f, s=s, F=F, m=m, T=T,
                              U0=U_0, V0=V_0, dt=dt)
    plot_u(u, t)


def application_ode_EC_linear_damping_sine_excitation():
    """ Linear damping with sine excitation demo"""
    b = 0.3
    f = lambda v: b * v
    s = lambda u: k * u
    from math import pi, sin
    w = 1
    A = 0.5
    F = lambda t: A * sin(w * t)

    m = 1
    k = 1
    U_0 = 1
    V_0 = 0

    T = 12 * pi
    dt = T / 5000.

    u, v, t = ode_EulerCromer(f=f, s=s, F=F, m=m, T=T,
                              U0=U_0, V0=V_0, dt=dt)
    plot_u(u, t)


def application_ode_EC_sliding_friction():
    """ Sliding friction demo"""
    from numpy import tanh, sign

    f = lambda v: mu * m * g * sign(v)
    alpha = 60.0
    s = lambda u: k / alpha * tanh(alpha * u)
    s = lambda u: k * u
    F = lambda t: 0

    g = 9.81
    mu = 0.4
    m = 1
    k = 1000

    U_0 = 0.1
    V_0 = 0

    T = 2
    dt = T / 5000.

    u, v, t = ode_EulerCromer(f=f, s=s, F=F, m=m, T=T,
                              U0=U_0, V0=V_0, dt=dt)
    plot_u(u, t)


def plot_u(u, t, percentage=100, heading='', labels=('t', 'u')):
    index = int(len(u) * percentage / 100)
    plt.plot(t[-index:], u[-index:], 'b-')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(heading)
    plt.show()


def application_ode_RK2():
    """ Oscillating system RK2 demo"""
    omega = 2
    P = 2 * pi / omega
    dt = P / 20
    T = 10 * P
    X0 = 2
    u, v, t = ode_RK2(X0, omega, dt, T)

    fig = plt.figure()
    l1, l2 = plt.plot(t, u, 'b-', t, X0 * cos(omega * t), 'r--')
    fig.legend((l1, l2), ('numerical', 'exact'), 'upper left')
    plt.xlabel('t')
    plt.show()
