from src.ode.ode import *


def test_ode_FE():
    """ Test that a linear u(t)=a*t+b is exactly reproduced """

    def exact_solution(t):
        return a * t + b

    def f(u, t):
        return a + (u - exact_solution(t)) ** m

    a = 4
    b = -1
    m = 6
    dt = 0.5
    T = 20.0

    u, t = ode_FE(f, exact_solution(0), dt, T)
    diff = abs(exact_solution(t) - u).max()
    tol = 1E-15
    success = diff < tol

    assert success


def test_ode_EC_undamped_linear():
    """Test Euler-Cromer method for a linear problem."""
    from numpy import pi
    omega = 2
    P = 2 * pi / omega
    dt = P / 20
    T = 40 * P
    exact_v = -3.5035725322034139
    exact_u = 0.7283057044967003
    computed_u, computed_v, t = ode_EulerCromer(
        f=lambda v: 0, s=lambda u: omega ** 2 * u,
        F=lambda t: 0, m=1, T=T, U0=2, V0=0, dt=dt)
    diff_u = abs(exact_u - computed_u[-1])
    diff_v = abs(exact_v - computed_v[-1])
    tol = 1E-14
    assert diff_u < tol and diff_v < tol


def _test_manufactured_solution(damping=True):
    import sympy as sp
    t, m, k, b = sp.symbols('t m k b')
    # Choose solution
    u = sp.sin(t)
    v = sp.diff(u, t)
    # Choose f, s, F
    f = b * v
    s = k * sp.tanh(u)
    F = sp.cos(2 * t)

    equation = m * sp.diff(v, t) + f + s - F

    # Adjust F (source term because of manufactured solution)
    F += equation
    print('F:', F)

    # Set values for the symbols m, b, k
    m = 0.5
    k = 1.5
    b = 0.5 if damping else 0
    F = F.subs('m', m).subs('b', b).subs('k', k)

    print(f, s, F)
    # Turn sympy expression into Python function
    F = sp.lambdify([t], F)
    # Define Python functions for f and s
    # (the expressions above are functions of t, we need
    # s(u) and f(v)
    from numpy import tanh
    s = lambda u: k * tanh(u)
    f = lambda v: b * v

    # Add modules='numpy' such that exact u and v work
    # with t as array argument
    exact_u = sp.lambdify([t], u, modules='numpy')
    exact_v = sp.lambdify([t], v, modules='numpy')

    # Solve problem for different dt
    from numpy import pi, sqrt, sum, log
    P = 2 * pi
    time_intervals_per_period = [20, 40, 80, 160, 240]
    h = []  # store discretization parameters
    E_u = []  # store errors in u
    E_v = []  # store errors in v

    for n in time_intervals_per_period:
        dt = P / n
        T = 8 * P
        computed_u, computed_v, t = ode_EulerCromer(
            f=f, s=s, F=F, m=m, T=T,
            U0=exact_u(0), V0=exact_v(0), dt=dt)

        error_u = sqrt(dt * sum((exact_u(t) - computed_u) ** 2))
        error_v = sqrt(dt * sum((exact_v(t) - computed_v) ** 2))
        h.append(dt)
        E_u.append(error_u)
        E_v.append(error_v)

    # Compute convergence rates
    r_u = [log(E_u[i] / E_u[i - 1]) / log(h[i] / h[i - 1])
           for i in range(1, len(h))]
    r_v = [log(E_u[i] / E_u[i - 1]) / log(h[i] / h[i - 1])
           for i in range(1, len(h))]
    tol = 0.02
    exact_r_u = 1.0 if damping else 2.0
    exact_r_v = 1.0 if damping else 2.0
    success = abs(exact_r_u - r_u[-1]) < tol and \
              abs(exact_r_v - r_v[-1]) < tol
    msg = ' u rate: %.2f, v rate: %.2f' % (r_u[-1], r_v[-1])
    assert success, msg


def test_manufactured_solution_ode_EC():
    _test_manufactured_solution(damping=True)
    _test_manufactured_solution(damping=False)
