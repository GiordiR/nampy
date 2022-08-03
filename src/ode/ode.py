from numpy import linspace, zeros, exp


def ode_FE(f, U0, dt, T):
    """
    Forward Euler method to compute the solution of ODE

    :param f: function
    :type f: lambda function

    :param U0: Initial Value
    :type U0: float

    :param dt: Time step
    :type dt: float

    :param T: Final time
    :type T: float
    """

    Nt = int(round(float(T)/dt))
    u = zeros(Nt+1)
    t = linspace(0, Nt*dt, len(u))
    u[0] = U0
    for n in range(Nt):
        u[n+1] = u[n] + dt*f(u[n], t[n])

    return u, t
