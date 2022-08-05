import numpy as np


def trapezoidal(f, a, b, n):
    r"""
    Composite trapezoidal method for integral numerical calculation.

    .. math ::
        \int_{a}^{b} f(x) dx \approx h \left[ \frac{1}{2} f(x_0) + \sum_{i=1}^{n-1} f(x_i) + \frac{1}{2} f(x_n) \right]

    :param f: function.
    :type f: lambda function

    :param a: Lower interval bound.
    :type a: float

    :param b: Upper interval bound.
    :type b: float

    :param n: Number of subdivision.
    :type n: int
    """

    h = float(b - a) / n
    result = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, n):
        result += f(a + i * h)
    result *= h

    return result


def midpoint(f, a, b, n):
    r"""
    Composite trapezoidal method for integral numerical calculation.

    .. math ::
        \int_{a}^{b} f(x) dx \approx h \sum_{i=0}^{n-1} f(x_i)

        where, x_i = a + \frac{h}{2} + ih

    :param f: function.
    :type f: lambda function

    :param a: Lower interval bound.
    :type a: float

    :param b: Upper interval bound.
    :type b: float

    :param n: Number of subdivision.
    :type n: int
    """

    h = float(b - a) / n
    result = 0
    for i in range(n):
        result += f((a + h / 2.0) + i*h)
    result *= h

    return result


def midpoint_double(f, a, b, c, d, nx, ny):
    r"""
    Composite trapezoidal method for double integral numerical calculation.

    .. math ::
        \int_{a}^{b} \int_{c}^{d} f(x, y) dydx \approx h_x h_y \sum_{i=0}^{n_x-1} \sum_{j=0}^{n_y-1} f(x_i, y_j)

        where, x_i = a + \frac{h_x}{2} + ih_x \\
               y_j = c + \frac{h_y}{2} + jh_y

    :param f: function.
    :type f: lambda function

    :param a: Lower interval bound in x.
    :type a: float

    :param b: Upper interval bound in x.
    :type b: float

    :param c: Lower interval bound in y.
    :type c: float

    :param d: Upper interval bound in y.
    :type d: float

    :param nx: Number of subdivision in x.
    :type nx: int

    :param ny: Number of subdivision in y.
    :type ny: int
    """

    hx = (b - a)/float(nx)
    hy = (d - c)/float(ny)
    Integral = 0
    for i in range(nx):
        for j in range(ny):
            xi = a + hx/2 + i*hx
            yj = c + hy/2 + j*hy
            Integral += hx * hy * f(xi, yj)

    return Integral


def midpoint_double2(f, a, b, c, d, nx, ny):
    r"""
    Composite trapezoidal method for double integral numerical calculation.
    Reusing 1D formulation

    :param f: function.
    :type f: lambda function

    :param a: Lower interval bound in x.
    :type a: float

    :param b: Upper interval bound in x.
    :type b: float

    :param c: Lower interval bound in y.
    :type c: float

    :param d: Upper interval bound in y.
    :type d: float

    :param nx: Number of subdivision in x.
    :type nx: int

    :param ny: Number of subdivision in y.
    :type ny: int
    """

    def g(x):
        return midpoint(lambda y: f(x, y), c, d, ny)

    return midpoint(g, a, b, nx)

def midpoint_triple(g, a, b, c, d, e, f, nx, ny, nz):
    r"""
    Composite trapezoidal method for triple integral numerical calculation.
    Reusing 1D formulation

    :param g: function.
    :type g: lambda function

    :param a: Lower interval bound in x.
    :type a: float

    :param b: Upper interval bound in x.
    :type b: float

    :param c: Lower interval bound in y.
    :type c: float

    :param d: Upper interval bound in y.
    :type d: float

    :param e: Lower interval bound in z.
    :type e: float

    :param f: Upper interval bound in z.
    :type f: float

    :param nx: Number of subdivision in x.
    :type nx: int

    :param ny: Number of subdivision in y.
    :type ny: int

    :param nz: Number of subdivision in z.
    :type nz: int
    """

    def p(x, y):
        return midpoint(lambda z: g(x, y, z), e, f, nz)

    def q(x):
        return midpoint(lambda y: p(x, y), c, d, ny)

    return midpoint(q, a, b, nx)


def MonteCarlo_double(f, g, x0, x1, y0, y1, n):
    r"""
    Monte Carlo integration of f over a domain g>=0, embedded
    in a rectangle [x0, x1]x[y0, y1]. n^2 is the number of random
    points.

    :param f: function.
    :type f: lambda function

    :param g: level-set function.
    :type g: lambda function

    :param x0: Lower interval bound in x.
    :type x0: float

    :param x1: Upper interval bound in x.
    :type x1: float

    :param y0: Lower interval bound in y.
    :type y0: float

    :param y1: Upper interval bound in y.
    :type y1: float

    :param n: Square root of the number of sample points.
    :type n: float
    """

    # Draw n**2 random points in the rectangle
    x = np.random.uniform(x0, x1, n)
    y = np.random.uniform(y0, y1, n)

    # Compute sum of f values inside the integration domain
    f_mean = 0
    num_inside = 0  # number of x,y points inside domain (g>=0)
    for i in range(len(x)):
        for j in range(len(y)):
            if g(x[i], y[j]) >= 0:
                num_inside += 1
                f_mean += f(x[i], y[j])
    f_mean = f_mean / float(num_inside)
    area = num_inside / float(n ** 2) * (x1 - x0) * (y1 - y0)

    return area * f_mean
