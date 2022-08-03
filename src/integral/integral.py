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

        where, x_i = (a + \frac{h}{2} + ih

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


# TODO: Simpson's rule, Gauss quadrature
