from nampy.integral.integral import *
from nampy.integral.integral_vec import *


def test_trapezoidal_one_exact_result():
    """Compare one hand-computed result"""
    from math import exp

    # Compute numerical result
    f_numerical = lambda t: 3 * (t ** 2) * exp(t ** 3)
    n = 2
    computed = trapezoidal(f_numerical, 0, 1, n)
    expected = 2.463642041244344
    error = abs(expected - computed)
    tol = 1E-14
    success = error < tol
    msg = 'error=%g > tol0%g' % (error, tol)

    assert (success, msg)


def test_trapezoidal_linear():
    """Check that linear functions are integrated exactly"""
    f = lambda x: 6 * x - 4
    F = lambda x: 3 * x ** 2 - 4 * x
    a = 1.2
    b = 4.4

    expected = F(b) - F(a)
    tol = 1E-14
    for n in 2, 20, 21:
        computed = trapezoidal(f, a, b, n)
        error = abs(expected - computed)
        success = error < tol
        msg = 'error=%g > tol0%g' % (error, tol)
        assert (success, msg)


def convergence_rates_trapezoidal(f, F, a, b, num_experiments=14):
    """ Calculate the convergence rates """
    from math import log
    from numpy import zeros

    expected = F(b) - F(a)
    n = zeros(num_experiments, dtype=int)
    E = zeros(num_experiments)
    r = zeros(num_experiments - 1)
    for i in range(num_experiments):
        n[i] = 2 ** (i + 1)
        computed = trapezoidal(f, a, b, n[i])
        E[i] = abs(expected - computed)
        if i > 0:
            r_im1 = log(E[i] / E[i - 1]) / log(float(n[i]) / n[i - 1])
            r[i - 1] = float('%.2f' % r_im1)  # Truncate to two decimals
    return r


def test_trapezoidal_conv_rate():
    """Check empirical convergence rates against the expected -2."""
    from math import exp

    v = lambda t: 3 * (t ** 2) * exp(t ** 3)
    V = lambda t: exp(t ** 3)
    a = 1.1
    b = 1.9
    r = convergence_rates_trapezoidal(v, V, a, b, 14)
    print(r)
    tol = 0.01
    msg = str(r[-4:])  # show last 4 estimated rates
    assert (abs(r[-1]) - 2) < tol, msg


def test_midpoint_one_exact_result():
    """Compare one hand-computed result"""
    from math import exp

    # Compute numerical result
    f_numerical = lambda t: 3 * (t ** 2) * exp(t ** 3)
    n = 2
    computed = midpoint(f_numerical, 0, 1, n)
    expected = 2.463642041244344
    error = abs(expected - computed)
    tol = 1E-14
    success = error < tol
    msg = 'error=%g > tol0%g' % (error, tol)

    assert (success, msg)


def test_midpoint_linear():
    """Check that linear functions are integrated exactly"""
    f = lambda x: 6 * x - 4
    F = lambda x: 3 * x ** 2 - 4 * x
    a = 1.2
    b = 4.4

    expected = F(b) - F(a)
    tol = 1E-14
    for n in 2, 20, 21:
        computed = midpoint(f, a, b, n)
        error = abs(expected - computed)
        success = error < tol
        msg = 'error=%g > tol0%g' % (error, tol)
        assert (success, msg)


def convergence_rates_midpoint(f, F, a, b, num_experiments=14):
    """ Calculate the convergence rates """
    from math import log
    from numpy import zeros

    expected = F(b) - F(a)
    n = zeros(num_experiments, dtype=int)
    E = zeros(num_experiments)
    r = zeros(num_experiments - 1)
    for i in range(num_experiments):
        n[i] = 2 ** (i + 1)
        computed = midpoint(f, a, b, n[i])
        E[i] = abs(expected - computed)
        if i > 0:
            r_im1 = log(E[i] / E[i - 1]) / log(float(n[i]) / n[i - 1])
            r[i - 1] = float('%.2f' % r_im1)  # Truncate to two decimals
    return r


def test_midpoint_conv_rate():
    """Check empirical convergence rates against the expected -2."""
    from math import exp

    v = lambda t: 3 * (t ** 2) * exp(t ** 3)
    V = lambda t: exp(t ** 3)
    a = 1.1
    b = 1.9
    r = convergence_rates_midpoint(v, V, a, b, 14)
    print(r)
    tol = 0.01
    msg = str(r[-4:])  # show last 4 estimated rates
    assert (abs(r[-1]) - 2) < tol, msg


def test_trapezoidal_vec_linear():
    """Check that linear functions are integrated exactly"""
    f = lambda x: 6 * x - 4
    F = lambda x: 3 * x ** 2 - 4 * x
    a = 1.2
    b = 4.4

    expected = F(b) - F(a)
    tol = 1E-14
    for n in 2, 20, 21:
        computed = trapezoidal_vec(f, a, b, n)
        error = abs(expected - computed)
        success = error < tol
        msg = 'error=%g > tol0%g' % (error, tol)
        assert (success, msg)


def test_midpoint_vec_linear():
    """Check that linear functions are integrated exactly"""
    f = lambda x: 6 * x - 4
    F = lambda x: 3 * x ** 2 - 4 * x
    a = 1.2
    b = 4.4

    expected = F(b) - F(a)
    tol = 1E-14
    for n in 2, 20, 21:
        computed = midpoint_vec(f, a, b, n)
        error = abs(expected - computed)
        success = error < tol
        msg = 'error=%g > tol0%g' % (error, tol)
        assert (success, msg)


def test_midpoint_double():
    """Test that a linear function is integrated exactly."""
    def f(x, y):
        return 2 * x + y

    a = 0
    b = 2
    c = 2
    d = 3
    import sympy
    x, y = sympy.symbols('x y')
    I_expected = sympy.integrate(f(x, y), (x, a, b), (y, c, d))
    # Test three cases: nx < ny, nx = ny, nx > ny
    for nx, ny in (3, 5), (4, 4), (5, 3):
        I_computed1 = midpoint_double(f, a, b, c, d, nx, ny)
        I_computed2 = midpoint_double2(f, a, b, c, d, nx, ny)
        tol = 1E-14
        assert (abs(I_computed1 - I_expected) < tol)
        assert (abs(I_computed2 - I_expected) < tol)


def test_midpoint_triple():
    """Test that a linear function is integrated exactly."""
    def g(x, y, z):
        return 2 * x + y - 4 * z

    a = 0
    b = 2
    c = 2
    d = 3
    e = -1
    f = 2

    import sympy

    x, y, z = sympy.symbols('x y z')
    I_expected = sympy.integrate(g(x, y, z), (x, a, b), (y, c, d), (z, e, f))
    for nx, ny, nz in (3, 5, 2), (4, 4, 4), (5, 3, 6):
        I_computed = midpoint_triple(g, a, b, c, d, e, f, nx, ny, nz)
        tol = 1E-14
        assert (abs(I_computed - I_expected) < tol)


def test_MonteCarlo_double_rectangle_area():
    """Check the area of a rectangle."""
    def g(x, y):
        return 1 if (0 <= x <= 2 and 3 <= y <= 4.5) else -1

    x0 = 0
    x1 = 3
    y0 = 2
    y1 = 5  # embedded rectangle
    n = 1000
    np.random.seed(8)  # must fix the seed!
    I_expected = 3.121092  # computed with this seed
    I_computed = MonteCarlo_double(lambda x, y: 1, g, x0, x1, y0, y1, n)
    assert (abs(I_expected - I_computed) < 1E-14)


def test_MonteCarlo_double_circle_r():
    """Check the integral of r over a circle with radius 2."""
    def g(x, y):
        xc, yc = 0, 0  # center
        R = 2  # radius
        return R ** 2 - ((x - xc) ** 2 + (y - yc) ** 2)

    # Exact: integral of r*r*dr over circle with radius R becomes
    # 2*pi*1/3*R**3
    import sympy
    r = sympy.symbols('r')
    I_exact = sympy.integrate(2 * sympy.pi * r * r, (r, 0, 2))
    print('\nExact integral: ', I_exact.evalf())
    x0 = -2
    x1 = 2
    y0 = -2
    y1 = 2
    n = 1000
    np.random.seed(6)
    I_expected = 16.7970837117376384  # Computed with this seed
    I_computed = MonteCarlo_double(lambda x, y: np.sqrt(x ** 2 + y ** 2),
                                   g, x0, x1, y0, y1, n)
    print('MC approximation %d samples: %.16f' % (n ** 2, I_computed))
    assert (abs(I_expected - I_computed) < 1E-15)
