# Test for trapezoidal
from nampyPrj import trapezoidal, midpoint
from math import exp


def application_trapezoidal():
    # Compute numerical result
    f_numerical = lambda t: 3 * (t ** 2) * exp(t ** 3)
    n = 400
    numerical = trapezoidal(f_numerical, 0, 1, n)

    # Compute exact result
    f_exact = lambda t: exp(t ** 3)
    exact = f_exact(1) - f_exact(0)

    # Compute relative error
    error = abs((exact - numerical) / exact) * 100

    print("Numerical solution: %.16f" % numerical)
    print("Exact solution: %.16f" % exact)
    print("Error: %g" % error)


def application_midpoint():
    # Compute numerical result
    f_numerical = lambda t: 3 * (t ** 2) * exp(t ** 3)
    n = 400
    numerical = midpoint(f_numerical, 0, 1, n)

    # Compute exact result
    f_exact = lambda t: exp(t ** 3)
    exact = f_exact(1) - f_exact(0)

    # Compute relative error
    error = abs((exact - numerical) / exact) * 100

    print("Numerical solution: %.16f" % numerical)
    print("Exact solution: %.16f" % exact)
    print("Error: %g" % error)


def application_integral():
    # Integrand
    f_numerical = lambda t: 3 * (t ** 2) * exp(t ** 3)

    # Compute exact result
    f_exact = lambda t: exp(t ** 3)
    exact = f_exact(1) - f_exact(0)

    print("Exact solution: %.16f" % exact)
    print('%7s %16s %12s %16s %12s' % (str('n'), str('trapezoidal'), str('err'), str('midpoint'), str('err')))
    for i in range(1, 21):
        n = 2 ** i
        numerical_trapezoidal = trapezoidal(f_numerical, 0, 1, n)
        numerical_midpoint = midpoint(f_numerical, 0, 1, n)

        # Compute relative error
        error_trapezoidal = abs((exact - numerical_trapezoidal) / exact) * 100
        error_midpoint = abs((exact - numerical_midpoint) / exact) * 100

        print("%7d %.16f %12g %.16f %12g" % (n, numerical_trapezoidal,
                                             error_trapezoidal, numerical_midpoint,
                                             error_midpoint))
