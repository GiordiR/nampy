from src.root.root import *


def application_root_Newton():
    """ Simple application for the Newton-Raphson's method """
    def f(x):
        return x**2 - 9

    #def dfdx(x):
    #    return 2*x
    dfdx = None
    solution, no_iterations = root_NewtonRaphson(f, 1000, dfdx, 1E-6, 100, return_x_list=True)

    if no_iterations > 0:  # Solution found
        print("Number of function calls: %d" % (1 + 2 * no_iterations))
        print("A solution is: %f" % (solution[-1]))
        print_rates('Newton', solution, 3)
    else:
        print("Solution not found!")


def application_root_secant():
    """ Simple application for the secant method """
    def f(x):
        return x**2 - 9

    x0 = 1000
    x1 = x0 - 1
    solution, no_iterations = root_secant(f, x0, x1, 1.0e-6, 100, return_x_list=True)

    if no_iterations > 0:  # Solution found
        print("Number of function calls: %d" % (2 + no_iterations))
        print("A solution is: %f" % (solution[-1]))
        print_rates('Secant', solution, 3)
    else:
        print("Solution not found!")


def application_root_bisection():
    """ Simple application for the bisection method """
    def f(x):
        return x ** 2 - 9

    a = 0
    b = 1000

    solution, no_iterations = root_bisection(f, a, b, eps=1.0e-6, return_x_list=True)

    print("Number of function calls: %d" % (1 + 2 * no_iterations))
    print("A solution is: %f" % (solution[-1]))
    print_rates('Bisection', solution, 3)


def print_rates(method, x, x_exact):
    q = ['%.2f' % q_ for q_ in rate(x, x_exact)]
    print(method + ': ')
    for q_ in q:
        print(q_)
