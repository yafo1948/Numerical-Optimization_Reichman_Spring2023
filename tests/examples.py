import numpy as np
from math import e


def Q1_circle(x, hess):
    Q = np.array([[1, 0], [0, 1]])

    # f = np.matmul(x.transpose(), np.matmul(Q, x))
    # g = 2 * np.matmul(Q, x)
    f = np.dot(x, Q.dot(x)) #x.T? 
    g = 2 * Q.dot(x)
    if hess:
        return f, g, 2 * Q

    return f, g


def Q2_ellipse(x, hess):
    Q = np.array([[1, 0], [0, 100]])

    # f = np.matmul(x.transpose(), np.matmul(Q, x))
    # g = 2 * np.matmul(Q, x)
    f = np.dot(x, Q.dot(x)) #x.T? 
    g = 2 * Q.dot(x)
    if hess:
        return f, g, 2 * Q

    return f, g


def Q3_rot_ellipse(x, hess):
    W = np.array([[(3**0.5) / 2, -0.5], [0.5, (3**0.5) / 2]])
    Q = np.array([[100, 0], [0, 1]])
    Q = W.T @ Q @ W

    # value = np.matmul(x.transpose(), np.matmul(Q, x))
    # grad = 2 * np.matmul(Q, x)
    f = np.dot(x, Q.dot(x))
    g = 2 * Q.dot(x)
    
    if hess:
        return f, g, 2 * Q

    return f, g


def Rosenbrock(x, hess):
    f = 100 * ((x[1] - x[0] ** 2) ** 2) + ((1 - x[0]) ** 2)
    g = np.array(
        [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
    )

    if hess:
        h = np.array(
            [[-400 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]]
        )
        return f, g.T, h

    return f, g.T


def linear(x, hess):
    a = np.array([4, 3])
    # a = a.T

    f = a.T @ x
    g = a

    if hess:
        return f, g, 0

    return f, g


def exp_tri(x, hess):
    exp1 = x[0] + 3 * x[1] - 0.1
    exp2 = x[0] - 3 * x[1] - 0.1
    exp3 = -x[0] - 0.1

    f = e ** exp1 + e ** exp2 + e ** exp3
    g = np.array([2 * e ** x[0] - e ** -x[0], 3 * e ** (3 * x[1]) - 3 * e ** (-3 * x[1])])

    if hess:
        h = np.array(
            [[e * e ** x[0] + e ** -x[0], 0],
                [0, 9 * e ** (3 * x[1]) + 9 * e ** (-3 * x[1])],]
        )
        return f, g.T, h

    return f, g.T

