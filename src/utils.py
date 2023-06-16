import matplotlib.pyplot as plt
import numpy as np

def contour(f, x_hist_gd, x_hist_nm, x_hist_bfgs, x_hist_sr1, x_limit, y_limit, title):

    x_p_gd = []*len(x_hist_gd)
    y_p_gd = []*len(x_hist_gd)
    for p in x_hist_gd:
        x_p_gd.append(p[0])
        y_p_gd.append(p[1])

    x_p_nm = [] * len(x_hist_nm)
    y_p_nm = [] * len(x_hist_nm)
    for p in x_hist_nm:
        x_p_nm.append(p[0])
        y_p_nm.append(p[1])

    x_p_bfgs = []*len(x_hist_bfgs)
    y_p_bfgs = []*len(x_hist_bfgs)
    for p in x_hist_bfgs:
        x_p_bfgs.append(p[0])
        y_p_bfgs.append(p[1])

    x_p_sr1 = [] * len(x_hist_sr1)
    y_p_sr1 = [] * len(x_hist_sr1)
    for p in x_hist_sr1:
        x_p_sr1.append(p[0])
        y_p_sr1.append(p[1])


    X = np.linspace(x_limit[0], x_limit[1], 100)
    Y = np.linspace(y_limit[0], y_limit[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            x = X[i, j]
            y = Y[i, j]
            position = np.array([x, y])
            Z[i, j], a, b = f(position, False)


    plt.figure()
    plt.contour(X, Y, Z, levels=10, title=title)

    plt.plot(x_p_gd, y_p_gd, "xr-", label="Gradient Descent")

    plt.plot(x_p_nm, y_p_nm, 'xb-', label="Newton Method")

    plt.plot(x_p_bfgs, y_p_bfgs, 'xy-', label="BFGS")

    plt.plot(x_p_sr1, y_p_sr1, 'xm-', label="sr1")
    plt.legend()
    plt.title(label=title)
    plt.show()


def plot(fx_hist_gd, fx_hist_nm, fx_hist_bfgs, fx_hist_sr1, title):
    i = 0
    iter_gd = []
    for i in range(len(fx_hist_gd)):
        iter_gd.append(i)

    iter_nm = []
    for i in range(len(fx_hist_nm)):
        iter_nm.append(i)

    iter_bfgs = []
    for i in range(len(fx_hist_bfgs)):
        iter_bfgs.append(i)

    iter_sr1 = []
    for i in range(len(fx_hist_sr1)):
        iter_sr1.append(i)

    plt.plot(iter_gd, fx_hist_gd, label="Gradient Descent")
    plt.plot(iter_nm, fx_hist_nm, label="Newton Method")

    plt.plot(iter_bfgs, fx_hist_bfgs, label="BFGS")
    plt.plot(iter_sr1, fx_hist_sr1, label="sr1")
    plt.title(label=title)
    plt.legend()
    plt.show()










