import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import warnings

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


def plot_iterations(
    title, obj_values_1=None, obj_values_2=None, label_1=None, label_2=None
):

    fig, ax = plt.subplots()
    if obj_values_1 is not None:
        ax.plot(range(len(obj_values_1)), obj_values_1, label=label_1)

    if obj_values_2 is not None:
        ax.plot(range(len(obj_values_2)), obj_values_2, label=label_2)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("# iterations")
    ax.set_ylabel("Objective function value")
    plt.show()


def plot_feasible_set_2d(path_points):
    # plot the feasible region
    d = np.linspace(-2, 4, 300)
    x, y = np.meshgrid(d, d)
    plt.imshow(
        ((y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)).astype(int),
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="Greys",
        alpha=0.3,
    )

    # plot the lines defining the constraints
    x = np.linspace(0, 4, 2000)
    # y >= -x + 1
    y1 = -x + 1
    # y <= 1
    y2 = np.ones(x.size)
    # y >= 0
    y3 = np.zeros(x.size)

    if path_points is not None:
        x_path = [path_points[i][0] for i in range(len(path_points))]
        y_path = [path_points[i][1] for i in range(len(path_points))]

    # Make plot
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(np.ones(x.size) * 2, x)
    plt.plot(
        x_path,
        y_path,
        label="algorithm's path",
        color="k",
        marker=".",
        linestyle="--",
    )
    plt.xlim(0, 3)
    plt.ylim(0, 2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.suptitle('Feasible region and path 2D')
    plt.show()


def plot_feasible_regions_3d(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='gold', marker='o', label='Final candidate')
    ax.set_title("Feasible Regions and Path")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    ax.view_init(45, 45)
    plt.show()






