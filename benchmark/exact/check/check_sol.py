# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random as r


def check_above(n_stacks, x_o, y_o, x_e, y_e, df_vehicles):
    stacks = range(n_stacks)

    verts = []
    for i in stacks:
        verts.append(
            ([x_o[i], y_o[i]], [x_e[i], y_o[i]], [x_e[i], y_e[i]], [x_o[i], y_e[i]])
        )

    plot = collections.PolyCollection(
        verts=verts,
        edgecolors="black",
        linewidths=1,
        facecolors=[
            "#" + "".join([r.choice("0123456789ABCDEF") for j in range(6)])
            for i in range(n_stacks)
        ],
    )

    x_max = df_vehicles[0, 0]  # max dato dalle dim del veicolo
    y_max = df_vehicles[0, 1]
    fig, ax = plt.subplots()
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))

    # ax.text(5, 5, 'prova')
    ax.set_title("Vehicle 1")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.add_collection(plot)
    plt.show()


def check_3D(coordinates, sizes, n_items, df_vehicles):
    def cuboid_data(o, size=(1, 1, 1)):
        X = [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        X += np.array(o)
        return X

    colors = [
        "#" + "".join([r.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(n_items)
    ]
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(coordinates)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(coordinates)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    g = []
    count = 0
    for p, s, c in zip(coordinates, sizes, colors):
        g.append(cuboid_data(p, size=s))
        # add label
        ax.text(p[0], p[1], p[2], f"{count}?", color="black")
        count += 1
    ax.set_aspect("auto")

    # Hide axes ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([0])

    """
    ax.xaxis._axinfo['juggled'] = (0, 0, 0)
    ax.yaxis._axinfo['juggled'] = (1, 1, 1)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)
    """

    pc = Poly3DCollection(
        np.concatenate(g),
        facecolors=np.repeat(colors, 6, axis=None),
        edgecolor="k",
        linewidth=0.5,
        alpha=1,
    )

    ax.add_collection3d(pc)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Vehicle 1")
    # Set axis limit
    ax.set_xlim(0, df_vehicles[0, 0])
    ax.set_ylim(0, df_vehicles[0, 1])
    ax.set_zlim(0, df_vehicles[0, 2])
    # set equal aspect
    # ax.set_aspect('equal')

    set_axes_equal(ax)

    plt.show()


def set_axes_equal(ax):
    # From: https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
