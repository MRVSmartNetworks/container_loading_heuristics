# -*- coding: utf-8 -*-
import random as r
import matplotlib.pyplot as plt
from matplotlib import collections
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def orthogonal_plane(df_items, df_vehicles, df_sol, idx_vehicle=0, out_file=None):
    """
    orthogonal_plane
    ---
    Display the 2D representation of the solution
    """
    df_sol_vehicle = df_sol[df_sol.idx_vehicle == idx_vehicle]
    id_truck = df_sol_vehicle["type_vehicle"].iloc[0]
    idx_stacks = df_sol_vehicle.id_stack.unique()
    n_stacks = len(df_sol_vehicle.id_stack.unique())
    verts = []
    for ele in idx_stacks:
        data_stack = df_sol_vehicle.query(f"id_stack == '{ele}'").iloc[0]
        df_items["id_item"] = df_items["id_item"].astype(str)
        data_item = df_items.query(f"id_item=='{data_stack.id_item}'").iloc[0]
        x_o = data_stack.x_origin
        y_o = data_stack.y_origin
        # check orientation
        if data_stack["orient"] == "l":
            x_e = x_o + data_item.length
            y_e = y_o + data_item.width
        else:
            x_e = x_o + data_item.width
            y_e = y_o + data_item.length

        verts.append(([x_o, y_o], [x_e, y_o], [x_e, y_e], [x_o, y_e]))

    plot = collections.PolyCollection(
        verts=verts,
        edgecolors="black",
        linewidths=1,
        facecolors=[
            "#" + "".join([r.choice("0123456789ABCDEF") for j in range(6)])
            for i in range(n_stacks)
        ],
    )

    fig, ax = plt.subplots()
    # Set axis limit, given from the dimensions of the vehicle
    ax.set_xlim((0, df_vehicles[df_vehicles["id_truck"] == id_truck].iloc[0]["length"]))
    ax.set_ylim((0, df_vehicles[df_vehicles["id_truck"] == id_truck].iloc[0]["width"]))
    ax.set_aspect("equal")

    ax.set_title(f"Vehicle {id_truck}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.add_collection(plot)
    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
    plt.show()


def stack_3D(df_items, df_vehicles, df_sol, idx_vehicle=0, out_file=None):
    """
    stack_3D
    ---
    Display the 3d representation of the solution using blocks
    of the specified dimensions.

    Input variables:
    - df_items: dataframe containing all the items
    - df_vehicles: dataframe representing the vehicles (bins)
    - df_sol: dataframe representing the solution
    - ids_vehicle: (default 0) it is the vehicle index for the current plot
    """
    # Make sure the item ids are stored as strings, else the queries don't work
    df_items["id_item"] = df_items["id_item"].astype(str)
    df_sol_vehicle = df_sol[df_sol.idx_vehicle == idx_vehicle]
    sol_vehicle_id = df_sol_vehicle.iloc[0]["type_vehicle"]
    sol_vehicle = df_vehicles[df_vehicles.id_truck == sol_vehicle_id]
    idx_stacks = (
        df_sol_vehicle.id_stack.unique()
    )  # Distinct elements in the 'id_stack' column of the solution
    n_stacks = len(
        df_sol_vehicle.id_stack.unique()
    )  # Number of distinct elements in the 'id_stack' column
    coordinates = np.zeros(
        (n_stacks, 3)
    )  # Initialize coordinates of the elements (x,y,z)
    sizes = np.zeros((n_stacks, 3))  # Initialize the sides of the elements (h,w,d)
    i = 0
    for ele in idx_stacks:
        n_items_stack = len(df_sol_vehicle.query(f"id_stack == '{ele}'"))
        sizes[i, 2] = 0
        for j in range(n_items_stack):  # Iterate over the elements in the solution
            data_stack = df_sol_vehicle.query(f"id_stack=='{ele}'").iloc[
                j
            ]  # Extract current element
            data_item = df_items.query(
                f"id_item=='{data_stack.id_item}'"
            )  # Extract current element from the items according to the ID
            sizes[i, 2] += data_item.height
        data_stack = df_sol_vehicle.query(f"id_stack=='{ele}'").iloc[0]
        data_item = df_items.query(f"id_item=='{data_stack.id_item}'")
        coordinates[i, 0] = data_stack.x_origin
        coordinates[i, 1] = data_stack.y_origin
        coordinates[i, 2] = data_stack.z_origin
        # Check orientation, then store the width and length of the box
        if data_stack["orient"] == "w":
            sizes[i, 0] = data_item.width
            sizes[i, 1] = data_item.length
        else:
            sizes[i, 0] = data_item.length
            sizes[i, 1] = data_item.width

        i += 1

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

    # Choose colors randomly
    colors = [
        "#" + "".join([r.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(n_stacks)
    ]
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(coordinates)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(coordinates)

    # Display blocks
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    g = []
    # count = 0
    for p, s, c in zip(coordinates, sizes, colors):
        g.append(cuboid_data(p, size=s))
        # add label
        # ax.text(p[0], p[1], p[2], f"{count}?", color='black')
        # count += 1

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

    ax.set_title(f"Vehicle {idx_vehicle}")
    # Set axis limit, given from the dimensions of the vehicle
    x_lim = ax.set_xlim(0, sol_vehicle.iloc[0]["length"])
    y_lim = ax.set_ylim(0, sol_vehicle.iloc[0]["width"])
    z_lim = ax.set_zlim(0, sol_vehicle.iloc[0]["height"])

    # ax.set_aspect('equal')

    set_axes_equal(ax)

    if out_file is not None:
        plt.savefig(out_file)

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
