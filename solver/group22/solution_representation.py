import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import collections
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from solver.group22.utilities import cuboid_data, set_axes_equal


def myStack3D(df_items, df_vehicles, df_sol, idx_vehicle):
    """
    myStack3D
    ---
    Display the 3D representation of the solution for a single vehicle.

    ### Input variables
    - df_items: dataframe of the items
    - df_vehicles: dataframe of the vehicles
    - df_sol: dataframe of the current solution
    - idx_vehicle: string indicating the specific vehicle
    """
    # Isolate vehicle:
    df_cons = df_sol[df_sol["idx_vehicle"] == str(idx_vehicle)]

    # NOTE: each stack ID is the same for all items that make it up
    idx_stacks = (
        df_cons.id_stack.unique()
    )  # Distinct elements in the 'id_stack' column of the solution
    n_stacks = len(
        df_cons.id_stack.unique()
    )  # Number of distinct elements in the 'id_stack' column

    coordinates = np.zeros(
        (n_stacks, 3)
    )  # Initialize coordinates of the elements (x,y,z)
    sizes = np.zeros((n_stacks, 3))  # Initialize the sides of the elements (h,w,d)

    i = 0
    for sid in idx_stacks:
        # Iterate over stack IDs
        curr_stack = df_cons[df_cons["id_stack"] == sid]
        n_items_stack = len(curr_stack.index)

        # Get 1st element in the stack (bottom element)
        data_stack = curr_stack.iloc[0]
        coordinates[i, 0] = data_stack.x_origin
        coordinates[i, 1] = data_stack.y_origin
        coordinates[i, 2] = data_stack.z_origin  # Always 0...

        assert coordinates[i, 2] == 0, "The stack origin z coordinate is not 0!"

        # Get item information
        data_item = df_items[df_items["id_item"] == data_stack.id_item]
        if data_stack["orient"] == "w":
            sizes[i, 0] = data_item.width.iloc[0]
            sizes[i, 1] = data_item.length.iloc[0]
        else:
            sizes[i, 0] = data_item.length.iloc[0]
            sizes[i, 1] = data_item.width.iloc[0]
        # Get the overall stack height by considering all items in the current stack
        sizes[i, 2] = 0
        # Extract single items
        for j in range(n_items_stack):
            data_stack = curr_stack.iloc[j]
            data_item = df_items[df_items["id_item"] == data_stack.id_item]
            sizes[i, 2] += data_item.height.iloc[0]

        i += 1
    # Up to now:
    # Obtained the stack position and dimensions, which are the necessary info
    # for representing the truck

    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
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
        # Take all 'related' values of
        # - Coordinates (rows)
        # - Sizes (rows)
        # - Colors (single items)

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

    # Extract actual vehicle type ID
    idx_vehicle_type = int(idx_vehicle[1])

    ax.set_title(f"Vehicle {idx_vehicle}")
    # Set axis limit, given from the dimensions of the vehicle
    x_lim = ax.set_xlim(0, df_vehicles.iloc[idx_vehicle_type]["length"])
    y_lim = ax.set_ylim(0, df_vehicles.iloc[idx_vehicle_type]["width"])
    z_lim = ax.set_zlim(0, df_vehicles.iloc[idx_vehicle_type]["height"])

    # ax.set_aspect('equal')

    # def set_aspect_equal_3d(ax):
    #     x_mean = np.mean(x_lim)
    #     y_mean = np.mean(y_lim)
    #     z_mean = np.mean(z_lim)

    #     plot_radius = max([abs(lim - mean_)
    #                     for lims, mean_ in ((x_lim, x_mean),
    #                                         (y_lim, y_mean),
    #                                         (z_lim, z_mean))
    #                     for lim in lims])

    #     ax.set_xlim3d([0, x_mean + plot_radius])
    #     ax.set_ylim3d([0, y_mean + plot_radius])
    #     ax.set_zlim3d([0, z_mean + plot_radius])

    # set_aspect_equal_3d(ax)

    set_axes_equal(ax)

    plt.show()
