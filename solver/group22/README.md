# Decision rule-based heuristics

Group 22 - Davide Macario

---

## TODO

- How to create the stacks? (introduce some random behavior - VERY IMPORTANT)
- End method for creating stacks (`create_stack`)
- End method for filling 2D slice (`fill_width`)
- Understand how to place the stacks (keeping track of the reference system)
- Put together the solution - how to use the one provided by the prof?

## Data structures

**df_items**: pandas dataframe

- Columns
  - id_item
  - length (> width)
  - width
  - height
  - weight
  - nesting_height: it is the height that has to be subtracted when stacking *another* element on top of the current one (*for most items it is 0*) - $\bar{h}_s$
  - stackability code: identify elements that can be stacked (same length and width *if same code*)
  - forced orientation: it represents how the object has to be placed, either width-wise or length-wise (or neither); values: 'w' - width-wise, 'n' - no constraints
  - max_stackability: maximum number of (same) elements in the stack

**df_vehicles**: pandas dataframe

- Columns
  - id_truck
  - length
  - width
  - height
  - max_weight: maximum *total* supported weight
  - max_weight_stack: maximum weight of each stack of items
  - cost: truck cost (the total cost is what needs to be optimized)
  - max_density: max weight/surface ratio that can be withstood by the trailer

## Reference system and solution representation

<img src="./img_md/truck_ref_sys.png" alt="Truck reference system" style="width: 500px"/>

The result of the optimization consists of the stacks.
Stacks are identified by:

- Total weight: $\omega_s = \sum_{i\in I_s} \omega_i$
- Orientation of the whole stack
- Coordinates of the *origin*: $(x_s^o, y_s^o, z_s^o)$ - point closest to the truck origin
- Coordinates of the *extremity point*: $(x_s^e, y_s^e, z_s^e)$ - point dfarthest from the truck origin
- Stack height: $h_s = \sum_{i\in I_s} h_i - \sum_{i\in I_s,\ i≠bottom\,item} \bar{h}_s$

<img src="./img_md/truck_stacks_info.png" alt="Stacks identification" style="width: 500px"/>

Constraints:

- $x_s^e - x_s^o = l_s$ and $y_s^e - y_s^o = w_s$ if stack $s$ is oriented length-wise
- $x_s^e - x_s^o = w_s$ and $y_s^e - y_s^o = l_s$ if stack $s$ is oriented width-wise
- $z_s^o = 0$ and $z_s^e = h_s$

Additionally, the truck must be filled starting from the origin (no spaces allowed between the stacks - adjacent stacks must 'touch' one another).

*Idea*: solution representation based on ordered lists of items (from bottom to top), whose position is identified by the origin of the stack only and the orientation.

## ILP model

The problem consists in a 3d extension of the knapsack problem. It is useful, however to look at it from the 2D perspective, as the $z$ dimension is developed according to the stacking of object of the same 2d size (e.g., stack until constraints are violated).

## Proposed heuristics

### *Some possible decisions/approaches*

Choice of trucks: evaluate the *volume/cost* ratio for each truck and choose trucks based on higher value.
(**Decision rule**)

An effective approach has been found to be that of proceeding in 'slices' along the Y directions, i.e., by filling the truck from the 'beginning' with different layers (see: "Peak Filing Slices Push" - ["A New Heuristic Algorithm for the 3D Bin Packing Problem"](https://link.springer.com/chapter/10.1007/978-1-4020-8735-6_64)).
The fundamental idea of the heuristic is to use decision rules to fill the slices, and then to 'push' the slices towards the 'beginning' of the truck.

This approach can be complicated by [?]

For what conserns the z dimension, it can be tried to simply create stacks when filling the slices... (might be easier said then done).

In general, maximize the amount of 'y' dimension occupied, e.g., when choosing the objects, evaluate all the possible stacking options and choose the one which minimizes the difference between $W_i$ (truck width) and the total width of the stacks.
Possible optimizations of this require the widest elements to be placed first (left, looking in the direction $-\textbf{x}$) in each slice.

### *Needed utility functions*

- Function for removing elements from the list of available ones
- Function for evaluating stack parameters

## Useful links

- [3D bin packing heuristics](https://github.com/bchaiks/3D_Bin_Packing_Heuristics) - useful for solution representation in python
- [Peak Filling Sices Push](~/Documents/Politecnico/A.A.2022_2023/II-semester/operational-research/project/a-new-heuristic-algorithm-for-the-3d-bin-packing-problem.pdf) - found online (similar idea I had)
- [HBP heuristics for 2D/3D bin packing](~/Documents/Politecnico/A.A.2022_2023/II-semester/operational-research/project/5-BPP-4OR-Part-II.pdf) - suggested by professor
- [OR tools website discussion of 3D bin packing](https://developers.google.com/optimization/pack/bin_packing?hl=en) - could be a good starting point either just for the program structure or for some subproblems
