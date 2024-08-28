# Ant Colony Optimization-based heuristic algorithm for the container loading problem

This repository contains the code for the paper "Ant Colony Optimization
Heuristics for the 3D-BPP with stackable items", where we provide and analyze a
new heuristic approach, based on Ant Colony Optimization (ACO) for the solution
of the "truck loading"/"container loading" problem.

## Table of Contents

<!--toc:start-->

- [Ant Colony Optimization-based heuristic algorithm for the container loading problem](#ant-colony-optimization-based-heuristic-algorithm-for-the-container-loading-problem)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [ACO-CG](#aco-cg)
    - [OR-Tools](#or-tools)
    - [Gurobi (WIP)](#gurobi-wip)
  - [Contributing](#contributing)
  - [License](#license)
  <!--toc:end-->

Abstract:

> Effective allocation and utilization of vehicle space is one of the most
> critical problems in logistics. Often this problem
> is modeled using the three-dimensional bin packing problem which allows for a
> high degree of flexibility in item shapes
> and orientations. Nevertheless, in practice items are built to be stacked. In
> this paper, we deal with this problem,
> called the three-dimensional bin packing problem with stackable items, and we
> propose two heuristics. One based on
> ant colony and one mixing ant colony with column generation. The results
> obtained by applying the technique to a set
> of benchmark instances and a set of realistic ones show promising results.

## Overview

This optimization problem consists in placing a predefined set of boxes
("items") inside a number of trucks, so that the overall cost associated with
the used vehicles is minimized.
In other words, the goal of the problem is to find the best placement of such
items inside trucks so that we minimize the total cost.

The items possess additional characteristics, such as the possibility (or not)
to be rotated along the horizontal plane (i.e., by keeping the same orientation
along the z axis), and the possibility to stack items that are assigned the
same "stackability code".
The latter property allows us to solve the overall optimization problem by
first creating "stacks" of items (i.e., place items that can be stacked along
the z axis), and then placing these stacks inside a specific truck, effectively
solving the problem as a 2D bin packing.
The ACO mechanism comes into play for the solution of 2D bin packing.
Additionally, we pair this approach with column generation to improve the
performance in terms of finding the optimal result when compared to a
standalone run of the ACO alone.

We provide a comparison of our heuristic algorithm with an exact solver of 3D
bin packing implemented using Google's [OR
Tools](https://developers.google.com/optimization) library.

## Installation

To be able to replicate the paper results the first steps to follow are to
download and intall the correct packages:

1. Clone the repository:

   ```bash
   git clone https://github.com/ (to complete)
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To see all the available options and/or automatically run all the tests using
selected solvers, run:

```bash
python test_all_datasets.py --help
```

### ACO-CG

To obtain the solutions of ACO with column generation:

```bash
python3 test_all_datasets.py --solver master-aco --dataset realistic-ds
```

ACO-CG is also able to execute the other datasets passing the arguments:

- `mod-ds`
- `ivancic-ds`
- `beng-ds`
- `exact-ds`

All dataset can then be executed with the following command:

```bash
python3 test_all_datasets.py --solver master-aco --dataset realistic-ds mod-ds ivancic-ds exact-ds
```

**Note:** 1 or more arguments can be passed.

### OR-Tools

To obtain the OR-Tools solutions:

```bash
python3 test_all_datasets.py --solver or-tools --dataset mod-ds
```

We suggest to use OR-Tools only on the smaller instances to prevent excessively
long runtimes.

### Gurobi (WIP)

We are working on supporting a Gurobi-based exact solver we designed earlier on
to provide another comparison.

To obtain the Exact Solver solutions: **(WIP)**

```bash
python3 test_all_datasets.py --solver exact-solver --dataset exact-ds
```

The Exact Solver with Gurobi can only run on the exact datasets because of ther
dimensions(specific created dataset to test perfect truck filling).

**Note**: If you want to run a single dataset `main.py` can be used. In the
`main` is possible to change the dataset name and the solver used.

## Contributing

If you'd like to contribute to this project:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes and commit them.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.

## License

This project is licensed under the [MIT License](https://github.com/MRVSmartNetworks/container_loading_heuristics/LICENSE).
