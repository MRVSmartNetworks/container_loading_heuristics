# Notes about exact solver

* [x] I'm missing some additional functions (`from check_sol import check_above, check_3D`)
* [x] No more constraint on length >= width?? - YES
* [x] Lines 176-183 constraints?? Why the height?? - SOLVED
* [x] Make sure on same page about orientation (TODO) - see constraints on R\[i\] - DONE ('normal' direction is *lengthwise*)
* [x] Add constraint $W_{i, j} \geq V_{i, j}\ \forall i, j$ - *Maybe W is not needed*
* [x] What's going on with test_6? There are superimposing items, meaning the constraint on the correct separation of the items is not working! - fixed with review of orientation flag
* [x] Add time limit to solver (able to get solution from 'set_small')

---

* [ ] It may be useful to 'weight' the coordinates in the objective function - favor stacking? IMO in this way the solution can be driven towards 'building up', but not sure if convergence time will improve
* [ ] Ensure all necessary constraints are present
* [ ] Write down full model (see below)
* [ ] Discuss possiblity of adding constraints $B_{i,j,0} = B_{j,i,0} = B_{i,j,1} = B_{j,i,1} = 0$ if $V_{i,j} = 1$

## Work

* Try to build easy datasets with known solution and analyze the solver behavior
* Updated constraint relating V and B (changed inequality direction)

### Questions

* What is the meaning of B? Does it mean item i *completely before* item j? Some constraints imply it, some others simply require coordinate of i <> coordinate of j
  * Constraints to be reviewed: 3, 6, 7, 9, 11, 12.

## Model

Description of the LP model.

### Variables

* $V_{i,j}$: 1 if item $i$ is *immediately* below item $j$ (**stacked**)
* ~~$W_{i,j}$: 1 if item $i$ is below item $j$~~ - **Removed**
* $B_{i,j,d}$: 1 if item $i$ is (*strictly*) *before* item $j$ along direction $d$ (**not necessarily stacked**)
* $R_i$: 1 if item $i$ is rotated width-wise (i.e., swap length and width in constraints)
* $X_{i,d}$: coordinates of item $i$ ($x, y, z$)

### Equations

Obj. function: $\min{\sum_i \sum_d X_{i, d}}$

1. Element $i$ fits entirely in truck:
   * $X_{i,0} + dim_i(0)\cdot (1-R_i) + dim_i(1)\cdot R_i \leq V_{length}\ \forall i$
   * $X_{i,1} + dim_i(1)\cdot (1-R_i) + dim_i(0)\cdot R_i \leq V_{width}\ \forall i$
   * $X_{i,2} = dim_i(2) \leq V_{height}\ \forall i$
2. No two items have the same coordinates:
   * $\sum_d B_{i,j,d} + B_{j,i,d} \geq 1\ \forall i,\ j>i$
3. Ensure items do not overlap along any direction (if item $i$ is before $j$, the coord. of item $j$ must be bigger by at least item $i$'s dimension):
   * $X_{j,0} \geq X_{i,0} + (dim_i(0)\cdot (1-R_i) + dim_i(1)\cdot R_i) - 10000\cdot (1-B_{i,j,0})\ \forall i\neq j$
   * $X_{j,1} \geq X_{i,1} + (dim_i(1)\cdot (1-R_i) + dim_i(0)\cdot R_i) - 10000\cdot (1-B_{i,j,1})\ \forall i\neq j$
   * $X_{j,2} \geq X_{i,2} + dim_i(2) - h_{nesting,i} - 10000\cdot (1-B_{i,j,2})\ \forall i\neq j$
4. If item $i$ is immediately below $j$, then item $i$ comes before $j$ along z:
   * <mark>[**UPDATED**]</mark> $V_{i,j} \leq B_{i,j,2}\ \forall i, j$  (switched verse)
5. Items cannot be 'before' themselves:
    * $B_{i,i,d} = 0\ \forall i$ (redundant - see above and below)
    * $V_{i,i} = 0\ \forall i$
6. <mark>[**NEW**]</mark> Adjust relative positioning if stacked items:
   * $B_{i,j,0} \leq (1-V_{i,j})\ \forall i,j$
   * $B_{j,i,0} \leq (1-V_{i,j})\ \forall i,j$ (note: still $V_{i,j}$)
   * $B_{i,j,1} \leq (1-V_{i,j})\ \forall i,j$
   * $B_{j,i,1} \leq (1-V_{i,j})\ \forall i,j$
7. <mark>[**UPDATED**]</mark> Enforce checks on stackability code $sc_i$ (Removed constraints on B, as items having $B_{i,j,2}=1$ may be on different stacks)
   * If $sc_i \neq sc_j$:
     * $V_{i,j} = 0$
     * $V_{j,i} = 0$
8. Similar to above, but enforce same coordinate ($M$: big-m):
   * $X_{i,d} - X_{j,d} \leq M \cdot (1-V_{i,j})\ \forall i\neq j, d=1,2$
   * $X_{i,d} - X_{j,d} \geq -M \cdot (1-V_{i,j})\ \forall i\neq j, d=1,2$
9. Max stackability <mark>[to be reviewed - here $B_{i,j} = 1$ assumes i in same stack as j]</mark>:
   * $\sum_j B_{i,j,2} \leq ms_i - 1$
10. Rotation:
    * $R_i = 1 \iff i\text{: "widthwise"}$ (else: 0)
11. Maximum stack weight constraint <mark>[to be reviewed - here $B_{i,j} = 1$ assumes i in same stack as j]</mark>:
    * $w_i + \sum_j w_j\cdot B_{i,j,2} \leq \text{maxWtStack}$
12. Maximum density constraint <mark>[to be reviewed - here $B_{i,j} = 1$ assumes i in same stack as j]</mark>:
    * $w_i + \sum_j w_j\cdot B_{i,j,2} \leq \text{maxDens}\cdot (\text{length}_i \cdot \text{width}_i)$

### Comments

* Looking at matrix $B_{i,j,d}$, fixed $d$:
  * If both 0: same coord. of origin
  * Else: separated
    * **Note**: at most one '1' between $B_{i,j,d}$ and $B_{j,i,d}$
