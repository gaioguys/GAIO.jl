# Relative Attractor

### Mathematical Background
The set 
```math
A_Q = \bigcap_{k \geq 0} f^k(Q)
```
is called the global attractor relative to ``Q``.
The relative global attractor can be seen as the set which is eventually approached by every orbit originating in ``Q``. In particular, ``A_Q`` contains each invariant set in ``Q`` and therefore all the potentially interesting dynamics. 
The idea of the algorithm is to cover the relative global attractor with boxes and recursively tighten the covering by refining appropriately selected boxes.

Mathematically, the algorithm to compute the global attractor relative to ``Q`` takes two input arguments: a compact set ``Q`` as well as a map ``f``, which describes the dynamics. Now in each iteration, two steps happen:
1. **subdivision step:** The box set `B` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, with double the amount of boxes. This is saved in `B`. 
2. **selection step:** All those boxes `b` in the new box set `B` whose image does not intersect the domain, ie ``f(b) \cap \left( \bigcup_{b' \in B} b' \right) \neq \emptyset``, get discarded. 

If we repeatedly refine the box set `B` through ``k`` subdivision steps, then as ``k \to \infty`` the collection of boxes ``B`` converges to the relative global attractor ``A_Q`` in the Hausdorff metric.

```@docs
relative_attractor
```

### Example

```@example
using GAIO

# the Henon map
a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(f, P)
S = cover(P, :)
A = relative_attractor(F, S, steps = 16)

using Plots: plot
#using WGLMakie: plot    # same result, just interactive

plot(A);
```
