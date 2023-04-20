# Relative Attractor

### Mathematical Background
The set 
```math
A_Q = \bigcap_{k \geq 0} f^k(Q)
```
is called the global attractor relative to ``Q`` [1].
The relative global attractor can be seen as the set which is eventually approached by every orbit originating in ``Q``. In particular, ``A_Q`` contains each backward invariant set in ``Q``. 
The idea of the algorithm [1] is to cover the relative global attractor with boxes and recursively tighten the covering by refining appropriately selected boxes.

Mathematically, the algorithm to compute the global attractor relative to ``Q`` takes two input arguments: a compact set ``Q`` as well as a map ``f``, which describes the dynamics. Now in each iteration, two steps happen:
1. **subdivision step:** The box set `B` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, with double the amount of boxes. This is saved in `B`. 
2. **selection step:** All those boxes `b` in the new box set `B` whose image does not intersect the domain, ie ``f(b) \cap \left( \bigcup_{b' \in B} b' \right) = \emptyset``, get discarded. 

If we repeatedly refine the box set `B` through ``k`` subdivision steps, then as ``k \to \infty`` the collection of boxes ``B`` converges to the relative global attractor ``A_Q`` in the Hausdorff metric.

```@docs
relative_attractor
```

### Example

```@example 1
using GAIO

# the Henon map
const a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(f, P)
S = cover(P, :)
A = relative_attractor(F, S, steps = 22)

using Plots: plot
#using WGLMakie: plot    # same result, just interactive

p = plot(A);

using Plots: savefig # hide
savefig("relative_attractor.svg"); nothing # hide
```

![Relative attractor](relative_attractor.svg)

### Implementation

```julia
function relative_attractor(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    # B₀ is a set of `N`-dimensional boxes
    B = B₀
    for k = 1:steps
        B = subdivide(B, (k % N) + 1)   # cycle through dimesions for subdivision
        B = B ∩ F(B)
    end
    return B
end
```

### References

[1] Michael Dellnitz and Adreas Hohmann. “The Computation of Unstable Manifolds Using Subdivision”. In: _Nonlinear Systems and Chaos_. Ed. by Haim Brezis. Vol. 19. Progress in Nonlinear Differential Equations and their Applications. 1996, pp. 449–459. doi: https://doi.org/10.1007/978-3-0348-7518-9.

