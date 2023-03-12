# Unstable Set

In the following we are presenting the algorithm to cover invariant manifolds within some domain ``Q``, which has to contain a fixed point.
!!! note "Note"
    For simplicity, we will explain the algorithm for the case of the *unstable manifold*. However one can compute the stable manifold as well by considering the boxmap describing the inverse map ``f^{-1}`` as input argument for the algorithm.

### Mathematical Background
The unstable manifold is defined as
```math
W^U(x_0) = \{x: \lim_{k \to - \infty} f^k(x) = x_0 \}
```
where ``x_0`` is a fixed point of ``f``.

The idea behind the algorithm to compute the unstable manifold can be explained in two steps. Before starting we need to identify a hyperbolic fixed point and the region ``Q``, which we are going to compute the manifold in. The region ``Q`` needs to be already partitioned into small boxes.
1. **initialization step** Since a fixed point is always part of the unstable manifold, we need to identify a small region/box containing this fixed point. This box may be known a-priori, or one can use the `relative_attractor` around a small region where one suspects a fixed point to exist. 
2. **continuation step** The small box containing the fixed point is then mapped forward by `F` and the boxes that are hit under the image are added to the box collection. Then those newly included boxes are mapped forward and the procedure is repeated until no new boxes are added. 

!!! warning "Note on Convergence"
    One might not be able to compute the parts of the unstable manifold whose preimage lies outside the domain ``Q``.
    Thus, it is important to choose ``Q`` large enough.

```@docs
unstable_set
```

### Example

```@example 1
using GAIO

# the Lorenz system
const σ, ρ, β = 10.0, 28.0, 0.4
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
f(x) = rk4_flow_map(v, x)

center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
F = BoxMap(:adaptive, f, P)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
S = cover(P, x)
W = unstable_set(F, S)

#using Plots: plot
using GLMakie: plot

fig, ax, ms = plot(W)

using GLMakie: save # hide
save("unstable_manifold.png", fig); nothing # hide
```

![Unstable manifold](unstable_manifold.png)

### Implementation

```julia
function unstable_set(F::BoxMap, B::BoxSet)
    B₀ = B
    B₁ = B
    while !isempty(B₁)
        B₁ = F(B₁)          # map the current interation forward
        setdiff!(B₁, B₀)    # remove boxes we've already seen
        union!(B₀, B₁)      # add the new boxes to the collection
    end
    return B₀
end
```
