# Box Dimension

### Mathematical Background

The box dimension, or Minkowski-Bouligand dimension, is a measure of fractal dimension for sets. Denote by ``N = N(\epsilon)`` the number of boxes with side length ``\epsilon`` required to cover a set ``A``. The box dimension is defined as 
```math
D = \underset{\epsilon \to 0}{\mathrm{lim\,inf}}\ \frac{\log N(\epsilon)}{\log (1/\epsilon)}. 
```
If the above limit exists, then we can equivalently write the asymptotic equation ``N(\epsilon) \sim K e^{-D}`` for some ``K``. Thus, writing ``d(\epsilon) = \log N(\epsilon) / \log (1/\epsilon)`` we have 
```math
d(\epsilon) - D \sim \frac{\log K}{\log (1/\epsilon)}.
```
The method used to compute ``D`` follows that of [1]: it is difficult to make ``d(\epsilon) - D`` small by shrinking ``\epsilon``. However, for small ``\epsilon`` the relationship between ``d(\epsilon)`` and ``1 / \log (1/\epsilon)`` will be approximately linear. Hence we extrapolate the value of ``d(\epsilon)`` for ``\epsilon \to 0`` by linear least-squares regression on ``d(\epsilon)`` vs ``1 / \log (1/\epsilon)``. 

Using this method we have everything we need to compute the box dimension for general objects. All that is required is a sequence of successively finer box sets which cover the object. 

```@docs
box_dimension
```

### Example

```@example 1
using GAIO

# the Henon map
const a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
Q = Box(center, radius)
P = BoxPartition(Q)
F = BoxMap(f, P)
S = cover(P, :)

box_dimension( relative_attractor(F, S, steps=k) for k in 1:20 )
```

This method is simple, however it is not the most efficient since in each iteration we have to recalculate the relative attractor to a new depth, even though we already calculated this. A slightly more complicated, but much more efficient method to do this is

```@example 1
# continuation of the example above

P = TreePartition(Q)
S = cover(P, :)

Base.@kwdef mutable struct SubdivisionIterator{B<:BoxSet}
    boxset::B
    step::Int = 1
    maxsteps::Int = 20
end

# Every time we iterate SubdivisionIterator, 
# perform one step of the relative_attractor algorithm
function Base.iterate(s::SubdivisionIterator, state...)
    s.step == s.maxsteps && return nothing
    s.boxset = relative_attractor(F, s.boxset, steps=1)
    s.step += 1
    return (s.boxset, s.step)
end

s = SubdivisionIterator(boxset = S)
box_dimension(s)
```

### References

[1] David A. Russell, James D. Hanson, and Edward Ott. “Dimension of Strange Attractors”. In: Phys. Rev. Lett. 45 (14 Oct. 1980), pp. 1175–1178. doi: 10.1103/PhysRevLett.45.1175. url: https://link.aps.org/doi/10.1103/PhysRevLett.45.1175
