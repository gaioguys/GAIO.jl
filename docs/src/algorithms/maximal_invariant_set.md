# Maximal Invariant Set

### Mathematical Background


We say that a set ``A`` is _forward invariant_ under the dynamics of ``f`` if 
```math
f (A) \subset A . 
```
Analogously we can define a _backward invariant_ set ``A`` as a set which satisfies
```math
f^{-1} (A) \subset A . 
```
Finally, a set which is both forward- and backward invariant, i.e. satisfying
```math
f (A) = f^{-1} (A) = A
```
is called _invariant_. 

A natural question to ask is "given a dynamical system, what is the maximal (in the sense of inclusion) set which is invariant?" The answer is given by the set ``\text{Inv} (Q)`` defined as follows. 

Let ``\mathcal{O}(x) = \left\{ \ldots,\, f^{-1}(x),\, x,\, f(x),\, \ldots \right\}`` denote the _full orbit_ of a point ``x`` in the domain ``Q``. Then the set 
```math
\text{Inv} (Q) = \left\{ x \in Q : \mathcal{O} (x) \subset Q \right\}
```
is the set of all orbits contained entirely in ``Q``. This is precisely the maximal invariant set. 

The idea of the algorithm [Dellnitz.Hohmann.1996](@cite) is to cover the desired set with boxes and recursively tighten the covering by refining appropriately selected boxes. The algorithm requires a `BoxMap` `F` as well as a `BoxSet` `B`, and performs two steps:
1. **subdivision step:** The box set `B` is subdivided once, i.e. every box is bisected along one axis, which gives rise to a new partition of the domain, with double the amount of boxes. This is saved as `B`. 
2. **selection step:** `B` is mapped forward under `F`. All boxes which do not satisfy the invariance condition ``F (B) = B = F^{-1} (B)`` are discarded, i.e. only the box set `C = F(B) ∩ B ∩ F⁻¹(B)` is kept. This set can be computed by considering the transfer graph `G` restricted to `B` (as described in [Chain Recurrent Set](@ref)). `C` is precisely the set of vertices of `G` which have both an incoming and outgoing edge. 

This algorithm can be analogously performed to find the _maximal forward invariant set_ by replacing the selection step with selecting `C = B ∩ F⁻¹(B)`, or the _maximal backward invariant set_ by selecting `C = F(B) ∩ B`. The astute documentation reader might notice that the latter is precisely the algorithm for the _relative attractor_. 

```@docs; canonical=false
maximal_invariant_set
maximal_forward_invariant_set
preimage
symmetric_image
```

### Example

```@example 1
using GAIO

# the Henon map
const a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = GridPartition(Box(center, radius))
F = BoxMap(f, P)
S = cover(P, :)
A = maximal_invariant_set(F, S, steps = 22)

using Plots: plot
#using WGLMakie: plot    # same result, just interactive

p = plot(A);

using Plots: savefig # hide
savefig("max_inv_set.svg"); nothing # hide
```

![Maximal Invariant Set](max_inv_set.svg)

### implementation

GAIO.jl makes subdivision-based algorithms as the one above very easy to implement. As demonstration, this is the code used for `maximal_invariant_set`:

```julia
function maximal_invariant_set(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
    # B₀ is a set of `N`-dimensional boxes
    B = B₀
    for k in 1:steps
        B = subdivide(B, (k % N) + 1)   # cycle through dimesions for subdivision
        B = symmetric_image(F, B, B)    # F(B) ∩ B ∩ F⁻¹(B)
    end
    return B
end
```
