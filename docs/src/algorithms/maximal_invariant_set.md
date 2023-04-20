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
f^{-1} (A) = A
```
is called _invariant_. 

A natural question to ask is "given a dynamical system, what is the maximal (in the sense of inclusion) set which is invariant?" The answer is given by the set ``\text{Inv} (Q)`` defined as follows. 

Let ``\mathcal{O}(x) = \left\{\ \ldots,ß, f^{-1}(x),\, x,\, f(x),\, \ldots right\}`` denote the _full orbit_ of a point ``x`` in the domain ``Q``. Then the set 
```math
\text{Inv} (Q) = \left\{ x \in Q : \mathcal{O} (x) \subset Q \right\}
```
is the set of all orbits contained entirely in ``Q``. This is precisely the maximal invariant set. 

```@docs
maximal_invariant_set
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

### References

[1] Zeng, S. On sample-based computations of invariant sets. _Nonlinear Dyn_ 94, 2613–2624 (2018). https://doi.org/10.1007/s11071-018-4512-7
