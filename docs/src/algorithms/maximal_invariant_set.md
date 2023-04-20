# Maximal Invariant Set

### Mathematical Background

(TODO)

Defs: forward- backward- invariant, invariant

The following is an intuitive explanation of the method used to compute maximal invairant sets. Inspiration for this description is due to [1]. 

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
