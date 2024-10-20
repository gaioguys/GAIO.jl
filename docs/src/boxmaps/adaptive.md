# `AdaptiveBoxMap`

The above approaches may not necessarily be effective for covering the setwise image of a box. For choosing test points more effectively, we can use some knowledge of the Lipschitz matrix for ``f`` in a box `Box(c, r)`, that is, a matrix ``L \in \mathbb{R}^{d \times d}`` such that 
```math
| f(y) - f(z) | \leq L \, | y - z | \quad \text{for all } y, z \in \text{Box}(c, r),
```
where the operations ``| \cdot |`` and `` \leq `` are to be understood elementwise. The function `AdaptiveBoxMap` attempts to approximate ``L`` before choosing an adaptive grid of test points in each box, as described in [Junge.2000](@cite). 

```@docs; canonical=false
AdaptiveBoxMap
```

### Example

```@setup 1
using GAIO
using Plots

# We choose a simple but expanding map
const α, β, γ, δ, ω = 2., 9.2, 10., 2., 10.
f((x, y)) = (α + β*x + γ*y*(1-y), δ + ω*y)

midpoint = round.(Int, ( 1+(α+β+γ/4)/2, 1+(δ+ω)/2 ), RoundUp)
domain = Box(midpoint, midpoint)

P = BoxGrid(domain, 2 .* midpoint)
p = plot(cover(P, :), linewidth=0.5, fillcolor=nothing, lab="", leg=:outerbottom)

# unit box
B = cover(P, (0,0))
p = plot!(p, B, linewidth=4, fillcolor=RGBA(0.,0.,1.,0.2), linecolor=RGBA(0.,0.,1.,0.4), lab="Box")

# Plot the true image of B under f.
z = zeros(100)
boundary = [
    0       0;
    1       0;
    z.+1    0.01:0.01:1;
    0       1;
    z       0.99:-0.01:0;
]
b = f.(eachrow(boundary))
boundary .= [first.(b) last.(b)]
p = plot!(p, boundary[:, 1], boundary[:, 2], linewidth=4, fill=(0, RGBA(0.,0.,1.,0.2)), color=RGBA(0.,0.,1.,0.4), lab="True image under f")
```

```@repl 1
F = BoxMap(:adaptive, f, domain)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.,1.,0.5), 
    lab="adaptive test points"
)

savefig("adaptive.svg"); nothing # hide
```

![Adaptive BoxMap](adaptive.svg)
