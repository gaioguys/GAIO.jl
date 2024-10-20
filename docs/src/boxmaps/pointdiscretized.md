# `PointDiscretizedBoxMap`

A generalization of `MonteCarloBoxMap` and `GridBoxMap` can be defined as follows: 
1. we provide a "global" set of test points within the unit cube ``[-1,1]^d``. 
2. For each box `Box(c,r)`, we rescale the global test points to lie within the box by calculating `c .+ r .* p` for each global test point `p`. 

```@docs; canonical=false
PointDiscretizedBoxMap
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

P = GridPartition(domain, 2 .* midpoint)
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
using StaticArrays
# create a map that tests the vertices of a box
global_test_points = SVector{2,Float64}[
    (1,  1),
    (1, -1),
    (-1, 1),
    (-1, -1)
]
F = BoxMap(:pointdiscretized, f, domain, global_test_points)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.,0.,0.5), 
    lab="Images of test points along the vertices"
)

savefig("pointdiscretized.svg"); nothing # hide
```

![Point Discretized BoxMap](pointdiscretized.svg)
