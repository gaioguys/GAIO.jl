# `SampledBoxMap`

We can even further generalize the concept of `MonteCarloBoxMap`, `GridBoxMap`, `PointDiscretizedBoxMap` as follows: we define two functions `domain_points(c, r)` and `image_points(c, r)` for any `Box(c, r)`. 
1. for each box `Box(c, r)` a set of test points within the box is initialized using `domain_points(C, r)` and mapped forward by the point map. 
2. For each of the pointwise images `fc`, an optional set of "perturbations" can be applied. These perturbations are generated with `image_points(fc, r)`. The boxes which are hit by these perturbations are recorded. 

```@docs
SampledBoxMap
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

P = BoxPartition(domain, 2 .* midpoint)
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

# we will recreate the AdaptiveBoxMap using SampledBoxMap
domain_points(center, radius) = sample_adaptive(f, center, radius)

# vertices of a box
vertex_test_points = SVector{2,Float64}[
    (1,  1),
    (1, -1),
    (-1, 1),
    (-1, -1)
]
image_points(center, radius) = (radius .* p .+ center for p in vertex_test_points)

F = BoxMap(:sampled, f, domain, domain_points, image_points)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.,0.,0.5), 
    lab="Recreation of AdaptiveBoxMap using SampledBoxMap"
)

savefig("sampled.svg"); nothing # hide
```

![Sampled BoxMap](sampled.svg)

### Example (continued)

```@setup 2
using GAIO
using Plots

# We choose a simple but expanding map
const α, β, γ, δ, ω = 2., 9.2, 10., 2., 10.
f((x, y)) = (α + β*x + γ*y*(1-y), δ + ω*y)

midpoint = round.(Int, ( 1+(α+β+γ/4)/2, 1+(δ+ω)/2 ), RoundUp)
domain = Box(midpoint, midpoint)

P = BoxPartition(domain, 2 .* midpoint)
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

```@repl 2
using StaticArrays # hide

# we will now extend AdaptiveBoxMap to use a set of "fallback" test points
fallback_points = SVector{2,Float64}[ 2 .* rand(2) .- 1 for _ in 1:30 ];

function domain_points(center, radius)
    try
        sample_adaptive(f, center, radius)
    catch exception
        (center .+ radius .* point for point in fallback_points)
    end
end

# vertices of a box
vertex_test_points = SVector{2,Float64}[(1,  1), (1, -1), (-1, 1), (-1, -1)] # hide
image_points(center, radius) = (radius .* p .+ center for p in vertex_test_points)

F = BoxMap(:sampled, f, domain, domain_points, image_points)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.,0.,0.5), 
    lab="Recreation of AdaptiveBoxMap using fallback points if an exception is thrown"
)

savefig("sampled2.svg"); nothing # hide
```

![Sampled BoxMap](sampled2.svg)

