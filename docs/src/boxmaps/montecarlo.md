# `MonteCarloBoxMap`

The simplest technique for discretization is a Monte-Carlo approach: choose a random set of sample points within a box and record which boxes are hit by the point map. 

```@docs
MonteCarloBoxMap
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
p = plot(cover(P, :), linecolor=:black, fillcolor=nothing, lab="", leg=:outerbottom)

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
no_of_points = 128
F = BoxMap(:montecarlo, f, domain, no_of_points=no_of_points)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.,0.,0.5), 
    lab="$no_of_points MonteCarlo test points"
)

savefig("montecarlo.svg"); nothing # hide
```

![MonteCarlo BoxMap](montecarlo.svg)
