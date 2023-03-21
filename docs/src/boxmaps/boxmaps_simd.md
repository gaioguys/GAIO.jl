# `CPUSampledBoxMap`

Naturally, if an increase in accuracy is desired in a `SampledBoxMap`, a larger set of test points may be chosen. This leads to a dilemma: the more accurate we wish our approximation to be, the more we need to map very similar test points forward, causing a considerable slow down for complicated dynamical systems. However, the process of mapping each test point forward is completely independent on other test points. This means we do not need to perform each calculation sequentially; we can parallelize. 

If the point map only uses "basic" instructions, then it is possible to simultaneously apply Single Instructions to Multiple Data (SIMD). This way multiple funnction calls can be made at the same time, increasing performance. For more details, see the [maximizing performance section](https://gaioguys.github.io/GAIO.jl/simd/). 

```@docs
GridBoxMap(c::Val{:simd}, map, domain::Box{N,T}; no_of_points) where {N,T}
MonteCarloBoxMap(c::Val{:simd}, map, domain::Box{N,T}; no_of_points) where {N,T}
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
n_points = 256
F = BoxMap(:montecarlo, :simd, f, domain, n_points = n_points)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.,0.,0.5), 
    lab="$n_points MonteCarlo test points"
)

savefig("simd.svg"); nothing # hide
```

![MonteCarlo BoxMap](simd.svg)
