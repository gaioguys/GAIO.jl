# `IntervalBoxMap`

All of the above techniques provide a fast, efficient way to cover setwise images of boxes, but are not necessarily guaranteed to provide an complete covering. To avoid this as well as other numerical inaccuracies inherent in floating point arithmetic, one can use _interval arithmetic_ to guarantee a rigorous outer covering of box images. Interval arithmetic is a technique from _validated numerics_ which performs calculations while simultaneously recording the error of such calculations. A more detailed discussion and julia-implementation of interval arithmetic can be found in [`IntervalArithmetic.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl). 

```@docs; canonical=false
IntervalBoxMap
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
p = plot!(p, B, linewidth=4, fillcolor=RGBA(0.,0.4,1.,0.2), linecolor=RGBA(0.,0.4,1.,0.4), lab="Box")

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
F = BoxMap(:interval, f, domain)
p = plot!(
    p, F(B), 
    color=RGBA(1.,0.5,0.,0.5), 
    lab="interval arithmetic", 
    dpi=500 # hide
)

savefig("interval.png"); nothing # hide
```

![Interval Arithmetic BoxMap](interval.png)
