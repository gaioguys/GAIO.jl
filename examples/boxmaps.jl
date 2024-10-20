using GAIO
using Plots

# We choose a simple but expanding map
const α, β, γ, δ, ω = 2., 9.2, 10., 2., 10.
f((x, y)) = (α + β*x + γ*y*(1-y), δ + ω*y)

midpoint = round.(Int, ( 1+(α+β+γ/4)/2, 1+(δ+ω)/2 ), RoundUp)
domain = Box(midpoint, midpoint)

P = GridPartition(domain, 2 .* midpoint)
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
p = plot!(p, boundary[:, 1], boundary[:, 2], linewidth=4, fill=(0, RGBA(0.,0.,1.,0.2)), color=RGBA(0.,0.,1.,0.4), lab="true image")


# Plot the discretized images, using various discretization techniques.
# Play around with the paramters! 
# See how many points are needed for a satisfying covering!
n_points = 32
F = BoxMap(:montecarlo, f, domain, n_points = n_points)
p = plot!(p, F(B), color=RGBA(1.,0.,0.,0.5), lab="$n_points MonteCarlo test points")

n_points = (6, 6)
F = BoxMap(:grid, f, domain, n_points = n_points)
p = plot!(p, F(B), color=RGBA(0.,1.,0.,0.5), lab="$(join(n_points, "x")) grid of test points")

F = BoxMap(:adaptive, f, domain)
p = plot!(p, F(B), color=RGBA(1.,0.,1.,0.5), lab="adaptive")

n_subintervals = (2, 2)
F = BoxMap(:interval, f, domain, n_subintervals = n_subintervals)
p = plot!(p, F(B), color=RGBA(1.,0.5,0.,0.5), lab="interval arithmetic with $(join(n_subintervals, "x")) subinterval grid")
