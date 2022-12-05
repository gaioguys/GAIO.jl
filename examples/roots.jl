using GAIO
using ForwardDiff

# domain (-40,40)^n, 3^n roots in domain
g(x) = 100*x + x .^ 2 - x .^ 3 .- sum(x)
Dg = x -> ForwardDiff.jacobian(g, x)

dim = 3
center, radius = zeros(dim), 40*ones(dim)
P = BoxPartition(Box(center, radius))

R = cover_roots(g, Dg, P[:], steps=dim*8)

#using Plots: plot       # plot a 2D projection
using WGLMakie: plot    # plot a 3D projection

plot(R)