using GAIO
using LinearAlgebra

# domain (-40,40)^n, 3^n roots in domain
g(x) = 100*x + x.^2 - x.^3 .- sum(x)
function Dg(x)
    n = length(x)
    100*I(n) + 2*Diagonal(x) - 3*Diagonal(x.^2) + ones(n,n)
end

dim = 3
center, radius = zeros(dim), 40*ones(dim)
P = BoxPartition(Box(center, radius))

S = cover(P, :)
R = cover_roots(g, Dg, S, steps=dim*8)

#using Plots: plot       # plot a 2D projection
using MakieCore
using WGLMakie: plot    # plot a 3D projection

plot(R)
