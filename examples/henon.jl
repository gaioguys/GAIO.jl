using GAIO
using StaticArrays
using ForwardDiff
using LinearAlgebra
using Base.Cartesian

n = 20

grid = LinRange(-1, 1, 3)
points1 = collect(Iterators.product(grid, grid))

f = x -> SVector(1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1])
domain = Box(SVector(0.0, 0.0), SVector(1.0, 1.0))

g_adaptive = AdaptiveBoxMap(f, domain)
g_points_1 = PointDiscretizedMap(f, points1)

partition = RegularPartition(domain, 12)
point = ntuple(i->rand(2), 1)
B = partition[point]
g1B = g_points_1(B); length(g1B)
gaB = g_adaptive(B); length(gaB)
   #A = relative_attractor(boxset, g, 10)
    #gstar = TransferOperator(g, A)
    #λ, v = eigs(gstar)
B