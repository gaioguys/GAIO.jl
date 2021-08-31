using GAIO

center, radius = (0.0, 0.0), (1.0, 1.0)
Q = Box(center, radius)

# the Henon map
f(x) = (1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1])
F = BoxMap(f, Q)

P = RegularPartition(Q)
steps = 20
@time A = relative_attractor(F, P[:], steps)

plot(A)


