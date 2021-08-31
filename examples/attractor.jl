using GAIO

# the Henon map
f(x) = (1 - 1.4*x[1]^2 + x[2], 0.3*x[1])

center, radius = (0.0, 0.0), (3.0, 3.0)
Q = Box(center, radius)
P = RegularPartition(Q)

F = BoxMap(f, Q)

steps = 20
@time A = relative_attractor(F, P[:], steps)

plot(A)


