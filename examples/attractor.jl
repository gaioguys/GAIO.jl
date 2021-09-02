using GAIO

# the Henon map
a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0.0, 0.0), (3.0, 3.0)
P = BoxPartition(Box(center, radius))
F = BoxMap(f, P)
A = relative_attractor(F, P[:], steps = 16)

plot(A)
