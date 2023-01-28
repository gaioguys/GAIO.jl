using GAIO

# the Henon map
a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = [0, 0], (3, 3)
P = TreePartition(Box(center, radius))
F = BoxMap(f, P)
S = cover(P, :)
A = relative_attractor(F, S, steps = 16)

using Plots: plot
#using WGLMakie: plot    # same result, just interactive

plot(A)
