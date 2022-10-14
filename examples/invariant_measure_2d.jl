using WGLMakie: plot!, Figure, Axis3
using GAIO

# the Henon map
a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(f, P)
A = relative_attractor(F, P[:], steps = 16)

T = TransferOperator(F, A)
λ, ev = eigs(T)

# Plot the map at a nice viewing angle
fig = Figure()
ax = Axis3(fig[1,1], azimuth = -3*pi/5)
plot!(ax, abs ∘ (x -> 0.1*x) ∘ ev[1])

fig
