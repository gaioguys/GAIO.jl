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
μ = abs ∘ (x -> 0.1*x) ∘ ev[1]

# --- choose either Plots or Makie ---

using Plots: plot
# Plot a heatmap
plot(μ)

# ------------------------------------

using WGLMakie: plot!, Figure, Axis3
# Plot an interactive 3D bar plot at a nice viewing angle
fig = Figure()
ax = Axis3(fig[1,1], azimuth = -3*pi/5)
plot!(ax, μ)
fig

# ------------------------------------
