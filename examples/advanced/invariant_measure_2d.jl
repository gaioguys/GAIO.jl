using GAIO

# the Henon map
const a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxGrid(Box(center, radius))
F = BoxMap(:adaptive, f, P)
S = cover(P, :)
A = relative_attractor(F, S, steps = 16)

T = TransferOperator(F, A, A)
λ, ev = eigs(T)
μ = abs ∘ ev[1]

# --- choose either Plots or Makie ---

using Plots: plot
# Plot a heatmap
plot(μ)

# ------------------------------------

using GLMakie
# Plot an interactive 3D bar plot at a nice viewing angle
fig = Figure()
ax = Axis3(fig[1,1], azimuth = -3*pi/5)
plot!(ax, μ)
fig

# ------------------------------------
