using GAIO

# the four wing system
const a, b, d = 0.2, -0.01, -0.4
v((x,y,z)) = @. (a*x+y*z, d*y+b*x-z*y, -z-x*y) 
f(x) = rk4_flow_map(v, x, 0.01, 20)

center, radius = (0,0,0), (5,5,5)
P = BoxGrid(Box(center, radius), (2,2,2))
F = BoxMap(:grid, f, P)

B = cover(P, :)
A = relative_attractor(F, B, steps=21)

using WGLMakie
plot(A)

T = TransferOperator(F, A, A)
λ, ev, n_converged = eigs(T, nev=10000)

scatter(real.(λ),imag.(λ))

μ = log ∘ abs ∘ ev[1]

# --- choose either Plots or Makie ---

using Plots: plot, scatter
# Plot some 2D projections
p1 = plot(μ, projection = x->x[1:2])
p2 = plot(μ, projection = x->x[2:3])
plot(p1, p2, size = (1200,600))

# ------------------------------------

using GLMakie
# Plot an interactive 3D heatmap
fig, ax, ms = plot(μ)
Colorbar(fig[1,2], ms)
fig

# ------------------------------------

