using GAIO

# the Lorenz system
const σ, ρ, β = 10.0, 28.0, 0.4
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
f(x) = rk4_flow_map(v, x)

center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
F = BoxMap(:adaptive, f, P)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
S = cover(P, x)
W = unstable_set(F, S)

T = TransferOperator(F, W, W)
(λ, ev) = eigs(T)
μ = log ∘ abs ∘ ev[1]

# --- choose either Plots or Makie ---

using Plots: plot
# Plot some 2D projections
p1 = plot(μ, projection = x->x[1:2])
p2 = plot(μ, projection = x->x[2:3])
plot(p1, p2, size = (1200,600))

# ------------------------------------

using WGLMakie: plot, Colorbar
# Plot an interactive 3D heatmap
fig, ax, ms = plot(μ)
Colorbar(fig[1,2], ms)
fig

# ------------------------------------

