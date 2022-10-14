using WGLMakie: plot, Colorbar
using GAIO

# the Lorenz system
const σ, ρ, β = 10.0, 28.0, 0.4
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
f(x) = rk4_flow_map(v, x)

center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
F = BoxMap(f, P, no_of_points=200)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
W = unstable_set!(F, P[x])

T = TransferOperator(F, W)
(λ, ev) = eigs(T)

fig, ax, ms = plot(log ∘ abs ∘ ev[1])
Colorbar(fig[1,2], ms)

fig
