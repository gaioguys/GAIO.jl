using GAIO

# Chua's circuit
const a, b, m0, m1 = 16.0, 33.0, -0.2, 0.01
v((x,y,z)) = (a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y)
f(x) = rk4_flow_map(v, x, 0.05, 5)

center, radius = (0,0,0), (12,3,20)
Q = Box(center, radius)
P = BoxPartition(Q, (128,128,128))
F = BoxMap(:montecarlo, f, P, no_of_points=200)

# computing the attractor by covering the 2d unstable manifold
# of two equilibria
x = [sqrt(-3*m0/m1), 0.0, -sqrt(-3*m0/m1)]     # equilibrium
W = unstable_set(F, P[[x, -x]])

# computing the eigenmeasures at the eigenvalues of largest real part
T = TransferOperator(F, W)
λ, ev, nconv = eigs(T; nev=5, which=:LR)

ev_seba = SEBA(ev)

using WGLMakie: plot!, Figure, Axis3, Cycled
fig = Figure()
ax = Axis3(fig[1,1], aspect=(1,1.2,1))

for i in 1:length(S)
    plot!(ax, BoxSet(S[i]), color=Cycled(i))
end

fig
