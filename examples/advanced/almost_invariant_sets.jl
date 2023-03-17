using GAIO

# Chua's circuit
const a, b, m0, m1 = 16.0, 33.0, -0.2, 0.01
v((x,y,z)) = (a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y)
f(x) = rk4_flow_map(v, x, 0.05, 5)

center, radius = (0,0,0), (12,3,20)
Q = Box(center, radius)
P = BoxPartition(Q, (128,128,128))
F = BoxMap(:montecarlo, f, P, n_points=200)

# computing the attractor by covering the 2d unstable manifold
# of two equilibria
x = [sqrt(-3*m0/m1), 0.0, -sqrt(-3*m0/m1)]     # equilibrium
S = cover(P, [x, -x])
W = unstable_set(F, S)
T = TransferOperator(F, W, W)

# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(size(T, 2))
λ, ev = eigs(T; nev=2, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

using WGLMakie: plot!, Figure, Axis3, Cycled

fig = Figure()
ax = Axis3(fig[1,1], aspect=(1,1.2,1))
plot!(ax, ev[2])
fig
