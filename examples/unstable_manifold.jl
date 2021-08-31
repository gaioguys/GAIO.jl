using GAIO

center, radius = (0.0, 0.0, 25.0), (30.0, 30.0, 30.0)
Q = Box(center, radius)

# the Lorenz system
const σ, ρ, β = 10.0, 28.0, 0.4
function v((x,y,z))
    return (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
end
f(x) = rk4_flow_map(v, x)
F = BoxMap(f, Q)

depth = 21
P = RegularPartition(Q, depth)
x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium

W = unstable_set!(F, P[x])

plot(W, color = :red, figure = (resolution = (1200, 900),))
