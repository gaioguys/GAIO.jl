using CUDA, SIMD
using GAIO

# the Lorenz system
const σ, ρ, β = 10f0, 28f0, 0.4f0
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
f(x) = rk4_flow_map(v, x)

center, radius = (0f0,0f0,25f0), (30i32,30i32,30i32)
P = BoxPartition(Box(center, radius), (128i32,128i32,128i32))
#F = BoxMap(f, P)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium
W = unstable_set!(F, P[x])

for accel in (nothing, :cpu), p in (40, 400, 4000)
    println("accel: $accel  p: $p")
    F = BoxMap(f, P, accel; no_of_points=p)
    W = unstable_set!(F, P[x])
    @time W = unstable_set!(F, P[x])
end
#plot(W)
