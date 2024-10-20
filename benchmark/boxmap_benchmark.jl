using CUDA, SIMD, StaticArrays
using GAIO

# the Lorenz system
const σ, ρ, β = 10f0, 28f0, 0.4f0
const step_size, steps = 1f-2, 20i32

center, radius = (0f0,0f0,25f0), (30i32,30i32,30i32)
P = GridPartition(Box(center, radius), (128i32,128i32,128i32))
x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium

#v(u::V) where V = V((σ*(u[2]-u[1]), ρ*u[1]-u[2]-u[1]*u[3], u[1]*u[2]-β*u[3]))
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
f(x) = rk4_flow_map(v, x, step_size, steps)

F = BoxMap(f, P, :gpu)
#= @profview =# W = unstable_set!(F, P[x])
plot(W)

for accel in (nothing, :cpu,:gpu,), p in (40, 400, 4000, 40000)
    println("accel: $accel    p: $p    step_size: $step_size    steps: $steps")
    F = BoxMap(f, P, accel; no_of_points=p)
    W = unstable_set!(F, P[x])
    @time W = unstable_set!(F, P[x])
end
