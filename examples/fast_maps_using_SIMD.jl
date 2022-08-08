using GAIO

# This example demonstrates how to get a ~2x speedup
# in your code using your CPU's SIMD capabilities.

N = 3
const σ, ρ, β = 10.0, 28.0, 0.4

# By default, GAIO is set up to accept functions of the form
function f(x)
    # Some map, here we use the Lorenz equation
    dx = (
           σ * x[2] -    σ * x[1],
           ρ * x[1] - x[1] * x[3] - x[2],
        x[1] * x[2] -    β * x[3]
    )
    return dx
end

F(x) = rk4_flow_map(f, x)

# Internally, GAIO calls this function on a set of test points
# within the domain. This means many function calls have to be made.
# If your function only uses "basic" instructions, then it is
# possible to simultaneously apply 
# Single Instructions to Multiple Data (SIMD).
# This way less total function calls have to be made, saving time.

# To see which instructions are supported, refer to 
# https://github.com/eschnett/SIMD.jl.git

# All we need to do is pass :cpu to the BoxMap command.
center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
G = BoxMap(F, P, :cpu)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)
@time W = unstable_set!(G, P[x])
