using GAIO

# This example demonstrates how to get a vast speedup
# in your code using nvidia CUDA. The speedup factor
# increases exponentially with the complexity of the map.

# !!! For best results, ensure that your functions only use 
# 32-bit operations, as GPUs are not efficient with 64-bit.

const σ, ρ, β = 10.0f0, 28.0f0, 0.4f0

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

# GAIO can convert your BoxPartition to 32-bit automatically
# when you use GPU acceleration, but preferred are still 
# explicit 32-bit literals like
# center, radius = (0f0,0f0,25f0), (30f0,30f0,30f0)
center, radius = (0,0,25), (30,30,30)

P = BoxPartition(Box(center, radius), (128,128,128))

# All we need to do now is pass :gpu to the BoxMap command.
G = BoxMap(F, P, :gpu)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)
@time W = unstable_set!(G, P[x])
