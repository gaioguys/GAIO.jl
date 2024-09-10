# Using the GPU (Metal)

## Tutorial

This example demonstrates how to get a vast speedup in your code using Apple's M-series GPUs. The speedup factor increases exponentially with the complexity of the map.

Consider the point map `f`:
```julia
using GAIO
const σ, ρ, β = 10.0f0, 28.0f0, 0.4f0
function v(x)
    # Some map, here we use the Lorenz equation
    dx = (
           σ * x[2] -    σ * x[1],
           ρ * x[1] - x[1] * x[3] - x[2],
        x[1] * x[2] -    β * x[3]
    )
    return dx
end

# set f as 100 steps of the classic 4th order RK method
f(x) = rk4_flow_map(v, x, 0.002f0, 100)
```

!!! tip "Single vs Double Precision Arithmetic"
    You MUST ensure that your map uses entirely 32-bit arithmetic! Apple's Metal explicitly disallows 64-bit precision arithmetic. 

    GAIO can convert your `BoxPartition` to 32-bit automatically when you use GPU acceleration, but preferred are still explicit 32-bit literals like
    ```julia
    center, radius = (0f0,0f0,25f0), (30f0,30f0,30f0)
    ```
    instead of `center, radius = (0,0,25), (30,30,30)`. 

All we need to do is load the Metal.jl package and pass `:gpu` as the second argument to one of the two supported box map constructors: `BoxMap(:montecarlo, ...)`, `BoxMap(:grid, ...)`. Other box mapping algorithms are currently not supported. 
```julia
using CUDA

center, radius = (0f0,0f0,25f0), (30f0,30f0,30f0)
Q = Box(center, radius)
P = BoxPartition(Q, (128,128,128))
F = BoxMap(:montecarlo, :gpu, f, Q)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)
S = cover(P, x)
@time W = unstable_set(F, S)
```
