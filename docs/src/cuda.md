# Using the GPU (Nvidia)

## Tutorial

This example demonstrates how to get a vast speedup in your code using nvidia CUDA. The speedup factor increases exponentially with the complexity of the map.

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
    For best results, ensure that your functions only use 32-bit operations, as GPUs are not efficient with 64-bit.

    GAIO can convert your `BoxGrid` to 32-bit automatically when you use GPU acceleration, but preferred are still explicit 32-bit literals like
    ```julia
    center, radius = (0f0,0f0,25f0), (30f0,30f0,30f0)
    ```
    instead of `center, radius = (0,0,25), (30,30,30)`. 

All we need to do is load the CUDA.jl package and pass `:gpu` as the second argument to one of the box map constructors, eg. `BoxMap(:montecarlo, ...)`, `BoxMap(:grid, ...)`. 
```julia
using CUDA

center, radius = (0f0,0f0,25f0), (30f0,30f0,30f0)
Q = Box(center, radius)
P = BoxGrid(Q, (128,128,128))
F = BoxMap(:montecarlo, :gpu, f, Q)

x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)
S = cover(P, x)
@time W = unstable_set(F, S)
```

Using CUDA, one can achieve a more than 100-fold increase in performance. However, the performance increase is dependent on the complexity of the map `f`. For "simple" maps (eg. `f` from above with 20 steps), the GPU accelerated version will actually perform _worse_ because computation time is dominated by the time required to transfer data across the (comparatively slow) PCIe bus. The GPU accelerated version only beats the CPU accelerated version if `f` is set to use more than 40 steps. Hence it is highly recommended to use the GPU if the map `f` is not dominated by memory transfer speed, but not recommended otherwise. For more detail, see [GAIO.jl](@cite). 

![performance metrics](assets/flops_gpu_loglog.png)

## I get `InvalidIRError` due to `unsupported dynamic function invocation`

CUDA.jl generally give somewhat cryptic error messages. An `unsupported dynamic function invocation` can be caused by a simple error in the code. Hence, first try algorithms with `f` WITHOUT using the GPU, and ensure that no errors occur. 

If you still recieve dynamic function invocations, there likely is an operation somewhere in `f` which is not supported in CUDA.jl. A deliberately unsupported function can be for example matrix factorization, matrix-matrix multiplication, etc. because this is typically a performance trap if done on a single GPU thread. One option for linear algebra based functions which cause unsupported dynamic function invocations is to use `StaticArrays`. `StaticArrays` implements specialized methods for many low-dimensional linear algebra routines, allowing one to escape the standard methods which may cause unsupported dynamic function invocations. However, this is not a solution for all such problems, so a read through the [CUDA.jl documentation](https://cuda.juliagpu.org/stable/), [opening an issue on the GAIO.jl repo](https://github.com/gaioguys/GAIO.jl/issues), or posting a question on the [GPU category of julia Discourse](https://discourse.julialang.org/c/domain/gpu/) for help may be necessary. 
