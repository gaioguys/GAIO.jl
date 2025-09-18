# Using GAIO with DynamicalSystems.jl

`BoxMap`s can also be generated via the `DynamicalSystems.jl` package: 
```@repl 1
using GAIO
c, r = (0.5, 0.5), (0.5, 0.5)
Q = Box(c, r)
f((x,y)) = (1-1.4*x^2+y, 0.3*x)   # the Hénon map
```

```@repl 1 
using DynamicalSystems: DiscreteDynamicalSystem
using StaticArrays

dynamic_rule(u::Vec, p, t) where Vec = Vec( f(u) ) # DynamicalSystems.jl syntax

u0 = SA_F64[0, 0]   # this is dummy data
p0 = SA_F64[0, 0]   # parameters, in case they are needed

system = DiscreteDynamicalSystem(dynamic_rule, u0, p0)

F̃ = BoxMap(system, Q)
```
The same works for a continuous dynamical system. 

!!! warning "Maps based on `DynamicalSystem`s cannot run on the GPU!"
    Currently, you must hard-code your systems, and cannot rely on `DifferentialEquations` or `DynamicalSystems` for GPU-acceleration. 

!!! warning "Check your time-steps!"
    GAIO.jl ALWAYS performs integration over **one** time unit for `ContinuousDynamicalSystem`s! To perform smaller steps, rescale your dynamical system accordingly!

```@repl 1
using DynamicalSystems: ContinuousDynamicalSystem
using OrdinaryDiffEq

function lorenz_dudx(u::Vec, p, t, Δt) where Vec
    x,y,z = u
    σ,ρ,β = p

    dudx = Vec(( σ*(y-x), ρ*x-y-x*z, x*y-β*z ))
    return Δt * dudx
end

Δt = 0.2
lorenz(u, p=p0, t=0) = lorenz_dudx(u, p, t, Δt)

u0 = SA_F64[0, 0, 0]
p0 = SA_F64[10, 28, 0.4]
diffeq = (alg = RK4(), dt = 1/20, adaptive = false)

lorenz_system = ContinuousDynamicalSystem(lorenz, u0, p0; diffeq)

Q̄ = Box((0,0,25), (30,30,30))
F̄ = BoxMap(lorenz_system, Q̄)
```
