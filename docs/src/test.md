test citation [subalg](@cite). 

```@example 1
using GAIO

# the Lorenz system
const σ, ρ, β = 10.0, 28.0, 0.4
v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)

# time-0.2 flow map
f(x) = rk4_flow_map(v, x)

center, radius = (0,0,25), (30,30,30)
Q = Box(center, radius)
P = GridPartition(Q, (256,256,256))
F = BoxMap(:adaptive, f, Q)

# equilibrium
x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)
S = cover(P, x)

W = unstable_set(F, S)
```
