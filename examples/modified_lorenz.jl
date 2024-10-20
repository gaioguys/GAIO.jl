

using GAIO, GLMakie, Metal
const Box = GAIO.Box

const a, b = 5f0, 1f-1
modified_lorenz((x,y,z)) = ( y-x, -x*z + b*abs(z), x*y - a )
f(x) = rk4_flow_map(modified_lorenz, x, 1f-2, 800)

c = (0,0,0)
r = (25,25,25)
domain = Box(c, r)
P = GridPartition(domain, (4,4,4))

F = BoxMap(f, domain)
A = cover(P, :)

A = relative_attractor(F, A, steps=12)

F = BoxMap(:gpusampled, f, domain, 750)
A = chain_recurrent_set(F, A, steps=3)

F♯ = TransferOperator(F, A, A)
λs, μs, _ = eigs(F♯, nev=100)
μ = log ∘ abs ∘ μs[1]
ν = scale ∘ real ∘ μs[perm[4]]

threshhold = 21#-7.2
B = BoxSet(
    μ.partition, 
    Set(key for key in keys(μ) if abs(ν[key]) > threshhold)
)
ν = BoxMeasure(ν.partition, Dict(key=>val for (key,val) in pairs(ν) if key in B.set))

fig = Figure();
ax = Axis3(fig[1,1])
ms = plot!(ax, ν, colormap=(:viridis, 0.1))
Colorbar(fig[1,2], ms)
fig