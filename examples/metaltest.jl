using GAIO
using Metal

const a, b, c, d, e, g, h = Float32.((0.2, -0.01, 1.0, -0.4, -1.0, -1.0, -1.0))
v((x,y,z)) = @. (a,d,e)*(x,y,z) + 
                    (0,b*x,0) + 
                    (c,h,g)*(y,z,x)*(z,x,y)

f(x) = rk4_flow_map(v, x)

center = (0., 0., 0.)
radius = (5., 5., 5.)
X = Box(center, radius)
P = BoxPartition(X, (128,128,128))
F = BoxMap(f, X)

B = cover(P, center)
A = unstable_set(F, B)

#ENV["JULIA_DEBUG"] = "all"

F = BoxMap(:gpusampled, f, X, 128)

#= @profview =# FA = F(A)

using GLMakie
fig = Figure();
ax = Axis3(fig[1,1], aspect=(1,1,1))
ms = plot!(ax, S)
fig




f(x) = rk4_flow_map(v, x, 1f-3, Int(10^2))
F = BoxMap(:montecarlo, f, X, n_points=128)
F = BoxMap(:gpusampled, f, X, 128)
@timev F(A)
