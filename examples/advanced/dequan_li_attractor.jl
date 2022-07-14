using GAIO

# the Thomas system
const β = 0.19
v((x,y,z)) = ( -β*x + sin(y) , -β*y + sin(z), -β*z + sin(x))
f(x) = rk4_flow_map(v, x, 0.01, 40)

center, radius = (0,0,0), (5,5,5)
P = BoxPartition(Box(center, radius), (200,200,200))
F = BoxMap(f, P, no_of_points=20)

x = (0,0,0)      
W = unstable_set!(F, P[x])

plot(W)

T = TransferOperator(F, W)
(λ, ev) = eigs(T)

fig = plot(log∘abs∘ev[1], show_axis=false)
Makie.save("Dequan_Li.2.png", fig)  