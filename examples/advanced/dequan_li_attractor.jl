using GAIO

# the Dequan Li system
const β = 0.19
v((x,y,z)) = ( -β*x + sin(y) , -β*y + sin(z), -β*z + sin(x))
f(x) = rk4_flow_map(v, x, 0.02, 40)

center, radius = (0,0,0), (5,5,5)
P = BoxPartition(Box(center, radius), ntuple(_ -> 500, 3))
F = BoxMap(f, P, no_of_points=24)

x = (0,0,0)  
Q = P[x]
@time W = unstable_set!(F, Q)
@time T = TransferOperator(F, W)
@time (λ, ev) = eigs(T, nev=1)

fig = plot(log∘abs∘ev[1], show_axis=false)
Makie.save("Dequan_Li.2.png", fig)  