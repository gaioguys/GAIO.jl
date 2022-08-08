using LinearAlgebra
using WGLMakie: plot
using GAIO

# the Dadras system
const a, b, c = 8.0, 40.0, 14.9
v((x,y,z,w)) = (a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y) ./ sqrt(norm((x,y,z,w)) + 1)
f(x) = rk4_flow_map(v, x)

center, radius = (0,0,0,0), (250,150,200,25)
P = BoxPartition(Box(center, radius), (128,128,128,128))
F = BoxMap(f, P, :cpu)

x = zeros(4)        # equilibrium
W = unstable_set!(F, P[x])

plot(W)

#T = TransferOperator(W)
#(λ, ev) = eigs(T)

#plot(log∘abs∘ev[1])
