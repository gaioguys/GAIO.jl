using LinearAlgebra, StaticArrays
using GLMakie: plot
using GAIO

# Dadras system
const a, b, c = 8.0, 40.0, 14.9
v((x,y,z,w)) = (a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)
f(x) = rk4_flow_map(v, x, 0.01, 10)

domain = Box((0,0,0,0), (250,150,200,25))
P = BoxPartition(domain, (128,128,128,128))
equillibrium = P[Box((0,0,0,0), (0.1,0.1,0.1,0.1))]

F = IntervalBoxMap(f, domain, no_subintervals=(8,4,8,2))

W = unstable_set(F, equillibrium)
plot(W)
