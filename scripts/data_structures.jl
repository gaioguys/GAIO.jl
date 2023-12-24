using GAIO
import CairoMakie
CairoMakie.activate!()

## Box

center = [1, 2.5]
radius = [0.5, 1.0]
Q = Box(center, radius)
fig,axis,p = plot(Q)
CairoMakie.limits!(axis, 0, 4, 0, 4)
fig
save("box.svg", fig)

volume(Q)
x = rand(2)
x ∈ Q

## BoxPartition
P = BoxPartition(Q, (10,20))
B = P[:]
empty(P)

X = [ 3*rand(2) for _ = 1:100 ]
C = P[X]
v = collect(C)
b = v[1]
b.center, b.radius

B ∪ C
B ∩ C
setdiff(B, C)

Ĉ = subdivide(C,1)





