using GAIO
import CairoMakie
CairoMakie.activate!()

## Box

center = [1, 2.5]
radius = [0.5, 1.0]
box = Box(center, radius)
fig,axis,_ = plot(box)
CairoMakie.limits!(axis, 0, 4, 0, 4)
fig
save("box.svg", fig)

volume(box)

x = rand(2)
x ∈ box

## BoxSet
Ω = Box(center, radius)
B = BoxSet(Ω, (10,20))

X = [ center + rand(2).*radius for _ = 1:100 ]
B = B[X]




