using GAIO
import CairoMakie
CairoMakie.activate!()

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


