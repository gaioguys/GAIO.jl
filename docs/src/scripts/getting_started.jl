# the Henon map
a, b = 1.35, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

points = Vec{2, Float64}[]
@show x = rand(2)
for k = 1:10000
    @show x = f(x)
    push!(points, x)
end

using CairoMakie
CairoMakie.activate!()
p = scatter(points[10:end])
save("henon-simulation.svg", p)

using GAIO

center, radius = (0, 0), (3, 3)
Ω = Box(center, radius)
B = BoxSet(Ω, (4,4)) 
F = BoxMap(f, B)
A = chain_recurrent_set(F, B[:], steps = 15)

p = plot(A)
save("henon-attractor.svg", p)
