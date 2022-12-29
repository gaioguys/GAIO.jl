using LinearAlgebra, StaticArrays
using GLMakie: plot
using GAIO

# Dadras system
const a, b, c = 8.0, 40.0, 14.9
v((x,y,z,w)) = (a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)

# (scaled) Dadras system
# coordinate transformation x̃ = μ(x)
# with μ(x) = x * η(x), η(x) = 1 / (sqrt ∘ norm)(x)
#function v(q̃::SVector{4,T}) where {T}
#    η = (sqrt ∘ sum)(x -> x*x, q̃)    #norm(q̃)
#    q = q̃ .* η
#    η = 1 / η
#    ∇η = -q̃ .* (η ^ 2) ./ 2
#    vq = let x = q[1], y = q[2], z = q[3], w = q[4]
#        SVector{4,T}(a*x-y*z+w, x*z-b*y, x*y-c*z+x*w, -y)
#    end
#    vq̃ = η .* vq .+ q .* (∇η'vq)
#end

f(x) = rk4_flow_map(v, x, 0.01, 10)

domain = Box((0,0,0,0), (250,150,200,25))
P = BoxPartition(domain, (96,96,96,96))
equillibrium = P[Box((0,0,0,0), (0.1,0.1,0.1,0.1))]

F = BoxMap(:montecarlo, f, domain, no_of_points=1024)

W = unstable_set(F, equillibrium)
plot(W)
