using GAIO

# the Henon map
const a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(:adaptive, f, P)
B = cover(P, :)

# We will modify the relative_attractor
# algorithm to compute the 'box-dimension'
# of the attractor. For well-behaved 
# fractal objects, this corresponds to 
# the more popular 'Hausdorff-dimension'

# For reference, the original 
# relative_attractor algorithm is shown: 

# function relative_attractor(F::BoxMap, B₀::BoxSet{Box{N,T}}; steps=12) where {N,T}
#     B = copy(B₀)
#      for k = 1:steps
#         B = subdivide(B, (k % N) + 1)
#         B = B ∩ F(B)
#     end
#     return B
# end

box_dim = 2
k = 1
ϵ = maximum(2 .* radius)
tol = eps() ^ (1/4)

while ϵ > tol
    B = subdivide(B, (k % 2) + 1)
    B = B ∩ F(B)

    N = length(B)
    c, r = first(B)        # grab one box from B
    ϵ = maximum(2 .* r)    # compute the side length of the box
    box_dim = log(N) / log(1/ϵ)

    k = k + 1
end

println("Estimated fractal dimension: $box_dim")
println("Number of iterations needed for computation: $k")
