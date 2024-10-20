using GAIO

# simple one dimensional linear regression
function linreg(xs, ys)
    n = length(xs)
    n == length(ys) || throw(DimensionMismatch())

    sum_x, sum_y = sum(xs), sum(ys)
    sum_xy, sum_x2 = xs'ys, xs'xs

    m = ( n*sum_xy - sum_x*sum_y ) / ( n*sum_x2 - sum_x^2 )
    b = ( sum_x*sum_y - m*sum_x^2 ) / ( n*sum_x )

    return m, b
end

# the Henon map
const a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxGrid(Box(center, radius))
F = BoxMap(:adaptive, f, P)
B = cover(P, :)

# We will modify the relative_attractor
# algorithm to compute the 'box-dimension'
# of the attractor which is defined as
#   D = lim(ϵ→0) N(ϵ) / log(1/ϵ),
# where N(ϵ) is the number of boxes 
# required to cover the attractor. 
# For well-behaved fractal objects, 
# this corresponds to the more popular 
# 'Hausdorff-dimension'. 

# The method we use follows the one from 
# Russel et al.
# For small ϵ, we expect
#    N(ϵ) ∼ Ke⁻ᴰ
# for some K. Thus, writing
#    d(ϵ) = log(N(ϵ)) / log(1/ϵ)
# we expect that
#    d(ϵ) - D ∼ log(K) / log(1/ϵ). 

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

# headstart on the attractor
initial_steps = 16
B = relative_attractor(F, B; steps=initial_steps)

box_dim = Float64[]
ϵ = Float64[]

k = initial_steps + 1
tol = eps() ^ (1/4)

while k==initial_steps+1 || ϵ[end] > tol
    B = subdivide(B, (k % 2) + 1)
    B = B ∩ F(B)

    N = length(B)
    c, r = first(B)              # grab one box from B
    push!(ϵ, maximum(2 .* r))    # compute the side length of the box
    push!(box_dim, log(N) / log(1/ϵ[end]))

    k = k + 1
end

logϵ = 1 ./ log.(1 ./ ϵ)
logK, D = linreg(logϵ, box_dim)

println("Estimated fractal dimension: $D")
println("Number of iterations needed for computation: $k")
