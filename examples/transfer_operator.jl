using GAIO
using Plots: plot, plot!

# the unit box [-1, 1]²
domain = Box((0.0, 0.0), (1.0, 1.0))
partition = BoxPartition(domain, (16,8))

# left / right halves of the domain
left  = cover(partition, Box((-0.5, 0.0), (0.5, 1.0)))
right = cover(partition, Box((0.5, 0.0), (0.5, 1.0)))
full  = cover(partition, :)

# plot the BoxSets
p = plot(full, linecolor=black, fillcolor=nothing, lab="")
plot!(p, left, fillcolor=:blue)
plot!(p, right, fillcolor=:red)

# create measures with constant weight 1 per box
μ_left  = BoxFun(left, ones(n))
μ_right = BoxFun(right, ones(n))
μ_full  = BoxFun(full, ones(2n))

# vector space operations are supported for measures
μ_left + μ_right     ==  μ_full
μ_full - μ_left      ==  μ_right
μ_left - μ_full      == -μ_right
2*μ_left + 2*μ_right ==  μ_full + μ_full
μ_left/2 + μ_right/2 ==  μ_full/2

# horizontal translation map
f((x, y)) = (x+1, y)

# BoxMap which uses one sample point in the center of each box
F = BoxMap(:sampled, f, domain, center, center)

# compute the transfer operator over the domain
T = TransferOperator(H, full, full)

# Compute the pushforward / pullback measures by using the transfer operator
T*μ_left  == μ_right
T'μ_right == μ_left

# integration w.r.t. the measures
μ_full(domain) == volume(domain)

g(x) = 2x
sum(g, μ_full) == 2*volume(domain)

(2*μ_full)(domain) == 2*volume(domain)
