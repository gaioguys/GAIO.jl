using GAIO
using ForwardDiff
using StaticArrays

# example functions
# domain (-5,5)^2
function example_g1(x)
    fac1 = x[1]*x[1] + x[2] - 11
    fac2 = x[1] + x[2]*x[2] - 7

    return SVector(4*x[1]*fac1 + 2*fac2, 2*fac1 + 4*x[2]*fac2)
end

# domain (-5,5)^2
function example_g2(x)
    return SVector(x[1]*x[1]*x[1] - 3*x[1]*x[2]*x[2] - x[1] + 1/sqrt(2),
                    -x[2]*x[2]*x[2] + 3*x[1]*x[1]*x[2] - x[2])
end

# demanding example, newton could diverge at some point
# domain (-3,3)^2
function example_g3(x)
    para = [
        map(i -> 0.1 + (-10 + i) / 1000, StaticArrays.SUnitRange(1, 20));
        map(i -> 0.9 + (-30 + i) / 1000, StaticArrays.SUnitRange(21, 40))
    ]

    return SVector(sin(4*x[2]) * prod(x[1] .- para), sin(4*x[1]) * prod(x[2] .- para))
end

# domain (-40,40)^n, 3^n roots in domain, 
function example_g4(x)
    return 100*x + x .^ 2 - x .^ 3 .- sum(x)
end

# domain (-0.3,0.8)^n
function example_g5(x)
    n = length(x)
    c = cos.(x)

    return StaticArrays.SUnitRange(1, n) .* (1 .- c) - sin.(x) .+ (n - sum(c))
end

dim = 3
g = example_g4
Dg = x -> ForwardDiff.jacobian(g, x)

center, radius = [0.0 for _ in 1:dim], [40.0 for _ in 1:dim]
P = BoxPartition(Box(center, radius))

R = cover_roots(g, Dg, P[:]; steps=dim*8)

plot(R)