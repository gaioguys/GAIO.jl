using GAIO
using ForwardDiff
using StaticArrays

grid = LinRange(-1, 1, 30)
points1 = collect(Iterators.product(grid, grid))

f(x) = SA[1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1]]

domain = Box((0.0, 0.0), (1.0, 1.0))
partition = RegularPartition(domain, 12)
point = ntuple(i->rand(2), 1)
B = partition[point]

g1 = PointDiscretizedMap(f, points)
g2 = AdaptiveBoxMap(f, domain)

g1B = g1(B)
g2B = g2(B)

# and for the Lorenz system

@inline function rk4(f, x, τ)
    τ½ = τ/2

    k = f(x)
    dx = @. k/6

    k = f(@. x+τ½*k)
    dx = @. dx + k/3

    k = f(@. x+τ½*k)
    dx = @. dx + k/3

    k = f(@. x+τ*k)
    dx = @. dx + k/6

    return @. x+τ*dx
end

function lorenz_v(x)
    s = 10.0
    rh = 28.0
    b = 0.4
    return s * (x[2] - x[1]), rh * x[1] - x[2] - x[1] * x[3], x[1] * x[2] - b * x[3]
end

function lorenz_f(x)
    h = 0.01
    n = 20

    for i = 1:n
        x = rk4(lorenz_v, x, h)
    end

    return x
end

grid = LinRange(-1, 1, 20)
points = collect(Iterators.product(grid, grid, grid))

domain = Box((0.0, 0.0, 27.0), (30.0, 30.0, 40.0))
partition = RegularPartition(domain, 15)

rh = 28.0
b = 0.4
x0 = (sqrt(b*(rh-1)), sqrt(b*(rh-1)), rh-1)

boxset = partition[x0]

g_adaptive = AdaptiveBoxMap(lorenz_f, domain)
g_points_1 = PointDiscretizedMap(lorenz_f, points)

point = ntuple(i->60.0*rand(3).-30.0, 1)
B = partition[point]; @show length(B)

g1B = g_points_1(B); @show length(g1B)
gaB = g_adaptive(B); @show length(gaB)
@show g_adaptive.image_points(point, point)

