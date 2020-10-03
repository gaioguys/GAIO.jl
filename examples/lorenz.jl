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

function lorenz()
    grid = LinRange(-1, 1, 7)
    points = collect(Iterators.product(grid, grid, grid))
    g = PointDiscretizedMap(lorenz_f, points)

    domain = Box(SVector(0.0, 0.0, 27.0), SVector(30.0, 30.0, 40.0))
    partition = RegularPartition(domain, 24)

    rh = 28.0
    b = 0.4

    x0 = (sqrt(b*(rh-1)), sqrt(b*(rh-1)), rh-1)

    boxset = partition[x0]
    return unstable_set!(g, boxset)
end
