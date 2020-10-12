using GAIO
using BenchmarkTools

function henon()
    n = 10
    grid = LinRange(-1, 1, n)
    points = collect(Iterators.product(grid, grid))
    f(x) = (1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1])

    g = PointDiscretizedMap(f, points)
    domain = Box((0.0, 0.0), (1.0, 1.0))
    partition = RegularPartition(domain)
    boxset = partition[:]

    A = relative_attractor(g, boxset, 26)
    @btime map_boxes($g,$A)
    @btime map_boxes_new($g,$A)
end

henon()
