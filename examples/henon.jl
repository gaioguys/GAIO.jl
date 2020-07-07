function henon()
    n = 20

    points = [
        [(x, -1.0) for x in LinRange(-1, 1, n)];
        [(x,  1.0) for x in LinRange(-1, 1, n)];
        [(-1.0, x) for x in LinRange(-1, 1, n)];
        [( 1.0, x) for x in LinRange(-1, 1, n)];
    ]

    f = x -> SVector(1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1])

    g = PointDiscretizedMap(f, points)
    partition = RegularPartition(Box(SVector(0.0, 0.0), SVector(1.0, 1.0)))
    boxset = partition[:]

    return relative_attractor(boxset, g, 20)
end
