using GAIO

n = 10
grid = LinRange(-1, 1, n)
points = collect(Iterators.product(grid, grid))

f(x) = (1/2 - 2*1.4*x[1]^2 + x[2], 0.3*x[1])
boxmap = PointDiscretizedMap(f, points)

domain = Box((0.0, 0.0), (1.0, 1.0))
partition = RegularPartition(domain)
boxset = partition[:]

steps = 18
A = relative_attractor(boxmap, boxset, steps)

plot(A)

