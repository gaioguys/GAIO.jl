using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    # a toy example with an easy attractor to test how well all the components
    # fit together
    f(x) = (x[1], x[2] * 0.5)
    test_points = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    center = (0.0, 0.0)
    radius = (1.0, 1.0)
    domain = Box(center, radius)
    g = PointDiscretizedBoxMap(f, domain, test_points)
    partition = BoxPartition(domain)
    n = 10
    dims = (32, 32)
    partition_at_depth_n = BoxPartition(domain, dims)
    rga = relative_attractor(g, partition[:], steps = n)
    unstable = unstable_set(g, partition_at_depth_n[:])
    # ground truths attractor and unstable set
    x_axis = [SVector(0, x) for x in range(-1, 1, length=100)]
    y_axis = [SVector(x, 0) for x in range(-1, 1, length=100)]
    gt_rga = partition_at_depth_n[y_axis]
    gt_unstable = partition_at_depth_n[x_axis]
    # make sure that the algorithms cover the ground truth (we won't have equality)
    @testset "relative attractor" begin
        @test length(gt_rga) > 0
        @test length(intersect(rga, gt_rga)) == length(gt_rga)
    end
    @testset "unstable set" begin
        @test length(gt_unstable) > 0
        @test length(intersect(unstable, gt_unstable)) == length(gt_unstable)
    end
end
