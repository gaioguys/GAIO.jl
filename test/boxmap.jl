using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    f(x) = x .^ 2
    test_points = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    g = boxmap(f, test_points)
    @testset "basics" begin
        @test typeof(g) <: SampledBoxMap
        partition = RegularPartition(Box(SVector(0.0, 0.0), SVector(1.0, 1.0)), 10)
        p1 = SVector(0.0, 0.0)
        p2 = SVector(0.5, 0.0)
        p3 = SVector(0.0, -0.5)
        boxset = partition[(p1, p2, p3)]
        # re-implement a straight forward box map to have a ground truth
        # terrible implementation in any real scenario
        mapped_points = []
        for box in boxset
            push!(mapped_points, f(box.center .- box.radius))
            push!(mapped_points, f(box.center .+ box.radius))
            x = f(SVector(box.center[1] - box.radius[1], box.center[2] + box.radius[2]))
            push!(mapped_points, x)
            y = f(SVector(box.center[1] + box.radius[1], box.center[2] - box.radius[2]))
            push!(mapped_points, y)
        end
        image = partition[mapped_points]
        full_boxset = partition[:]
        mapped1 = g(boxset)
        mapped2 = g(boxset; target=boxset)
        mapped3 = g(boxset; target=image)
        mapped4 = g(boxset, target=full_boxset)

        @test !isempty(boxset) && !isempty(image)
        @test !isempty(mapped1) && !isempty(mapped2)
        @test !isempty(mapped3) && !isempty(mapped4)
        # easiest way right now to check for equality
        @test length(union(image, mapped1)) == length(intersect(image, mapped1))
        @test length(union(image, mapped2)) != length(intersect(image, mapped2))
        @test length(union(image, mapped3)) == length(intersect(image, mapped4))
        @test length(union(image, mapped4)) == length(intersect(image, mapped3))
        # check for subset
        @test length(intersect(boxset, mapped2)) == length(mapped2)
    end
    @testset "points in boxmaps" begin
        f(x) = x .^ 2
        test_points = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        g = boxmap(f, test_points)
        x = (-2.0, 3.0)
        y = SVector(4.0, 1)
        @test_throws MethodError g(x)
        @test_throws MethodError g(y)
    end
end
