using GAIO
using StaticArrays
using Test

using Base.Sys
if isapple() && ARCH === :aarch64
    using Metal
else
    using CUDA
end

@testset "exported functionality" begin
    f(x) = x .^ 2
    test_points = SVector{2,Float32}[
        (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0),
        (-1.0,  0.0), (1.0, 0.0),  (0.0, -1.0), (0.0, 1.0)
    ]
    center = SA_F32[0,0]
    radius = SA_F32[1,1]
    domain = Box{2,Float32}(center, radius)
    g = BoxMap(:pointdiscretized, :gpu, f, domain, test_points)
    @testset "basics with :gpu" begin
        #@test typeof(g) <: GPUSampledBoxMap
        partition = BoxPartition(domain, (32,32))
        p1 = SVector(0.0, 0.0)
        p2 = SVector(0.5, 0.0)
        p3 = SVector(0.0, -0.5)
        boxset = cover(partition, (p1, p2, p3))
        # re-implement a straight forward box map to have a ground truth
        # terrible implementation in any real scenario
        mapped_points = []
        for box in boxset
            push!(mapped_points, f(box.center .- box.radius))
            push!(mapped_points, f(box.center .+ box.radius))
            x = f(SVector(box.center[1] .- box.radius[1], box.center[2] .+ box.radius[2]))
            push!(mapped_points, x)
            y = f(SVector(box.center[1] .+ box.radius[1], box.center[2] .- box.radius[2]))
            push!(mapped_points, y)
        end
        image = cover(partition, mapped_points)
        full_boxset = cover(partition, :)
        mapped1 = g(boxset)

        @test !isempty(boxset) && !isempty(image)
        # easiest way right now to check for equality
        @test length(union(image, mapped1)) == length(intersect(image, mapped1))
    end
end
