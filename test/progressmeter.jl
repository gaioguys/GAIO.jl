using GAIO
using ProgressMeter
using IntervalArithmetic
using StaticArrays
using HostCPUFeatures
using Test

@testset "exported functionality" begin
    f(x) = x .^ 2
    test_points = [
        (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0),
        (-1.0,  0.0), (1.0, 0.0),  (0.0, -1.0), (0.0, 1.0)
    ]
    center = SVector(0.0, 0.0)
    radius = SVector(1.0, 1.0)
    domain = Box(center, radius)

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

    @testset "sampled" begin
        g = BoxMap(:pointdiscretized, f, domain, test_points)

        mapped1 = g(boxset; show_progress=true)

        @test !isempty(boxset) && !isempty(image)
        # easiest way right now to check for equality
        @test length(union(image, mapped1)) == length(intersect(image, mapped1))
    end
    @testset "interval" begin
        g = BoxMap(:interval, f, domain, n_subintervals=(1,1))

        mapped1 = g(boxset; show_progress=true)

        boxarr = collect(IntervalBox(c .± r ...) for (c,r) in boxset)
        image_arr = collect(Box(f(int)) for int in boxarr)
        image_set = cover(partition, image_arr)

        @test image_set == mapped1
    end
end 
