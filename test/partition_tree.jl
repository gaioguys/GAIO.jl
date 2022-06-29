using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    partition = TreePartition(Box(SVector(0.0, 0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0, 1.0)))
    @testset "basics" begin
        @test depth(partition) == 0
        @test ndims(partition) == 4
    end
    @testset "subdivision" begin
        keys_to_subdivide = [(0, 1), (1, 1), (1, 2), (2, 2), (2, 4)]
        for key in keys_to_subdivide
            subdivide!(partition, key)
        end
        @test depth(partition) == 3
        @test ndims(partition) == 4
    end
    @testset "domain with zero radius" begin
        center = SVector(0.0, 0.0)
        radius = SVector(1.0, 0.0)
        @test_throws DomainError Box(center, radius)
    end
end
