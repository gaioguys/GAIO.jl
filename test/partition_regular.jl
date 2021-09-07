using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    partition = BoxPartition(Box(SVector(0.0, 0.0, 0.0, 0.0),
                                     SVector(1.0, 1.0, 1.0, 1.0)))
    @testset "basics" begin
        @test dimension(partition) == 4
    end
    @testset "subdivision" begin
        n = 10
        for k in 1:n
            partition = subdivide(partition, (k%3)+1 )
        end
        @test dimension(partition) == 4
    end
    @testset "size" begin
        partition = BoxPartition(Box(SVector(0.0, 1.0), SVector(1.0, 1.0)), (4,2))
        @test size(partition) == (4, 2)
    end
    @testset "domain with zero radius" begin
        center = SVector(0.0, 0.0)
        radius = SVector(1.0, 0.0)
        @test_throws ErrorException Box(center, radius)
    end
end
@testset "internal functionality" begin
    partition = BoxPartition(Box(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)), (4,2,2))
    @testset "keys all" begin
        @test size(GAIO.keys_all(partition)) == (4*2*2,)
    end
    inside = SVector(0.5, 0.5, 0.5)
    left = SVector(-1.0, -1.0, -1.0)
    right = SVector(1.0, 1.0, 1.0)
    on_boundary_left = SVector(0.0, 0.0, -1.0)
    on_boundary_right = SVector(0.0, 1.0, 0.0)
    outside_left = SVector(0.0, 0.0, -2.0)
    outside_right = SVector(0.0, 2.0, 0.0)
    @testset "point to key" begin
        @test !isnothing(GAIO.point_to_key(partition, inside))
        @test !isnothing(GAIO.point_to_key(partition, left))
        @test isnothing(GAIO.point_to_key(partition, right))
        @test !isnothing(GAIO.point_to_key(partition, on_boundary_left))
        @test isnothing(GAIO.point_to_key(partition, on_boundary_right))
        @test isnothing(GAIO.point_to_key(partition, outside_left))
        @test isnothing(GAIO.point_to_key(partition, outside_right))
        key = GAIO.point_to_key(partition, partition.domain.center)
        @test typeof(key) <: GAIO.keytype(typeof(partition))
    end
    key_inside = GAIO.point_to_key(partition, inside)
    key_left = GAIO.point_to_key(partition, left)
    @testset "key to point" begin
        @test typeof(GAIO.key_to_box(partition, key_inside)) <: typeof(partition.domain)
        @test GAIO.key_to_box(partition, key_inside) != GAIO.key_to_box(partition, key_left)
    end
    @testset "roundtrip" begin
        point = SVector(0.3, 0.3, 0.3)
        key = GAIO.point_to_key(partition, point)
        box = GAIO.key_to_box(partition, key)
        @test point ∈ box
        key_2 = GAIO.point_to_key(partition, box.center)
        @test key == key_2
    end
    @testset "points with wrong dimension" begin
        point_2d = SVector(0.0, 0.0)
        point_4d = SVector(0.0, 0.0, 0.0, 0.0)
        @test_throws Exception GAIO.point_to_key(partition, point_2d)
        @test_throws Exception GAIO.point_to_key(partition, point_4d)
    end
    @testset "non existing keys" begin
        @test_throws Exception GAIO.key_to_box(partition, -1)
        @test_throws Exception GAIO.key_to_box(partition, 2^partiton.depth + 1)
    end
end
