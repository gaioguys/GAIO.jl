using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    partition = BoxTree(Box(SVector(0.0, 0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0, 1.0)))
    @testset "basics" begin
        @test depth(partition) == 1
        @test ndims(partition) == 4
    end
    @testset "subdivision" begin
        keys_to_subdivide = [(1, (1,1,1,1)), (2, (1,1,1,1)), (2, (2,1,1,1)), (3, (1,1,1,1)), (3, (2,2,1,1))]
        for key in keys_to_subdivide
            subdivide!(partition, key)
        end
        @test depth(partition) == 4
        @test ndims(partition) == 4
        @test size(partition) == (1, 2, 4, 4) 
    end
    @testset "internal functionality" begin
        #           test partition:
        #     1  ----------------------- 
        #       |           |           |
        #       |           |           |
        #       |           |           |
        #       |-----------|           |
        #       |     |     |           |
        #       |     |-----|           |
        #       |     |     |           |
        #    -1  ----------------------- 
        #       -1                      1 
        partition = BoxTree(Box(SVector(0.0, 0.0), SVector(1.0, 1.0)))
        keys_to_subdivide = [(1, (1,1)), (2, (1,1)), (3, (1,1)), (4, (2,1))]
        for key in keys_to_subdivide
            subdivide!(partition, key)
        end
        @testset "keys" begin
            @test length(keys(partition)) == length(partition)
        end
        inside = SVector(0.5, 0.5)
        left = SVector(-1.0, -1.0)
        right = SVector(1.0, 1.0)
        on_boundary_left = SVector(-0.2, -1.0)
        on_boundary_right = SVector(-0.2, 1.0)
        outside_left = SVector(0.0, -2.0)
        outside_right = SVector(0.0, 2.0)
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
            point = SVector(0.5, 0.5)
            key = GAIO.point_to_key(partition, point)
            box = GAIO.key_to_box(partition, key)
            @test point ∈ box
            key_2 = GAIO.point_to_key(partition, box.center)
            @test key == key_2
            point = SVector(-0.2, -0.2)
            key = GAIO.point_to_key(partition, point)
            box = GAIO.key_to_box(partition, key)
            @test point ∈ box
            key_2 = GAIO.point_to_key(partition, box.center)
            @test key == key_2
        end
        @testset "points with wrong dimension" begin
            point_1d = SVector(0.0)
            point_3d = SVector(0.0, 0.0, 0.0)
            @test_throws Exception GAIO.point_to_key(partition, point_1d)
            @test_throws Exception GAIO.point_to_key(partition, point_3d)
        end
        @testset "non existing keys" begin
            @test_throws BoundsError GAIO.key_to_box(partition, (1, (2,1)))
            @test_throws BoundsError GAIO.key_to_box(partition, (2, (-1,1)))
            @test_throws BoundsError GAIO.key_to_box(partition, (3, (2,1)))
        end
    end
end
