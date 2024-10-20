using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    @testset "regular partition" begin
        partition = GridPartition(Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (16,8,8))
        @testset "empty" begin
            empty_boxset = BoxSet(partition)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0

            empty_boxset = subdivide(empty_boxset, 1)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0
        end
        @testset "full" begin
            full_boxset = cover(partition, :)
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^10

            full_boxset = subdivide(full_boxset, 1)
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^11
        end
        p1 = SVector(0.5, 0.5, 0.5)
        p2 = SVector(-0.5, 0.5, 0.5)
        p3 = SVector(0.5, -0.5, 0.5)
        p4 = SVector(0.51, 0.51, 0.51)
        @testset "boxsets created on points" begin
            box_set = cover(partition, (p1, p2, p3, p4))
            @test length(box_set) == 3
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
            n = 10
            for k in 1:n
                box_set = subdivide(box_set, (k%3)+1 )
            end
            @test length(box_set) == 3 * 2^n
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
        end
        @testset "boxsets created on boxes" begin
            @test cover(partition, GAIO.point_to_box(partition, p1)) == cover(partition, p1)
            boxes = [GAIO.point_to_box(partition, p) for p in (p1, p2, p3, p4)]
            box_set = cover(partition, boxes)
            box_set_points = cover(partition, (p1, p2, p3, p4))
            @test box_set == box_set_points
        end
        @testset "set operations" begin
            p1_box_set = cover(partition, p1)
            p1p2_box_set = cover(partition, (p1, p2))
            p3_box_set = cover(partition, p3)
            p2p3_box_set = cover(partition, (p2, p3))
            @test length(p1_box_set) == 1
            @test length(p1p2_box_set) == 2
            @test length(p3_box_set) == 1
            @test length(p2p3_box_set) == 2
            @test length(union(p1_box_set, p3_box_set)) == 2
            @test length(union(p1_box_set, p1p2_box_set)) == 2
            @test length(union(p1p2_box_set, p2p3_box_set)) == 3
            @test length(intersect(p1_box_set, p3_box_set)) == 0
            @test length(intersect(p1_box_set, p1p2_box_set)) == 1
            @test length(intersect(p2p3_box_set, p2p3_box_set)) == 2
            @test length(setdiff(p1_box_set, p3_box_set)) == 1
            @test length(setdiff(p1_box_set, p1p2_box_set)) == 0
            @test length(setdiff(p1p2_box_set, p1_box_set)) == 1
            union!(p1_box_set, p2p3_box_set)
            setdiff!(p1p2_box_set, p2p3_box_set)
            intersect!(p2p3_box_set, p3_box_set)
            @test length(p1_box_set) == 3
            @test length(p2p3_box_set) == 1
            @test length(p1p2_box_set) == 1
        end
        @testset "accessing boxes" begin
            B = cover(partition, (p1, p2, p3, p4))
            boxes = collect(B)
            @test boxes isa Vector{Box{3,Float64}}
            @test length(boxes) == 3

            centers = collect(box.center for box in B)
            @test centers isa Vector{SVector{3,Float64}}

            mat = reinterpret(reshape, Float64, centers)
            @test size(mat) == (3,3)
        end
    end
    @testset "tree partition" begin
        partition = TreePartition(Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 10)
        @testset "empty" begin
            empty_boxset = BoxSet(partition)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0

            empty_boxset = subdivide(empty_boxset, 1)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0
        end
        @testset "full" begin
            full_boxset = cover(partition, :)
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^10

            full_boxset = subdivide(full_boxset, 1)
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^11
        end
        p1 = SVector(0.5, 0.5, 0.5)
        p2 = SVector(-0.5, 0.5, 0.5)
        p3 = SVector(0.5, -0.5, 0.5)
        p4 = SVector(0.51, 0.51, 0.51)
        @testset "boxsets created on points" begin
            box_set = cover(partition, (p1, p2, p3, p4))
            @test length(box_set) == 3
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
            n = 10
            for k in 1:n
                box_set = subdivide(box_set, (k%3)+1 )
            end
            @test length(box_set) == 3 * 2^n
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
        end
        @testset "boxsets created on boxes" begin
            @test cover(partition, GAIO.point_to_box(partition, p1)) == cover(partition, p1)
            boxes = [GAIO.point_to_box(partition, p) for p in (p1, p2, p3, p4)]
            box_set = cover(partition, boxes)
            box_set_points = cover(partition, (p1, p2, p3, p4))
            @test box_set == box_set_points
        end
        @testset "set operations" begin
            p1_box_set = cover(partition, p1)
            p1p2_box_set = cover(partition, (p1, p2))
            p3_box_set = cover(partition, p3)
            p2p3_box_set = cover(partition, (p2, p3))
            @test length(p1_box_set) == 1
            @test length(p1p2_box_set) == 2
            @test length(p3_box_set) == 1
            @test length(p2p3_box_set) == 2
            @test length(union(p1_box_set, p3_box_set)) == 2
            @test length(union(p1_box_set, p1p2_box_set)) == 2
            @test length(union(p1p2_box_set, p2p3_box_set)) == 3
            @test length(intersect(p1_box_set, p3_box_set)) == 0
            @test length(intersect(p1_box_set, p1p2_box_set)) == 1
            @test length(intersect(p2p3_box_set, p2p3_box_set)) == 2
            @test length(setdiff(p1_box_set, p3_box_set)) == 1
            @test length(setdiff(p1_box_set, p1p2_box_set)) == 0
            @test length(setdiff(p1p2_box_set, p1_box_set)) == 1
            union!(p1_box_set, p2p3_box_set)
            setdiff!(p1p2_box_set, p2p3_box_set)
            intersect!(p2p3_box_set, p3_box_set)
            @test length(p1_box_set) == 3
            @test length(p2p3_box_set) == 1
            @test length(p1p2_box_set) == 1
        end
        @testset "accessing boxes" begin
            B = cover(partition, (p1, p2, p3, p4))
            boxes = collect(B)
            @test boxes isa Vector{Box{3,Float64}}
            @test length(boxes) == 3

            centers = collect(box.center for box in B)
            @test centers isa Vector{SVector{3,Float64}}

            mat = reinterpret(reshape, Float64, centers)
            @test size(mat) == (3,3)
        end
    end
end
