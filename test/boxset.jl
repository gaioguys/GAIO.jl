using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    @testset "regular partition" begin
        partition = BoxPartition(Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), depth=10)
        @testset "empty" begin
            empty_boxset = boxset_empty(partition)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0

            empty_boxset = subdivide(empty_boxset)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0
        end
        @testset "full" begin
            full_boxset = partition[:]
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^10

            full_boxset = subdivide(full_boxset)
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^11
        end
        p1 = SVector(0.5, 0.5, 0.5)
        p2 = SVector(-0.5, 0.5, 0.5)
        p3 = SVector(0.5, -0.5, 0.5)
        p4 = SVector(0.51, 0.51, 0.51)
        @testset "boxsets created on points" begin
            box_set = partition[(p1, p2, p3, p4)]
            @test length(box_set) == 3
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
            n = 10
            for _ in 1:n
                box_set = subdivide(box_set)
            end
            @test length(box_set) == 3 * 2^n
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
        end
        @testset "set operations" begin
            p1_box_set = partition[p1]
            p1p2_box_set = partition[(p1, p2)]
            p3_box_set = partition[p3]
            p2p3_box_set = partition[(p2, p3)]
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
    end
    @testset "tree partition" begin
        partition = TreePartition(Box(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)))
        @testset "empty" begin
            empty_boxset = boxset_empty(partition)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0

            empty_boxset = subdivide(empty_boxset)
            @test isempty(empty_boxset)
            @test length(empty_boxset) == 0
        end
        full_boxset = partition[:]
        @testset "full" begin
            @test !isempty(full_boxset)
            @test length(full_boxset) == 1
            n = 10
            for _ in 1:10
                full_boxset = subdivide(full_boxset)
            end
            @test !isempty(full_boxset)
            @test length(full_boxset) == 2^n
        end
        # this seems to be the easiest way right now
        # to get a full tree at a certain depth. Not
        # really very clean
        partition = full_boxset.partition
        p1 = SVector(0.5, 0.5, 0.5)
        p2 = SVector(-0.5, 0.5, 0.5)
        p3 = SVector(0.5, -0.5, 0.5)
        p4 = SVector(0.51, 0.51, 0.51)
        @testset "box sets created on points" begin
            box_set = partition[(p1, p2, p3, p4)]
            @test length(box_set) == 3
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
            n = 10
            for _ in 1:n
                box_set = subdivide(box_set)
            end
            @test length(box_set) == 3 * 2^n
            @test any(box -> p1 ∈ box, box_set)
            @test any(box -> p2 ∈ box, box_set)
            @test any(box -> p3 ∈ box, box_set)
            @test any(box -> p4 ∈ box, box_set)
        end
        @testset "set operations" begin
            p1_box_set = partition[p1]
            p1p2_box_set = partition[(p1, p2)]
            p3_box_set = partition[p3]
            p2p3_box_set = partition[(p2, p3)]
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
    end
end
