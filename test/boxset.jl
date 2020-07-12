using GAIO, StaticArrays, Test

@testset "exported functionality" begin
    @testset "regular partition" begin
        partition = RegularPartition(Box(SVector(0.,0.,0.),SVector(1.,1.,1.)),10)

        empty_boxset = boxset_empty(partition)
        @test Base.isempty(empty_boxset)
        @test Base.length(empty_boxset) == 0

        empty_boxset = subdivide(empty_boxset)
        @test Base.isempty(empty_boxset)
        @test Base.length(empty_boxset) == 0

        full_boxset = partition[:]
        @test !Base.isempty(full_boxset)
        @test Base.length(full_boxset) == 2^10

        full_boxset = subdivide(full_boxset)
        @test !Base.isempty(full_boxset)
        @test Base.length(full_boxset) == 2^11

        p1 = SVector(0.5,0.5,0.5)
        p2 = SVector(-0.5,0.5,0.5)
        p3 = SVector(0.5,-0.5,0.5)
        p4 = SVector(0.51,0.51,0.51)
        box_set = partition[(p1,p2,p3,p4)]
        @test Base.length(box_set) == 3
        @test any(box -> GAIO.contains(box,p1), box_set)
        @test any(box -> GAIO.contains(box,p2), box_set)
        @test any(box -> GAIO.contains(box,p3), box_set)
        @test any(box -> GAIO.contains(box,p4), box_set)
        n = 10
        for _ in 1:n box_set = subdivide(box_set) end
        @test Base.length(box_set) == 3*2^n
        @test any(box -> GAIO.contains(box,p1), box_set)
        @test any(box -> GAIO.contains(box,p2), box_set)
        @test any(box -> GAIO.contains(box,p3), box_set)
        @test any(box -> GAIO.contains(box,p4), box_set)

        p1_box_set = partition[p1]
        p1p2_box_set = partition[(p1,p2)]
        p3_box_set = partition[p3]
        p2p3_box_set = partition[(p2,p3)]
        @test Base.length(p1_box_set)                           == 1
        @test Base.length(p1p2_box_set)                         == 2
        @test Base.length(p3_box_set)                           == 1
        @test Base.length(p2p3_box_set)                         == 2
        @test Base.length(union(p1_box_set,p3_box_set))         == 2
        @test Base.length(union(p1_box_set,p1p2_box_set))       == 2
        @test Base.length(union(p1p2_box_set,p2p3_box_set))     == 3
        @test Base.length(intersect(p1_box_set,p3_box_set))     == 0
        @test Base.length(intersect(p1_box_set,p1p2_box_set))   == 1
        @test Base.length(intersect(p2p3_box_set,p2p3_box_set)) == 2
        @test Base.length(setdiff(p1_box_set,p3_box_set))       == 1
        @test Base.length(setdiff(p1_box_set,p1p2_box_set))     == 0
        @test Base.length(setdiff(p1p2_box_set,p1_box_set))     == 1
        union!(p1_box_set,p2p3_box_set)
        setdiff!(p1p2_box_set,p2p3_box_set)
        intersect!(p2p3_box_set,p3_box_set)
        @test Base.length(p1_box_set)                           == 3
        @test Base.length(p2p3_box_set)                         == 1
        @test Base.length(p1p2_box_set)                         == 1
    end
    @testset "tree partition" begin
        partition = TreePartition(Box(SVector(0.,0.,0.),SVector(1.,1.,1.)))

        empty_boxset = boxset_empty(partition)
        @test Base.isempty(empty_boxset)
        @test Base.length(empty_boxset) == 0

        empty_boxset = subdivide(empty_boxset)
        @test Base.isempty(empty_boxset)
        @test Base.length(empty_boxset) == 0

        full_boxset = partition[:]
        @test !Base.isempty(full_boxset)
        @test Base.length(full_boxset) == 1

        n = 10
        for _ in 1:10 full_boxset = subdivide(full_boxset) end
        @test !Base.isempty(full_boxset)
        @test Base.length(full_boxset) == 2^n

        # this seems to be the easiest way right now
        # to get a full tree at a certain depth
        partition = full_boxset.partition
        p1 = SVector(0.5,0.5,0.5)
        p2 = SVector(-0.5,0.5,0.5)
        p3 = SVector(0.5,-0.5,0.5)
        p4 = SVector(0.51,0.51,0.51)
        box_set = partition[(p1,p2,p3,p4)]
        @test Base.length(box_set) == 3
        @test any(box -> GAIO.contains(box,p1), box_set)
        @test any(box -> GAIO.contains(box,p2), box_set)
        @test any(box -> GAIO.contains(box,p3), box_set)
        @test any(box -> GAIO.contains(box,p4), box_set)
        n = 10
        for _ in 1:n box_set = subdivide(box_set) end
        @test Base.length(box_set) == 3*2^n
        @test any(box -> GAIO.contains(box,p1), box_set)
        @test any(box -> GAIO.contains(box,p2), box_set)
        @test any(box -> GAIO.contains(box,p3), box_set)
        @test any(box -> GAIO.contains(box,p4), box_set)

        p1_box_set = partition[p1]
        p1p2_box_set = partition[(p1,p2)]
        p3_box_set = partition[p3]
        p2p3_box_set = partition[(p2,p3)]
        @test Base.length(p1_box_set)                           == 1
        @test Base.length(p1p2_box_set)                         == 2
        @test Base.length(p3_box_set)                           == 1
        @test Base.length(p2p3_box_set)                         == 2
        @test Base.length(union(p1_box_set,p3_box_set))         == 2
        @test Base.length(union(p1_box_set,p1p2_box_set))       == 2
        @test Base.length(union(p1p2_box_set,p2p3_box_set))     == 3
        @test Base.length(intersect(p1_box_set,p3_box_set))     == 0
        @test Base.length(intersect(p1_box_set,p1p2_box_set))   == 1
        @test Base.length(intersect(p2p3_box_set,p2p3_box_set)) == 2
        @test Base.length(setdiff(p1_box_set,p3_box_set))       == 1
        @test Base.length(setdiff(p1_box_set,p1p2_box_set))     == 0
        @test Base.length(setdiff(p1p2_box_set,p1_box_set))     == 1
        union!(p1_box_set,p2p3_box_set)
        setdiff!(p1p2_box_set,p2p3_box_set)
        intersect!(p2p3_box_set,p3_box_set)
        @test Base.length(p1_box_set)                           == 3
        @test Base.length(p2p3_box_set)                         == 1
        @test Base.length(p1p2_box_set)                         == 1

        # partition = TreePartition(Box(SVector(0.,0.,0.),SVector(1.,1.,1.)))
        # box_set = partition[:]
        # @test Base.length(box_set) == 1
        # subdivide!(box_set,(0,1))
        # subdivide!(box_set,(1,1))
        # subdivide!(box_set,(2,2))
        # @test Base.length(box_set) == 7
        # @test_throws Exception subdivide!(box_set,(4,1))
        # @test_throws Exception subdivide!(box_set,(2,3))
    end
end
