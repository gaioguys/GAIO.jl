using GAIO, StaticArrays, Test

@testset "exported functionality" begin
    partition = RegularPartition(Box(SVector(0.,0.,0.,0.),SVector(1.,1.,1.,1.)))
    @test depth(partition) == 0
    @test dimension(partition) == 4

    n = 10
    for _ = 1:n partition = subdivide(partition) end
    @test depth(partition) == n
    @test dimension(partition) == 4


    partition = RegularPartition(Box(SVector(0.0,1.),SVector(1.,0.)),3)
    @test depth(partition) == 3
    @test Base.size(partition) == (4,2)
    @test prod(Base.size(partition)) == 2^depth(partition)

    center = SVector(0.0,0.0)
    radius = SVector(1.0,0.0)
    @test_throws ErrorException RegularPartition(Box(center,radius))

    int_box = Box(SVector(0,0),SVector(1,1))
    # this should either throw a clearer error message or convert int to float
    @test_throws MethodError RegularPartition(int_box)
end


@testset "internal functionality" begin
    partition = RegularPartition(Box(SVector(0.,0.,0.),SVector(1.,1.,1.)),5)
    @test size(GAIO.keys_all(partition)) == (2^depth(partition),)

    inside = SVector(0.5,0.5,0.5)
    left = SVector(-1.,-1.,-1.)
    right = SVector(1.,1.,1.)
    on_boundary_left = SVector(0.,0.,-1.)
    on_boundary_right = SVector(0.,1.,0.)
    outside_left = SVector(0.,0.,-2.)
    outside_right = SVector(0.,2.,0.)

    @test !isnothing(GAIO.point_to_key(partition,inside))
    @test !isnothing(GAIO.point_to_key(partition,left))
    @test isnothing(GAIO.point_to_key(partition,right))
    @test !isnothing(GAIO.point_to_key(partition,on_boundary_left))
    @test isnothing(GAIO.point_to_key(partition,on_boundary_right))
    @test isnothing(GAIO.point_to_key(partition,outside_left))
    @test isnothing(GAIO.point_to_key(partition,outside_right))

    key_inside = GAIO.point_to_key(partition,inside)
    key_left = GAIO.point_to_key(partition,left)
    @test typeof(GAIO.key_to_box(partition,key_inside)) <: typeof(partition.domain)
    @test GAIO.key_to_box(partition,key_inside) != GAIO.key_to_box(partition,key_left)

    # round trip
    point = SVector(0.3,0.3,0.3)
    key = GAIO.point_to_key(partition,point)
    box = GAIO.key_to_box(partition,key)
    @test GAIO.contains(box,point)
    key_2 = GAIO.point_to_key(partition,box.center)
    @test key == key_2

    point_2D = SVector(0.,0.)
    point_4D = SVector(0.,0.,0.,0.)
    @test_throws Exception GAIO.point_to_key(partition,point_2D)
    @test_throws Exception GAIO.point_to_key(partition,point_4D)

    @test_throws Exception GAIO.key_to_box(partition,-1)
    @test_throws Exception GAIO.key_to_box(partition,2^partiton.depth+1)
end
