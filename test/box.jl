using GAIO, StaticArrays, Test

@testset "exported functionality" begin
    center = SVector(0.0,0.1)
    radius = SVector(10.0,10.0)
    box = Box(center,radius)
    @test box.center == center
    @test box.radius == radius

    center = SVector(0,0,1)
    radius = SVector(1.0,0.1,1.0)
    box = Box(center,radius)
    @test typeof(box.center) <: typeof(box.radius)
    @test typeof(box.radius) <: typeof(box.center)
    @test !(typeof(box.center) <: typeof(center))

    center = SVector(0.,0.,0.)
    radius = SVector(1.,1.,1.)
    box = Box(center,radius)
    inside = SVector(0.5,0.5,0.5)
    left = SVector(-1.,-1.,-1.)
    right = SVector(1.,1.,1.)
    on_boundary_left = SVector(0.,0.,-1.)
    on_boundary_right = SVector(0.,1.,0.)
    outside_left = SVector(0.,0.,-2.)
    outside_right = SVector(0.,2.,0.)
    @test GAIO.contains(box,inside)
    @test GAIO.contains(box,box.center)
    #boxes are half open to the right
    @test GAIO.contains(box,left)
    @test !GAIO.contains(box,right)
    @test GAIO.contains(box,on_boundary_left)
    @test !GAIO.contains(box,on_boundary_right)
    @test !GAIO.contains(box,outside_left)
    @test !GAIO.contains(box,outside_right)


    center = SVector(0.0,0.0,0.0)
    radius = SVector(1.0,1.0)
    @test_throws Exception Box(center,radius)

    center = SVector(0.0,0.0)
    radius = SVector(1.0,-1.0)
    @test_throws Exception Box(center,radius)
end


@testset "internal functionality" begin
    box = Box(SVector(0.,0.,),SVector(1.,1.))
    point_3D = SVector(0.,0.,0.)
    point_int = SVector(1,1)
    @test_throws Exception GAIO.contains(box,point_3D)
    @test_throws Exception GAIO.contains(box,point_int)
end
