using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    @testset "basics" begin
        center = SVector(0.0, 0.1)
        radius = SVector(10.0, 10.0)
        box = Box(center, radius)
        @test box.center == center
        @test box.radius == radius
    end
    @testset "types" begin
        center = SVector(0, 0, 1)
        radius = SVector(1.0, 0.1, 1.0)
        box = Box(center, radius)
        @test typeof(box.center) <: typeof(box.radius)
        @test typeof(box.radius) <: typeof(box.center)
        @test !(typeof(box.center) <: typeof(center))
    end
    @testset "containment" begin
        center = SVector(0.0, 0.0, 0.0)
        radius = SVector(1.0, 1.0, 1.0)
        box = Box(center, radius)
        inside = SVector(0.5, 0.5, 0.5)
        left = SVector(-1.0, -1.0, -1.0)
        right = SVector(1.0, 1.0, 1.0)
        on_boundary_left = SVector(0.0, 0.0, -1.0)
        on_boundary_right = SVector(0.0, 1.0, 0.0)
        outside_left = SVector(0.0, 0.0, -2.0)
        outside_right = SVector(0.0, 2.0, 0.0)
        @test inside ∈ box
        @test box.center ∈ box
        #boxes are half open to the right
        @test left ∈ box
        @test right ∉ box
        @test on_boundary_left ∈ box
        @test on_boundary_right ∉ box
        @test outside_left ∉ box
        @test outside_right ∉ box
    end
    @testset "non matching dimensions" begin
        center = SVector(0.0, 0.0, 0.0)
        radius = SVector(1.0, 1.0)
        @test_throws Exception Box(center, radius)
    end
    @testset "nonpositive radii" begin
        center = SVector(0.0, 0.0)
        radius = SVector(1.0, -1.0)
        @test_throws DomainError Box(center, radius)
        center = SVector(0.0, 0.0)
        radius = SVector(1.0, 0.0)
        @test_throws DomainError Box(center, radius)
    end
end
@testset "internal functionality" begin
    box = Box(SVector(0.0, 0.0), SVector(1.0, 1.0))
    @testset "integer point in box" begin
        point_int_outside = SVector(2, 2)
        point_int_inside = SVector(0, 0)
        @test point_int_inside ∈ box
        @test point_int_outside ∉ box
    end
    @test_throws DimensionMismatch SVector(0.0, 0.0, 0.0) ∈ box
end
