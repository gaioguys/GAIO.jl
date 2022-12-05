using GAIO
using Test

@testset "exported functionality" begin
    # the Lorenz system
    σ, ρ, β = 10.0, 28.0, 0.4
    v((x,y,z)) = (σ*(y-x), ρ*x-y-x*z, x*y-β*z)
    f(x) = rk4_flow_map(v, x)

    center, radius = (0,0,25), (30,30,30)
    domain = Box(center, radius)
    x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)         # equilibrium

    @testset "BoxMap with linear indices" begin 
        P = BoxPartition(domain, (128,128,128); indextype=IndexLinear())
        F = BoxMap(f, domain)
        W = unstable_set(F, P[x])
        @test W isa BoxSet  # passes if no error is thrown
    end
    @testset "BoxMap with cartesian indices" begin 
        P = BoxPartition(domain, (128,128,128); indextype=IndexCartesian())
        F = BoxMap(f, domain)
        W = unstable_set(F, P[x])
        @test W isa BoxSet
    end
    @testset "AdaptiveBoxMap with linear indices" begin 
        P = BoxPartition(domain, (128,128,128); indextype=IndexLinear())
        F = AdaptiveBoxMap(f, domain)
        W = unstable_set(F, P[x])
        @test W isa BoxSet
    end
    @testset "AdaptiveBoxMap with cartesian indices" begin 
        P = BoxPartition(domain, (128,128,128); indextype=IndexCartesian())
        F = AdaptiveBoxMap(f, domain)
        W = unstable_set(F, P[x])
        @test W isa BoxSet
    end
end
