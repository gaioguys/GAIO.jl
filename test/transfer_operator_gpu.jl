using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    function f(u)   # the Baker transformation
        x, y = u
        if x < 0.5
            (2x, y/2)
        else
            (2x - 1, y/2 + 1/2)
        end
    end

    c = r = SA_F32[0.5, 0.5]
    domain = Box{3,Float32}(c, r)

    F = BoxMap(:grid, :gpu, f, domain)
    P = GridPartition(domain, (32,32))

    S = cover(P, (0,0))
    F♯ = TransferOperator(F, S)
    @test F♯.codomain == F(S)

    μ = BoxMeasure(S, (1,))
    @test BoxSet(μ) == S
    ν = F♯ * μ
    @test BoxSet(ν) == F(S)

    S = cover(P, :)
    F♯ = TransferOperator(F, S, S)
    
    λ, μs, nconv = eigs(F♯, tol=100*eps(), v0=ones(size(F♯,2)))
    @test λ[1] ≈ 1

    μ = μs[1]
    u = values(μ)
    @test all( u .≈ sum(u) / length(u) )
end
